import torch
import torch.nn as nn
import torch.optim as optim
from TransformerTorch.Transformer import Transformer
from TransformerTorch.generate_data import *
import os
import time
import string
import random
import matplotlib.pyplot as plt
import numpy as np
import time

SEQLENGTH = 20
EPOCHS = 500
SAMPLE_SIZE = 64



def log(*arrs, logfile=f"logs/log_{time.time():.0f}.txt"):
    print(*arrs)
    with open(logfile, 'a') as file:
        file.write(str(' '.join([str(s) for s in arrs])) + '\n')

def main():
    compression_test()

def compression_test():
    charset = "abcd"
    string_length = SEQLENGTH
    control_data = []
    samples = 0
    while samples < SAMPLE_SIZE:
        datapoint = generate_string(charset, length=string_length)
        if len(datapoint) == SEQLENGTH:
            samples += 1
            control_data.append(datapoint)
    experimental_data = [compress_string(s) for s in control_data]

    log("\nExample of training data:")
    log("Control Data Sample:", control_data[0])
    log("Experimental Data Sample:", experimental_data[0])
    log("Compression ratio:", len(experimental_data[0]) / len(control_data[0]))

    log("Control Transformer Training:")
    control_transformer, ctrl_losses, ctrl_times = train_transformer(control_data)
    log("\nExperimental Transformer Training:")
    experimental_transformer, exp_losses, exp_times = train_transformer(experimental_data, padding=True)

    plot_results(ctrl_losses, ctrl_times, exp_losses, exp_times)


def compress_string(s):
    compressed = ""
    i = 0
    while i < len(s):
        count = 1
        while i + 1 < len(s) and s[i] == s[i+1]:
            i += 1
            count += 1
        if count == 1:
            compressed += s[i]
        elif count == 2:
            compressed += s[i].upper()
        else:
            compressed += s[i] + str(count)
        i += 1
    return compressed

def train_transformer(data, padding=False):
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = SEQLENGTH
    dropout = 0.1
    epochs = EPOCHS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    transformer.to(device)

    if padding:
        max_length = max(len(s) for s in data)
        data = [s.ljust(max_length, ' ') for s in data]
    src_data = torch.tensor([list(map(ord, s)) for s in data]).to(device)
    tgt_data = torch.tensor([list(map(ord, s)) for s in data]).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()
    losses = []
    times = []
    cumulative_time = 0
    for epoch in range(epochs):
        start_time = time.time()
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        elapsed_time = end_time - start_time
        cumulative_time += elapsed_time
        losses.append(loss.item())
        times.append(cumulative_time)
        estimated_time_remaining = ((cumulative_time / (epoch + 1)) * epochs - cumulative_time)
        log(f"Epoch: {epoch+1}, Loss: {loss.item()}, Cumulative Time: {cumulative_time:.3f}s, ETA: {estimated_time_remaining:.3f}")

    return transformer, losses, times




def plot_results(ctrl_losses, ctrl_times, exp_losses, exp_times):
    # Convert losses to log scale
    ctrl_log_losses = np.log(ctrl_losses)
    exp_log_losses = np.log(exp_losses)

    # Determine suitable y-ticks based on the range of log(loss)
    min_loss = min(min(ctrl_log_losses), min(exp_log_losses))
    max_loss = max(max(ctrl_log_losses), max(exp_log_losses))
    y_ticks = np.linspace(min_loss, max_loss, num=10)  # Adjust num for more or fewer ticks

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Loss vs Cumulative Time
    axs[0].plot(np.cumsum(ctrl_times), ctrl_log_losses, label='Control Loss')
    axs[0].plot(np.cumsum(exp_times), exp_log_losses, label='Experimental Loss')
    axs[0].set_title('Log(Loss) vs Cumulative Time')
    axs[0].set_xlabel('Cumulative Time (s)')
    axs[0].set_ylabel('Log(Loss)')
    axs[0].set_yticks(y_ticks)
    axs[0].legend()

    # Loss vs Epoch
    axs[1].plot(ctrl_log_losses, label='Control Loss')
    axs[1].plot(exp_log_losses, label='Experimental Loss')
    axs[1].set_title('Log(Loss) vs Epoch')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Log(Loss)')
    axs[1].set_yticks(y_ticks)
    axs[1].legend()

    # Adding faint grey lines for each y-tick
    for ax in axs:
        for y in y_ticks:
            ax.axhline(y=y, color='grey', linewidth=0.5, linestyle='--')

    plt.show()



if __name__ == "__main__":
    main()