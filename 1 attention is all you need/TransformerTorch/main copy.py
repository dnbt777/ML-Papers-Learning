from TransformerTorch.Transformer import *
import os
import time
import string
import random

def main():
    compression_test()

def compression_test():
    # Generate random strings
    control_data = [''.join(random.choices(string.ascii_lowercase, k=40)) for _ in range(64)]
    experimental_data = [compress_string(s) for s in control_data]

    
    print("\nExperimental Transformer Training:")
    experimental_transformer = train_transformer(experimental_data, padding=True)
    print("Control Transformer Training:")
    control_transformer = train_transformer(control_data)


def compress_string(s):
    compressed = ""
    i = 0
    while i < len(s):
        count = 1
        while i + 1 < len(s) and s[i] == s[i+1]:
            i += 1
            count += 1
        compressed += s[i] + str(count)
        i += 1
    return compressed



def train_transformer(data, padding=False):
    ### Prep data
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:",device)

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    transformer.to(device)

    # Convert string data to tensor
    if padding:
        max_length = max(len(s) for s in data)
        data = [s.ljust(max_length, ' ') for s in data]
    src_data = torch.tensor([list(map(ord, s)) for s in data])  # (batch_size, seq_length)
    src_data = src_data.to(device)
    tgt_data = torch.tensor([list(map(ord, s)) for s in data])  # (batch_size, seq_length)
    tgt_data = tgt_data.to(device)

    ### Train model
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()

    for epoch in range(100):
        start_time = time.time()
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}, Time: {end_time - start_time}s")

    return transformer





def standard():
    ### Prep data

    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:",device)

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    transformer.to(device)

    # Generate random sample data
    src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    src_data = src_data.to(device)
    tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    tgt_data = tgt_data.to(device)


    ### Train model

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()

    for epoch in range(100):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")