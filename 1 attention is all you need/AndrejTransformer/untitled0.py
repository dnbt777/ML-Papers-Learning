import requests
import numpy as np
import torch


# new parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
max_iters = 3000 # max learning iterations
eval_interval = 300 # the interval of iterations over which to evaluate the loss
learning_rate = 1e-2
eval_iters = 200 # no idea...
n_embed = 32 # number of channels in each embedding
batch_size = 4 # how many sequences to train on
block_size = 8 # size of each sequence to train on

# get training data
if "response" not in locals():
    response = requests.get("https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt")
    text = response.text


# token scheme
tokens = sorted(list(set(text)))
vocab_size = len(tokens)
stoi = { char : i for i, char in enumerate(tokens)}
itos = { i : char for i, char in enumerate(tokens)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join([itos[x] for x in xs])

#print(encode(train_data[:10]))

# tensorify and encode data
data = torch.tensor(encode(text), dtype=torch.long)

# save 10% for validation
split = int(len(data)*0.9)
print(split)
train_data, validate_data = data[:split], data[split:]

torch.manual_seed(1337)

# get batch of data
def get_batch(split="train"):
    data = train_data if split == "train" else validate_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


# function to evaluate model after a few runs
@torch.no_grad() # means we dont gaf about back propogation
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'validate']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            out[split] = losses.mean()
            model.train()
    return out



# run through batches
xb, yb = get_batch()

for batch in range(batch_size):
    for t in range(block_size): # think of it as time series
        context = xb[batch, :t+1]
        target = yb[batch, t] # because yb is already shifted !!! durr
        print(f"input {context.tolist()} target {target}")

# bigram language model
import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) # head of the language model which takes in the embeddings?

    def forward(self, idx, targets=None):
        # get scores for next character in sequence
        token_embeddings = self.token_embedding_table(idx) # (Batch, Time, Channels (vocab size, 65 rn))
        logits = self.lm_head(token_embeddings) # (B, T, vocab_size)
        
        # B = batch = [24, 43, 58, 5, 57, 1, 46, 43]
        # T = time = [[24, 43, 58]
        # [24, 43, 58, 5]
        # [24, 43, 58, 5, 57]
        # [24, 43, 58, 5, 57, 1]
        # [24, 43, 58, 5, 57, 1, 46]
        # [24, 43, 58, 5, 57, 1, 46, 43]]
        # C = channels = embedding channels for each token

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx) # self.forward. get the channel/embeddings of the predictions of each sequence + its newly generated token, for each sequence (batch) in idxk
            # focus only on last time step
            logits = logits[:, -1, :] #becomes (B, C)
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) # batch and single prediction for each batch
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) # add prediction to the sequence in each batch!
        return idx # return extended (B, T) (i.e. a batch of sequences)

model = BigramLanguageModel(vocab_size)
logits, loss = model(xb, yb)
print(logits.shape)
print(loss)



# generate batch of starting sequences (batch size is just one in this case, and the sequence is [[0],], just a single starting token )
idx = torch.zeros((1, 1), dtype=torch.long)
# generate 100 new tokens from idx
generated = model.generate(idx, max_new_tokens=100)
# decoded generated tokens
decoded = decode(generated[0].tolist())
print(decoded)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) #1e-4 is the normal one

batch_size = 32
for steps in range(10000):
    # get a sample batch of data
    xb, yb = get_batch()

    # evaluate the loss
    logits, loss = model(xb, yb) # gets the loss and output logits
    optimizer.zero_grad(set_to_none=True) # resets the gradients to 0 that are stored in the optimizer
    loss.backward() # calculates the gradients for the current loss/training step
    optimizer.step() # moves the parameters by this amount

print(loss.item())

# generate batch of starting sequences (batch size is just one in this case, and the sequence is [[0],], just a single starting token )
idx = torch.zeros((1, 1), dtype=torch.long)
# generate 100 new tokens from idx
generated = model.generate(idx, max_new_tokens=10000)
# decoded generated tokens
decoded = decode(generated[0].tolist())
print(decoded)