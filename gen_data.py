import torch

from config import config

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
config['vocab_size'] = len(chars)
config['vocab_embed_size'] = 64
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# encoder: take a string, output a list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: take a list of integers, output a string
def decode(l): return ''.join([itos[i] for i in l])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long, device=config["device"])
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(train_or_eval):
    # generate a small batch of data of inputs x and targets y
    data = train_data if train_or_eval == 'train' else val_data
    ix = torch.randint(len(
        data) - config["sentence_len"], (config["batch_size"],), device=config["device"])
    x = torch.stack([data[i:i+config["sentence_len"]] for i in ix])
    y = torch.stack([data[i+1:i+config["sentence_len"]+1] for i in ix])
    # x, y = x.to(device), y.to(device)
    return x, y
