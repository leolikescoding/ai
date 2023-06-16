
from gen_data import get_batch
import torch.nn as nn
from config import config


print("config",config)

# train_data = get_batch("train")
# print(train_data[0].shape,train_data[1].shape)


class LanModelManual(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(config["vocab_size"], config["vocab_embed_size"])
        self.position_embedding_table = nn.Embedding(config["sentence_len"], config["vocab_embed_size"])
        
        # build manually
        