import torch

config = {}

##
config["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'

##
config["batch_size"] = 64

##
config["sentence_len"] = 32
