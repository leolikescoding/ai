import torch

config = {}

##
config["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'

##
config["batch_size"] = 64

##
config["sentence_len"] = 32

#
config["learning_rate"] = 1e-3

#
config["max_iters"] = 500000
config["eval_interval"] = 100
config["eval_iters"] = 200

config["output_size"] = 32
config["layers_size"] = [20, 8, 4, 30]
config["layers_antenna_max_size"] = [1, 6, 5, 10] 
config["layers_antenna_probability_inv"] = [6, 4, 2]