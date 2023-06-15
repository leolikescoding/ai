import sys
import torch
import torch.nn as nn
from torch.nn import functional as F


class Element(nn.Module):

    def __init__(self, input_dim, input_n, sentence_len):
        super().__init__()
        self.input_n = input_n
        self.input_dim = input_dim
        self.sentence_len = sentence_len

        self.output_ln = nn.LayerNorm(input_dim)

        self.key = nn.Linear(input_dim*input_n, input_dim, bias=False)
        self.query = nn.Linear(input_dim*input_n, input_dim, bias=False)
        self.value = nn.Linear(input_dim*input_n, input_dim, bias=False)

        self.register_buffer('tril', torch.tril(
            torch.ones(sentence_len, sentence_len)))
        # self.dropout = nn.Dropout(dropout)
        # self.proj = nn.Linear(input_dim, input_dim)

    def forward(self, input_list):
        if self.input_n != len(input_list):
            sys.exit('input_n != len(input_list)')

        x = torch.cat(input_list, -1)
        k = self.key(x)  # (B,T,H)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * self.input_dim ** -0.5
        T = self.sentence_len
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        # wei = self.dropout(wei)
 
        out = wei @ v
        # out = self.proj(out)
        return self.output_ln(out)


# test Element

def test_ele():
    batchsize = 4
    sentence_len = 5
    input_dim = 3

    input_n = 2


    input = torch.rand((batchsize, sentence_len, input_dim))

    #
    ele1 = Element(input_dim, 1, sentence_len)
    input_1 = ele1([input])

    #
    ele2 = Element(input_dim, 1, sentence_len)
    input_2 = ele2([input])

    #
    ele3 = Element(input_dim, 1, sentence_len)
    input_3 = ele3([input])


    ####
    ele4 = Element(input_dim, 3, sentence_len)
    out = ele4([input_1, input_2, input_3])

    print(out.shape)
