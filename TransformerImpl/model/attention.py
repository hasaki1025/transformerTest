import math

import torch
from torch import nn


class DotProductAttention(nn.Module):

    def __init__(self, dropout) -> None:
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(query.shape[2])
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        return torch.bmm(
            torch.softmax(
                scores, dim=-1
            ), value
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dropout, query_size, key_size, value_size, num_hiddens) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens)
        self.W_k = nn.Linear(key_size, num_hiddens)
        self.W_v = nn.Linear(value_size, num_hiddens)
        self.W_out = nn.Linear(num_hiddens, num_hiddens)
        self.attention = DotProductAttention(dropout)

    def split(self, x):
        # TODO 是否需要考虑三维以上的情况
        shape = x.shape
        x = x.reshape(shape[0], shape[1], self.num_heads, -1)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, x.shape[2], x.shape[3])
        return x

    def unsplit(self, x):
        shape = x.shape
        x = x.reshape(-1, self.num_heads, shape[2], shape[3])
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)

    def forward(self, query, key, value, mask=None):
        query, key, value = self.W_q(query), self.W_k(key), self.W_v(value)
        out = self.attention(self.split(query), self.split(key), self.split(value), self.split(mask))
        return self.W_out(self.unsplit(out))
