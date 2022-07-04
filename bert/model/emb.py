from torch import nn

from .base import Module


class Embedding(nn.Embedding, Module):
    def __init__(self, main_vocab_size, unknown_size, dim, pad_id):
        super(Embedding, self).__init__(main_vocab_size, unknown_size, dim, pad_id)
        self.dim = dim
