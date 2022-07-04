import torch
from torch import nn
from torch.nn import functional as F

from bert.record import Record
from .base import Module

from bert.const import (
    MAIN_VOCAB, UNK_SIZE,
    SEQ_LEN, EMB_DIM, HIDDEN_DIM,
    LAYERS_NUM, HEADS_NUM,
    DROPOUT, NORM_EPS, CPU
)


class BERTConfig(Record):
    __attributes__ = [
        'main_vocab_size', 'unknown_size', 'seq_len',
        'emb_dim', 'layers_num', 'heads_num',
        'hidden_dim', 'dropout', 'norm_eps'
    ]


class CBNewsBERTConfig(BERTConfig):
    def __init__(self,
                 main_vocab_size=MAIN_VOCAB,
                 unknown_size=UNK_SIZE,
                 seq_len=SEQ_LEN,
                 emb_dim=EMB_DIM,
                 layers_num=LAYERS_NUM,
                 heads_num=HEADS_NUM,
                 hidden_dim=HIDDEN_DIM,
                 dropout=DROPOUT,
                 norm_eps=NORM_EPS):
        super(CBNewsBERTConfig, self).__init__(
            main_vocab_size, unknown_size, seq_len, emb_dim,
            layers_num, heads_num, hidden_dim,
            dropout, norm_eps
        )


class BERTEmbedding(Module):
    def __init__(self,
                 main_vocab_size,
                 unknown_size,
                 seq_len,
                 emb_dim,
                 dropout=DROPOUT,
                 norm_eps=NORM_EPS):
        super(BERTEmbedding, self).__init__()
        self.vocab_size = main_vocab_size + unknown_size + 1
        self.word = nn.Embedding(num_embeddings=self.vocab_size,
                                 embedding_dim=emb_dim, padding_idx=-1)
        self.position = nn.Embedding(seq_len, emb_dim)
        self.norm = nn.LayerNorm(emb_dim, eps=norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, _input):
        idx1, idx2 = self.preprocess_input(_input)

        assert idx1.shape == idx2.shape, "Weird vocabulary index array shape"
        batch_size, seq_len = idx1.shape

        position = torch.arange(seq_len).expand_as(idx1).to(idx1.device)
        emb = self.word(idx1) + self.word(idx2) + self.position(position)
        emb = self.norm(emb)
        return self.dropout(emb)

    @classmethod
    def from_config(cls, config):
        return cls(
            config.main_vocab_size, config.unknown_size, config.seq_len, config.emb_dim,
            config.dropout, config.norm_eps
        )

    def preprocess_input(self, _input):
        input_idx1 = _input[:, :, 0]
        input_idx1 = input_idx1.to(torch.device(CPU))
        input_idx1.apply_(lambda _id: self.positive_ind(_id))
        input_idx1 = input_idx1.to(_input.device)

        input_idx2 = _input[:, :, 1]
        input_idx2 = input_idx2.to(torch.device(CPU))
        input_idx2.apply_(lambda _id: self.positive_ind(_id))
        input_idx2 = input_idx2.to(_input.device)
        return input_idx1, input_idx2

    def positive_ind(self, _index):
        return (self.vocab_size + _index) % self.vocab_size


def BERTLayer(emb_dim, heads_num, hidden_dim, dropout=DROPOUT, norm_eps=NORM_EPS):
    layer = nn.TransformerEncoderLayer(
        d_model=emb_dim,
        nhead=heads_num,
        dim_feedforward=hidden_dim,
        dropout=dropout,
        activation='gelu'
    )

    layer.norm1.eps = norm_eps
    layer.norm2.eps = norm_eps
    return layer


class BERTEncoder(Module):
    def __init__(self,
                 layers_num,
                 emb_dim,
                 heads_num,
                 hidden_dim,
                 dropout=DROPOUT,
                 norm_eps=NORM_EPS):
        super(BERTEncoder, self).__init__()
        self.layers = nn.ModuleList([
            BERTLayer(emb_dim, heads_num, hidden_dim, dropout, norm_eps) for _ in range(layers_num)
        ])

    @classmethod
    def from_config(cls, config):
        return cls(
            config.layers_num, config.emb_dim, config.heads_num, config.hidden_dim,
            config.dropout, config.norm_eps
        )

    def forward(self, _input, pad_mask=None):
        _input = _input.transpose(0, 1)  # seq_len x batch_size x emb_dim
        for layer in self.layers:
            _input = layer(_input, src_key_padding_mask=pad_mask)
        return _input.transpose(0, 1)


class BERTMLMHead(Module):
    def __init__(self, emb_dim, vocab_size, norm_eps=NORM_EPS):
        super(BERTMLMHead, self).__init__()
        self.linear1 = nn.Linear(emb_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim, eps=norm_eps)
        self.linear2 = nn.Linear(emb_dim, vocab_size)

    def forward(self, _input):
        x = self.linear1(_input)
        x = F.gelu(x)
        x = self.norm(x)
        return self.linear2(x)


class BERTMLM(Module):
    def __init__(self, emb, encoder, head):
        super(BERTMLM, self).__init__()
        self.emb = emb
        self.encoder = encoder
        self.head = head

    def forward(self, _input):
        x = self.emb(_input)
        x = self.encoder(x)
        return self.head(x)
