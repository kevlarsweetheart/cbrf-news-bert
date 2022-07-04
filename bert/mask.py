import torch

from .record import Record
from .pad import pad_bool_sequence


class Masked(Record):
    __attributes__ = ['value', 'mask']


def mask_like(_input):
    return torch.ones_like(_input, dtype=torch.bool)


def split_masked(_input, mask):
    sizes = mask.sum(dim=-1).tolist()
    return _input[mask].split(sizes)


def pad_masked(_input, mask, fill=0):
    seqs = split_masked(_input, mask)
    return pad_bool_sequence(seqs, fill)


def fill_masked(_input, mask, fill=0):
    return fill * mask + _input * ~mask
