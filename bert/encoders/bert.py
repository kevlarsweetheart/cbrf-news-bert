import torch

from bert.chop import chop, chop_drop
from bert.batch import Batch
from bert.mask import Masked
from dataset.data import lemmatize_text

from .buffer import ShuffleBuffer


##########
#
#   MLM
#
########


class BERTMLMTrainEncoder:
    def __init__(self, vocab,
                 seq_len=512, batch_size=8, shuffle_size=1,
                 mask_prob=0.15):
        self.vocab = vocab
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.shuffle = ShuffleBuffer(shuffle_size)

        self.mask_prob = mask_prob

    def items(self, texts):
        for text in texts:
            try:
                lemmatized = lemmatize_text(text)
                for lemma in lemmatized:
                    yield self.vocab.encode(lemma)
            except Exception:
                print('Lemmatization error')
            finally:
                continue

    def seqs(self, items):
        for chunk in chop_drop(items, self.seq_len - 2):
            yield [(self.vocab.cls_id, -1)] + chunk + [(self.vocab.sep_id, -1)]

    def mask(self, _input):
        prob = torch.full((_input.shape[0], _input.shape[1]), self.mask_prob)

        spec = torch.tensor([[self.is_spec(id1, id2) for id1, id2 in seq]
                            for seq in _input]).bool()
        prob.masked_fill_(spec, 0)  # do not mask cls, sep and unk

        return torch.bernoulli(prob).bool()

    def batch(self, chunk):
        _input = torch.tensor(chunk).long()
        target = _input.clone()

        mask = self.mask(_input)
        _input[mask] = torch.tensor([self.vocab.mask_id, -1]).long()

        return Batch(_input, Masked(target, mask))

    def is_spec(self, id1, id2):
        return (id1 == self.vocab.cls_id or id1 == self.vocab.sep_id or id1 == self.vocab.oov_id) and id2 == -1

    def __call__(self, texts):
        items = self.items(texts)
        seqs = self.seqs(items)
        seqs = self.shuffle(seqs)
        chunks = chop(seqs, self.batch_size)
        for chunk in chunks:
            yield self.batch(chunk)
