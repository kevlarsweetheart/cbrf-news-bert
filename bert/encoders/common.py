from bert.record import Record
from bert.pad import pad_sequence
from bert.chop import chop


class WordInferInput(Record):
    __attributes__ = ['word_id', 'pad_mask']


class WordInferEncoder:
    def __init__(self, words_vocab, batch_size=8):
        self.words_vocab = words_vocab
        self.batch_size = batch_size

    def item(self, words):
        word_ids = []
        for word in words:
            word_id = self.words_vocab.encode(word.lower())
            word_ids.append(word_id)
        return word_ids

    def input(self, items):
        word_id = [word_ids for word_ids in items]
        word_id = pad_sequence(word_id, (self.words_vocab.pad_id, -1))
        pad_mask = word_id == self.words_vocab.pad_id
        return WordInferInput(word_id, pad_mask)

    def __call__(self, items):
        items = (self.item(_) for _ in items)
        chunks = chop(items, self.batch_size)
        for chunk in chunks:
            yield self.input(chunk)
