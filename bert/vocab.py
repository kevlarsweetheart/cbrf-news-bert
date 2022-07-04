import mmh3

from dataset.utils import load_lines, dump_lines
from .record import Record

from .const import (
    UNK, PAD,
    CLS, SEP,
    MASK, UNK_SIZE
)


class BERTVocab(Record):
    __attributes__ = ['items', 'oov_size']

    def __init__(self, tokens, m):
        """
        :param tokens: список из топ-N частотных токенов, отсортированных по убыванию частоты встречаемости в корпусе
                       + служебные токены <cls>, <sep>, <mask>, <unk> и <pad> в конце
        :param m: число индексов, зарезервированных для неизвестных слов
        """
        super(BERTVocab, self).__init__(tokens, m)
        self.items = tokens
        self.oov_size = m
        self.token2id = {token: _id for _id, token in enumerate(tokens)}
        self.id2token = {self.token2id[token]: token for token in tokens}
        self.pad_id = self.token2id.get(PAD)
        self.oov_id = self.token2id.get(UNK)
        self.cls_id = self.token2id.get(CLS)
        self.sep_id = self.token2id.get(SEP)
        self.mask_id = self.token2id.get(MASK)

    def encode(self, token):
        """
        Для заданного токена получить индекс в таблице эмбеддингов.
        Если токена в словаре нет, возвращается пара хэшей токена по модулю M - числа индексов, зарезервированных
        для неизвестных слов (идея Bloom Embeddings).
        Если токен словарный, возвращается его индекс в словаре со смещением на M и -1.

        :param token: исходный токен
        :return: пара индексов в таблице эмбеддингов
        """
        putative_id = self.token2id.get(token, self.oov_id)
        if putative_id == self.oov_id:
            index1 = mmh3.hash(token, seed=0) % self.oov_size
            index2 = mmh3.hash(token, seed=1) % self.oov_size
            return index1, index2
        return putative_id + self.oov_size, -1

    def decode(self, id_pair):
        if id_pair[1] == -1:
            return self.id2token.get(id_pair[0])
        else:
            return UNK

    def __len__(self):
        return len(self.token2id)

    def __repr__(self):
        list_tokens = [token + ': ' + str(self.token2id[token]) for token in self.token2id.keys()]
        return '%s\n%s' % (self.__class__.__name__, '\n'.join(list_tokens))

    @classmethod
    def load(cls, path):
        items = list(load_lines(path))
        return cls(items, UNK_SIZE)

    def dump(self, path):
        dump_lines(self.items, path)
