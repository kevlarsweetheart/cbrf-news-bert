import re
import conf
import os
from collections import defaultdict

import stanza
import spacy_stanza

from bert.vocab import BERTVocab
from bert.const import (
    UNK, PAD,
    CLS, SEP,
    MASK
)


stanza.download('ru')
nlp = spacy_stanza.load_pipeline("ru")


def clean_text(_text):
    _text = re.sub("\n|(</?[^>]*>)", " ", _text)
    _text = re.sub(r"\s+", " ", _text)
    return _text


def lemmatize_text(_text):
    parsing_result = nlp(clean_text(_text))
    for token in parsing_result:
        yield token.lemma_


def get_vocab_items(path,
                    max_size=25000,
                    max_doc_frequency=0.8,
                    min_count=5):
    word2count = defaultdict(int)
    doc_n = 0

    # Посчитать количество документов, в которых употребляется каждое слово,
    # и общее количество документов
    _, _, filenames = next(os.walk(path))
    for _name in filenames:
        _text = ''
        with open(path + _name, 'r', encoding='cp1251') as _file:
            for line in _file:
                _text += ' ' + clean_text(line)
        if not _text:
            continue
        lemmatized_text = [lemma for lemma in lemmatize_text(clean_text(_text))]
        doc_n += 1
        unique_text_lemmas = set(lemmatized_text)
        for lemma in unique_text_lemmas:
            word2count[lemma] += 1

    # Убрать слишком редкие и слишком частые слова
    word2count = {word: cnt for word, cnt in word2count.items()
                  if cnt >= min_count and cnt / doc_n <= max_doc_frequency}

    # отсортировать слова по убыванию частоты
    sorted_word2count = sorted(word2count.items(),
                               reverse=True,
                               key=lambda _: _[1])
    if len(sorted_word2count) > max_size:
        sorted_word2count = sorted_word2count[:max_size]
    lemma_list = [_word for _word, _ in sorted_word2count]
    lemma_list += [CLS, SEP, MASK, UNK, PAD]

    return lemma_list, len(lemma_list), len(lemma_list) // 10 - 1


# tokens, _, unknown_size = get_vocab_items(conf.TEXT_FILES)
# bert_vocab = BERTVocab(tokens, unknown_size)


def encode_lemmatized(lemmatized_text, bert_vocab):
    ids = [bert_vocab.encode(lemma)
           for lemma in lemmatized_text]
    return ids
