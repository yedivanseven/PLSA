import re
from typing import Iterable, Tuple, Callable

from nltk.stem import WordNetLemmatizer


__all__ = ['remove_non_ascii', 'to_lower', 'remove_numbers', 'remove_tags',
           'remove_punctuation', 'tokenize', 'remove_stopwords',
           'lemmatize_words', 'remove_short_words',
           'Str2StrT', 'StrIter2TupleT']

Str2StrT = Callable[[str], str]
StrIter2TupleT = Callable[[Iterable[str]], Tuple[str, ...]]


def remove_non_ascii(doc: str) -> str:
    return ''.join(char if ord(char) < 128 else ' ' for char in doc)


def to_lower(doc: str) -> str:
    return doc.lower()


def remove_numbers(doc: str) -> str:
    removed = filter(lambda character: not character.isdigit(), doc)
    return ''.join(removed)


def remove_tags(exclude_regex: str) -> Str2StrT:
    exclude_regex = re.compile(exclude_regex)

    def tag_remover(doc: str) -> str:
        return re.sub(exclude_regex, ' ', doc)

    return tag_remover


def remove_punctuation(punctuation: Iterable[str]) -> Str2StrT:
    translation = str.maketrans({character: ' ' for character in punctuation})

    def punctuation_remover(doc: str) -> str:
        return doc.translate(translation)

    return punctuation_remover


def tokenize(doc: str) -> Tuple[str, ...]:
    return tuple(doc.split())


def remove_stopwords(stopwords: Iterable[str]) -> StrIter2TupleT:

    def stopword_remover(doc: Iterable[str]) -> Tuple[str, ...]:
        removed = filter(lambda word: word not in stopwords, doc)
        return tuple(removed)

    return stopword_remover


def remove_short_words(min_word_len: int) -> StrIter2TupleT:

    def short_word_remover(doc: Iterable[str]) -> Tuple[str, ...]:
        removed = filter(lambda word: len(word) >= min_word_len, doc)
        return tuple(removed)

    return short_word_remover


def lemmatize_words(*incl_pos: str) -> StrIter2TupleT:
    pos_tag = {'JJ': 'a', 'VB': 'v', 'NN': 'n', 'RB': 'r'}
    lemmatize = WordNetLemmatizer().lemmatize

    def word_lemmatizer(doc: Iterable[str]) -> Tuple[str, ...]:
        filtered = filter(lambda tag: tag[1][:2] in incl_pos, doc)
        return tuple(lemmatize(f[0], pos_tag[f[1][:2]]) for f in filtered)

    return word_lemmatizer

