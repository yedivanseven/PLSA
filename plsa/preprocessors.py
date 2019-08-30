import re
from typing import Iterable, Tuple, Callable, Union, Iterator

from nltk.stem import WordNetLemmatizer


__all__ = ['remove_non_ascii', 'to_lower', 'remove_numbers', 'remove_tags',
           'remove_punctuation', 'tokenize', 'RemoveStopwords',
           'LemmatizeWords', 'remove_short_words', 'Preprocessor']

StrOrIter = Union[str, Iterable[str]]
Str2Str = Callable[[str], str]
StrIter2Tuple = Callable[[Iterable[str]], Tuple[str, ...]]
Preprocessor = Union[Str2Str, StrIter2Tuple]


def remove_non_ascii(doc: str) -> str:
    return ''.join(char if ord(char) < 128 else ' ' for char in doc)


def to_lower(doc: str) -> str:
    return doc.lower()


def remove_numbers(doc: str) -> str:
    removed = filter(lambda character: not character.isdigit(), doc)
    return ''.join(removed)


def remove_tags(exclude_regex: str) -> Str2Str:
    exclude_regex = re.compile(exclude_regex)

    def tag_remover(doc: str) -> str:
        return re.sub(exclude_regex, ' ', doc)

    return tag_remover


def remove_punctuation(punctuation: Iterable[str]) -> Str2Str:
    translation = str.maketrans({character: ' ' for character in punctuation})

    def punctuation_remover(doc: str) -> str:
        return doc.translate(translation)

    return punctuation_remover


def tokenize(doc: str) -> Tuple[str, ...]:
    return tuple(doc.split())


def remove_short_words(min_word_len: int) -> StrIter2Tuple:

    def short_word_remover(doc: Iterable[str]) -> Tuple[str, ...]:
        removed = filter(lambda word: len(word) >= min_word_len, doc)
        return tuple(removed)

    return short_word_remover


class RemoveStopwords:
    def __init__(self, stopwords: StrOrIter) -> None:
        self.__stopwords = self.__normed(stopwords)

    def __repr__(self) -> str:
        title = self.__class__.__name__
        header = f'{title}:\n'
        divider = '=' * len(title) + '\n'
        return header + divider + str(self.__stopwords)

    def __call__(self, doc: Iterable[str]) -> Tuple[str, ...]:
        removed = filter(lambda word: word not in self.__stopwords, doc)
        return tuple(removed)

    def __add__(self, stopword: StrOrIter) -> 'RemoveStopwords':
        stopwords = tuple(set(self.__stopwords + self.__normed(stopword)))
        return RemoveStopwords(stopwords)

    def __iadd__(self, stopword: StrOrIter) -> 'RemoveStopwords':
        self.__stopwords = tuple(set(self.__stopwords+self.__normed(stopword)))
        return self

    def __iter__(self) -> Iterator[str]:
        return iter(self.__stopwords)

    @property
    def words(self) -> Tuple[str, ...]:
        return self.__stopwords

    @words.setter
    def words(self, stopwords: StrOrIter) -> None:
        self.__stopwords = self.__normed(stopwords)

    def __normed(self, stopwords: StrOrIter) -> Tuple[str, ...]:
        if hasattr(stopwords, '__iter__') and not isinstance(stopwords, str):
            return tuple(set(map(lambda x: str(x).lower(), stopwords)))
        return str(stopwords).lower(),


class LemmatizeWords:
    def __init__(self, *incl_pos: str) -> None:
        self.__pos_tag = {'JJ': 'a', 'VB': 'v', 'NN': 'n', 'RB': 'r'}
        self.__incl_pos = self.__check(*incl_pos)
        self.__lemmatize = WordNetLemmatizer().lemmatize

    def __repr__(self) -> str:
        title = self.__class__.__name__
        header = f'{title}:\n'
        divider = '=' * len(title) + '\n'
        legend = ('\n\n'
                  'where:\n'
                  'JJ ... adjectives\n'
                  'VB ... verbs\n'
                  'NN ... nouns\n'
                  'RB ... adverb')
        return header + divider + str(self.__incl_pos) + legend

    def __call__(self, doc: Iterable[str]) -> Tuple[str, ...]:
        filtered = filter(lambda tag: tag[1][:2] in self.__incl_pos, doc)
        return tuple(self.__lemmatize(f[0], self.__pos_tag[f[1][:2]])
                     for f in filtered)

    def __add__(self, pos_tag: StrOrIter) -> 'LemmatizeWords':
        pos_tags = tuple(set(self.__incl_pos + self.__checked(pos_tag)))
        return LemmatizeWords(*pos_tags)

    def __iadd__(self, pos_tag: StrOrIter) -> 'LemmatizeWords':
        self.__incl_pos = tuple(set(self.__incl_pos + self.__checked(pos_tag)))
        return self

    def __iter__(self) -> Iterator[str]:
        return iter(self.__incl_pos)

    @property
    def types(self) -> Tuple[str, ...]:
        return self.__incl_pos

    @types.setter
    def types(self, pos_tag: StrOrIter) -> None:
        self.__incl_pos = self.__checked(pos_tag)

    def __checked(self, pos_tag: StrOrIter) -> Tuple[str, ...]:
        if hasattr(pos_tag, '__iter__') and not isinstance(pos_tag, str):
            return self.__check(*pos_tag)
        return self.__check(pos_tag)

    def __check(self, *tags: str) -> Tuple[str, ...]:
        tags = tuple(set(map(lambda x: str(x).upper(), tags)))
        for tag in tags:
            if tag not in self.__pos_tag:
                allowed_tags = tuple(self.__pos_tag.keys())
                msg = f'Unknown pos tag {tag}. Must be one of {allowed_tags}!'
                raise KeyError(msg)
        return tags
