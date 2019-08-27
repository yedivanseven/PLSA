import string
from typing import Tuple
from functools import reduce
from .preprocessors import *

from nltk.corpus import stopwords
from nltk import pos_tag


DEFAULT_PIPELINE = (
    remove_non_ascii,
    to_lower,
    remove_numbers,
    remove_tags('<[^>]*>'),
    remove_punctuation(string.punctuation),
    tokenize,
    pos_tag,
    LemmatizeWords('NN'),  # any or all of 'JJ', 'VB', 'NN', 'RB'
    RemoveStopwords(stopwords.words('english') + ['nbsp', 'amp', 'urllink']),
    remove_short_words(3)
)


class Pipeline:
    def __init__(self, *preprocessors: Preprocessor) -> None:
        self.__pipeline = reduce(lambda f, g: lambda x: g(f(x)), preprocessors)
        enumerated = enumerate(preprocessors)
        self.__preprocessors = {self.__name(p): (p, i) for i, p in enumerated}

    def __repr__(self) -> str:
        title = self.__class__.__name__
        header = f'{title}:\n'
        divider = '=' * len(title) + '\n'
        enumerated = sorted(self.__preprocessors.items(), key=lambda x: x[1][1])
        body = (f'{i}: {name}' for name, (_, i) in enumerated)
        return header + divider + '\n'.join(body)

    def __getattr__(self, name) -> Preprocessor:
        return self.__preprocessors[name][0]

    def __getitem__(self, name) -> Preprocessor:
        return self.__preprocessors[name][0]

    def process(self, doc: str) -> Tuple[str, ...]:
        return self.__pipeline(doc)

    @staticmethod
    def __name(thing: object) -> str:
        try:
            name = thing.__name__
        except AttributeError:
            name = thing.__class__.__name__
        return name

