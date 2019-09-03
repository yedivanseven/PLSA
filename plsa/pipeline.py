import string
from typing import Tuple
from functools import reduce
from .preprocessors import *

from nltk.corpus import stopwords


DEFAULT_PIPELINE = (
    remove_non_ascii,
    to_lower,
    remove_numbers,
    remove_tags('<[^>]*>'),
    remove_punctuation(string.punctuation),
    tokenize,
    LemmatizeWords('NN'),
    RemoveStopwords(stopwords.words('english') + ['nbsp', 'amp', 'urllink']),
    remove_short_words(3)
)


class Pipeline:
    """Encapsulates and applies multiple document preprocessors.

    Each preprocessor is assumed to be a callable that takes a single document
    as input and produces a single document as output. Importantly, each
    document fed to the first preprocessor in the chain is delivered as a
    single string, while the last preprocessor is required to return it
    as an iterable over strings with each element representing one word
    of that document.

    Other than that, preprocessors can be combined in any which way, provided
    that the return value of one matches the call signature of the next. The
    order in which they are applied is the order in which they are specified,
    `i.e.`, from left to right.


    Parameters
    ----------
    *preprocessors: callable
        Function(s) or other callable object(s) that each take a single
        document as input and produce a (processed) document as output.

    See Also
    --------
    plsa.preprocessors

    """
    def __init__(self, *preprocessors: PreprocessorT) -> None:
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

    def __getattr__(self, name) -> PreprocessorT:
        return self.__preprocessors[name][0]

    def __getitem__(self, name) -> PreprocessorT:
        return self.__preprocessors[name][0]

    def process(self, doc: str) -> Tuple[str, ...]:
        """Applies a chain of one or more preprocessors to a document.

        Parameters
        ----------
        doc: str
            A text document given as a single string.

        Returns
        -------
        tuple of str
            Each element represents one word of the document.

        """
        return self.__pipeline(str(doc))

    @staticmethod
    def __name(thing: object) -> str:
        try:
            name = thing.__name__
        except AttributeError:
            name = thing.__class__.__name__
        return name

