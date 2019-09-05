"""Preprocessors for documents and words.

These preprocessors come in three flavours (functions, closures that return
functions, and classes defining callable objects). The choice for the respective
flavour is motivated by the complexity of the preprocessor. If it doesn't need
any parameters, a simple function will do. If it is simple, does not need to be
manipulated interactively, but needs some parameter(s), then a closure is fine.
If it would be convenient to alter parameters of the preprocessor interactively,
then a class is a good choice.

Preprocessors act either on an entire document string or, after splitting
documents into individual words, on an iterable over the words contained in a
single document. Therefore, they cannot be combined in arbitrary order but
care must be taken to ensure that the return value of one matches the
call signature of the next.

"""

__all__ = ['remove_non_ascii', 'to_lower', 'remove_numbers', 'remove_tags',
           'remove_punctuation', 'tokenize', 'RemoveStopwords',
           'LemmatizeWords', 'remove_short_words', 'PreprocessorT']

import re
from typing import Iterable, Tuple, Callable, Union, Iterator
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

StrOrIterT = Union[str, Iterable[str]]
Str2StrT = Callable[[str], str]
StrIter2TupleT = Callable[[Iterable[str]], Tuple[str, ...]]
PreprocessorT = Union[Str2StrT, StrIter2TupleT]


def remove_non_ascii(doc: str) -> str:
    """Removes non-ASCII characters (i.e., with unicode > 127) from a string.

    Parameters
    ----------
    doc: str
        A document given as a single string.

    Returns
    -------
    str
        The document as a single string with all characters of unicode > 127
        removed.

    """
    return ''.join(char if ord(char) < 128 else ' ' for char in str(doc))


def to_lower(doc: str) -> str:
    """Converts a string to all-lowercase.

    Parameters
    ----------
    doc: str
        A document given as a single string.

    Returns
    -------
    str
        The document as a single string with all characters
        converted to lowercase.

    """
    return str(doc).lower()


def remove_numbers(doc: str) -> str:
    """Removes digit/number characters from a string.

    Parameters
    ----------
    doc: str
        A document given as a single string.

    Returns
    -------
    str
        The document as a single string with all number/digit characters
        removed.

    """
    removed = filter(lambda character: not character.isdigit(), str(doc))
    return ''.join(removed)


def remove_tags(exclude_regex: str) -> Str2StrT:
    """Returns callable that removes matches to the given regular expression.

    Parameters
    ----------
    exclude_regex: str
        A regular expression specifying specific patterns to remove from a
        document.

    Returns
    -------
    function
        A callable that removes patterns matching the given regular expression
        from a string.

    """
    exclude_regex = re.compile(str(exclude_regex))

    def tag_remover(doc: str) -> str:
        """Removes matches to the given regular expression from a string.

        Parameters
        ----------
        doc: str
            A document given as a single string.

        Returns
        -------
        str
            The document as a single string with all matches to the pattern
            specified by `exclude_regex` removed.

        """
        return re.sub(exclude_regex, ' ', str(doc))

    return tag_remover


def remove_punctuation(punctuation: Iterable[str]) -> Str2StrT:
    """Returns callable that removes punctuation characters from a string."

    Parameters
    ----------
    punctuation: iterable of str
        An iterable over single-character strings specifying punctuation
        characters to remove from a document.

    Returns
    -------
    function
        A callable that removes the given punctuation characters from a string.

    """
    translation = str.maketrans({str(char): ' ' for char in punctuation})

    def punctuation_remover(doc: str) -> str:
        """ Removes the given punctuation characters from a string.

        Parameters
        ----------
        doc: str
            A document given as a single string.

        Returns
        -------
        str
            The document as a single string with all punctuation characters
            removed.

        """
        return str(doc).translate(translation)

    return punctuation_remover


def tokenize(doc: str) -> Tuple[str, ...]:
    """Splits a string into individual words.

    Parameters
    ----------
    doc: str
        A document given as a single string.

    Returns
    -------
    tuple of str
        The document as tuple of individual words.


    """
    return tuple(str(doc).split())


def remove_short_words(min_word_len: int) -> StrIter2TupleT:
    """Returns a callable that removes short words from an iterable of strings.

    Parameters
    ----------
    min_word_len: int
        Minimum number of characters in a word for it to be retained.

    Returns
    -------
    function
        A callable that removes words shorter than the given threshold from
        an iterable over strings.

    """

    def short_word_remover(doc: Iterable[str]) -> Tuple[str, ...]:
        """

        Parameters
        ----------
        doc: iterable of str
            A document given as an iterable over words.

        Returns
        -------
        tuple of str
            The document as tuple of strings with all words shorter than the
            given threshold removed.

        """
        removed = filter(lambda word: len(word) >= int(min_word_len), doc)
        return tuple(removed)

    return short_word_remover


class RemoveStopwords:
    """Instantiate callable objects that remove stopwords from a document.

    Parameters
    ----------
    stopwords: str or iterable of str
        Stopword(s) to remove from a document given as an iterable
        over words.

    Examples
    --------
    >>> from plsa.preprocessors import RemoveStopwords
    >>> remover = RemoveStopwords('is')
    >>> remover.words
    ('is',)

    >>> remover.words = 'the', 'are'
    >>> remover.words
    ('the', 'are')

    >>> remover += 'is', 'we'
    >>> remover.words
    ('is', 'we', 'the', 'are')

    >>> new_instance = remover + 'do'
    >>> new_instance.words
    ('are', 'we', 'is', 'do', 'the')

    """

    def __init__(self, stopwords: StrOrIterT) -> None:
        self.__stopwords = self.__normed(stopwords)

    def __repr__(self) -> str:
        title = self.__class__.__name__
        header = f'{title}:\n'
        divider = '=' * len(title) + '\n'
        return header + divider + str(self.__stopwords)

    def __call__(self, doc: Iterable[str]) -> Tuple[str, ...]:
        """Remove stopwords from a document given as iterable over words.

        Parameters
        ----------
        iterable of str
            A document given as an iterable over words.

        Returns
        -------
        tuple of str
            The document as tuple of strings with all words on the stopword
            list removed.

        """
        removed = filter(lambda word: str(word) not in self.__stopwords, doc)
        return tuple(removed)

    def __add__(self, stopword: StrOrIterT) -> 'RemoveStopwords':
        stopwords = tuple(set(self.__stopwords + self.__normed(stopword)))
        return RemoveStopwords(stopwords)

    def __iadd__(self, stopword: StrOrIterT) -> 'RemoveStopwords':
        self.__stopwords = tuple(set(self.__stopwords+self.__normed(stopword)))
        return self

    def __iter__(self) -> Iterator[str]:
        return iter(self.__stopwords)

    @property
    def words(self) -> Tuple[str, ...]:
        """The current stopwords."""
        return self.__stopwords

    @words.setter
    def words(self, stopwords: StrOrIterT) -> None:
        self.__stopwords = self.__normed(stopwords)

    def __normed(self, stopwords: StrOrIterT) -> Tuple[str, ...]:
        """Lowercase stopwords for both strings and iterables over strings."""
        if hasattr(stopwords, '__iter__') and not isinstance(stopwords, str):
            return tuple(set(map(lambda x: str(x).lower(), stopwords)))
        return str(stopwords).lower(),


class LemmatizeWords:
    """Instantiate callable objects that find the root form of words.

    Parameters
    ----------
    *inc_pos: str
        One or more positional tag(s) indicating the type(s) of words to retain
        and to find the root form of. Must be one of 'JJ' (adjectives), 'NN'
        (nouns), 'VB' (verbs), or 'RB' (adverbs).

    Raises
    ------
    KeyError
        If the given positional tags are not among the list of allowed ones.

    Examples
    --------
    >>> from plsa.preprocessors import LemmatizeWords
    >>> lemmatizer = LemmatizeWords('VB')
    >>> lemmatizer.types
    ('VB',)

    >>> lemmatizer.types = 'jj', 'nn'
    >>> lemmatizer.types
    ('JJ', 'NN')

    >>> lemmatizer += 'VB', 'NN'
    >>> lemmatizer.types
    ('JJ', 'NN', 'VB')

    >>> new_instance = lemmatizer + 'RB'
    >>> new_instance.types
    ('JJ', 'RB', 'NN', 'VB')

    """
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
        """Find root forms of the words in a given document.

        Parameters
        ----------
        doc: iterable over 2-tuples of str
            A document given as an iterable over over 2-tuples of strings with
            the first string a word and the second a positional tag. Only words
            with positional tags 'JJ', 'VB', 'NN', and/or 'RB' are retained.

        Returns
        -------
        tuple of str
            The document as tuple of strings with all words matching
            the given positional tag(s) replaced by their root form.

        """
        tagged = filter(lambda tag: tag[1][:2] in self.__incl_pos, pos_tag(doc))
        return tuple(self.__lemmatize(word[0], self.__pos_tag[word[1][:2]])
                     for word in tagged)

    def __add__(self, incl_pos: StrOrIterT) -> 'LemmatizeWords':
        incl_pos = tuple(set(self.__incl_pos + self.__checked(incl_pos)))
        return LemmatizeWords(*incl_pos)

    def __iadd__(self, incl_pos: StrOrIterT) -> 'LemmatizeWords':
        self.__incl_pos = tuple(set(self.__incl_pos + self.__checked(incl_pos)))
        return self

    def __iter__(self) -> Iterator[str]:
        return iter(self.__incl_pos)

    @property
    def types(self) -> Tuple[str, ...]:
        """The current type(s) of words to retain."""
        return self.__incl_pos

    @types.setter
    def types(self, incl_pos: StrOrIterT) -> None:
        self.__incl_pos = self.__checked(incl_pos)

    def __checked(self, incl_pos: StrOrIterT) -> Tuple[str, ...]:
        """Differentiate between a single string and an iterable of strings."""
        if hasattr(incl_pos, '__iter__') and not isinstance(incl_pos, str):
            return self.__check(*incl_pos)
        return self.__check(incl_pos)

    def __check(self, *tags: str) -> Tuple[str, ...]:
        """Convert pos tags to upper case and check if they are allowed."""
        tags = tuple(set(map(lambda x: str(x).upper(), tags)))
        for tag in tags:
            if tag not in self.__pos_tag:
                allowed_tags = tuple(self.__pos_tag.keys())
                msg = f'Unknown pos tag {tag}. Must be one of {allowed_tags}!'
                raise KeyError(msg)
        return tags
