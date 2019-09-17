import os
import csv

from collections import defaultdict
from typing import Iterable, Dict, Tuple
from numpy import zeros, ndarray, log, sign, abs

from .pipeline import Pipeline


class Corpus:
    """Processes raw document collections and provides numeric representations.

    Parameters
    ----------
    corpus: iterable of str
        An iterable over documents given as a single string each.
    pipeline: Pipeline
        The preprocessing pipeline.

    See Also
    --------
    plsa.pipeline

    """
    def __init__(self, corpus: Iterable[str], pipeline: Pipeline) -> None:
        self.__pipeline = pipeline
        self.__raw = []
        self.__index = defaultdict(lambda: len(self.__index))
        self.__vocabulary = {}
        self.__n_occurrences = 0
        self.__n_docs = 0
        self.__n_words = 0
        self.__doc_word = None
        self.__generate_doc_word(corpus)

    def __repr__(self):
        title = self.__class__.__name__
        header = f'{title}:\n'
        divider = '=' * len(title) + '\n'
        n_docs = f'Number of documents: {self.n_docs}\n'
        n_words = f'Number of words:     {self.n_words}'
        return header + divider + n_docs + n_words

    @classmethod
    def from_csv(cls, path: str,
                 pipeline: Pipeline,
                 col: int = -1,
                 encoding: str = 'latin_1',
                 max_docs: int = 1000,
                 **kwargs) -> 'Corpus':
        """Instantiate a corpus from documents in a column of a CSV file.

        Parameters
        ----------
        path: str
            Full path (incl. file name) to a CSV file with one column
            containing documents.
        pipeline:
            The preprocessing pipeline.
        col: int
            Which column contains the documents. Numbering starts with 0 for
            the first column. Negative numbers count back from the last
            column (`e.g.`, -1 for last, -2 just before the last, `etc.`).
        encoding: str
            A valid python encoding used to read the documents.
        max_docs: int
            The maximum number of documents to read from file.
        **kwargs
            Keyword arguments are passed on to Python's own ``csv.reader``
            function.

        Raises
        ------
        StopIteration
            If you do not have at least two lines in your CSV file.

        Notes
        -----
        If you set a ``col`` to a value outside the range present in the CSV
        file, it will be silently reset to the first or last column, depending
        on which side you exceed the permitted range.

        A list of available encodings can be found at
        https://docs.python.org/3/library/codecs.html

        Formatting parameters for the Python's ``csv.reader`` can be found at
        https://docs.python.org/3/library/csv.html#csv-fmt-params

        """
        docs, n_docs = [], 0
        with open(str(path), encoding=encoding, newline='') as stream:
            file = csv.reader(stream, **kwargs)
            try:
                n_cols = len(next(file))
            except StopIteration:
                raise StopIteration('Not enough lines in CSV file. '
                                    'Must be at least 2!')
            col = min(sign(col) * min(abs(col), n_cols), n_cols - 1)
            for line in file:
                docs.append(line[col])
                n_docs += 1
                if n_docs >= max_docs:
                    break
        return cls(docs, pipeline)

    @classmethod
    def from_xml(cls, directory: str,
                 pipeline: Pipeline,
                 tag: str = 'post',
                 encoding: str = 'latin_1',
                 max_files: int = 100) -> 'Corpus':
        """Instantiate a corpus from elements of XML files in a directory.

        Parameters
        ----------
        directory: str
            Path to the directory with the XML files.
        pipeline: Pipeline
           The preprocessing pipeline.
        tag:
            The XML tag that opens (<...>) and closes (</...>) the elements
            containing documents.
        encoding:
            A valid python encoding used to read the documents.
        max_files
            The maximum number of XML files to read.

        Notes
        --------
        A list of available encodings can be found at
        https://docs.python.org/3/library/codecs.html

        """
        directory = directory if directory.endswith('/') else directory + '/'
        filenames = os.listdir(directory)
        n_files = min(len(filenames), max_files)
        docs = []
        for filename in filenames[:n_files]:
            with open(directory + filename, encoding=encoding) as file:
                we_are_within_tagged_element = False
                for line in file:
                    if f'<{tag}>' in line:
                        doc = ''
                        we_are_within_tagged_element = True
                        continue
                    elif f'</{tag}>' in line:
                        docs.append(doc)
                        we_are_within_tagged_element = False
                        continue
                    if we_are_within_tagged_element:
                        doc += line.strip()
        return cls(docs, pipeline)

    @property
    def raw(self) -> Tuple[str, ...]:
        """The raw documents as they were read from the source."""
        return tuple(self.__raw)

    @property
    def pipeline(self) -> Pipeline:
        """The pipeline of preprocessors for each document."""
        return self.__pipeline

    @property
    def n_docs(self) -> int:
        """The number of non-empty documents."""
        return self.__n_docs

    @property
    def n_words(self) -> int:
        """The number of unique words retained after preprocessing."""
        return self.__n_words

    @property
    def vocabulary(self) -> Dict[int, str]:
        """Mapping from numeric word index to actual word."""
        return self.__vocabulary

    @property
    def index(self) -> Dict[str, int]:
        """Mapping from actual word to numeric word index."""
        return self.__index

    @property
    def n_occurrences(self) -> int:
        """Total number of times any word occurred in any document."""
        return self.__n_occurrences

    @property
    def idf(self) -> ndarray:
        """Logarithm of inverse fraction of documents each word occurs in."""
        return log(self.__n_docs / (self.__doc_word > 0.0).sum(axis=0))

    def get_doc_word(self, tf_idf: bool) -> ndarray:
        """The normalized document-word counts matrix.

        Also referred to as the `term-frequency` matrix. Because words (or
        `terms`) that occur in the majority of documents are the least helpful
        in discriminating types of documents, each column of this matrix can be
        multiplied by the logarithm of the total number of documents divided
        by the number of documents containing the given word. The result is
        then referred to as the `term-frequency inverse-document-frequency`
        or `TF-IDF` matrix.

        Either way, the returned matrix is always `normalized` such that it
        can be interpreted as the joint document-word probability `p(d, w)`.

        Parameters
        ----------
        tf_idf: bool
            Whether to return the term-frequency inverse-document-frequency
            or just the term-frequency matrix.

        Returns
        -------
        ndarray
            The normalized document (rows) - word (columns) matrix, either
            as pure counts (if ``tf_idf`` = ``False``) or weighted by the
            inverse document frequency (if ``tf_idf`` is ``False``).

        """
        if tf_idf:
            not_normalized = self.__doc_word * self.idf
            return not_normalized / not_normalized.sum()
        return self.__doc_word / self.__n_occurrences

    def get_doc(self, tf_idf: bool) -> ndarray:
        """The marginal probability that any word comes from a given document.

        This probability `p(d)` is obtained by summing the joint document-
        word probability `p(d, w)` over all words.

        Parameters
        ----------
        tf_idf: bool
            Whether to marginalize the term-frequency inverse-document-frequency
            or just the term-frequency matrix.

        Returns
        -------
        ndarray
            The document probability `p(d)`.

        """
        return self.get_doc_word(tf_idf).sum(axis=1)

    def get_word(self, tf_idf: bool) -> ndarray:
        """The marginal probability of a particular word.

        This probability `p(w)` is obtained by summing the joint document-
        word probability `p(d, w)` over all documents.

        Parameters
        ----------
        tf_idf: bool
            Whether to marginalize the term-frequency inverse-document-frequency
            or just the term-frequency matrix.

        Returns
        -------
        ndarray
            The word probability `p(w)`.

        """
        return self.get_doc_word(tf_idf).sum(axis=0)

    def get_doc_given_word(self, tf_idf: bool) -> ndarray:
        """The conditional probability of a particular word in a given document.

        This probability `p(d|w)` is obtained by dividing the joint document-
        word probability `p(d, w)` by the marginal word probability `p(w)`.

        Parameters
        ----------
        tf_idf: bool
            Whether to base the conditional probability on the term-frequency
            inverse-document-frequency or just the term-frequency matrix.

        Returns
        -------
        ndarray
            The conditional word probability `p(d|w)`.

        """
        return self.get_doc_word(tf_idf) / self.get_word(tf_idf)

    def __generate_doc_word(self, corpus: Iterable[str]) -> None:
        doc_word_dict = defaultdict(int)
        for doc in corpus:
            self.__raw.append(doc)
            doc = self.__pipeline.process(doc)
            for word in doc:
                doc_word_dict[(self.__n_docs, self.__index[word])] += 1
                self.__vocabulary[self.__index[word]] = word
            self.__n_docs = self.__n_docs + 1 if len(doc) else self.__n_docs
        self.__n_words = len(self.__vocabulary)
        self.__index = dict(self.__index)
        self.__doc_word = zeros((self.__n_docs, self.__n_words))
        for (doc, word), count in doc_word_dict.items():
            self.__doc_word[doc, word] = count
        self.__n_occurrences = int(self.__doc_word.sum())
