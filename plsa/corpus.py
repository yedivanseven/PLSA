from collections import defaultdict
from typing import Iterable, Dict
from numpy import zeros, ndarray, log

from .pipeline import Pipeline


class Corpus:
    def __init__(self, corpus: Iterable[str], pipeline: Pipeline) -> None:
        self.__corpus = corpus
        self.__pipeline = pipeline
        self.__index = defaultdict(lambda: len(self.__index))
        self.__vocabulary = {}
        self.__norm = 0
        self.__n_docs = 0
        self.__n_words = 0
        self.__doc_word = None
        self.__generate_doc_word()

    def __repr__(self):
        title = self.__class__.__name__
        header = f'{title}:\n'
        divider = '=' * len(title) + '\n'
        n_docs = f'Number of documents: {self.n_docs}\n'
        n_words = f'Number of words:     {self.n_words}'
        return header + divider + n_docs + n_words

    @property
    def raw(self) -> Iterable[str]:
        return self.__corpus

    @property
    def n_docs(self) -> int:
        return self.__n_docs

    @property
    def n_words(self) -> int:
        return self.__n_words

    @property
    def vocabulary(self) -> Dict[int, str]:
        return self.__vocabulary

    @property
    def index(self) -> Dict[str, int]:
        return self.__index

    @property
    def norm(self) -> int:
        return self.__norm

    def get_doc_word(self, tf_idf: bool) -> ndarray:
        if tf_idf:
            idf = log(self.__n_docs / (self.__doc_word > 0.0).sum(axis=0))
            tf_idf = self.__doc_word * idf
            return tf_idf / tf_idf.sum()
        return self.__doc_word / self.__norm

    def get_doc(self, tf_idf: bool) -> ndarray:
        return self.get_doc_word(tf_idf).sum(axis=1)

    def get_word(self, tf_idf: bool) -> ndarray:
        return self.get_doc_word(tf_idf).sum(axis=0)

    def get_doc_given_word(self, tf_idf: bool) -> ndarray:
        return self.get_doc_word(tf_idf) / self.get_word(tf_idf)

    def __generate_doc_word(self) -> None:
        doc_word_dict = defaultdict(int)
        for doc in self.__corpus:
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
        self.__norm = int(self.__doc_word.sum())
