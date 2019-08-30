import os
import csv

from collections import defaultdict
from typing import Iterable, Dict, Tuple
from numpy import zeros, ndarray, log, sign, abs

from .pipeline import Pipeline


class Corpus:
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
                 max_docs: int = 1000) -> 'Corpus':
        docs, n_docs = [], 0
        with open(path, encoding=encoding, newline='') as stream:
            file = csv.reader(stream)
            n_cols = len(next(file))
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
        return tuple(self.__raw)

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
    def n_occurrences(self) -> int:
        return self.__n_occurrences

    def get_doc_word(self, tf_idf: bool) -> ndarray:
        if tf_idf:
            idf = log(self.__n_docs / (self.__doc_word > 0.0).sum(axis=0))
            tf_idf = self.__doc_word * idf
            return tf_idf / tf_idf.sum()
        return self.__doc_word / self.__n_occurrences

    def get_doc(self, tf_idf: bool) -> ndarray:
        return self.get_doc_word(tf_idf).sum(axis=1)

    def get_word(self, tf_idf: bool) -> ndarray:
        return self.get_doc_word(tf_idf).sum(axis=0)

    def get_doc_given_word(self, tf_idf: bool) -> ndarray:
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
        self.__n_occurrence = int(self.__doc_word.sum())
