from typing import Union
from numpy import empty, ndarray, einsum, abs, log, inf, finfo
from numpy.random import rand

from .result import PlsaResult
from ..corpus import Corpus

Norm = Union[ndarray, None]
Divisor = Union[int, float, ndarray]
MACHINE_PRECISION = finfo(float).eps


class BasePLSA:
    def __init__(self, corpus: Corpus, n_topics: int, tf_idf: bool = False):
        self.__n_topics = n_topics
        self._vocabulary = corpus.vocabulary
        self._doc_word = corpus.get_doc_word(tf_idf)
        self._joint = empty((n_topics, corpus.n_docs, corpus.n_words))
        self._conditional = self.__random(corpus.n_docs, corpus.n_words)
        self.__norm = empty((corpus.n_docs, n_topics))
        self._doc_given_topic = empty((corpus.n_docs, n_topics))
        self._topic = empty(n_topics)
        self._likelihoods = []

    def __repr__(self) -> str:
        title = self.__class__.__name__
        header = f'{title}:\n'
        divider = '=' * len(title) + '\n'
        n_topics = f'Number of topics:     {self.__n_topics}\n'
        n_docs = f'Number of documents:  {self._doc_word.shape[0]}\n'
        n_words = f'Number of words:      {self._doc_word.shape[1]}\n'
        iterations = f'Number of iterations: {len(self._likelihoods)}'
        body = n_topics + n_docs + n_words + iterations
        return header + divider + body

    @property
    def n_topics(self) -> int:
        return self.__n_topics

    def fit(self, eps: float = 1e-5,
            max_iter: int = 200,
            warmup: int = 5) -> PlsaResult:
        n_iter = 0
        while n_iter < max_iter:
            self._m_step()
            self.__e_step()
            likelihood = (self._doc_word * log(self.__norm)).sum()
            n_iter += 1
            if n_iter > warmup and self.__rel_change(likelihood) < eps:
                break
            self._likelihoods.append(likelihood)
        return self._result()

    def _m_step(self) -> None:
        raise NotImplementedError

    def __e_step(self) -> None:
        self._conditional, self.__norm = self.__normalize(self._joint)

    def __random(self, n_docs: int, n_words: int) -> ndarray:
        conditional = rand(self.__n_topics, n_docs, n_words)
        return self.__normalize(conditional)[0]

    def _norm_sum(self, index_pattern: str) -> ndarray:
        probability = einsum(index_pattern, self._doc_word, self._conditional)
        return self.__normalize(probability)[0]

    def __normalize(self, array: ndarray, norm: Norm = None) -> (ndarray, Norm):
        norm = norm or array.sum(axis=0)
        mask = norm < MACHINE_PRECISION
        array[..., mask] = 0.0
        norm[mask] = 1.0
        return self._safe_divide(array, norm), norm

    @staticmethod
    def _safe_divide(array: ndarray, number: Divisor) -> ndarray:
        array[array < MACHINE_PRECISION] = 0.0
        return array / number

    def __rel_change(self, new: float) -> float:
        if self._likelihoods:
            old = self._likelihoods[-1]
            return abs((new - old) / new)
        return inf

    def _result(self) -> PlsaResult:
        raise NotImplementedError

    def _invert(self, conditional: ndarray, marginal: ndarray) -> ndarray:
        inverted = conditional * marginal
        return self.__normalize(inverted.T)[0]
