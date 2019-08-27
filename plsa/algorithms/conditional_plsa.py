from numpy import empty, einsum, log, abs

from ..corpus import Corpus
from .result import PlsaResult
from .base import BasePLSA


class ConditionalPLSA(BasePLSA):
    def __init__(self, corpus: Corpus, n_topics: int):
        super().__init__(corpus, n_topics)
        self._target = corpus.doc_given_word
        self.__topic_given_word = empty((n_topics, corpus.n_words))
        self.__word = corpus.word
        self.__n_words = corpus.n_words

    def _update(self) -> float:
        self._doc_given_topic, _ = self._norm_sum('dw,tdw->dt')
        self.__topic_given_word, _ = self._norm_sum('dw,tdw->tw')
        self._joint = einsum('dt,tw->tdw',
                             self._doc_given_topic,
                             self.__topic_given_word) / self.__n_words
        self._joint = self._safe_divide(self._joint, self.__n_words)
        self._topic = einsum('tdw->t', self._joint)
        self._conditional, self._norm = self._normalize(self._joint)
        return (self._doc_word * log(self._norm)).sum()

    def _result(self) -> PlsaResult:
        return PlsaResult(self._invert(self._doc_given_topic, self._topic),
                          self._invert(self.__topic_given_word, self.__word),
                          self._topic,
                          self._likelihoods,
                          self._vocabulary)
