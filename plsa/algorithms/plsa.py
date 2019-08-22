from numpy import empty, einsum, log

from ..corpus import Corpus
from .result import PlsaResult
from .base import BasePLSA


class PLSA(BasePLSA):
    def __init__(self, corpus: Corpus, n_topics: int):
        super().__init__(corpus, n_topics)
        self.__word_given_topic = empty((corpus.n_words, n_topics))

    def _update(self) -> float:
        self._doc_given_topic, _ = self._norm_sum('dw,tdw->dt')
        self.__word_given_topic, _ = self._norm_sum('dw,tdw->wt')
        self._topic = einsum('dw,tdw->t', self._target, self._conditional)
        self._joint = einsum('dt,wt,t->tdw',
                             self._doc_given_topic,
                             self.__word_given_topic,
                             self._topic)
        self._conditional, self._norm = self._normalize(self._joint)
        return (self._doc_word * log(self._norm)).sum()

    def _result(self) -> PlsaResult:
        return PlsaResult(self._invert(self._doc_given_topic, self._topic),
                          self.__word_given_topic,
                          self._topic,
                          self._likelihoods,
                          self._vocabulary)
