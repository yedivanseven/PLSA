from numpy import empty, einsum

from ..corpus import Corpus
from .result import PlsaResult
from .base import BasePLSA


class PLSA(BasePLSA):
    def __init__(self, corpus: Corpus, n_topics: int, tf_idf: bool = False):
        super().__init__(corpus, n_topics, tf_idf)
        self._word_given_topic = empty((corpus.n_words, n_topics))

    def _update(self) -> None:
        self._doc_given_topic = self._norm_sum('dw,tdw->dt')
        self._word_given_topic = self._norm_sum('dw,tdw->wt')
        self._topic = einsum('dw,tdw->t', self._doc_word, self._conditional)
        self._joint = einsum('dt,wt,t->tdw',
                             self._doc_given_topic,
                             self._word_given_topic,
                             self._topic)

    def _result(self) -> PlsaResult:
        return PlsaResult(self._invert(self._doc_given_topic, self._topic),
                          self._word_given_topic,
                          self._topic,
                          self._likelihoods,
                          self._vocabulary)
