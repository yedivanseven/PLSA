from numpy import empty, einsum

from ..corpus import Corpus
from .result import PlsaResult
from .base import BasePLSA


class ConditionalPLSA(BasePLSA):
    def __init__(self, corpus: Corpus, n_topics: int, tf_idf: bool = False):
        super().__init__(corpus, n_topics, tf_idf)
        self._topic_given_word = empty((n_topics, corpus.n_words))
        self._word = corpus.get_word(tf_idf)

    def _update(self) -> None:
        self._doc_given_topic = self._norm_sum('dw,tdw->dt')
        self._topic_given_word = self._norm_sum('dw,tdw->tw')
        self._joint = einsum('dt,tw,w->tdw',
                             self._doc_given_topic,
                             self._topic_given_word,
                             self._word)
        self._topic = einsum('tdw->t', self._joint)

    def _result(self) -> PlsaResult:
        return PlsaResult(self._invert(self._doc_given_topic, self._topic),
                          self._invert(self._topic_given_word, self._word),
                          self._topic,
                          self._likelihoods,
                          self._vocabulary)
