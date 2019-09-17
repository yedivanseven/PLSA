from numpy import empty, einsum

from ..corpus import Corpus
from .result import PlsaResult
from .base import BasePLSA


class ConditionalPLSA(BasePLSA):
    """Implements conditional probabilistic latent semantic analysis (PLSA).

    Given that the normalized document-word (or term-frequency) matrix
    `p(d, w)`, weighted with the inverse document frequency or not, can
    always be written as,

    .. math:: p(d, w) = p(d|w)p(w)

    the core of conditional PLSA is the assumption that the conditional
    `p(d|w)` can be factorized as:

    .. math:: p(d|w) \\approx \sum_t \\tilde{p}(d|t)\\tilde{p}(t|w)

    Parameters
    ----------
    corpus: Corpus
        The corpus of preprocessed and numerically represented documents.
    n_topics: int
        The number of latent topics to identify.
    tf_idf: bool
        Whether to use the term-frequency inverse-document-frequency
        or just the term-frequency matrix as joint probability `p(d, w)` of
        documents and words.

    Raises
    ------
    ValueError
        If the number of topics is < 2 or the number of both, words and
        documents, in the corpus isn't greater than the number of topics.

    Notes
    -----
    Importantly, the present implementation does `not` follow algorithm 15.3
    in Barber's book [1]_. The update equations there appear non-sensical.
    Following through the derivation that gives (non-conditional) PLSA, one
    arrives at the following updates:

    .. math::
        \\tilde{p}(d|t) &= \sum_w p(d, w)q(t|d, w) \\\\
        \\tilde{p}(t|w) &= \sum_d p(d, w)q(t|d, w) \\\\
        \\tilde{p}(t, d, w) &= p(w)\sum_t\\tilde{p}(d|t)\\tilde{p}(t|w) \\\\
        q(t| d, w) &= \\tilde{p}(t, d, w) / \\tilde{p}(d, w)

    References
    ----------
    .. [1] "Bayesian Reasoning and Machine Learning", David Barber (Cambridge
       Press, 2012).


    """
    def __init__(self, corpus: Corpus, n_topics: int, tf_idf: bool = True):
        super().__init__(corpus, n_topics, tf_idf)
        self._topic_given_word = empty((n_topics, corpus.n_words))
        self._word = corpus.get_word(tf_idf)

    def _m_step(self) -> None:
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
                          self._topic_given_word,
                          self._topic,
                          self._kl_divergences,
                          self._corpus,
                          self.tf_idf)
