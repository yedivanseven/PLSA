from numpy import empty, einsum

from ..corpus import Corpus
from .result import PlsaResult
from .base import BasePLSA


class PLSA(BasePLSA):
    """Implements probabilistic latent semantic analysis (PLSA).

    At its core lies the assumption that the normalized document-word
    (or term-frequency) matrix `p(d, w)`, weighted with the inverse document
    frequency or not, can be factorized as:

    .. math:: p(d, w)\\approx\sum_t \\tilde{p}(d|t)\\tilde{p}(w|t)\\tilde{p}(t)

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
    The implementation follows algorithm 15.2 in Barber's book [1]_ to the
    letter. What is not said there is that, in order to update the conditional
    probability `p(t|d, w)` of a certain topic given a certain word in
    a certain document, one first needs to find the joint probability of
    all random variables as

    .. math:: \\tilde{p}(t, d, w) = \\tilde{p}(d|t)\\tilde{p}(w|t)\\tilde{p}(t)

    and then divide by the marginal :math:`\\tilde{p}(d, w)`.

    References
    ----------
    .. [1] "Bayesian Reasoning and Machine Learning", David Barber (Cambridge
       Press, 2012).

    """
    def __init__(self, corpus: Corpus, n_topics: int, tf_idf: bool = True):
        super().__init__(corpus, n_topics, tf_idf)
        self._word_given_topic = empty((corpus.n_words, self.n_topics))

    def _m_step(self) -> None:
        """Implements the M-step of EM-style algorithm to train PLSA model."""
        self._doc_given_topic = self._norm_sum('dw,tdw->dt')
        self._word_given_topic = self._norm_sum('dw,tdw->wt')
        self._topic = einsum('dw,tdw->t', self._doc_word, self._conditional)
        self._joint = einsum('dt,wt,t->tdw',
                             self._doc_given_topic,
                             self._word_given_topic,
                             self._topic)

    def _result(self) -> PlsaResult:
        """Prepares result with inverted doc-given-topic probability."""
        return PlsaResult(self._invert(self._doc_given_topic, self._topic),
                          self._word_given_topic,
                          self._invert(self._word_given_topic, self._topic),
                          self._topic,
                          self._kl_divergences,
                          self._corpus,
                          self.tf_idf)
