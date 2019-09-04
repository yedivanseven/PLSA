from typing import Union
from numpy import empty, ndarray, einsum, abs, log, inf, finfo
from numpy.random import rand

from .result import PlsaResult
from ..corpus import Corpus

Norm = Union[ndarray, None]
Divisor = Union[int, float, ndarray]
MACHINE_PRECISION = finfo(float).eps


class BasePLSA:
    """Base class for all flavours of PLSA algorithms.

    Since the base class for all algorithms is not supposed to ever be
    instantiated directly, it is also not documented. For more information,
    please refer to the docstrings of the individual algorithms.

    """
    def __init__(self, corpus: Corpus, n_topics: int, tf_idf: bool = False):
        self.__n_topics = abs(int(n_topics))
        self._vocabulary = corpus.vocabulary
        self._doc_word = corpus.get_doc_word(tf_idf)
        self._joint = empty((self.__n_topics, corpus.n_docs, corpus.n_words))
        self._conditional = self.__random(corpus.n_docs, corpus.n_words)
        self.__norm = empty((corpus.n_docs, self.__n_topics))
        self._doc_given_topic = empty((corpus.n_docs, self.__n_topics))
        self._topic = empty(self.__n_topics)
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
        """The number of topics to find."""
        return self.__n_topics

    def fit(self, eps: float = 1e-5,
            max_iter: int = 200,
            warmup: int = 5) -> PlsaResult:
        """Run EM-style training to find latent topic in documents.

        Expectation-maximization (EM) iterates until either the maximum number
        of iterations is reached or if relative changes of the log-likelihood
        fall below a certain threshold, whichever occurs first.

        Since all quantities are update in-place, calling the ``fit`` method
        again after a successful run (possibly with changed convergence
        criteria) will continue to add more iterations on top of the status
        quo rather than starting all over again from scratch.

        Because a few EM iterations are needed to get things going, you can
        specify an initial `warm-up` period during which progress in the
        log-likelihood is not tracked and which does not count towards the
        maximum number of iterations.


        Parameters
        ----------
        eps: float, optional
            The convergence cutoff for relative changes in the log-likelihood.
            Defaults to 1e-5
        max_iter: int, optional
            The maximum number of iterations to perform. Defaults to 200.
        warmup: int, optional
            The number of iterations to perform before changes in the
            log-likelihood are tracked for convergence.

        Returns
        -------
        PlsaResult
            Container class for the results of the latent semantic analysis.

        """
        eps = abs(float(eps))
        max_iter = abs(int(max_iter))
        warmup = abs(int(warmup))
        n_iter = 0
        while n_iter < max_iter + warmup:
            self._m_step()
            self.__e_step()
            likelihood = (self._doc_word * log(self.__norm)).sum()
            n_iter += 1
            if n_iter > warmup and self.__rel_change(likelihood) < eps:
                break
            self._likelihoods.append(likelihood)
        return self._result()

    def _m_step(self) -> None:
        """This must be implemented for each specific PLSA flavour."""
        raise NotImplementedError

    def _result(self) -> PlsaResult:
        """This must be implemented for each specific PLSA flavour."""
        raise NotImplementedError

    def __e_step(self) -> None:
        """The E-step of the EM algorithm is the same for all PLSA flavours.

        From the joint probability `p(t, d, w)` of latent topics, documents,
        and words, we need to get a new conditional probability `p(t|d, w)`
        by dividing the joint by the marginal `p(d, w)`.

        """
        self._conditional, self.__norm = self.__normalize(self._joint)

    def __random(self, n_docs: int, n_words: int) -> ndarray:
        """Randomly initialize the conditional probability p(t|d, w)."""
        conditional = rand(self.__n_topics, n_docs, n_words)
        return self.__normalize(conditional)[0]

    def _norm_sum(self, index_pattern: str) -> ndarray:
        """Update individual probability factors in the M-step."""
        probability = einsum(index_pattern, self._doc_word, self._conditional)
        return self.__normalize(probability)[0]

    def __normalize(self, array: ndarray, norm: Norm = None) -> (ndarray, Norm):
        """Normalize probability without underflow or divide-by-zero errors."""
        norm = norm or array.sum(axis=0)
        mask = norm < MACHINE_PRECISION
        array[..., mask] = 0.0
        norm[mask] = 1.0
        return self._safe_divide(array, norm), norm

    @staticmethod
    def _safe_divide(array: ndarray, divisor: Divisor) -> ndarray:
        """Divide a numpy array without running into underflow errors."""
        array[array < MACHINE_PRECISION] = 0.0
        return array / divisor

    def __rel_change(self, new: float) -> float:
        """Return the relative change in the log-likelihood."""
        if self._likelihoods:
            old = self._likelihoods[-1]
            return abs((new - old) / new)
        return inf

    def _invert(self, conditional: ndarray, marginal: ndarray) -> ndarray:
        """Perform a Bayesian inversion of a conditional probability."""
        inverted = conditional * marginal
        return self.__normalize(inverted.T)[0]
