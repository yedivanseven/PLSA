from typing import List, Dict, Tuple, Callable
from numpy import newaxis, arange, ndarray, zeros

from ..corpus import Corpus

TuplesT = Tuple[Tuple[str, float], ...]


class PlsaResult:
    def __init__(self, topic_given_doc: ndarray,
                 word_given_topic: ndarray,
                 topic_given_word: ndarray,
                 topic: ndarray,
                 kl_divergences: List[float],
                 corpus: Corpus,
                 tf_idf: bool) -> None:
        self.__n_topics = topic.size
        self.__kl_divergences = kl_divergences
        self.__corpus = corpus
        self.__tf_idf = tf_idf
        self.__topic, topic_order = self.__ordered_topic(topic)
        self.__topic_given_doc = topic_given_doc[topic_order]
        self.__topic_given_word = topic_given_word[topic_order]
        word_given_topic = word_given_topic[:, topic_order]
        word_given_topic, word_order = self.__sorted(word_given_topic)
        zipped = self.__zipped(word_order, word_given_topic)
        tuples = self.__tuples(corpus.vocabulary)
        topics = range(topic.size)
        self.__word_given_topic = tuple(tuples(zipped(t)) for t in topics)

    def __repr__(self) -> str:
        title = self.__class__.__name__
        header = f'{title}:\n'
        divider = '=' * len(title) + '\n'
        n_topics = f'Number of topics:    {self.__n_topics}\n'
        n_docs = f'Number of documents: {len(self.__topic_given_doc[0])}\n'
        n_words = f'Number of words:     {len(self.__word_given_topic[0])}'
        body = n_topics + n_docs + n_words
        return header + divider + body

    @property
    def n_topics(self) -> int:
        return self.__n_topics

    @property
    def tf_idf(self) -> bool:
        return self.__tf_idf

    @property
    def topic(self) -> ndarray:
        return self.__topic

    @property
    def word_given_topic(self) -> Tuple[TuplesT, ...]:
        return self.__word_given_topic

    @property
    def topic_given_doc(self) -> ndarray:
        return self.__topic_given_doc.T

    @property
    def convergence(self) -> List[float]:
        return self.__kl_divergences

    def predict(self, doc: str) -> (ndarray, int, Tuple[str, ...]):
        processed = self.__corpus.pipeline.process(doc)
        encoded = zeros(self.__corpus.n_words + 1)
        new_words = []
        for word in processed:
            index = self.__corpus.index.get(word, self.__corpus.n_words)
            encoded[index] += 1
            if index == self.__corpus.n_words:
                new_words.append(word)
        encoded, n_new_words = encoded[:-1], int(encoded[-1])
        encoded = encoded * self.__corpus.idf if self.__tf_idf else encoded
        encoded /= encoded.sum()
        new_words = tuple(new_words)
        return self.__topic_given_word.dot(encoded), n_new_words, new_words

    def __ordered_topic(self, topic: ndarray) -> (ndarray, ndarray):
        topic, topic_order = self.__sorted(topic[:, newaxis])
        return self.__raveled(topic, topic_order)

    @staticmethod
    def __sorted(array: ndarray) -> (ndarray, ndarray):
        sorting_indices = (-array).argsort(axis=0)
        return array[sorting_indices, arange(array.shape[1])], sorting_indices.T

    @staticmethod
    def __raveled(*arrays: ndarray) -> Tuple[ndarray, ...]:
        return tuple(array.ravel() for array in arrays)

    @staticmethod
    def __zipped(first: ndarray, second: ndarray) -> Callable[[int], zip]:

        def zipped(topic: int) -> zip:
            return zip(first[topic], second[:, topic])

        return zipped

    @staticmethod
    def __tuples(vocabulary: Dict[int, str]) -> Callable[[zip], TuplesT]:

        def tuples(zipped: zip) -> TuplesT:
            return tuple((vocabulary[index], proba) for index, proba in zipped)

        return tuples
