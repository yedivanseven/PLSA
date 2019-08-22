from typing import List, Dict, Tuple, Callable
from numpy import newaxis, arange, ndarray

TuplesT = Tuple[Tuple[str, float], ...]


class PlsaResult:
    def __init__(self, topic_given_doc: ndarray,
                 word_given_topic: ndarray,
                 topic: ndarray,
                 likelihoods: List[float],
                 vocabulary: Dict[int, str]) -> None:
        self.__n_topics = topic.size
        self.__likelihoods = likelihoods
        self.__topic, topic_order = self.__ordered_topic(topic)
        self.__topic_given_doc = topic_given_doc[topic_order]
        word_given_topic = word_given_topic[:, topic_order]
        word_given_topic, word_order = self.__sorted(word_given_topic)
        zipped = self.__zipped(word_order, word_given_topic)
        tuples = self.__tuples(vocabulary)
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
    def topic(self) -> ndarray:
        return self.__topic

    @property
    def word_given_topic(self) -> Tuple[TuplesT, ...]:
        return self.__word_given_topic

    @property
    def topic_given_doc(self) -> ndarray:
        return self.__topic_given_doc

    @property
    def convergence(self) -> List[float]:
        return self.__likelihoods

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
