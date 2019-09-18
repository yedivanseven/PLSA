from typing import List
from wordcloud import WordCloud
from matplotlib.lines import Line2D
from matplotlib.axes import Subplot
from matplotlib.image import AxesImage
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

from .algorithms import PlsaResult


class Visualize:
    """Visualize the results of probabilistic latent semantic analysis.

    Parameters
    ----------
    result: PlsaResult
        The results object returned by the ``fit`` method of a PLSA
        model object.

    """
    def __init__(self, result: PlsaResult) -> None:
        self.__convergence = result.convergence
        self.__topics = result.topic
        self.__word_given_topic = result.word_given_topic
        self.__n_topics = result.n_topics
        self.__topic_range = range(result.n_topics)
        self.__topic_given_doc = result.topic_given_doc
        self.__predict = result.predict
        self.__wordcloud = WordCloud(background_color='white')

    def __repr__(self) -> str:
        title = self.__class__.__name__
        header = f'{title}:\n'
        divider = '=' * len(title) + '\n'
        n_topics = f'Number of topics:    {self.__topics.size}\n'
        n_docs = f'Number of documents: {self.__topic_given_doc.shape[0]}\n'
        n_words = f'Number of words:     {len(self.__word_given_topic[0])}'
        body = n_topics + n_docs + n_words
        return header + divider + body

    def convergence(self, axis: Subplot) -> List[Line2D]:
        """Plot the convergence of the PLSA run.

        The quantity to be minimized is the Kullback-Leibler divergence
        between the original document-word matrix and its approximation
        given by the (conditional) PLSA factorization.

        Parameters
        ----------
        axis: Subplot
            The matplotlib axis to plot into.

        Returns
        -------
        list of Line2D
            The line object plotted into the given axis.

        """
        axis.set(xlabel='Iteration', ylabel='Kullback-Leibler divergence')
        return axis.plot(self.__convergence)

    def topics(self, axis: Subplot) -> BarContainer:
        """Plot the relative importance of the individual topics.

        Parameters
        ----------
        axis: Subplot
            The matplotlib axis to plot into.

        Returns
        -------
        BarContainer
            The container for the bars plotted into the given axis.

        """
        colors = [f'C{color}' for color in self.__topic_range]
        axis.set(xlabel='Topic', ylabel='Importance')
        axis.set_title('Topic distribution')
        return axis.bar(self.__topic_range, self.__topics, color=colors,
                        tick_label=self.__topic_range)

    def words_in_topic(self, i_topic: int, axis: Subplot) -> AxesImage:
        """Plot the relative importance of words in a given topic.

        Parameters
        ----------
        i_topic: int
            Index of the topic to plot. Numbering starts at 0.
        axis: Subplot
            The matplotlib axis to plot into.

        Returns
        -------
        AxesImage
            The image with the produced word cloud.

        """
        word_given_topic = dict(self.__word_given_topic[i_topic])
        wordcloud = self.__wordcloud.generate_from_frequencies(word_given_topic)
        axis.set_title(f'Topic {i_topic}')
        axis.set_axis_off()
        return axis.imshow(wordcloud, interpolation='bilinear')

    def topics_in_doc(self, i_doc: int, axis: Subplot) -> BarContainer:
        """Plot the relative weights of topics in a given document.

        Parameters
        ----------
        i_doc: int
            Index of the document to plot. Numbering starts at 0.
        axis: Subplot
            The matplotlib axis to plot into.

        Returns
        -------
        BarContainer
            The container for the bars plotted into the given axis.

        """
        colors = [f'C{color}' for color in self.__topic_range]
        axis.set(xlabel='Topic', ylabel='Importance', title=f'Document {i_doc}')
        return axis.bar(self.__topic_range, self.__topic_given_doc[i_doc],
                        color=colors, tick_label=self.__topic_range)

    def wordclouds(self, figure: Figure) -> List[AxesImage]:
        """Plot the relative importance of words in all topics.

        Parameters
        ----------
        figure: Figure
            An empty matplotlib figure to plot into.

        Returns
        -------
        list of AxisImage
            List of images with the created word clouds.

        """
        n_rows = (self.__n_topics + 1) // 2
        axes = (figure.add_subplot(n_rows, 2, topic+1)
                for topic in self.__topic_range)
        zipped = zip(self.__topic_range, axes)
        imgs = list(self.words_in_topic(topic, axis) for topic, axis in zipped)
        figure.tight_layout()
        return imgs

    def prediction(self, doc: str, axis: Subplot) -> BarContainer:
        """Plot the predicted relative weights of topics in a new document.

        Parameters
        ----------
        doc: str
            A new document given as a single string.
        axis: Subplot
            The matplotlib axis to plot into.

        Returns
        -------
        BarContainer
            The container for the bars plotted into the given axis.

        """
        colors = [f'C{color}' for color in self.__topic_range]
        prediction, n_unknown_words, _ = self.__predict(doc)
        axis.set(xlabel='Topic', ylabel='Importance')
        axis.set_title(f'Number of unknown words: {n_unknown_words}')
        return axis.bar(self.__topic_range, prediction, color=colors,
                        tick_label=self.__topic_range)
