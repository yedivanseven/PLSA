[![Documentation Status](https://readthedocs.org/projects/probabilistic-latent-semantic-analysis/badge/?version=latest)](https://probabilistic-latent-semantic-analysis.readthedocs.io/en/latest/?badge=latest)
# PLSA
_A `python` implementation of Probabilistic Latent Semantic Analysis_

## What PLSA can do for you
Broadly speaking,
[PLSA](https://en.wikipedia.org/wiki/Probabilistic_latent_semantic_analysis)
is a tool of
[Natural Language Processing](https://en.wikipedia.org/wiki/Natural_language_processing)
(NLP). It analyses a collection of text documents (a _corpus_) under the assumption
that there are (by far) fewer _topics_ to write about than there are documents
in the corpus. It then tries to identify these topics (in terms of words and
their relative importance to each topic) and to give you the relative
importance of a pre-specified number of topics in each document.

In doing so, it does not actually try to "make sense" of each document (or
"understand" it) by contextually analysing it. Rather, it simply counts how
often which word occurs in each document, regardless of the context in which
they occur. As such, it belongs to the family of so-called
[bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model) models. 


In reducing a large number of documents to a much smaller number of topics,
PSLA can be seen as an example of unsupervised
[dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction),
most related to
[non-negative matrix factorization](https://en.wikipedia.org/wiki/Dimensionality_reduction).

To give an example, a bunch of documents might frequently contain words like
"eating", "nutrition", "health", _etc._ Others might contain words like "state",
"party", "ministry", _etc_. Yet others might contain words like "tournament",
"ranking", "win", _etc._ It is easy to imagine there being documents that
contain a mixture of these words. Not knowing in advance how many topics there
are, one would have to run PLSA with several different numbers of topics and
see the results to judge how many is a good choice. Picking three in our example
would yield topics that could be described as "food", "politics", and "sports"
and, while a number of documents will emerge as being purely about one of these
topics, it is easy to imagine that there are others that have contributions
from more than one topic (_e.g._, about a new initiative from the ministry
of health, combining "food" and "politics"). PLSA will give you that
mixture.

## Installation
This code is available on the python package index [PyPi](https://pypi.org/).
To install, I _strongly_ recommend setting up a new virtual python environment,
and then type
```bash
pip install plsa
```
on the console.

__WARNING__: _On first use, some components of `nltk` that don't come with it out-of-the-box
wil be downloaded. Should you install (against my express recommendation) install
the `plsa` package system-wide (with `sudo`), then you lack the access rights
to write the required `nltk` data to where it is supposed to go (into a subfolder of
the `plsa` package directory)._

#### Dependencies
This package depends on the following python packages:
- `numpy`
- `matplotlib`
- `wordcould`
- `nltk`

If you want to run the
[example notebook](https://github.com/yedivanseven/PLSA/blob/master/notebooks/Examples.ipynb),
you will also need to install the `jupyter` package.

## Getting Started
Clone the
[GitHub repository](https://github.com/yedivanseven/PLSA)
and run the `jupyter` notebook
[Examples.ipynb](https://github.com/yedivanseven/PLSA/blob/master/notebooks/Examples.ipynb)
in the
[notebooks](https://github.com/yedivanseven/PLSA/tree/master/notebooks)
folder.

## Documentation
Read the [API documentation](https://probabilistic-latent-semantic-analysis.readthedocs.io/en/latest/index.html) on [Read the Docs](https://readthedocs.org/)

## Technical considerations
The matrices to store and manipulate data can easily get quite large. That
means you will soon run out of memory when toying with a large corpus. This
could be mitigated to some extent by using sparse matrices. But since there
is no built-in support for sparse matrices of more than 2 dimensions
(we need 3) in `scipy`, this is not implemented.


