from pkg_resources import resource_filename
from numpy import seterr

import nltk

nltk_data_dir = resource_filename(__name__, 'nltk_data')
nltk.data.path.append(nltk_data_dir)
try:
    from .pipeline import Pipeline
    from .corpus import Corpus
    from .visualize import Visualize
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('wordnet', download_dir=nltk_data_dir)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
    from .pipeline import Pipeline
    from .corpus import Corpus
    from .visualize import Visualize

seterr(all='raise')
