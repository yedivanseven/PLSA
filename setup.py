from os import path
from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='plsa',
    version='0.1.1',
    description='Probabilistic Latent Semantic Analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yedivanseven/PLSA',
    download_url='https://pypi.python.org/pypi/plsa',
    author='Georg Heimel',
    author_email='georg@muckisnspirit.com',
    license='',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing'
    ],
    keywords='nlp bag-of-words',
    packages=find_packages(include=['plsa', 'plsa.*']),
    project_urls={
        'Documentation': 'https://probabilistic-latent-semantic-analysis.readthedocs.io/en/latest/index.html'
    },
    install_requires=[
        'matplotlib>=3.0',
        'nltk>=3.4.5',
        'numpy>=1.16',
        'wordcloud>=1.5'
    ]
)
