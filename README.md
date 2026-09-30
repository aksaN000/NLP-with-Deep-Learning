# NLP with Deep Learning

My natural language processing coursework in three notebooks. It runs from classic text processing with NLTK, through word embeddings, to neural networks for sentiment analysis. It also includes a small PyTorch transformer and preprocessing utilities I wrote alongside it.

## Notebooks

| Notebook | What it covers |
|---|---|
| `part1.ipynb` | **NLTK foundations:** corpora, tokenisation, stemming, part-of-speech tagging and parsing |
| `part2.ipynb` | **Classic NLP and embeddings:**<br>(1) a TF-IDF sentiment classifier with evaluation, feature analysis and custom-review testing;<br>(2) word analogies with pre-trained GloVe vectors (*king − man + woman*);<br>(3) Word2Vec trained on the Reuters corpus with gensim, evaluated on similarity and analogy tasks and visualised with PCA |
| `part3.ipynb` | **Neural networks:**<br>a feed-forward classifier for the Wine dataset;<br>IMDB sentiment with a shallow network, then a deeper network with dropout, compared on training and validation curves |

## Supporting code

| File | Contents |
|---|---|
| `models/advanced_models.py` | A transformer encoder in PyTorch: multi-head attention, positional encoding, transformer blocks and a training loop |
| `utils/nlp_utils.py` | Text preprocessing, corpus statistics, embedding helpers and evaluation metrics |
| `utils/advanced_preprocessing.py` | A configurable preprocessing pipeline with language detection (langdetect), spaCy processing and text-quality checks |

`PROJECT_REPORT.md` summarises the methods and results from all three notebooks.

## Tech stack

Python · NLTK · scikit-learn · gensim · TensorFlow / Keras · PyTorch · spaCy · pandas · Matplotlib · Jupyter

## Running locally

```bash
git clone https://github.com/aksaN000/NLP-with-Deep-Learning.git
cd NLP-with-Deep-Learning
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger reuters
jupyter notebook
```

Part 2 also needs the GloVe vectors (`glove.6B.100d.txt` from the Stanford NLP site); the notebook's first cells show where it expects the file.

## License

MIT, see [LICENSE](LICENSE).
