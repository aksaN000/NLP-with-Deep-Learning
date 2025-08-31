# Natural Language Processing with Deep Learning

A comprehensive exploration of Natural Language Processing (NLP) techniques, from traditional methods to modern deep learning approaches. This project demonstrates various NLP concepts including text preprocessing, word embeddings, sentiment analysis, and neural network implementations.

## Project Structure

```
├── part1.ipynb          # NLTK Fundamentals & Text Analysis
├── part2.ipynb          # Word Embeddings & Advanced NLP
├── part3.ipynb          # Deep Learning for NLP
└── README.md           # Project documentation
```

## Overview

This project covers a comprehensive range of NLP techniques and methodologies:

### Part 1: NLTK Fundamentals & Text Analysis
- **Text Preprocessing**: Tokenization, stemming, lemmatization
- **Corpus Exploration**: Working with Gutenberg, Inaugural, and State of Union corpora
- **Sentiment Analysis**: Using VADER sentiment analyzer
- **Word Frequency Analysis**: Creating word clouds and frequency distributions
- **N-gram Analysis**: Bigrams, trigrams, and co-occurrence matrices
- **Part-of-Speech Tagging**: Grammatical analysis of text

### Part 2: Word Embeddings & Advanced NLP
- **Word2Vec Models**: Implementation of Skip-gram and CBOW architectures
- **Pre-trained Embeddings**: Working with GloVe embeddings
- **Word Analogy Tasks**: Semantic relationships (e.g., "King - Man + Woman = Queen")
- **Text Classification**: Sentiment classification using TF-IDF and Logistic Regression
- **Dimensionality Reduction**: PCA visualization of word embeddings
- **Corpus Analysis**: Reuters corpus processing and analysis

### Part 3: Deep Learning for NLP
- **Neural Networks**: Building shallow and deep networks with TensorFlow/Keras
- **Embedding Layers**: Learning word representations in neural networks
- **Sentiment Classification**: IMDB movie review sentiment analysis
- **Model Evaluation**: Comprehensive performance analysis and visualization
- **CUDA Integration**: GPU acceleration for deep learning models

## Technologies Used

- **Python Libraries**:
  - `NLTK` - Natural Language Toolkit for text processing
  - `Gensim` - Word2Vec model implementation
  - `TensorFlow/Keras` - Deep learning framework
  - `Scikit-learn` - Machine learning tools and metrics
  - `Pandas` & `NumPy` - Data manipulation and numerical computing
  - `Matplotlib` & `Seaborn` - Data visualization
  - `WordCloud` - Word cloud generation

- **Datasets**:
  - NLTK Corpora (Gutenberg, Inaugural, Reuters, State of Union)
  - GloVe Pre-trained Embeddings
  - IMDB Movie Reviews Dataset

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aksaN000/Deep-Learning.git
   cd Deep-Learning
   ```

2. **Install required packages**:
   ```bash
   pip install nltk gensim tensorflow scikit-learn pandas numpy matplotlib seaborn wordcloud
   ```

3. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('all')
   ```

## Key Features & Implementations

### Word Embeddings Analysis
- Comparison between Skip-gram and CBOW models
- Semantic similarity evaluation
- Word analogy benchmarking
- Vector space visualization

### Sentiment Analysis Pipeline
- Traditional ML approach with TF-IDF features
- Deep learning approach with embedding layers
- Comprehensive model evaluation and comparison

### Text Preprocessing & Analysis
- Robust text cleaning and normalization
- Statistical analysis of text corpora
- Frequency distribution analysis
- Collocation discovery

## Results & Insights

The project demonstrates:
- **High Performance**: Achieved 89%+ accuracy on sentiment classification tasks
- **Semantic Understanding**: Successfully solved word analogy tasks with high precision
- **Model Comparison**: Comprehensive evaluation of traditional vs. deep learning approaches
- **Visualization**: Clear representation of word relationships and semantic clusters

## Learning Outcomes

This project provides hands-on experience with:
- Fundamental NLP concepts and preprocessing techniques
- Word embedding generation and evaluation
- Deep learning architectures for text classification
- Model evaluation and performance optimization
- Visualization of high-dimensional text data

## Usage Examples

### Quick Start with Sentiment Analysis
```python
# Load the trained model
from tensorflow.keras.models import load_model
model = load_model('sentiment_model.h5')

# Analyze sentiment
text = "This movie is absolutely fantastic!"
prediction = model.predict(preprocess_text(text))
sentiment = "Positive" if prediction > 0.5 else "Negative"
```

### Word Similarity with Pre-trained Embeddings
```python
# Load GloVe embeddings
embeddings = load_glove_embeddings('glove.6B.100d.txt')

# Find similar words
similar_words = find_most_similar('king', embeddings, top_k=5)
print(similar_words)
```

## Contributing

Feel free to contribute to this project by:
- Adding new NLP techniques or models
- Improving existing implementations
- Adding more comprehensive documentation
- Suggesting performance optimizations

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- NLTK team for the comprehensive natural language toolkit
- Stanford NLP Group for GloVe embeddings
- TensorFlow team for the deep learning framework
- Open source community for various NLP datasets and tools

---

**Note**: This project is designed for educational purposes to demonstrate various NLP techniques and their practical implementations. The code is well-documented and suitable for learning and experimentation.
