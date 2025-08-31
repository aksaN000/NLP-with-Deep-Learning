"""
Comprehensive NLP Utilities and Helper Functions

This module provides a collection of utility functions for natural language processing
tasks. These functions are designed to be reusable, efficient, and well-documented
to support educational and research purposes.

Author: NLP Education Community
License: MIT
"""

import re
import string
import unicodedata
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
import warnings


class TextPreprocessor:
    """
    A comprehensive text preprocessing class with multiple cleaning options.
    
    This class provides various text cleaning and normalization methods
    that can be chained together for flexible preprocessing pipelines.
    """
    
    def __init__(self, language: str = 'english'):
        """
        Initialize the preprocessor.
        
        Args:
            language: Language for stopword removal (default: 'english')
        """
        self.language = language
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                nltk.download(data, quiet=True)
    
    def clean_text(self, text: str, 
                   lowercase: bool = True,
                   remove_punctuation: bool = True,
                   remove_numbers: bool = False,
                   remove_extra_whitespace: bool = True) -> str:
        """
        Basic text cleaning operations.
        
        Args:
            text: Input text to clean
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation marks
            remove_numbers: Remove numeric characters
            remove_extra_whitespace: Remove extra spaces and newlines
        
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        if lowercase:
            text = text.lower()
        
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        if remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_filter(self, text: str,
                           min_length: int = 2,
                           remove_stopwords: bool = True,
                           apply_stemming: bool = False,
                           apply_lemmatization: bool = False) -> List[str]:
        """
        Tokenize text and apply various filtering options.
        
        Args:
            text: Input text to tokenize
            min_length: Minimum token length
            remove_stopwords: Remove stopwords
            apply_stemming: Apply Porter stemming
            apply_lemmatization: Apply WordNet lemmatization
        
        Returns:
            List of filtered tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter by length
        tokens = [token for token in tokens if len(token) >= min_length]
        
        # Remove stopwords
        if remove_stopwords:
            stop_words = set(stopwords.words(self.language))
            tokens = [token for token in tokens if token.lower() not in stop_words]
        
        # Apply stemming
        if apply_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Apply lemmatization
        if apply_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def preprocess_pipeline(self, text: str, **kwargs) -> List[str]:
        """
        Complete preprocessing pipeline combining cleaning and tokenization.
        
        Args:
            text: Input text
            **kwargs: Arguments for clean_text and tokenize_and_filter
        
        Returns:
            List of preprocessed tokens
        """
        # Separate kwargs for different methods
        clean_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['lowercase', 'remove_punctuation', 'remove_numbers', 'remove_extra_whitespace']}
        token_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['min_length', 'remove_stopwords', 'apply_stemming', 'apply_lemmatization']}
        
        cleaned_text = self.clean_text(text, **clean_kwargs)
        tokens = self.tokenize_and_filter(cleaned_text, **token_kwargs)
        
        return tokens


class TextAnalyzer:
    """
    Statistical analysis tools for text data.
    """
    
    @staticmethod
    def frequency_analysis(tokens: List[str], top_k: int = 20) -> Dict[str, int]:
        """
        Analyze token frequencies.
        
        Args:
            tokens: List of tokens
            top_k: Number of top frequent tokens to return
        
        Returns:
            Dictionary of token frequencies
        """
        counter = Counter(tokens)
        return dict(counter.most_common(top_k))
    
    @staticmethod
    def vocabulary_stats(tokens: List[str]) -> Dict[str, Union[int, float]]:
        """
        Calculate vocabulary statistics.
        
        Args:
            tokens: List of tokens
        
        Returns:
            Dictionary with vocabulary statistics
        """
        unique_tokens = set(tokens)
        return {
            'total_tokens': len(tokens),
            'unique_tokens': len(unique_tokens),
            'vocabulary_diversity': len(unique_tokens) / len(tokens) if tokens else 0,
            'average_token_length': np.mean([len(token) for token in tokens]) if tokens else 0
        }
    
    @staticmethod
    def ngram_analysis(tokens: List[str], n: int = 2, top_k: int = 10) -> List[Tuple[str, int]]:
        """
        Extract and analyze n-grams.
        
        Args:
            tokens: List of tokens
            n: N-gram size
            top_k: Number of top n-grams to return
        
        Returns:
            List of (n-gram, frequency) tuples
        """
        if len(tokens) < n:
            return []
        
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        ngram_strings = [' '.join(ngram) for ngram in ngrams]
        counter = Counter(ngram_strings)
        
        return counter.most_common(top_k)
    
    @staticmethod
    def pos_tag_analysis(text: str) -> Dict[str, int]:
        """
        Analyze part-of-speech tag distribution.
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with POS tag frequencies
        """
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        tag_freq = Counter([tag for _, tag in pos_tags])
        
        return dict(tag_freq)


class EmbeddingUtils:
    """
    Utilities for working with word embeddings.
    """
    
    @staticmethod
    def load_glove_embeddings(filepath: str, vocab_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Load GloVe embeddings from file.
        
        Args:
            filepath: Path to GloVe file
            vocab_size: Maximum vocabulary size to load
        
        Returns:
            Dictionary mapping words to embedding vectors
        """
        embeddings = {}
        loaded_count = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if vocab_size and loaded_count >= vocab_size:
                        break
                    
                    values = line.strip().split()
                    if len(values) < 2:
                        continue
                    
                    word = values[0]
                    vector = np.array(values[1:], dtype=np.float32)
                    embeddings[word] = vector
                    loaded_count += 1
        
        except FileNotFoundError:
            warnings.warn(f"GloVe file not found at {filepath}")
            return {}
        
        return embeddings
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def find_most_similar(target_word: str, 
                         embeddings: Dict[str, np.ndarray], 
                         top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar words to target word.
        
        Args:
            target_word: Word to find similarities for
            embeddings: Dictionary of word embeddings
            top_k: Number of similar words to return
        
        Returns:
            List of (word, similarity_score) tuples
        """
        if target_word not in embeddings:
            return []
        
        target_vector = embeddings[target_word]
        similarities = []
        
        for word, vector in embeddings.items():
            if word != target_word:
                similarity = EmbeddingUtils.cosine_similarity(target_vector, vector)
                similarities.append((word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    @staticmethod
    def word_analogy(word_a: str, word_b: str, word_c: str,
                    embeddings: Dict[str, np.ndarray],
                    top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Solve word analogies: A is to B as C is to ?
        
        Args:
            word_a: First word in analogy
            word_b: Second word in analogy
            word_c: Third word in analogy
            embeddings: Dictionary of word embeddings
            top_k: Number of candidates to return
        
        Returns:
            List of (word, similarity_score) tuples
        """
        required_words = [word_a, word_b, word_c]
        for word in required_words:
            if word not in embeddings:
                return []
        
        # Calculate: B - A + C
        result_vector = (embeddings[word_b] - embeddings[word_a] + embeddings[word_c])
        
        similarities = []
        exclude_words = set(required_words)
        
        for word, vector in embeddings.items():
            if word not in exclude_words:
                similarity = EmbeddingUtils.cosine_similarity(result_vector, vector)
                similarities.append((word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class EvaluationMetrics:
    """
    Evaluation metrics for NLP tasks.
    """
    
    @staticmethod
    def classification_metrics(y_true: List, y_pred: List) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary with metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
    
    @staticmethod
    def embedding_evaluation(embeddings: Dict[str, np.ndarray],
                           analogy_pairs: List[Tuple[str, str, str, str]]) -> float:
        """
        Evaluate embeddings on word analogy tasks.
        
        Args:
            embeddings: Word embeddings dictionary
            analogy_pairs: List of (A, B, C, D) analogy tuples
        
        Returns:
            Accuracy score on analogy tasks
        """
        correct = 0
        total = 0
        
        for a, b, c, d in analogy_pairs:
            if all(word in embeddings for word in [a, b, c, d]):
                predictions = EmbeddingUtils.word_analogy(a, b, c, embeddings, top_k=1)
                if predictions and predictions[0][0] == d:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0


# Example usage and demonstrations
if __name__ == "__main__":
    # Example usage of the utilities
    
    # Text preprocessing example
    preprocessor = TextPreprocessor()
    sample_text = "This is a sample text for demonstration! It contains various words, punctuation, and Numbers123."
    
    # Basic cleaning
    cleaned = preprocessor.clean_text(sample_text)
    print(f"Cleaned text: {cleaned}")
    
    # Full preprocessing pipeline
    tokens = preprocessor.preprocess_pipeline(
        sample_text,
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        apply_lemmatization=True
    )
    print(f"Processed tokens: {tokens}")
    
    # Text analysis
    analyzer = TextAnalyzer()
    freq_analysis = analyzer.frequency_analysis(tokens)
    print(f"Frequency analysis: {freq_analysis}")
    
    vocab_stats = analyzer.vocabulary_stats(tokens)
    print(f"Vocabulary statistics: {vocab_stats}")
    
    print("\nNLP Utilities loaded successfully!")
    print("These utilities are designed for educational and research purposes.")
    print("Contribute to make NLP more accessible worldwide!")
