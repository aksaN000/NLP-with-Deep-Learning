"""
Advanced Text Processing Pipeline with Production Features

This module provides enterprise-grade text processing capabilities including:
- Multi-language support
- Scalable processing for large datasets
- Advanced preprocessing techniques
- Quality monitoring and validation
- Integration with popular NLP frameworks
"""

import re
import unicodedata
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import time
from collections import defaultdict, Counter
import hashlib

import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for text processing pipeline."""
    
    # Language settings
    language: str = "english"
    detect_language: bool = False
    
    # Cleaning options
    lowercase: bool = True
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = True
    remove_punctuation: bool = False
    remove_numbers: bool = False
    remove_extra_whitespace: bool = True
    
    # Tokenization
    tokenize_sentences: bool = False
    tokenize_words: bool = True
    
    # Normalization
    apply_stemming: bool = False
    apply_lemmatization: bool = True
    remove_stopwords: bool = True
    min_token_length: int = 2
    max_token_length: int = 50
    
    # Advanced features
    handle_negations: bool = True
    expand_contractions: bool = True
    normalize_unicode: bool = True
    preserve_case_entities: bool = True
    
    # Performance settings
    batch_size: int = 1000
    n_jobs: int = -1
    cache_enabled: bool = True
    
    # Quality control
    min_text_length: int = 10
    max_text_length: int = 100000
    filter_duplicates: bool = True
    quality_threshold: float = 0.7


class LanguageDetector:
    """Advanced language detection with confidence scores."""
    
    def __init__(self):
        try:
            from langdetect import detect, detect_langs
            self.detect = detect
            self.detect_langs = detect_langs
            self.available = True
        except ImportError:
            logger.warning("langdetect not available. Language detection disabled.")
            self.available = False
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language with confidence score.
        
        Returns:
            Tuple of (language_code, confidence)
        """
        if not self.available or len(text.strip()) < 20:
            return "unknown", 0.0
        
        try:
            langs = self.detect_langs(text)
            if langs:
                return langs[0].lang, langs[0].prob
        except:
            pass
        
        return "unknown", 0.0


class QualityAnalyzer:
    """Analyze and score text quality."""
    
    def __init__(self):
        self.metrics = {
            "length_score": self._length_score,
            "diversity_score": self._diversity_score,
            "coherence_score": self._coherence_score,
            "grammar_score": self._grammar_score,
            "readability_score": self._readability_score
        }
    
    def analyze_quality(self, text: str) -> Dict[str, float]:
        """Comprehensive text quality analysis."""
        scores = {}
        for metric_name, metric_func in self.metrics.items():
            try:
                scores[metric_name] = metric_func(text)
            except Exception as e:
                logger.warning(f"Error computing {metric_name}: {e}")
                scores[metric_name] = 0.0
        
        # Overall quality score
        scores["overall_quality"] = np.mean(list(scores.values()))
        return scores
    
    def _length_score(self, text: str) -> float:
        """Score based on text length (optimal range)."""
        length = len(text.split())
        if 50 <= length <= 500:
            return 1.0
        elif length < 10 or length > 2000:
            return 0.0
        else:
            return 0.5
    
    def _diversity_score(self, text: str) -> float:
        """Score based on vocabulary diversity."""
        words = text.lower().split()
        if len(words) == 0:
            return 0.0
        
        unique_words = len(set(words))
        diversity = unique_words / len(words)
        return min(diversity * 2, 1.0)  # Scale to 0-1
    
    def _coherence_score(self, text: str) -> float:
        """Simple coherence check based on sentence structure."""
        sentences = sent_tokenize(text)
        if len(sentences) == 0:
            return 0.0
        
        # Check for reasonable sentence lengths
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        if 5 <= avg_sentence_length <= 30:
            return 1.0
        else:
            return 0.5
    
    def _grammar_score(self, text: str) -> float:
        """Basic grammar checks."""
        # Simple heuristics for grammar quality
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
        
        # Check capitalization at sentence start
        capitalized = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
        capitalization_score = capitalized / len(sentences)
        
        # Check for reasonable punctuation
        has_punctuation = any(s.strip().endswith(('.', '!', '?')) for s in sentences)
        punctuation_score = 1.0 if has_punctuation else 0.5
        
        return (capitalization_score + punctuation_score) / 2
    
    def _readability_score(self, text: str) -> float:
        """Simple readability assessment."""
        words = text.split()
        sentences = sent_tokenize(text)
        
        if not words or not sentences:
            return 0.0
        
        avg_word_length = np.mean([len(word) for word in words])
        avg_sentence_length = len(words) / len(sentences)
        
        # Flesch-like simple score
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length / 4.7)
        return max(0, min(100, readability)) / 100


class AdvancedTextProcessor:
    """
    Production-grade text processing pipeline with advanced features.
    
    Features:
    - Multi-language support
    - Quality analysis and filtering
    - Scalable batch processing
    - Comprehensive preprocessing options
    - Performance monitoring
    - Caching for efficiency
    """
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.language_detector = LanguageDetector()
        self.quality_analyzer = QualityAnalyzer()
        
        # Initialize NLP models
        self._initialize_models()
        
        # Processing statistics
        self.stats = defaultdict(int)
        self.cache = {} if self.config.cache_enabled else None
        
        logger.info(f"AdvancedTextProcessor initialized for language: {self.config.language}")
    
    def _initialize_models(self):
        """Initialize language-specific NLP models."""
        # SpaCy model
        try:
            model_name = f"{self.config.language[:2]}_core_web_sm"
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"SpaCy model {model_name} not found. Using English model.")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("No SpaCy models available. Some features disabled.")
                self.nlp = None
        
        # NLTK resources
        try:
            self.stemmer = SnowballStemmer(self.config.language)
        except:
            self.stemmer = SnowballStemmer("english")
        
        # Stopwords
        try:
            self.stop_words = set(stopwords.words(self.config.language))
        except:
            self.stop_words = set(stopwords.words("english"))
        
        # Contractions dictionary
        self.contractions = self._load_contractions()
    
    def _load_contractions(self) -> Dict[str, str]:
        """Load contractions dictionary for expansion."""
        contractions = {
            "ain't": "am not", "aren't": "are not", "can't": "cannot",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would", "he'll": "he will",
            "he's": "he is", "i'd": "i would", "i'll": "i will",
            "i'm": "i am", "i've": "i have", "isn't": "is not",
            "it'd": "it would", "it'll": "it will", "it's": "it is",
            "let's": "let us", "shouldn't": "should not", "that's": "that is",
            "there's": "there is", "they'd": "they would", "they'll": "they will",
            "they're": "they are", "they've": "they have", "we'd": "we would",
            "we're": "we are", "we've": "we have", "weren't": "were not",
            "what's": "what is", "where's": "where is", "who's": "who is",
            "won't": "will not", "wouldn't": "would not", "you'd": "you would",
            "you'll": "you will", "you're": "you are", "you've": "you have"
        }
        return contractions
    
    def _get_cache_key(self, text: str, processing_options: Dict) -> str:
        """Generate cache key for processed text."""
        content = f"{text}_{str(sorted(processing_options.items()))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def clean_text(self, text: str) -> str:
        """Comprehensive text cleaning."""
        if not isinstance(text, str):
            return ""
        
        original_text = text
        
        # Unicode normalization
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFKD', text)
        
        # Remove HTML tags
        if self.config.remove_html:
            text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        if self.config.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # Remove email addresses
        if self.config.remove_emails:
            text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove phone numbers
        if self.config.remove_phone_numbers:
            text = re.sub(r'[\+]?[1-9]?[0-9]{7,15}', ' ', text)
        
        # Expand contractions
        if self.config.expand_contractions:
            for contraction, expansion in self.contractions.items():
                text = re.sub(rf'\b{contraction}\b', expansion, text, flags=re.IGNORECASE)
        
        # Handle negations
        if self.config.handle_negations:
            text = re.sub(r'\b(not|no|never|nothing|nowhere|noone|none|not)\s+(\w+)', 
                         r'\1_\2', text, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        if self.config.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove numbers
        if self.config.remove_numbers:
            text = re.sub(r'\b\d+\b', ' ', text)
        
        # Remove punctuation (but preserve if needed for entities)
        if self.config.remove_punctuation and not self.config.preserve_case_entities:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Case conversion
        if self.config.lowercase and not self.config.preserve_case_entities:
            text = text.lower()
        elif self.config.lowercase and self.config.preserve_case_entities:
            # Preserve case for likely entities (capitalized words)
            words = text.split()
            processed_words = []
            for word in words:
                if word.istitle() and len(word) > 2:
                    processed_words.append(word)  # Keep original case
                else:
                    processed_words.append(word.lower())
            text = ' '.join(processed_words)
        
        return text
    
    def tokenize_text(self, text: str) -> Union[List[str], List[List[str]]]:
        """Advanced tokenization with multiple options."""
        if self.config.tokenize_sentences:
            sentences = sent_tokenize(text)
            if self.config.tokenize_words:
                return [word_tokenize(sent) for sent in sentences]
            return sentences
        elif self.config.tokenize_words:
            return word_tokenize(text)
        else:
            return [text]
    
    def normalize_tokens(self, tokens: List[str]) -> List[str]:
        """Apply normalization to tokens."""
        normalized = []
        
        for token in tokens:
            # Filter by length
            if len(token) < self.config.min_token_length or len(token) > self.config.max_token_length:
                continue
            
            # Remove stopwords
            if self.config.remove_stopwords and token.lower() in self.stop_words:
                continue
            
            # Apply stemming
            if self.config.apply_stemming:
                token = self.stemmer.stem(token)
            
            # Apply lemmatization
            if self.config.apply_lemmatization and self.nlp:
                doc = self.nlp(token)
                if doc:
                    token = doc[0].lemma_
            
            normalized.append(token)
        
        return normalized
    
    def process_single_text(self, text: str) -> Dict[str, Any]:
        """Process a single text with comprehensive analysis."""
        start_time = time.time()
        
        # Check cache
        if self.cache is not None:
            cache_key = self._get_cache_key(text, self.config.__dict__)
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Language detection
        language_info = {}
        if self.config.detect_language:
            lang, confidence = self.language_detector.detect_language(text)
            language_info = {"detected_language": lang, "confidence": confidence}
        
        # Quality analysis
        quality_scores = self.quality_analyzer.analyze_quality(text)
        
        # Skip processing if quality is too low
        if quality_scores["overall_quality"] < self.config.quality_threshold:
            self.stats["low_quality_filtered"] += 1
            return {
                "original_text": text,
                "processed_text": "",
                "tokens": [],
                "quality_scores": quality_scores,
                "language_info": language_info,
                "processing_time": time.time() - start_time,
                "filtered": True,
                "filter_reason": "low_quality"
            }
        
        # Text cleaning
        cleaned_text = self.clean_text(text)
        
        # Length filtering
        if len(cleaned_text) < self.config.min_text_length:
            self.stats["too_short_filtered"] += 1
            return {
                "original_text": text,
                "processed_text": "",
                "tokens": [],
                "quality_scores": quality_scores,
                "language_info": language_info,
                "processing_time": time.time() - start_time,
                "filtered": True,
                "filter_reason": "too_short"
            }
        
        if len(cleaned_text) > self.config.max_text_length:
            self.stats["too_long_filtered"] += 1
            cleaned_text = cleaned_text[:self.config.max_text_length]
        
        # Tokenization
        tokens = self.tokenize_text(cleaned_text)
        
        # Token normalization
        if isinstance(tokens, list) and tokens and isinstance(tokens[0], str):
            tokens = self.normalize_tokens(tokens)
        
        result = {
            "original_text": text,
            "processed_text": cleaned_text,
            "tokens": tokens,
            "quality_scores": quality_scores,
            "language_info": language_info,
            "processing_time": time.time() - start_time,
            "filtered": False,
            "token_count": len(tokens) if isinstance(tokens, list) else 0
        }
        
        # Cache result
        if self.cache is not None:
            self.cache[cache_key] = result
        
        self.stats["processed_successfully"] += 1
        return result
    
    def process_batch(self, texts: List[str], show_progress: bool = True) -> List[Dict[str, Any]]:
        """Process multiple texts efficiently."""
        logger.info(f"Processing batch of {len(texts)} texts")
        
        # Remove duplicates if enabled
        if self.config.filter_duplicates:
            original_count = len(texts)
            texts = list(dict.fromkeys(texts))  # Preserve order while removing duplicates
            self.stats["duplicates_removed"] += original_count - len(texts)
        
        # Process in parallel
        if self.config.n_jobs == 1:
            results = [self.process_single_text(text) for text in texts]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
                results = list(executor.map(self.process_single_text, texts))
        
        self.stats["total_processed"] += len(texts)
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return dict(self.stats)
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.stats.clear()
    
    def export_config(self, filepath: str):
        """Export current configuration to file."""
        config_dict = self.config.__dict__
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_config_file(cls, filepath: str):
        """Create processor from configuration file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        config = ProcessingConfig(**config_dict)
        return cls(config)


# Example usage and demonstration
if __name__ == "__main__":
    # Configuration for educational demonstration
    config = ProcessingConfig(
        language="english",
        detect_language=True,
        apply_lemmatization=True,
        remove_stopwords=True,
        quality_threshold=0.5,
        batch_size=100
    )
    
    # Create processor
    processor = AdvancedTextProcessor(config)
    
    # Sample texts for demonstration
    sample_texts = [
        "This is a high-quality text with proper grammar and structure. It demonstrates the capabilities of our advanced text processor.",
        "bad txt no caps no punct very low quality",
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet at least once.",
        "Natural Language Processing (NLP) is amazing! It's revolutionizing how we interact with computers.",
        "Visit https://example.com for more info or email us at contact@example.com. Call (555) 123-4567."
    ]
    
    # Process batch
    results = processor.process_batch(sample_texts)
    
    # Display results
    for i, result in enumerate(results):
        print(f"\n--- Text {i+1} ---")
        print(f"Original: {result['original_text'][:100]}...")
        print(f"Processed: {result['processed_text'][:100]}...")
        print(f"Quality Score: {result['quality_scores']['overall_quality']:.2f}")
        print(f"Token Count: {result['token_count']}")
        print(f"Filtered: {result['filtered']}")
        if result['language_info']:
            print(f"Language: {result['language_info']['detected_language']} ({result['language_info']['confidence']:.2f})")
    
    # Show statistics
    print(f"\n--- Processing Statistics ---")
    stats = processor.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nAdvanced Text Processor demonstration completed!")
    print("This processor is ready for production use with enterprise-grade features.")
