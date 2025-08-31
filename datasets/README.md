# Dataset Collection for NLP Education

This directory contains curated datasets for educational purposes, along with scripts for preprocessing and analysis.

## Available Datasets

### Text Classification
- **Movie Reviews**: IMDB sentiment analysis dataset
- **News Categories**: Reuters news categorization
- **Spam Detection**: Email spam classification
- **Product Reviews**: Amazon product review sentiment

### Named Entity Recognition
- **CoNLL-2003**: Standard NER benchmark dataset
- **OntoNotes 5.0**: Multilingual named entity recognition
- **WikiNER**: Automatically annotated NER dataset

### Machine Translation
- **OPUS Collections**: Parallel corpora for multiple language pairs
- **WMT Datasets**: Workshop on Machine Translation datasets
- **OpenSubtitles**: Movie subtitle translations

### Question Answering
- **SQuAD**: Stanford Question Answering Dataset
- **Natural Questions**: Real Google search queries
- **MS MARCO**: Microsoft dataset for machine reading comprehension

### Text Summarization
- **CNN/DailyMail**: News article summarization
- **XSum**: BBC article summarization
- **Multi-News**: Multi-document summarization

## Data Loading Utilities

```python
from datasets import load_dataset
import pandas as pd

class DatasetLoader:
    """Utility class for loading and preprocessing NLP datasets."""
    
    @staticmethod
    def load_imdb_reviews():
        """Load IMDB movie review dataset."""
        dataset = load_dataset('imdb')
        return dataset
    
    @staticmethod
    def load_conll_ner():
        """Load CoNLL-2003 NER dataset."""
        dataset = load_dataset('conll2003')
        return dataset
    
    @staticmethod
    def create_custom_dataset(texts, labels, split_ratio=0.8):
        """Create a custom dataset with train/test split."""
        df = pd.DataFrame({'text': texts, 'label': labels})
        train_size = int(len(df) * split_ratio)
        
        train_df = df[:train_size]
        test_df = df[train_size:]
        
        return train_df, test_df
```

## Preprocessing Scripts

### Text Cleaning
```python
def clean_dataset(dataset, text_column='text'):
    """Clean text data in a dataset."""
    preprocessor = TextPreprocessor()
    
    dataset[text_column] = dataset[text_column].apply(
        lambda x: preprocessor.clean_text(x, lowercase=True, remove_punctuation=True)
    )
    
    return dataset
```

### Data Augmentation
```python
def augment_text_data(texts, augmentation_factor=2):
    """Augment text data using various techniques."""
    augmented_texts = []
    
    for text in texts:
        # Original text
        augmented_texts.append(text)
        
        # Synonym replacement
        augmented_texts.append(synonym_replacement(text))
        
        # Random insertion
        if augmentation_factor > 2:
            augmented_texts.append(random_insertion(text))
    
    return augmented_texts
```

## Evaluation Datasets

### Benchmark Tasks
- **GLUE**: General Language Understanding Evaluation
- **SuperGLUE**: More challenging language understanding tasks
- **XTREME**: Cross-lingual benchmark for multilingual models

### Domain-Specific
- **BioBERT**: Biomedical text mining datasets
- **FinBERT**: Financial domain datasets
- **LegalBERT**: Legal document analysis datasets

## Synthetic Data Generation

```python
class SyntheticDataGenerator:
    """Generate synthetic datasets for educational purposes."""
    
    def generate_sentiment_data(self, num_samples=1000):
        """Generate synthetic sentiment analysis data."""
        positive_templates = [
            "I love {item}, it's {adjective}!",
            "This {item} is {adjective} and {adjective2}.",
            "{item} exceeded my expectations, very {adjective}."
        ]
        
        negative_templates = [
            "I hate {item}, it's {adjective}.",
            "This {item} is {adjective} and {adjective2}.",
            "{item} disappointed me, very {adjective}."
        ]
        
        # Generate samples using templates
        # Implementation details...
    
    def generate_ner_data(self, num_samples=500):
        """Generate synthetic NER training data."""
        # Implementation for synthetic NER data generation
        pass
```

## Multi-language Support

### Language-Specific Datasets
- **English**: Comprehensive coverage across all tasks
- **Spanish**: News, reviews, and social media data
- **Chinese**: Weibo sentiment, news classification
- **Arabic**: News articles, social media posts
- **Hindi**: Movie reviews, news categorization

### Cross-lingual Datasets
- **XNLI**: Cross-lingual Natural Language Inference
- **XQuAD**: Cross-lingual Question Answering
- **WikiANN**: Cross-lingual Named Entity Recognition

## Data Ethics and Privacy

### Guidelines
1. **Consent**: Ensure all data is properly licensed
2. **Privacy**: Remove or anonymize personal information
3. **Bias**: Document known biases in datasets
4. **Representation**: Ensure diverse representation

### Privacy-Preserving Techniques
```python
def anonymize_dataset(dataset, sensitive_columns):
    """Anonymize sensitive information in datasets."""
    import hashlib
    
    for column in sensitive_columns:
        if column in dataset.columns:
            dataset[column] = dataset[column].apply(
                lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:10]
            )
    
    return dataset
```

## Contributing Datasets

### Guidelines for Contributors
1. **Quality**: Ensure high-quality annotations
2. **Documentation**: Provide comprehensive metadata
3. **Licensing**: Use appropriate open-source licenses
4. **Validation**: Include quality checks and statistics

### Dataset Submission Process
1. Prepare dataset according to standard formats
2. Create metadata file with task description
3. Include baseline results and evaluation scripts
4. Submit via pull request with documentation

## Usage Examples

### Loading and Exploring Data
```python
# Load a dataset
from utils.dataset_loader import DatasetLoader

loader = DatasetLoader()
imdb_data = loader.load_imdb_reviews()

# Explore the data
print(f"Dataset size: {len(imdb_data['train'])}")
print(f"Sample text: {imdb_data['train'][0]['text'][:100]}...")
print(f"Label: {imdb_data['train'][0]['label']}")

# Basic statistics
from utils.data_analysis import analyze_dataset
stats = analyze_dataset(imdb_data['train'])
print(f"Average text length: {stats['avg_length']}")
print(f"Vocabulary size: {stats['vocab_size']}")
```

### Creating Custom Datasets
```python
# Create a custom sentiment dataset
texts = ["This movie is great!", "I didn't like this film.", "Amazing story!"]
labels = [1, 0, 1]  # 1 = positive, 0 = negative

train_data, test_data = DatasetLoader.create_custom_dataset(
    texts, labels, split_ratio=0.8
)

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
```

## Research Applications

These datasets enable research in:
- **Model Comparison**: Benchmark different architectures
- **Transfer Learning**: Study domain adaptation
- **Multilingual NLP**: Cross-lingual model evaluation
- **Bias Analysis**: Study fairness in NLP models
- **Efficiency Studies**: Compare model efficiency

## Educational Value

The datasets in this collection are designed to:
- Provide hands-on experience with real-world data
- Demonstrate data preprocessing techniques
- Enable reproducible research
- Support comparative studies
- Foster collaborative learning

---

**Note**: All datasets are provided for educational and research purposes. Please respect the original licenses and terms of use.
