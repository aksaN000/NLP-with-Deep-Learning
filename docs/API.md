# API Documentation

## NLP Education Platform API

### Base URL
```
https://api.nlp-education.org/v1
```

### Authentication
```http
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### Text Processing

#### Process Single Text
```http
POST /process/text
```

**Request Body:**
```json
{
  "text": "Your text to process",
  "options": {
    "language": "english",
    "apply_lemmatization": true,
    "remove_stopwords": true,
    "quality_threshold": 0.7
  }
}
```

**Response:**
```json
{
  "processed_text": "processed text result",
  "tokens": ["processed", "text", "result"],
  "quality_score": 0.85,
  "language_detected": "english",
  "processing_time": 0.123
}
```

#### Batch Text Processing
```http
POST /process/batch
```

**Request Body:**
```json
{
  "texts": ["First text", "Second text"],
  "options": {
    "language": "english",
    "batch_size": 100
  }
}
```

### Text Classification

#### Sentiment Analysis
```http
POST /classify/sentiment
```

**Request Body:**
```json
{
  "text": "I love this product!",
  "model": "bert-base"
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.95,
  "scores": {
    "positive": 0.95,
    "negative": 0.03,
    "neutral": 0.02
  }
}
```

### Word Embeddings

#### Get Word Similarity
```http
GET /embeddings/similarity
```

**Parameters:**
- `word1`: First word
- `word2`: Second word  
- `model`: Embedding model (word2vec, glove, fasttext)

**Response:**
```json
{
  "similarity": 0.87,
  "word1": "king",
  "word2": "queen",
  "model": "glove"
}
```

#### Word Analogies
```http
POST /embeddings/analogy
```

**Request Body:**
```json
{
  "word_a": "king",
  "word_b": "man", 
  "word_c": "woman",
  "model": "word2vec",
  "top_k": 5
}
```

### Model Training

#### Start Training Job
```http
POST /training/start
```

**Request Body:**
```json
{
  "model_type": "classification",
  "dataset_id": "imdb_reviews",
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10
  }
}
```

## WebSocket API

### Real-time Processing
```javascript
const ws = new WebSocket('wss://api.nlp-education.org/ws/process');

ws.onopen = function() {
    ws.send(JSON.stringify({
        action: 'process_text',
        text: 'Your text here'
    }));
};

ws.onmessage = function(event) {
    const result = JSON.parse(event.data);
    console.log('Processed:', result);
};
```

## SDKs

### Python SDK
```python
from nlp_education import NLPClient

client = NLPClient(api_key='your_key')

# Process text
result = client.process_text(
    "Hello world!", 
    apply_lemmatization=True
)

# Classify sentiment
sentiment = client.classify_sentiment("I love this!")

# Get word similarity
similarity = client.word_similarity("king", "queen")
```

### JavaScript SDK
```javascript
import { NLPClient } from 'nlp-education-js';

const client = new NLPClient({ apiKey: 'your_key' });

// Process text
const result = await client.processText('Hello world!');

// Classify sentiment  
const sentiment = await client.classifySentiment('I love this!');

// Get embeddings
const embeddings = await client.getEmbeddings(['word1', 'word2']);
```

## Rate Limits

| Endpoint | Rate Limit |
|----------|------------|
| Text Processing | 1000/hour |
| Classification | 500/hour |
| Embeddings | 2000/hour |
| Training | 10/day |

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Text is too long",
    "details": {
      "max_length": 10000,
      "provided_length": 15000
    }
  },
  "request_id": "req_123456"
}
```

### Common Error Codes
- `INVALID_INPUT`: Invalid request parameters
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `AUTHENTICATION_FAILED`: Invalid API key
- `MODEL_NOT_FOUND`: Requested model unavailable
- `PROCESSING_ERROR`: Internal processing error

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized |
| 429 | Rate Limit Exceeded |
| 500 | Internal Server Error |

## Webhooks

Register webhooks to receive notifications:

```http
POST /webhooks/register
```

**Request Body:**
```json
{
  "url": "https://your-app.com/webhook",
  "events": ["training_complete", "batch_processed"],
  "secret": "your_webhook_secret"
}
```

**Webhook Payload:**
```json
{
  "event": "training_complete",
  "data": {
    "job_id": "job_123",
    "status": "completed",
    "metrics": {
      "accuracy": 0.92,
      "f1_score": 0.89
    }
  },
  "timestamp": "2023-08-31T12:00:00Z"
}
```
