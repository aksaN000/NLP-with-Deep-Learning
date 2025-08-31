# Production Deployment Guide

This guide covers deploying NLP models and applications to production environments with best practices for scalability, reliability, and maintainability.

## Deployment Architectures

### 1. Microservices Architecture
**Recommended for**: Large-scale applications with multiple NLP components

```yaml
# docker-compose.yml
version: '3.8'
services:
  # Text preprocessing service
  preprocessing-service:
    build: ./services/preprocessing
    ports:
      - "8001:8000"
    environment:
      - WORKERS=4
      - MAX_REQUESTS=1000
    volumes:
      - ./data:/app/data
    
  # Model inference service
  inference-service:
    build: ./services/inference
    ports:
      - "8002:8000"
    environment:
      - MODEL_PATH=/models
      - BATCH_SIZE=32
      - GPU_MEMORY_FRACTION=0.8
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  # API gateway
  api-gateway:
    build: ./gateway
    ports:
      - "8000:8000"
    depends_on:
      - preprocessing-service
      - inference-service
    environment:
      - PREPROCESSING_URL=http://preprocessing-service:8000
      - INFERENCE_URL=http://inference-service:8000
  
  # Monitoring
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### 2. Serverless Deployment
**Recommended for**: Variable workloads and cost optimization

```python
# AWS Lambda deployment
import json
import boto3
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def lambda_handler(event, context):
    """AWS Lambda handler for NLP inference."""
    
    # Load model (use S3 for model storage)
    model_path = '/tmp/model'
    if not os.path.exists(model_path):
        download_model_from_s3(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Process request
    text = json.loads(event['body'])['text']
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'prediction': predictions.tolist(),
            'confidence': float(torch.max(predictions))
        })
    }

# Serverless.yml configuration
service: nlp-inference

provider:
  name: aws
  runtime: python3.9
  memorySize: 3008
  timeout: 30
  environment:
    MODEL_BUCKET: ${env:MODEL_BUCKET}

functions:
  classify:
    handler: handler.lambda_handler
    events:
      - http:
          path: classify
          method: post
    layers:
      - arn:aws:lambda:us-east-1:123456789:layer:pytorch-layer:1
```

### 3. Kubernetes Deployment
**Recommended for**: Enterprise environments requiring orchestration

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nlp-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nlp-api
  template:
    metadata:
      labels:
        app: nlp-api
    spec:
      containers:
      - name: nlp-api
        image: nlp-education/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models"
        - name: WORKERS
          value: "4"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: model-storage
          mountPath: /models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: nlp-api-service
spec:
  selector:
    app: nlp-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nlp-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nlp-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Performance Optimization

### Model Optimization Techniques

```python
# Model quantization for faster inference
import torch
from transformers import AutoModelForSequenceClassification

class OptimizedModel:
    """Optimized model wrapper with quantization and caching."""
    
    def __init__(self, model_path: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Apply quantization
        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        # Compile model for faster inference (PyTorch 2.0+)
        self.model = torch.compile(self.model)
        
        # Cache for frequent inputs
        self.cache = {}
        self.cache_size = 1000
    
    def predict(self, texts: List[str]) -> List[Dict]:
        """Optimized batch prediction."""
        # Check cache first
        cached_results = {}
        uncached_texts = []
        
        for i, text in enumerate(texts):
            text_hash = hash(text)
            if text_hash in self.cache:
                cached_results[i] = self.cache[text_hash]
            else:
                uncached_texts.append((i, text))
        
        # Process uncached texts
        if uncached_texts:
            indices, texts_to_process = zip(*uncached_texts)
            
            # Batch tokenization
            inputs = self.tokenizer(
                list(texts_to_process),
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Update cache and results
            for i, idx in enumerate(indices):
                result = {
                    'prediction': predictions[i].tolist(),
                    'confidence': float(torch.max(predictions[i]))
                }
                cached_results[idx] = result
                
                # Update cache
                text_hash = hash(texts_to_process[i])
                if len(self.cache) < self.cache_size:
                    self.cache[text_hash] = result
        
        # Return results in original order
        return [cached_results[i] for i in range(len(texts))]
```

### Caching Strategies

```python
# Redis-based caching for distributed systems
import redis
import json
import hashlib
from typing import Optional

class DistributedCache:
    """Redis-based caching for NLP predictions."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.ttl = 3600  # 1 hour TTL
    
    def _get_key(self, text: str, model_version: str) -> str:
        """Generate cache key."""
        content = f"{text}_{model_version}"
        return f"nlp_cache:{hashlib.md5(content.encode()).hexdigest()}"
    
    def get(self, text: str, model_version: str) -> Optional[Dict]:
        """Get cached prediction."""
        key = self._get_key(text, model_version)
        cached = self.redis_client.get(key)
        
        if cached:
            return json.loads(cached)
        return None
    
    def set(self, text: str, model_version: str, result: Dict):
        """Cache prediction result."""
        key = self._get_key(text, model_version)
        self.redis_client.setex(key, self.ttl, json.dumps(result))
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        keys = self.redis_client.keys(pattern)
        if keys:
            self.redis_client.delete(*keys)
```

## Monitoring and Observability

### Comprehensive Monitoring Setup

```python
# monitoring.py
import time
import psutil
import GPUtil
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import logging

# Metrics
REQUEST_COUNT = Counter('nlp_requests_total', 'Total requests', ['endpoint', 'status'])
REQUEST_DURATION = Histogram('nlp_request_duration_seconds', 'Request duration')
MODEL_LOAD_TIME = Histogram('nlp_model_load_time_seconds', 'Model loading time')
MEMORY_USAGE = Gauge('nlp_memory_usage_bytes', 'Memory usage')
GPU_UTILIZATION = Gauge('nlp_gpu_utilization_percent', 'GPU utilization')
CACHE_HIT_RATE = Gauge('nlp_cache_hit_rate', 'Cache hit rate')

class PerformanceMonitor:
    """Comprehensive performance monitoring."""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.cache_hits = 0
        self.cache_total = 0
        
        # Start Prometheus metrics server
        start_http_server(8001)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def record_request(self, endpoint: str, status: str, duration: float):
        """Record request metrics."""
        REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()
        REQUEST_DURATION.observe(duration)
        self.request_count += 1
        
        self.logger.info(f"Request: {endpoint} - {status} - {duration:.3f}s")
    
    def record_cache_access(self, hit: bool):
        """Record cache access metrics."""
        self.cache_total += 1
        if hit:
            self.cache_hits += 1
        
        hit_rate = self.cache_hits / self.cache_total if self.cache_total > 0 else 0
        CACHE_HIT_RATE.set(hit_rate)
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        # Memory usage
        memory = psutil.virtual_memory()
        MEMORY_USAGE.set(memory.used)
        
        # GPU utilization
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                GPU_UTILIZATION.set(gpus[0].load * 100)
        except:
            pass
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        uptime = time.time() - self.start_time
        memory = psutil.virtual_memory()
        
        health_status = {
            "status": "healthy",
            "uptime_seconds": uptime,
            "requests_processed": self.request_count,
            "memory_usage_percent": memory.percent,
            "cache_hit_rate": self.cache_hits / self.cache_total if self.cache_total > 0 else 0
        }
        
        # Check for issues
        if memory.percent > 90:
            health_status["status"] = "warning"
            health_status["issues"] = ["High memory usage"]
        
        return health_status
```

### API Implementation with Monitoring

```python
# api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
from typing import List, Dict, Any
import uvicorn

app = FastAPI(title="NLP Education API", version="1.0.0")
monitor = PerformanceMonitor()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str
    model_version: str = "latest"

class BatchTextRequest(BaseModel):
    texts: List[str]
    model_version: str = "latest"

@app.middleware("http")
async def monitor_requests(request, call_next):
    """Middleware to monitor all requests."""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        monitor.record_request(
            endpoint=request.url.path,
            status=str(response.status_code),
            duration=duration
        )
        
        return response
    except Exception as e:
        duration = time.time() - start_time
        monitor.record_request(
            endpoint=request.url.path,
            status="error",
            duration=duration
        )
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return monitor.health_check()

@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    # Check if models are loaded
    if not hasattr(app.state, 'model_loaded'):
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {"status": "ready"}

@app.post("/classify")
async def classify_text(request: TextRequest, background_tasks: BackgroundTasks):
    """Single text classification."""
    try:
        # Check cache first
        cached_result = cache.get(request.text, request.model_version)
        if cached_result:
            monitor.record_cache_access(hit=True)
            return cached_result
        
        monitor.record_cache_access(hit=False)
        
        # Process text
        result = model.predict([request.text])[0]
        
        # Cache result in background
        background_tasks.add_task(
            cache.set, request.text, request.model_version, result
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/batch")
async def classify_batch(request: BatchTextRequest):
    """Batch text classification."""
    try:
        results = model.predict(request.texts)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize models and services."""
    global model, cache
    
    # Load model
    start_time = time.time()
    model = OptimizedModel("path/to/model")
    load_time = time.time() - start_time
    MODEL_LOAD_TIME.observe(load_time)
    
    # Initialize cache
    cache = DistributedCache()
    
    # Mark as loaded
    app.state.model_loaded = True
    
    # Start background monitoring
    import threading
    def background_monitoring():
        while True:
            monitor.update_system_metrics()
            time.sleep(30)
    
    monitoring_thread = threading.Thread(target=background_monitoring, daemon=True)
    monitoring_thread.start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
```

## CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy NLP API

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t nlp-api:${{ github.sha }} .
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push nlp-api:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to Kubernetes
      run: |
        # Update deployment with new image
        kubectl set image deployment/nlp-api nlp-api=nlp-api:${{ github.sha }}
        kubectl rollout status deployment/nlp-api
```

## Security Best Practices

### Input Validation and Sanitization

```python
from pydantic import BaseModel, validator
import re

class SecureTextRequest(BaseModel):
    text: str
    
    @validator('text')
    def validate_text(cls, v):
        # Length limits
        if len(v) > 10000:
            raise ValueError('Text too long')
        
        # Content filtering
        if re.search(r'<script|javascript:|data:', v, re.IGNORECASE):
            raise ValueError('Potentially malicious content detected')
        
        return v

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/classify")
@limiter.limit("100/minute")
async def classify_text_secure(request: Request, text_request: SecureTextRequest):
    """Secure text classification with rate limiting."""
    pass
```

This production deployment guide provides comprehensive coverage of deploying NLP applications at scale with enterprise-grade reliability, security, and observability.
