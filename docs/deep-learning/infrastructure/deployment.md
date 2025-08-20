---
sidebar_position: 1
---

# Model Deployment

## Introduction

Model deployment is the process of making trained deep learning models available for inference in production environments. It involves considerations of scalability, reliability, latency, and maintainability.

## Deployment Strategies

### 1. Batch Inference

**Use Case:** Non-real-time predictions on large datasets

**Characteristics:**
- Process data in batches
- Higher throughput, higher latency
- Cost-effective for large-scale processing

**Implementation:**
```python
# Batch processing example
def batch_inference(model, data_batch):
    predictions = model(data_batch)
    return predictions

# Process large dataset in chunks
for batch in data_loader:
    results = batch_inference(model, batch)
    save_results(results)
```

### 2. Real-time Inference

**Use Case:** Low-latency predictions for user-facing applications

**Characteristics:**
- Single request processing
- Low latency requirements
- Higher cost per prediction

**Implementation:**
```python
# Real-time inference API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model(data)
    return jsonify({'prediction': prediction})
```

### 3. Edge Deployment

**Use Case:** On-device inference for mobile/IoT applications

**Characteristics:**
- Limited computational resources
- Network independence
- Privacy benefits

**Technologies:**
- TensorFlow Lite
- ONNX Runtime
- Core ML (iOS)
- TensorRT (NVIDIA)

## Deployment Architectures

### 1. Monolithic Deployment

**Structure:**
```
[Load Balancer] → [Web Server + Model]
```

**Pros:**
- Simple to deploy and manage
- Low latency (no network calls)

**Cons:**
- Difficult to scale independently
- Resource coupling
- Single point of failure

### 2. Microservices Architecture

**Structure:**
```
[Load Balancer] → [API Gateway] → [Model Service] → [Database]
```

**Pros:**
- Independent scaling
- Technology flexibility
- Fault isolation

**Cons:**
- Increased complexity
- Network latency
- More moving parts

### 3. Serverless Deployment

**Structure:**
```
[API Gateway] → [Lambda Functions] → [Model]
```

**Pros:**
- Auto-scaling
- Pay-per-use
- No server management

**Cons:**
- Cold start latency
- Memory limitations
- Vendor lock-in

## Model Serving Technologies

### 1. TensorFlow Serving

**Features:**
- High-performance serving
- Model versioning
- A/B testing support
- REST and gRPC APIs

**Example:**
```bash
# Start TensorFlow Serving
tensorflow_model_server \
  --port=8500 \
  --rest_api_port=8501 \
  --model_name=my_model \
  --model_base_path=/path/to/model
```

### 2. TorchServe

**Features:**
- PyTorch model serving
- Model archiving
- Custom handlers
- Metrics and monitoring

**Example:**
```bash
# Archive model
torch-model-archiver --model-name my_model \
  --version 1.0 \
  --model-file model.py \
  --serialized-file model.pth \
  --handler handler.py

# Start TorchServe
torchserve --start --model-store model-store \
  --models my_model=my_model.mar
```

### 3. Triton Inference Server

**Features:**
- Multi-framework support
- Dynamic batching
- GPU optimization
- Concurrent model execution

### 4. Custom REST APIs

**Framework Options:**
- Flask/FastAPI (Python)
- Express.js (Node.js)
- Spring Boot (Java)

**Example with FastAPI:**
```python
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.load('model.pth')
model.eval()

@app.post("/predict")
async def predict(data: dict):
    input_tensor = torch.tensor(data['input'])
    with torch.no_grad():
        prediction = model(input_tensor)
    return {"prediction": prediction.tolist()}
```

## Containerization

### Docker Deployment

**Dockerfile Example:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model.pth .
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Benefits:**
- Consistent environments
- Easy deployment
- Resource isolation
- Version control

### Kubernetes Deployment

**Deployment YAML:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: ml-model
        image: my-model:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

## Model Optimization

### 1. Model Compression

**Techniques:**
- **Quantization**: Reduce precision (FP32 → INT8)
- **Pruning**: Remove unnecessary weights
- **Knowledge Distillation**: Train smaller model

**Example (PyTorch Quantization):**
```python
import torch.quantization

# Quantize model
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### 2. Model Format Conversion

**ONNX (Open Neural Network Exchange):**
```python
import torch
import onnx

# Export to ONNX
torch.onnx.export(model, dummy_input, "model.onnx")
```

**TensorRT:**
```python
import tensorrt as trt

# Convert ONNX to TensorRT
engine = trt.ICudaEngine()
```

## Monitoring and Observability

### 1. Metrics to Track

**Performance Metrics:**
- Latency (P50, P95, P99)
- Throughput (requests/second)
- Error rate
- Resource utilization

**Business Metrics:**
- Prediction accuracy
- User engagement
- Revenue impact

### 2. Logging and Tracing

**Structured Logging:**
```python
import logging
import json

def predict_with_logging(data):
    logging.info("Prediction request", extra={
        "input_size": len(data),
        "timestamp": datetime.now().isoformat()
    })
    
    try:
        result = model(data)
        logging.info("Prediction successful", extra={
            "prediction": result.tolist()
        })
        return result
    except Exception as e:
        logging.error("Prediction failed", extra={
            "error": str(e)
        })
        raise
```

### 3. Health Checks

**Implementation:**
```python
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }
```

## Security Considerations

### 1. Input Validation
```python
from pydantic import BaseModel, validator

class PredictionRequest(BaseModel):
    input_data: List[float]
    
    @validator('input_data')
    def validate_input(cls, v):
        if len(v) != expected_features:
            raise ValueError('Invalid input size')
        return v
```

### 2. Authentication and Authorization
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/predict")
async def predict(data: dict, token: str = Depends(security)):
    if not validate_token(token):
        raise HTTPException(status_code=401)
    return model(data)
```

## Practice Questions

1. **What are the trade-offs between batch and real-time inference?**
2. **How would you design a system to handle 1000 requests per second?**
3. **What considerations are important for edge deployment?**
4. **How do you handle model versioning in production?**
5. **What monitoring metrics are most important for ML systems?**
6. **How would you implement A/B testing for model deployments?**

## Further Reading

- [Machine Learning Systems Design](https://github.com/chiphuyen/machine-learning-systems-design)
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781492042659/)
- [Kubeflow Documentation](https://www.kubeflow.org/)
- [MLOps: Continuous Delivery for Machine Learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
