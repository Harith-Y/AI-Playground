# AI-Playground API Documentation

Complete API documentation for the AI-Playground machine learning platform.

## 📋 Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Authentication](#authentication)
- [API Endpoints](#api-endpoints)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)
- [SDKs and Client Libraries](#sdks-and-client-libraries)

## Overview

The AI-Playground API provides a comprehensive RESTful interface for building, training, and deploying machine learning models without writing code.

### Base URL

```
Development: http://localhost:8000/api/v1
Production:  https://api.ai-playground.com/api/v1
```

### API Version

Current version: **v1.0.0**

### Interactive Documentation

- **Swagger UI**: `/docs` - Interactive API documentation with try-it-out functionality
- **ReDoc**: `/redoc` - Alternative documentation with better readability

## Getting Started

### Quick Start

1. **Upload a dataset**:
```bash
curl -X POST "http://localhost:8000/api/v1/datasets/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@dataset.csv"
```

2. **Create preprocessing steps**:
```bash
curl -X POST "http://localhost:8000/api/v1/preprocessing/steps" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "uuid",
    "step_type": "imputation",
    "parameters": {"strategy": "mean"}
  }'
```

3. **Train a model**:
```bash
curl -X POST "http://localhost:8000/api/v1/models/train" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "uuid",
    "model_type": "random_forest",
    "hyperparameters": {"n_estimators": 100}
  }'
```

### Prerequisites

- Python 3.11+
- Valid authentication token
- Dataset in supported format (CSV, Excel, JSON)

## Authentication

### JWT Bearer Token

Most endpoints require authentication using JWT tokens in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

### API Key (Service-to-Service)

For service-to-service authentication, use an API key:

```
X-API-Key: <your_api_key>
```

### Obtaining Tokens

```bash
# Login endpoint (if implemented)
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "your_password"
  }'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

## API Endpoints

### Datasets

#### Upload Dataset

```http
POST /api/v1/datasets/upload
```

Upload a dataset file and extract metadata.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (CSV, XLSX, XLS, or JSON)

**Response:**
```json
{
  "id": "uuid",
  "name": "dataset.csv",
  "rows": 1000,
  "cols": 10,
  "dtypes": {...},
  "missing_values": {...},
  "uploaded_at": "2024-01-15T10:30:00Z"
}
```

#### Get Dataset Preview

```http
GET /api/v1/datasets/{dataset_id}/preview?rows=10
```

Get a preview of the dataset.

**Query Parameters:**
- `rows` (int, optional): Number of rows to return (default: 10)

**Response:**
```json
{
  "columns": ["col1", "col2", "col3"],
  "data": [[1, 2, 3], [4, 5, 6]],
  "total_rows": 1000
}
```

#### Get Dataset Statistics

```http
GET /api/v1/datasets/{dataset_id}/stats
```

Get comprehensive statistics for the dataset.

**Response:**
```json
{
  "shape": {"rows": 1000, "columns": 10},
  "dtypes": {...},
  "missing_values": {...},
  "duplicates": 5,
  "memory_usage": "80.5 KB",
  "column_stats": [...]
}
```

#### Delete Dataset

```http
DELETE /api/v1/datasets/{dataset_id}
```

Delete a dataset and all associated data.

**Response:**
```json
{
  "message": "Dataset deleted successfully"
}
```

### Preprocessing

#### Create Preprocessing Step

```http
POST /api/v1/preprocessing/steps
```

Create a new preprocessing step.

**Request Body:**
```json
{
  "dataset_id": "uuid",
  "step_type": "imputation",
  "parameters": {
    "strategy": "mean"
  },
  "column_name": "age",
  "order": 0
}
```

**Response:**
```json
{
  "id": "uuid",
  "dataset_id": "uuid",
  "step_type": "imputation",
  "parameters": {...},
  "order": 0,
  "is_active": true
}
```

#### List Preprocessing Steps

```http
GET /api/v1/preprocessing/steps?dataset_id={uuid}
```

Get all preprocessing steps for a dataset.

**Response:**
```json
{
  "steps": [
    {
      "id": "uuid",
      "step_type": "imputation",
      "parameters": {...},
      "order": 0
    }
  ],
  "total": 1
}
```

#### Update Preprocessing Step

```http
PUT /api/v1/preprocessing/steps/{step_id}
```

Update a preprocessing step.

**Request Body:**
```json
{
  "parameters": {
    "strategy": "median"
  },
  "is_active": true
}
```

#### Delete Preprocessing Step

```http
DELETE /api/v1/preprocessing/steps/{step_id}
```

Delete a preprocessing step.

#### Reorder Steps

```http
POST /api/v1/preprocessing/steps/reorder
```

Reorder preprocessing steps.

**Request Body:**
```json
{
  "dataset_id": "uuid",
  "step_order": ["step_id_1", "step_id_2", "step_id_3"]
}
```

#### Apply Preprocessing

```http
POST /api/v1/preprocessing/apply
```

Apply preprocessing pipeline to dataset.

**Request Body:**
```json
{
  "dataset_id": "uuid",
  "save_result": true
}
```

### Models

#### Train Model

```http
POST /api/v1/models/train
```

Train a machine learning model.

**Request Body:**
```json
{
  "dataset_id": "uuid",
  "model_type": "random_forest",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
  },
  "train_split": 0.8,
  "validation_split": 0.1
}
```

**Response:**
```json
{
  "run_id": "uuid",
  "status": "running",
  "message": "Model training started"
}
```

#### Get Training Status

```http
GET /api/v1/models/runs/{run_id}/status
```

Get the status of a training run.

**Response:**
```json
{
  "run_id": "uuid",
  "status": "completed",
  "progress": 100,
  "metrics": {
    "accuracy": 0.95,
    "f1_score": 0.93
  },
  "training_time": 45.2
}
```

#### Get Training Results

```http
GET /api/v1/models/runs/{run_id}/results
```

Get detailed training results.

**Response:**
```json
{
  "run_id": "uuid",
  "model_type": "random_forest",
  "metrics": {
    "train": {...},
    "validation": {...},
    "test": {...}
  },
  "feature_importance": [...],
  "confusion_matrix": [...],
  "training_time": 45.2
}
```

#### Compare Models

```http
POST /api/v1/models/compare
```

Compare multiple trained models.

**Request Body:**
```json
{
  "run_ids": ["uuid1", "uuid2", "uuid3"]
}
```

### Hyperparameter Tuning

#### Start Tuning

```http
POST /api/v1/tuning/optimize
```

Start hyperparameter tuning.

**Request Body:**
```json
{
  "dataset_id": "uuid",
  "model_type": "random_forest",
  "tuning_method": "grid_search",
  "param_grid": {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15]
  },
  "cv_folds": 5
}
```

**Response:**
```json
{
  "tuning_id": "uuid",
  "status": "running",
  "total_combinations": 9
}
```

#### Get Tuning Status

```http
GET /api/v1/tuning/{tuning_id}/status
```

Get tuning progress and status.

#### Get Tuning Results

```http
GET /api/v1/tuning/{tuning_id}/results
```

Get best parameters and results.

**Response:**
```json
{
  "best_params": {
    "n_estimators": 100,
    "max_depth": 10
  },
  "best_score": 0.95,
  "all_results": [...]
}
```

### Code Generation

#### Generate Python Code

```http
POST /api/v1/code-generation/python
```

Generate Python script from experiment.

**Request Body:**
```json
{
  "experiment_id": "uuid",
  "include_preprocessing": true,
  "include_training": true,
  "include_evaluation": true
}
```

**Response:**
```json
{
  "code": "# Generated Python code...",
  "filename": "experiment_code.py"
}
```

#### Generate Jupyter Notebook

```http
POST /api/v1/code-generation/notebook
```

Generate Jupyter notebook.

#### Generate FastAPI Service

```http
POST /api/v1/code-generation/fastapi
```

Generate FastAPI inference service.

#### Download Code Package

```http
GET /api/v1/code-generation/{generation_id}/download
```

Download complete code package as ZIP.

### Experiments

#### Get Experiment Configuration

```http
GET /api/v1/experiments/{experiment_id}/config
```

Get complete experiment configuration.

**Query Parameters:**
- `include_results` (bool): Include training results
- `include_artifacts` (bool): Include model artifact paths

#### Download Configuration

```http
GET /api/v1/experiments/{experiment_id}/config/download
```

Download configuration as JSON file.

#### Export Experiment Package

```http
GET /api/v1/experiments/{experiment_id}/export
```

Export complete reproduction package as ZIP.

#### Compare Experiments

```http
POST /api/v1/experiments/compare?experiment_id_1={uuid1}&experiment_id_2={uuid2}
```

Compare two experiments.

#### Get Experiment Summary

```http
GET /api/v1/experiments/{experiment_id}/summary
```

Get high-level experiment summary.

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "detail": "Error message",
  "error_code": "ERROR_CODE",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request succeeded |
| 201 | Created | Resource created successfully |
| 204 | No Content | Request succeeded with no response body |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error occurred |
| 503 | Service Unavailable | Service temporarily unavailable |

### Common Error Codes

| Error Code | Description |
|------------|-------------|
| `UNAUTHORIZED` | Authentication required or token invalid |
| `FORBIDDEN` | Insufficient permissions |
| `NOT_FOUND` | Resource not found |
| `VALIDATION_ERROR` | Request validation failed |
| `DATASET_NOT_FOUND` | Dataset not found |
| `MODEL_NOT_FOUND` | Model not found |
| `TRAINING_FAILED` | Model training failed |
| `INVALID_FILE_TYPE` | Unsupported file type |
| `FILE_TOO_LARGE` | File exceeds size limit |
| `RATE_LIMIT_EXCEEDED` | Too many requests |

### Error Handling Examples

```python
import requests

try:
    response = requests.post(
        "http://localhost:8000/api/v1/models/train",
        headers={"Authorization": f"Bearer {token}"},
        json={"dataset_id": "uuid", "model_type": "random_forest"}
    )
    response.raise_for_status()
    result = response.json()
except requests.exceptions.HTTPError as e:
    error = e.response.json()
    print(f"Error: {error['detail']}")
    print(f"Code: {error.get('error_code')}")
```

## Rate Limiting

### Limits

- **Standard tier**: 100 requests per minute
- **Premium tier**: 1000 requests per minute

### Rate Limit Headers

Response headers include rate limit information:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248000
```

### Handling Rate Limits

```python
import time
import requests

def make_request_with_retry(url, **kwargs):
    while True:
        response = requests.get(url, **kwargs)
        
        if response.status_code == 429:
            # Rate limited, wait and retry
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            wait_time = max(reset_time - time.time(), 1)
            print(f"Rate limited. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            continue
        
        return response
```

## Examples

### Complete Workflow Example

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"
TOKEN = "your_jwt_token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# 1. Upload dataset
with open("data.csv", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/datasets/upload",
        headers=HEADERS,
        files={"file": f}
    )
dataset = response.json()
dataset_id = dataset["id"]

# 2. Create preprocessing steps
steps = [
    {
        "dataset_id": dataset_id,
        "step_type": "imputation",
        "parameters": {"strategy": "mean"},
        "order": 0
    },
    {
        "dataset_id": dataset_id,
        "step_type": "scaling",
        "parameters": {"method": "standard"},
        "order": 1
    }
]

for step in steps:
    requests.post(
        f"{BASE_URL}/preprocessing/steps",
        headers=HEADERS,
        json=step
    )

# 3. Train model
train_request = {
    "dataset_id": dataset_id,
    "model_type": "random_forest",
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10
    }
}

response = requests.post(
    f"{BASE_URL}/models/train",
    headers=HEADERS,
    json=train_request
)
run_id = response.json()["run_id"]

# 4. Check training status
import time

while True:
    response = requests.get(
        f"{BASE_URL}/models/runs/{run_id}/status",
        headers=HEADERS
    )
    status = response.json()
    
    if status["status"] == "completed":
        print("Training completed!")
        print(f"Metrics: {status['metrics']}")
        break
    elif status["status"] == "failed":
        print("Training failed!")
        break
    
    time.sleep(5)

# 5. Get results
response = requests.get(
    f"{BASE_URL}/models/runs/{run_id}/results",
    headers=HEADERS
)
results = response.json()
print(f"Final metrics: {results['metrics']}")

# 6. Generate code
response = requests.post(
    f"{BASE_URL}/code-generation/python",
    headers=HEADERS,
    json={"experiment_id": dataset_id}
)
code = response.json()["code"]
print("Generated code:")
print(code)
```

### Batch Processing Example

```python
import requests
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "http://localhost:8000/api/v1"
TOKEN = "your_jwt_token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

def train_model(config):
    response = requests.post(
        f"{BASE_URL}/models/train",
        headers=HEADERS,
        json=config
    )
    return response.json()

# Train multiple models in parallel
configs = [
    {"dataset_id": "uuid", "model_type": "random_forest"},
    {"dataset_id": "uuid", "model_type": "logistic_regression"},
    {"dataset_id": "uuid", "model_type": "xgboost"}
]

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(train_model, configs))

print(f"Started {len(results)} training runs")
```

## SDKs and Client Libraries

### Python SDK

```python
from ai_playground import Client

client = Client(api_key="your_api_key")

# Upload dataset
dataset = client.datasets.upload("data.csv")

# Create preprocessing pipeline
pipeline = client.preprocessing.create_pipeline(dataset.id)
pipeline.add_step("imputation", strategy="mean")
pipeline.add_step("scaling", method="standard")

# Train model
model = client.models.train(
    dataset_id=dataset.id,
    model_type="random_forest",
    hyperparameters={"n_estimators": 100}
)

# Wait for completion
model.wait_for_completion()

# Get results
results = model.get_results()
print(f"Accuracy: {results.metrics['accuracy']}")
```

### JavaScript/TypeScript SDK

```typescript
import { AIPlaygroundClient } from '@ai-playground/sdk';

const client = new AIPlaygroundClient({
  apiKey: 'your_api_key'
});

// Upload dataset
const dataset = await client.datasets.upload('data.csv');

// Train model
const model = await client.models.train({
  datasetId: dataset.id,
  modelType: 'random_forest',
  hyperparameters: {
    n_estimators: 100
  }
});

// Get results
const results = await model.getResults();
console.log(`Accuracy: ${results.metrics.accuracy}`);
```

## Support and Resources

- **Documentation**: https://docs.ai-playground.com
- **API Reference**: https://api.ai-playground.com/docs
- **GitHub**: https://github.com/your-org/ai-playground
- **Discord**: https://discord.gg/ai-playground
- **Email**: support@ai-playground.com

## Changelog

### v1.0.0 (2024-01-15)
- Initial API release
- Dataset management endpoints
- Preprocessing pipeline
- Model training and evaluation
- Hyperparameter tuning
- Code generation
- Experiment configuration serialization

---

**Last Updated**: January 15, 2024  
**API Version**: 1.0.0
