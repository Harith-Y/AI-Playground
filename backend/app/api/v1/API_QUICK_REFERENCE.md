# API Quick Reference

Quick reference guide for AI-Playground API endpoints.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

```bash
Authorization: Bearer <token>
```

## Datasets

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/datasets/upload` | Upload dataset |
| GET | `/datasets/{id}/preview` | Preview data |
| GET | `/datasets/{id}/stats` | Get statistics |
| DELETE | `/datasets/{id}` | Delete dataset |

## Preprocessing

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/preprocessing/steps` | Create step |
| GET | `/preprocessing/steps` | List steps |
| PUT | `/preprocessing/steps/{id}` | Update step |
| DELETE | `/preprocessing/steps/{id}` | Delete step |
| POST | `/preprocessing/steps/reorder` | Reorder steps |
| POST | `/preprocessing/apply` | Apply pipeline |

## Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/models/train` | Train model |
| GET | `/models/runs/{id}/status` | Get status |
| GET | `/models/runs/{id}/results` | Get results |
| POST | `/models/compare` | Compare models |
| DELETE | `/models/runs/{id}` | Delete run |

## Tuning

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tuning/optimize` | Start tuning |
| GET | `/tuning/{id}/status` | Get status |
| GET | `/tuning/{id}/results` | Get results |
| POST | `/tuning/{id}/apply` | Apply best params |

## Code Generation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/code-generation/python` | Generate Python |
| POST | `/code-generation/notebook` | Generate notebook |
| POST | `/code-generation/fastapi` | Generate FastAPI |
| GET | `/code-generation/{id}/download` | Download package |

## Experiments

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/experiments/{id}/config` | Get config |
| GET | `/experiments/{id}/config/download` | Download config |
| GET | `/experiments/{id}/export` | Export package |
| POST | `/experiments/compare` | Compare experiments |
| GET | `/experiments/{id}/summary` | Get summary |

## Common Parameters

### Query Parameters

- `skip` (int): Pagination offset
- `limit` (int): Items per page
- `include_results` (bool): Include results
- `include_artifacts` (bool): Include artifacts

### Response Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `422` - Validation Error
- `500` - Server Error

## Quick Examples

### Upload Dataset

```bash
curl -X POST "http://localhost:8000/api/v1/datasets/upload" \
  -H "Authorization: Bearer TOKEN" \
  -F "file=@data.csv"
```

### Train Model

```bash
curl -X POST "http://localhost:8000/api/v1/models/train" \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "uuid",
    "model_type": "random_forest",
    "hyperparameters": {"n_estimators": 100}
  }'
```

### Get Results

```bash
curl -X GET "http://localhost:8000/api/v1/models/runs/{run_id}/results" \
  -H "Authorization: Bearer TOKEN"
```

## Rate Limits

- Standard: 100 req/min
- Premium: 1000 req/min

## Support

- Docs: `/docs`
- ReDoc: `/redoc`
- Health: `/health`
