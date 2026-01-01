"""
Enhanced API Documentation Configuration.

This module provides comprehensive OpenAPI/Swagger documentation configuration
for the AI-Playground API, including detailed descriptions, examples, and tags.
"""

from typing import Dict, Any

# API Metadata
API_TITLE = "AI-Playground API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
# AI-Playground API

**AI-Playground** is a comprehensive full-stack machine learning platform for automated ML workflows.
Build, train, and deploy ML models without writing code, then export production-ready pipelines.

## Features

- 📊 **Dataset Management** - Upload, explore, and visualize datasets
- 🔧 **Preprocessing Pipeline** - Automated data cleaning and transformation
- 🤖 **Model Training** - Support for regression, classification, and clustering
- 🎯 **Hyperparameter Tuning** - Grid search, random search, Bayesian optimization
- 📈 **Model Evaluation** - Comprehensive metrics and visualizations
- 💾 **Experiment Tracking** - Track and compare model experiments
- 📦 **Code Generation** - Export production-ready ML pipeline code
- 🔄 **Serialization** - Save and load models, pipelines, and workflows

## Authentication

Most endpoints require authentication using JWT tokens. Include the token in the `Authorization` header:

```
Authorization: Bearer <your_jwt_token>
```

To obtain a token, use the `/auth/login` endpoint (if implemented) or configure your authentication provider.

## Rate Limiting

API requests are rate-limited to prevent abuse:
- **Standard tier**: 100 requests per minute
- **Premium tier**: 1000 requests per minute

Rate limit information is included in response headers:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when the rate limit resets

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages:

### Success Codes
- `200 OK` - Request succeeded
- `201 Created` - Resource created successfully
- `204 No Content` - Request succeeded with no response body

### Client Error Codes
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `422 Unprocessable Entity` - Validation error

### Server Error Codes
- `500 Internal Server Error` - Server error occurred
- `503 Service Unavailable` - Service temporarily unavailable

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong",
  "error_code": "SPECIFIC_ERROR_CODE",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Pagination

List endpoints support pagination using query parameters:
- `skip` (int): Number of items to skip (default: 0)
- `limit` (int): Maximum number of items to return (default: 100, max: 1000)

Response includes pagination metadata:

```json
{
  "items": [...],
  "total": 1500,
  "skip": 0,
  "limit": 100,
  "has_more": true
}
```

## Versioning

The API is versioned using URL path versioning:
- Current version: `/api/v1/`
- Future versions: `/api/v2/`, `/api/v3/`, etc.

## Support

- **Documentation**: https://docs.ai-playground.com
- **GitHub**: https://github.com/your-org/ai-playground
- **Email**: support@ai-playground.com

## License

This API is provided for educational and development purposes.
"""

# API Tags with descriptions
API_TAGS = [
    {
        "name": "datasets",
        "description": """
**Dataset Management Operations**

Upload, preview, and manage datasets. Supports multiple formats including CSV, Excel, and JSON.

**Key Features:**
- Upload datasets with automatic metadata extraction
- Preview data with pagination
- Get comprehensive statistics
- Delete datasets
        """,
    },
    {
        "name": "preprocessing",
        "description": """
**Data Preprocessing Operations**

Create and manage preprocessing pipelines with various transformation steps.

**Available Steps:**
- Missing value imputation (mean, median, mode)
- Outlier detection (IQR, Z-score)
- Feature scaling (standard, min-max, robust)
- Encoding (one-hot, label, ordinal)
- Feature selection (variance, correlation, mutual information)

**Key Features:**
- CRUD operations for preprocessing steps
- Step reordering
- Preview transformations
- Apply pipeline to dataset
        """,
    },
    {
        "name": "models",
        "description": """
**Model Training and Management**

Train machine learning models with various algorithms and configurations.

**Supported Models:**
- **Regression**: Linear, Ridge, Lasso, Random Forest, XGBoost, etc.
- **Classification**: Logistic Regression, SVM, Random Forest, XGBoost, etc.
- **Clustering**: K-Means, DBSCAN, Hierarchical, etc.

**Key Features:**
- Train models with custom hyperparameters
- Monitor training progress
- Get training results and metrics
- Compare multiple models
- Save trained models
        """,
    },
    {
        "name": "tuning",
        "description": """
**Hyperparameter Tuning Operations**

Optimize model hyperparameters using various search strategies.

**Tuning Methods:**
- Grid Search - Exhaustive search over parameter grid
- Random Search - Random sampling of parameter space
- Bayesian Optimization - Smart parameter search

**Key Features:**
- Configure search space
- Monitor tuning progress
- Get best parameters
- Apply best configuration to model
        """,
    },
    {
        "name": "tuning-orchestration",
        "description": """
**Advanced Tuning Orchestration**

Orchestrate complex hyperparameter tuning workflows with multiple models.

**Key Features:**
- Multi-model tuning
- Parallel execution
- Progress tracking
- Result comparison
        """,
    },
    {
        "name": "code-generation",
        "description": """
**Code Generation and Export**

Generate production-ready code from trained models and pipelines.

**Export Formats:**
- Python script (.py)
- Jupyter Notebook (.ipynb)
- FastAPI service
- Requirements file

**Generated Code Includes:**
- Data loading and preprocessing
- Model training
- Evaluation and metrics
- Prediction functions
- Deployment templates
        """,
    },
    {
        "name": "experiments",
        "description": """
**Experiment Configuration Management**

Serialize, export, and compare complete experiment configurations.

**Key Features:**
- Save experiment configurations
- Export reproduction packages
- Compare experiments
- Download configurations
- Get experiment summaries

**Use Cases:**
- Experiment reproducibility
- Team collaboration
- Version control
- Production deployment
        """,
    },
    {
        "name": "monitoring",
        "description": """
**System Monitoring and Metrics**

Monitor system health and performance metrics.

**Available Metrics:**
- API request metrics
- Model training metrics
- System resource usage
- Error rates

**Formats:**
- Prometheus metrics
- JSON metrics
        """,
    },
]

# Common response examples
COMMON_RESPONSES = {
    "unauthorized": {
        "description": "Authentication required",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Not authenticated",
                    "error_code": "UNAUTHORIZED"
                }
            }
        }
    },
    "forbidden": {
        "description": "Insufficient permissions",
        "content": {
            "application/json": {
                "example": {
                    "detail": "You don't have permission to access this resource",
                    "error_code": "FORBIDDEN"
                }
            }
        }
    },
    "not_found": {
        "description": "Resource not found",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Resource not found",
                    "error_code": "NOT_FOUND"
                }
            }
        }
    },
    "validation_error": {
        "description": "Validation error",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "loc": ["body", "field_name"],
                            "msg": "field required",
                            "type": "value_error.missing"
                        }
                    ]
                }
            }
        }
    },
    "internal_error": {
        "description": "Internal server error",
        "content": {
            "application/json": {
                "example": {
                    "detail": "An internal error occurred",
                    "error_code": "INTERNAL_ERROR"
                }
            }
        }
    },
}

# OpenAPI configuration
def get_openapi_config() -> Dict[str, Any]:
    """
    Get OpenAPI configuration for FastAPI.
    
    Returns:
        Dictionary with OpenAPI configuration
    """
    return {
        "title": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "openapi_tags": API_TAGS,
        "servers": [
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            },
            {
                "url": "https://api.ai-playground.com",
                "description": "Production server"
            },
            {
                "url": "https://staging-api.ai-playground.com",
                "description": "Staging server"
            }
        ],
        "contact": {
            "name": "AI-Playground Support",
            "url": "https://github.com/your-org/ai-playground",
            "email": "support@ai-playground.com"
        },
        "license_info": {
            "name": "Educational Use",
            "url": "https://github.com/your-org/ai-playground/blob/main/LICENSE"
        },
        "terms_of_service": "https://ai-playground.com/terms",
    }


# Security schemes
SECURITY_SCHEMES = {
    "Bearer": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "Enter your JWT token in the format: Bearer <token>"
    },
    "ApiKey": {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "API key for service-to-service authentication"
    }
}


def get_custom_openapi_schema(app):
    """
    Generate custom OpenAPI schema with enhanced documentation.
    
    Args:
        app: FastAPI application instance
    
    Returns:
        OpenAPI schema dictionary
    """
    from fastapi.openapi.utils import get_openapi
    
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        routes=app.routes,
        tags=API_TAGS,
        servers=[
            {"url": "http://localhost:8000", "description": "Development"},
            {"url": "https://api.ai-playground.com", "description": "Production"}
        ]
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = SECURITY_SCHEMES
    
    # Add contact and license info
    openapi_schema["info"]["contact"] = {
        "name": "AI-Playground Support",
        "url": "https://github.com/your-org/ai-playground",
        "email": "support@ai-playground.com"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "Educational Use",
        "url": "https://github.com/your-org/ai-playground/blob/main/LICENSE"
    }
    
    # Add external documentation
    openapi_schema["externalDocs"] = {
        "description": "Full Documentation",
        "url": "https://docs.ai-playground.com"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema
