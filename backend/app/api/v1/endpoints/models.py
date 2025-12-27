"""
Model training endpoints.

Provides REST API for machine learning model operations including
listing available models, training, evaluation, and prediction.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from app.ml_engine.model_registry import model_registry, TaskType, ModelCategory

router = APIRouter()


@router.get("/available")
async def get_available_models(
    task_type: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get list of available machine learning models.

    Query parameters:
        task_type: Filter by task type (regression, classification, clustering)
        category: Filter by model category (linear, tree_based, boosting, etc.)
        search: Search query for model name, description, or tags

    Returns:
        Dictionary containing available models organized by task type

    Example:
        GET /api/v1/models/available
        GET /api/v1/models/available?task_type=classification
        GET /api/v1/models/available?category=boosting
        GET /api/v1/models/available?search=random forest
    """
    try:
        # Apply filters
        if search:
            # Search across all fields
            models = model_registry.search_models(search)

            # Group by task type
            result = {
                "regression": [m.to_dict() for m in models if m.task_type == TaskType.REGRESSION],
                "classification": [m.to_dict() for m in models if m.task_type == TaskType.CLASSIFICATION],
                "clustering": [m.to_dict() for m in models if m.task_type == TaskType.CLUSTERING],
            }

            return {
                "search_query": search,
                "total_results": len(models),
                "models": result
            }

        elif task_type:
            # Filter by task type
            try:
                task_enum = TaskType(task_type.lower())
                models = model_registry.get_models_by_task(task_enum)

                return {
                    "task_type": task_type,
                    "count": len(models),
                    "models": [model.to_dict() for model in models]
                }
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid task_type '{task_type}'. Must be one of: regression, classification, clustering"
                )

        elif category:
            # Filter by category
            try:
                category_enum = ModelCategory(category.lower())
                models = model_registry.get_models_by_category(category_enum)

                return {
                    "category": category,
                    "count": len(models),
                    "models": [model.to_dict() for model in models]
                }
            except ValueError:
                valid_categories = [c.value for c in ModelCategory]
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid category '{category}'. Must be one of: {', '.join(valid_categories)}"
                )

        else:
            # Return all models grouped by task type
            all_models = model_registry.get_all_models()

            result = {
                "regression": [model.to_dict() for model in all_models["regression"]],
                "classification": [model.to_dict() for model in all_models["classification"]],
                "clustering": [model.to_dict() for model in all_models["clustering"]],
            }

            total_count = sum(len(models) for models in all_models.values())

            return {
                "total_models": total_count,
                "models_by_task": result,
                "summary": {
                    "regression_models": len(all_models["regression"]),
                    "classification_models": len(all_models["classification"]),
                    "clustering_models": len(all_models["clustering"]),
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving models: {str(e)}"
        )


@router.get("/available/{model_id}")
async def get_model_details(model_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model.

    Path parameters:
        model_id: Unique identifier of the model

    Returns:
        Detailed model information including hyperparameters and capabilities

    Example:
        GET /api/v1/models/available/random_forest_classifier
    """
    try:
        model = model_registry.get_model(model_id)
        return model.to_dict()
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model details: {str(e)}"
        )


@router.get("/categories")
async def get_model_categories() -> Dict[str, Any]:
    """
    Get all available model categories.

    Returns:
        List of model categories with descriptions

    Example:
        GET /api/v1/models/categories
    """
    return {
        "categories": [
            {
                "id": "linear",
                "name": "Linear Models",
                "description": "Models that assume linear relationships between features and target"
            },
            {
                "id": "tree_based",
                "name": "Tree-Based Models",
                "description": "Models based on decision trees and tree ensembles"
            },
            {
                "id": "boosting",
                "name": "Boosting Models",
                "description": "Sequential ensemble models that correct previous errors"
            },
            {
                "id": "support_vector",
                "name": "Support Vector Machines",
                "description": "Models that find optimal hyperplanes for classification/regression"
            },
            {
                "id": "instance_based",
                "name": "Instance-Based Models",
                "description": "Models that make predictions based on nearest training examples"
            },
            {
                "id": "neural_network",
                "name": "Neural Networks",
                "description": "Multi-layer perceptron and deep learning models"
            },
            {
                "id": "probabilistic",
                "name": "Probabilistic Models",
                "description": "Models based on probability theory and Bayes theorem"
            },
            {
                "id": "density_based",
                "name": "Density-Based Clustering",
                "description": "Clustering based on density of data points"
            },
            {
                "id": "hierarchical",
                "name": "Hierarchical Clustering",
                "description": "Clustering that builds a hierarchy of clusters"
            }
        ]
    }


@router.get("/task-types")
async def get_task_types() -> Dict[str, Any]:
    """
    Get all available task types.

    Returns:
        List of task types with descriptions and model counts

    Example:
        GET /api/v1/models/task-types
    """
    all_models = model_registry.get_all_models()

    return {
        "task_types": [
            {
                "id": "regression",
                "name": "Regression",
                "description": "Predict continuous numerical values",
                "model_count": len(all_models["regression"]),
                "examples": ["house prices", "sales forecasting", "temperature prediction"]
            },
            {
                "id": "classification",
                "name": "Classification",
                "description": "Predict discrete categories or classes",
                "model_count": len(all_models["classification"]),
                "examples": ["spam detection", "disease diagnosis", "customer churn"]
            },
            {
                "id": "clustering",
                "name": "Clustering",
                "description": "Group similar data points together (unsupervised)",
                "model_count": len(all_models["clustering"]),
                "examples": ["customer segmentation", "anomaly detection", "document grouping"]
            }
        ]
    }
