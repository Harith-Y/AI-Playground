"""
Tuning Orchestration Schemas.

Pydantic models for orchestration API requests and responses.
"""

from typing import Optional, Dict, List, Any
from uuid import UUID
from pydantic import BaseModel, Field


# Progressive Search Schemas

class ProgressiveSearchRequest(BaseModel):
    """Request schema for progressive search workflow."""
    
    model_run_id: UUID
    initial_param_grid: Optional[Dict[str, List[Any]]] = None
    refinement_factor: float = Field(default=0.3, ge=0.0, le=1.0)
    cv_folds: int = Field(default=5, ge=2, le=20)
    scoring_metric: Optional[str] = None
    n_iter_random: int = Field(default=50, ge=10, le=500)
    n_iter_bayesian: int = Field(default=30, ge=10, le=200)
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_run_id": "123e4567-e89b-12d3-a456-426614174000",
                "initial_param_grid": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },
                "refinement_factor": 0.3,
                "cv_folds": 5,
                "scoring_metric": "accuracy",
                "n_iter_random": 50,
                "n_iter_bayesian": 30
            }
        }


class StageInfo(BaseModel):
    """Information about a tuning stage."""
    
    stage: str
    tuning_run_id: UUID
    method: str
    task_id: Optional[str] = None
    status: str


class ProgressiveSearchResponse(BaseModel):
    """Response schema for progressive search initiation."""
    
    orchestration_id: str
    model_run_id: str
    workflow: str
    stages: List[Dict[str, Any]]
    grid_search: Dict[str, Any]
    random_search: Dict[str, Any]
    bayesian_optimization: Dict[str, Any]
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "orchestration_id": "abc-123-def-456",
                "model_run_id": "123e4567-e89b-12d3-a456-426614174000",
                "workflow": "progressive_search",
                "stages": [
                    {
                        "stage": "grid_search",
                        "tuning_run_id": "111-222-333",
                        "method": "grid_search"
                    },
                    {
                        "stage": "random_search",
                        "tuning_run_id": "444-555-666",
                        "method": "random_search"
                    },
                    {
                        "stage": "bayesian_optimization",
                        "tuning_run_id": "777-888-999",
                        "method": "bayesian"
                    }
                ],
                "grid_search": {
                    "tuning_run_id": "111-222-333",
                    "task_id": "task-abc-123",
                    "status": "RUNNING"
                },
                "random_search": {
                    "tuning_run_id": "444-555-666",
                    "status": "PENDING"
                },
                "bayesian_optimization": {
                    "tuning_run_id": "777-888-999",
                    "status": "PENDING"
                },
                "message": "Progressive search workflow initiated. Grid search is running."
            }
        }


# Multi-Model Comparison Schemas

class MultiModelComparisonRequest(BaseModel):
    """Request schema for multi-model comparison."""
    
    model_run_ids: List[UUID] = Field(min_length=2)
    tuning_method: str = Field(default="bayesian")
    param_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None
    cv_folds: int = Field(default=5, ge=2, le=20)
    scoring_metric: Optional[str] = None
    n_iter: int = Field(default=30, ge=10, le=200)
    parallel: bool = Field(default=True)
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_run_ids": [
                    "123e4567-e89b-12d3-a456-426614174000",
                    "123e4567-e89b-12d3-a456-426614174001",
                    "123e4567-e89b-12d3-a456-426614174002"
                ],
                "tuning_method": "bayesian",
                "param_grids": {
                    "123e4567-e89b-12d3-a456-426614174000": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [5, 10, 20]
                    }
                },
                "cv_folds": 5,
                "scoring_metric": "accuracy",
                "n_iter": 30,
                "parallel": True
            }
        }


class MultiModelComparisonResponse(BaseModel):
    """Response schema for multi-model comparison initiation."""
    
    orchestration_id: str
    workflow: str
    n_models: int
    tuning_method: str
    parallel: bool
    group_task_id: Optional[str] = None
    tuning_runs: List[Dict[str, str]]
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "orchestration_id": "abc-123-def-456",
                "workflow": "multi_model_comparison",
                "n_models": 3,
                "tuning_method": "bayesian",
                "parallel": True,
                "group_task_id": "group-task-xyz-789",
                "tuning_runs": [
                    {
                        "model_run_id": "123e4567-e89b-12d3-a456-426614174000",
                        "tuning_run_id": "111-222-333"
                    },
                    {
                        "model_run_id": "123e4567-e89b-12d3-a456-426614174001",
                        "tuning_run_id": "444-555-666"
                    },
                    {
                        "model_run_id": "123e4567-e89b-12d3-a456-426614174002",
                        "tuning_run_id": "777-888-999"
                    }
                ],
                "message": "Multi-model comparison initiated for 3 models"
            }
        }


# Orchestration Status Schemas

class OrchestrationProgress(BaseModel):
    """Progress information for an orchestration."""
    
    completed: int
    total: int
    percentage: float


class OrchestrationStatusResponse(BaseModel):
    """Response schema for orchestration status."""
    
    orchestration_id: str
    workflow_type: Optional[str] = None
    overall_status: str
    progress: OrchestrationProgress
    statuses: Dict[str, int]
    stages: List[Dict[str, Any]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "orchestration_id": "abc-123-def-456",
                "workflow_type": "progressive_search",
                "overall_status": "RUNNING",
                "progress": {
                    "completed": 1,
                    "total": 3,
                    "percentage": 33.33
                },
                "statuses": {
                    "completed": 1,
                    "running": 1,
                    "failed": 0,
                    "pending": 1
                },
                "stages": [
                    {
                        "tuning_run_id": "111-222-333",
                        "model_run_id": "123e4567-e89b-12d3-a456-426614174000",
                        "method": "grid_search",
                        "status": "COMPLETED",
                        "stage": "grid_search",
                        "task_id": "task-abc-123",
                        "best_score": 0.95
                    },
                    {
                        "tuning_run_id": "444-555-666",
                        "model_run_id": "123e4567-e89b-12d3-a456-426614174000",
                        "method": "random_search",
                        "status": "RUNNING",
                        "stage": "random_search",
                        "task_id": "task-def-456"
                    },
                    {
                        "tuning_run_id": "777-888-999",
                        "model_run_id": "123e4567-e89b-12d3-a456-426614174000",
                        "method": "bayesian",
                        "status": "PENDING",
                        "stage": "bayesian_optimization"
                    }
                ]
            }
        }


class BestModelInfo(BaseModel):
    """Information about the best model."""
    
    model_run_id: str
    tuning_run_id: str
    model_type: Optional[str] = None
    best_score: float
    best_params: Dict[str, Any]
    tuning_method: str


class ModelComparisonResult(BaseModel):
    """Result for a single model in comparison."""
    
    model_run_id: str
    tuning_run_id: str
    score: Optional[float] = None
    method: str


class BestModelResponse(BaseModel):
    """Response schema for best model from comparison."""
    
    orchestration_id: str
    best_model: BestModelInfo
    all_models: List[Dict[str, Any]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "orchestration_id": "abc-123-def-456",
                "best_model": {
                    "model_run_id": "123e4567-e89b-12d3-a456-426614174001",
                    "tuning_run_id": "444-555-666",
                    "model_type": "random_forest_classifier",
                    "best_score": 0.96,
                    "best_params": {
                        "n_estimators": 150,
                        "max_depth": 15
                    },
                    "tuning_method": "bayesian"
                },
                "all_models": [
                    {
                        "model_run_id": "123e4567-e89b-12d3-a456-426614174000",
                        "tuning_run_id": "111-222-333",
                        "score": 0.94,
                        "method": "bayesian"
                    },
                    {
                        "model_run_id": "123e4567-e89b-12d3-a456-426614174001",
                        "tuning_run_id": "444-555-666",
                        "score": 0.96,
                        "method": "bayesian"
                    },
                    {
                        "model_run_id": "123e4567-e89b-12d3-a456-426614174002",
                        "tuning_run_id": "777-888-999",
                        "score": 0.92,
                        "method": "bayesian"
                    }
                ]
            }
        }


# Next Stage Trigger Schema

class TriggerNextStageRequest(BaseModel):
    """Request schema for triggering next stage."""
    
    orchestration_id: str
    completed_tuning_run_id: UUID
    
    class Config:
        json_schema_extra = {
            "example": {
                "orchestration_id": "abc-123-def-456",
                "completed_tuning_run_id": "111-222-333"
            }
        }


class NextStageResponse(BaseModel):
    """Response schema for next stage trigger."""
    
    stage: Optional[str] = None
    tuning_run_id: Optional[str] = None
    task_id: Optional[str] = None
    status: Optional[str] = None
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "stage": "random_search",
                "tuning_run_id": "444-555-666",
                "task_id": "task-def-456",
                "status": "RUNNING",
                "message": "Next stage (random_search) triggered successfully"
            }
        }
