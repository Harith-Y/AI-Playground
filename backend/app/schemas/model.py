from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


class ModelTrainingRequest(BaseModel):
	"""Request schema for training a model"""
	experiment_id: UUID
	dataset_id: UUID
	model_type: str
	hyperparameters: Optional[Dict[str, Any]] = None
	target_column: Optional[str] = None
	feature_columns: Optional[List[str]] = None
	test_size: float = Field(default=0.2, ge=0.0, le=1.0)
	random_state: int = 42

	class Config:
		json_schema_extra = {
			"example": {
				"experiment_id": "123e4567-e89b-12d3-a456-426614174000",
				"dataset_id": "123e4567-e89b-12d3-a456-426614174001",
				"model_type": "random_forest_classifier",
				"target_column": "species",
				"test_size": 0.2,
				"random_state": 42,
				"hyperparameters": {
					"n_estimators": 100,
					"max_depth": 10
				}
			}
		}


class ModelTrainingResponse(BaseModel):
	"""Response schema after initiating training"""
	model_run_id: UUID
	task_id: str
	status: str
	message: str
	created_at: datetime

	class Config:
		json_schema_extra = {
			"example": {
				"model_run_id": "123e4567-e89b-12d3-a456-426614174002",
				"task_id": "abc123def456",
				"status": "PENDING",
				"message": "Model training initiated successfully",
				"created_at": "2025-12-29T10:00:00Z"
			}
		}


class ModelTrainingStatus(BaseModel):
	"""Status of a training task"""
	model_run_id: UUID
	task_id: Optional[str] = None
	status: str  # PENDING, PROGRESS, SUCCESS, FAILURE
	progress: Optional[Dict[str, Any]] = None
	result: Optional[Dict[str, Any]] = None
	error: Optional[str] = None

	class Config:
		json_schema_extra = {
			"example": {
				"model_run_id": "123e4567-e89b-12d3-a456-426614174002",
				"task_id": "abc123def456",
				"status": "PROGRESS",
				"progress": {
					"current": 75,
					"total": 100,
					"status": "Evaluating model performance..."
				}
			}
		}
