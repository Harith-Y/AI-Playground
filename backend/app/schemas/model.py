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



class ModelRunDeletionSummary(BaseModel):
	"""Summary of model run deletion operations"""
	model_run_id: str
	model_type: str
	status: str
	task_revoked: bool
	artifact_deleted: bool
	database_record_deleted: bool

	class Config:
		json_schema_extra = {
			"example": {
				"model_run_id": "123e4567-e89b-12d3-a456-426614174002",
				"model_type": "random_forest_classifier",
				"status": "completed",
				"task_revoked": False,
				"artifact_deleted": True,
				"database_record_deleted": True
			}
		}


class ModelRunDeletionResponse(BaseModel):
	"""Response schema for model run deletion"""
	message: str
	deletion_summary: ModelRunDeletionSummary
	timestamp: str

	class Config:
		json_schema_extra = {
			"example": {
				"message": "Model run deleted successfully",
				"deletion_summary": {
					"model_run_id": "123e4567-e89b-12d3-a456-426614174002",
					"model_type": "random_forest_classifier",
					"status": "completed",
					"task_revoked": False,
					"artifact_deleted": True,
					"database_record_deleted": True
				},
				"timestamp": "2025-12-29T10:00:00Z"
			}
		}



class TrainingMetadata(BaseModel):
	"""Training metadata for metrics response"""
	training_time: Optional[float] = None
	created_at: str
	hyperparameters: Dict[str, Any]
	train_samples: Optional[int] = None
	test_samples: Optional[int] = None
	n_features: Optional[int] = None

	class Config:
		json_schema_extra = {
			"example": {
				"training_time": 45.5,
				"created_at": "2025-12-29T10:00:00Z",
				"hyperparameters": {
					"n_estimators": 100,
					"max_depth": 10
				},
				"train_samples": 120,
				"test_samples": 30,
				"n_features": 4
			}
		}


class ModelMetricsResponse(BaseModel):
	"""Response schema for model metrics endpoint"""
	model_run_id: str
	model_type: str
	task_type: str
	metrics: Dict[str, Any]
	training_metadata: TrainingMetadata
	feature_importance: Optional[Dict[str, float]] = None

	class Config:
		json_schema_extra = {
			"example": {
				"model_run_id": "123e4567-e89b-12d3-a456-426614174002",
				"model_type": "random_forest_classifier",
				"task_type": "classification",
				"metrics": {
					"accuracy": 0.95,
					"precision": 0.94,
					"recall": 0.93,
					"f1_score": 0.935
				},
				"training_metadata": {
					"training_time": 45.5,
					"created_at": "2025-12-29T10:00:00Z",
					"hyperparameters": {
						"n_estimators": 100,
						"max_depth": 10
					},
					"train_samples": 120,
					"test_samples": 30,
					"n_features": 4
				},
				"feature_importance": {
					"sepal_length": 0.35,
					"sepal_width": 0.25,
					"petal_length": 0.30,
					"petal_width": 0.10
				}
			}
		}


class FeatureImportanceItem(BaseModel):
	"""Individual feature importance item"""
	feature: str
	importance: float
	rank: int

	class Config:
		json_schema_extra = {
			"example": {
				"feature": "sepal_length",
				"importance": 0.35,
				"rank": 1
			}
		}


class FeatureImportanceResponse(BaseModel):
	"""Response schema for feature importance endpoint"""
	model_run_id: str
	model_type: str
	task_type: str
	has_feature_importance: bool
	feature_importance: Optional[List[FeatureImportanceItem]] = None
	feature_importance_dict: Optional[Dict[str, float]] = None
	total_features: int
	top_features: Optional[List[FeatureImportanceItem]] = None
	importance_method: Optional[str] = None
	message: Optional[str] = None

	class Config:
		json_schema_extra = {
			"example": {
				"model_run_id": "123e4567-e89b-12d3-a456-426614174002",
				"model_type": "random_forest_classifier",
				"task_type": "classification",
				"has_feature_importance": True,
				"feature_importance": [
					{"feature": "sepal_length", "importance": 0.35, "rank": 1},
					{"feature": "petal_length", "importance": 0.30, "rank": 2},
					{"feature": "sepal_width", "importance": 0.25, "rank": 3},
					{"feature": "petal_width", "importance": 0.10, "rank": 4}
				],
				"feature_importance_dict": {
					"sepal_length": 0.35,
					"petal_length": 0.30,
					"sepal_width": 0.25,
					"petal_width": 0.10
				},
				"total_features": 4,
				"top_features": [
					{"feature": "sepal_length", "importance": 0.35, "rank": 1},
					{"feature": "petal_length", "importance": 0.30, "rank": 2},
					{"feature": "sepal_width", "importance": 0.25, "rank": 3}
				],
				"importance_method": "feature_importances_",
				"message": None
			}
		}
