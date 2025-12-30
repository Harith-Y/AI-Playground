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


# ============================================================================
# Hyperparameter Tuning Schemas
# ============================================================================


class HyperparameterTuningRequest(BaseModel):
	"""Request schema for hyperparameter tuning"""
	model_run_id: UUID
	tuning_method: str = Field(default="grid_search", pattern="^(grid_search|random_search|bayesian)$")
	param_grid: Dict[str, List[Any]]
	cv_folds: int = Field(default=5, ge=2, le=10)
	scoring_metric: Optional[str] = None
	n_iter: Optional[int] = Field(default=10, ge=1, le=100)  # For random_search and bayesian
	n_jobs: int = Field(default=-1, ge=-1)
	random_state: int = 42

	class Config:
		json_schema_extra = {
			"example": {
				"model_run_id": "123e4567-e89b-12d3-a456-426614174002",
				"tuning_method": "grid_search",
				"param_grid": {
					"n_estimators": [50, 100, 200],
					"max_depth": [5, 10, 15, None],
					"min_samples_split": [2, 5, 10]
				},
				"cv_folds": 5,
				"scoring_metric": "accuracy",
				"n_jobs": -1,
				"random_state": 42
			}
		}


class HyperparameterTuningResponse(BaseModel):
	"""Response schema after initiating tuning"""
	tuning_run_id: UUID
	task_id: str
	status: str
	message: str
	created_at: datetime

	class Config:
		json_schema_extra = {
			"example": {
				"tuning_run_id": "123e4567-e89b-12d3-a456-426614174003",
				"task_id": "xyz789abc123",
				"status": "PENDING",
				"message": "Hyperparameter tuning initiated successfully",
				"created_at": "2025-12-29T10:00:00Z"
			}
		}


class TuningResultItem(BaseModel):
	"""Individual tuning result"""
	rank: int
	params: Dict[str, Any]
	mean_score: float
	std_score: float
	scores: List[float]

	class Config:
		json_schema_extra = {
			"example": {
				"rank": 1,
				"params": {
					"n_estimators": 100,
					"max_depth": 10,
					"min_samples_split": 2
				},
				"mean_score": 0.95,
				"std_score": 0.02,
				"scores": [0.94, 0.96, 0.95, 0.94, 0.96]
			}
		}


class HyperparameterTuningStatus(BaseModel):
	"""Status of a tuning task"""
	tuning_run_id: UUID
	task_id: Optional[str] = None
	status: str  # PENDING, PROGRESS, SUCCESS, FAILURE
	progress: Optional[Dict[str, Any]] = None
	result: Optional[Dict[str, Any]] = None
	error: Optional[str] = None

	class Config:
		json_schema_extra = {
			"example": {
				"tuning_run_id": "123e4567-e89b-12d3-a456-426614174003",
				"task_id": "xyz789abc123",
				"status": "PROGRESS",
				"progress": {
					"current": 15,
					"total": 36,
					"status": "Testing parameter combination 15/36..."
				}
			}
		}


class HyperparameterTuningResults(BaseModel):
	"""Complete tuning results"""
	tuning_run_id: str
	model_run_id: str
	tuning_method: str
	best_params: Dict[str, Any]
	best_score: float
	total_combinations: int
	top_results: List[TuningResultItem]
	cv_folds: int
	scoring_metric: str
	tuning_time: Optional[float] = None
	created_at: str

	class Config:
		json_schema_extra = {
			"example": {
				"tuning_run_id": "123e4567-e89b-12d3-a456-426614174003",
				"model_run_id": "123e4567-e89b-12d3-a456-426614174002",
				"tuning_method": "grid_search",
				"best_params": {
					"n_estimators": 100,
					"max_depth": 10,
					"min_samples_split": 2
				},
				"best_score": 0.95,
				"total_combinations": 36,
				"top_results": [
					{
						"rank": 1,
						"params": {"n_estimators": 100, "max_depth": 10},
						"mean_score": 0.95,
						"std_score": 0.02,
						"scores": [0.94, 0.96, 0.95, 0.94, 0.96]
					}
				],
				"cv_folds": 5,
				"scoring_metric": "accuracy",
				"tuning_time": 120.5,
				"created_at": "2025-12-29T10:00:00Z"
			}
		}


# ============================================================================
# Model Comparison Schemas
# ============================================================================


class CompareModelsRequest(BaseModel):
	"""Request schema for comparing multiple model runs."""
	
	model_run_ids: List[UUID] = Field(..., min_length=2, max_length=10)
	comparison_metrics: Optional[List[str]] = None  # Metrics to compare (auto-detected if None)
	ranking_criteria: Optional[str] = None  # Primary metric for ranking
	include_statistical_tests: bool = Field(default=False)
	
	class Config:
		json_schema_extra = {
			"example": {
				"model_run_ids": [
					"123e4567-e89b-12d3-a456-426614174001",
					"123e4567-e89b-12d3-a456-426614174002",
					"123e4567-e89b-12d3-a456-426614174003"
				],
				"comparison_metrics": ["accuracy", "f1_score", "precision", "recall"],
				"ranking_criteria": "f1_score",
				"include_statistical_tests": True
			}
		}


class ModelComparisonItem(BaseModel):
	"""Comparison data for a single model run."""
	
	model_run_id: str
	model_type: str
	experiment_id: str
	status: str
	metrics: Dict[str, Any]
	hyperparameters: Dict[str, Any]
	training_time: Optional[float] = None
	created_at: str
	rank: Optional[int] = None
	ranking_score: Optional[float] = None
	
	class Config:
		json_schema_extra = {
			"example": {
				"model_run_id": "123e4567-e89b-12d3-a456-426614174001",
				"model_type": "random_forest_classifier",
				"experiment_id": "123e4567-e89b-12d3-a456-426614174000",
				"status": "completed",
				"metrics": {
					"accuracy": 0.95,
					"precision": 0.94,
					"recall": 0.93,
					"f1_score": 0.935
				},
				"hyperparameters": {
					"n_estimators": 100,
					"max_depth": 10
				},
				"training_time": 45.5,
				"created_at": "2025-12-29T10:00:00Z",
				"rank": 1,
				"ranking_score": 0.935
			}
		}


class MetricStatistics(BaseModel):
	"""Statistical summary for a metric across models."""
	
	metric_name: str
	mean: float
	std: float
	min: float
	max: float
	best_model_id: str
	worst_model_id: str
	
	class Config:
		json_schema_extra = {
			"example": {
				"metric_name": "accuracy",
				"mean": 0.93,
				"std": 0.02,
				"min": 0.90,
				"max": 0.95,
				"best_model_id": "123e4567-e89b-12d3-a456-426614174001",
				"worst_model_id": "123e4567-e89b-12d3-a456-426614174003"
			}
		}


class ModelComparisonResponse(BaseModel):
	"""Response schema for model comparison."""
	
	comparison_id: str
	task_type: str
	total_models: int
	compared_models: List[ModelComparisonItem]
	best_model: ModelComparisonItem
	metric_statistics: List[MetricStatistics]
	ranking_criteria: str
	recommendations: List[str]
	timestamp: str
	
	class Config:
		json_schema_extra = {
			"example": {
				"comparison_id": "comp-abc-123",
				"task_type": "classification",
				"total_models": 3,
				"compared_models": [
					{
						"model_run_id": "123e4567-e89b-12d3-a456-426614174001",
						"model_type": "random_forest_classifier",
						"status": "completed",
						"metrics": {"accuracy": 0.95, "f1_score": 0.935},
						"rank": 1,
						"ranking_score": 0.935
					}
				],
				"best_model": {
					"model_run_id": "123e4567-e89b-12d3-a456-426614174001",
					"model_type": "random_forest_classifier",
					"rank": 1,
					"ranking_score": 0.935
				},
				"metric_statistics": [
					{
						"metric_name": "accuracy",
						"mean": 0.93,
						"std": 0.02,
						"min": 0.90,
						"max": 0.95,
						"best_model_id": "123e4567-e89b-12d3-a456-426614174001",
						"worst_model_id": "123e4567-e89b-12d3-a456-426614174003"
					}
				],
				"ranking_criteria": "f1_score",
				"recommendations": [
					"random_forest_classifier achieved the best f1_score of 0.935",
					"Consider ensembling the top 2 models for improved performance"
				],
				"timestamp": "2025-12-30T10:00:00Z"
			}
		}


class ModelRankingRequest(BaseModel):
	"""Request schema for ranking models by custom criteria."""
	
	model_run_ids: List[UUID] = Field(..., min_length=2, max_length=20)
	ranking_weights: Dict[str, float] = Field(
		...,
		description="Metric weights for composite ranking score. Must sum to 1.0"
	)
	higher_is_better: Optional[Dict[str, bool]] = None
	
	class Config:
		json_schema_extra = {
			"example": {
				"model_run_ids": [
					"123e4567-e89b-12d3-a456-426614174001",
					"123e4567-e89b-12d3-a456-426614174002"
				],
				"ranking_weights": {
					"f1_score": 0.5,
					"precision": 0.3,
					"recall": 0.2
				},
				"higher_is_better": {
					"f1_score": True,
					"precision": True,
					"recall": True
				}
			}
		}


class RankedModel(BaseModel):
	"""A ranked model with composite score."""
	
	model_run_id: str
	model_type: str
	rank: int
	composite_score: float
	individual_scores: Dict[str, float]
	weighted_contributions: Dict[str, float]
	
	class Config:
		json_schema_extra = {
			"example": {
				"model_run_id": "123e4567-e89b-12d3-a456-426614174001",
				"model_type": "random_forest_classifier",
				"rank": 1,
				"composite_score": 0.935,
				"individual_scores": {
					"f1_score": 0.935,
					"precision": 0.94,
					"recall": 0.93
				},
				"weighted_contributions": {
					"f1_score": 0.4675,
					"precision": 0.282,
					"recall": 0.186
				}
			}
		}


class ModelRankingResponse(BaseModel):
	"""Response schema for model ranking."""
	
	ranking_id: str
	ranked_models: List[RankedModel]
	ranking_weights: Dict[str, float]
	best_model: RankedModel
	score_range: Dict[str, Any]
	timestamp: str
	
	class Config:
		json_schema_extra = {
			"example": {
				"ranking_id": "rank-xyz-789",
				"ranked_models": [
					{
						"model_run_id": "123e4567-e89b-12d3-a456-426614174001",
						"model_type": "random_forest_classifier",
						"rank": 1,
						"composite_score": 0.935,
						"individual_scores": {"f1_score": 0.935, "precision": 0.94},
						"weighted_contributions": {"f1_score": 0.4675, "precision": 0.282}
					}
				],
				"ranking_weights": {"f1_score": 0.5, "precision": 0.3, "recall": 0.2},
				"best_model": {
					"model_run_id": "123e4567-e89b-12d3-a456-426614174001",
					"rank": 1,
					"composite_score": 0.935
				},
				"score_range": {
					"min": 0.85,
					"max": 0.935,
					"spread": 0.085
				},
				"timestamp": "2025-12-30T10:00:00Z"
			}
		}
