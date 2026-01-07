from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel


class PreprocessingStepBase(BaseModel):
	dataset_id: UUID
	step_type: str  # e.g., 'missing_value_imputation'
	parameters: Optional[Dict[str, Any]] = None
	column_name: Optional[str] = None
	order: Optional[int] = None


class PreprocessingStepCreate(PreprocessingStepBase):
	pass


class PreprocessingStepUpdate(BaseModel):
	step_type: Optional[str] = None
	parameters: Optional[Dict[str, Any]] = None
	column_name: Optional[str] = None
	order: Optional[int] = None


class PreprocessingStepRead(PreprocessingStepBase):
	id: UUID

	class Config:
		from_attributes = True


class ReorderRequest(BaseModel):
	"""Request to reorder preprocessing steps"""
	dataset_id: UUID
	step_ids: List[str]


# Apply preprocessing pipeline schemas
class PreprocessingApplyRequest(BaseModel):
	"""Request to apply preprocessing pipeline to a dataset"""
	dataset_id: UUID
	save_output: Optional[bool] = True
	output_name: Optional[str] = None


class PreprocessingApplyResponse(BaseModel):
	"""Response from applying preprocessing pipeline"""
	success: bool
	message: str
	steps_applied: int
	original_shape: List[int]  # [rows, cols]
	transformed_shape: List[int]  # [rows, cols]
	output_dataset_id: Optional[UUID] = None
	preview: Optional[List[Dict[str, Any]]] = None  # Sample rows
	statistics: Optional[Dict[str, Any]] = None  # Basic stats


# Async preprocessing schemas
class PreprocessingAsyncResponse(BaseModel):
	"""Response when starting async preprocessing task"""
	task_id: str
	message: str
	status: str


class PreprocessingTaskStatus(BaseModel):
	"""Status of a preprocessing task"""
	task_id: str
	state: str  # PENDING, STARTED, PROGRESS, SUCCESS, FAILURE
	status: Optional[str] = None  # Human-readable status message
	progress: Optional[int] = None  # 0-100
	current_step: Optional[str] = None
	result: Optional[Dict[str, Any]] = None  # Final result if SUCCESS
	error: Optional[str] = None  # Error message if FAILURE


# Preview schemas for before/after comparison
class DataPreview(BaseModel):
	"""Data preview for a single dataset state"""
	shape: List[int]  # [rows, cols]
	columns: List[str]
	dtypes: Dict[str, str]
	data: List[Dict[str, Any]]  # Sample rows (default 10 rows)
	missing_counts: Dict[str, int]
	null_counts: Dict[str, int]
	statistics: Optional[Dict[str, Any]] = None


class PreprocessingPreviewResponse(BaseModel):
	"""Before/After preview comparison for preprocessing pipeline"""
	success: bool
	message: str
	before: DataPreview
	after: DataPreview
	steps_applied: int
	transformations: Optional[List[Dict[str, Any]]] = None  # Details of what changed


# Preprocessing history schemas
class PreprocessingHistoryBase(BaseModel):
	"""Base schema for preprocessing history"""
	dataset_id: UUID
	operation_type: str  # 'apply', 'preview', 'execute'
	steps_applied: List[Dict[str, Any]]  # Array of step details
	original_shape: List[int]  # [rows, cols]
	transformed_shape: List[int]  # [rows, cols]
	execution_time: Optional[float] = None
	status: str = 'success'  # 'success', 'failure'
	error_message: Optional[str] = None
	output_dataset_id: Optional[UUID] = None
	statistics: Optional[Dict[str, Any]] = None


class PreprocessingHistoryCreate(PreprocessingHistoryBase):
	"""Schema for creating preprocessing history"""
	user_id: UUID


class PreprocessingHistoryRead(PreprocessingHistoryBase):
	"""Schema for reading preprocessing history"""
	id: UUID
	user_id: UUID
	created_at: datetime

	class Config:
		from_attributes = True


class PreprocessingHistoryList(BaseModel):
	"""Schema for listing preprocessing history"""
	total: int
	items: List[PreprocessingHistoryRead]
	page: int
	page_size: int
	has_next: bool
	has_prev: bool
