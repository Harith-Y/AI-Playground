from typing import Optional, Dict, Any, List
from uuid import UUID
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
