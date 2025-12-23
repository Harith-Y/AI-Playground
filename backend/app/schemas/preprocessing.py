from typing import Optional, Dict, Any
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
