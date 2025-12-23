from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, field_validator


class DatasetShape(BaseModel):
	rows: int
	cols: int

	@field_validator("rows", "cols")
	@classmethod
	def non_negative(cls, v: int) -> int:
		if v < 0:
			raise ValueError("shape values must be >= 0")
		return v


class DatasetBase(BaseModel):
	user_id: UUID
	name: str
	file_path: Optional[str] = None
	shape: Optional[DatasetShape] = None
	dtypes: Optional[Dict[str, Any]] = None
	missing_values: Optional[Dict[str, Any]] = None


class DatasetCreate(DatasetBase):
	pass


class DatasetUpdate(BaseModel):
	name: Optional[str] = None
	file_path: Optional[str] = None
	shape: Optional[DatasetShape] = None
	dtypes: Optional[Dict[str, Any]] = None
	missing_values: Optional[Dict[str, Any]] = None


class DatasetRead(DatasetBase):
	id: UUID
	uploaded_at: Optional[datetime] = None

	class Config:
		from_attributes = True
