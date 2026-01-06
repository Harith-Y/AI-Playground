from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, field_validator, computed_field


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
	rows: Optional[int] = None
	cols: Optional[int] = None
	
	# Computed fields for frontend compatibility
	@computed_field
	@property
	def rowCount(self) -> int:
		"""Alias for rows to match frontend expectations"""
		return self.rows or (self.shape.rows if self.shape else 0)
	
	@computed_field
	@property
	def columnCount(self) -> int:
		"""Alias for cols to match frontend expectations"""
		return self.cols or (self.shape.cols if self.shape else 0)
	
	@computed_field
	@property
	def size(self) -> int:
		"""Estimate file size for frontend display"""
		# This is an approximation - ideally should come from actual file size
		if self.rows and self.cols:
			return self.rows * self.cols * 8  # rough estimate
		return 0
	
	@computed_field
	@property
	def createdAt(self) -> Optional[str]:
		"""Alias for uploaded_at to match frontend expectations"""
		return self.uploaded_at.isoformat() if self.uploaded_at else None
	
	@computed_field
	@property
	def updatedAt(self) -> Optional[str]:
		"""Alias for uploaded_at to match frontend expectations"""
		return self.uploaded_at.isoformat() if self.uploaded_at else None
	
	@computed_field
	@property
	def filename(self) -> str:
		"""Extract filename from file_path"""
		if self.file_path:
			return self.file_path.split('/')[-1]
		return ""
	
	@computed_field
	@property
	def status(self) -> str:
		"""Dataset status - always 'ready' for existing datasets"""
		return "ready"

	class Config:
		from_attributes = True


class ColumnInfo(BaseModel):
	"""Column metadata for dataset preview"""
	name: str
	dataType: str
	dtype: str  # Alias for dataType (required by frontend)
	nullCount: int
	uniqueCount: int
	sampleValues: List[Any]


class DatasetPreviewResponse(BaseModel):
	"""Dataset preview with column info and sample rows"""
	preview: List[List[Any]]  # 2D array of values
	columns: List[ColumnInfo]
	totalRows: int
	displayedRows: int


class DatasetStatsResponse(BaseModel):
	"""Dataset statistics summary"""
	rowCount: int
	columnCount: int
	numericColumns: int
	categoricalColumns: int
	missingValues: int
	duplicateRows: int
	memoryUsage: int  # in bytes
	columns: List[ColumnInfo]
