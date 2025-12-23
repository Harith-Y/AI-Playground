from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel


class ExperimentStatus(str):
	# Running, completed, failed as examples; using str for simplicity
	pass


class ExperimentBase(BaseModel):
	user_id: UUID
	dataset_id: UUID
	name: Optional[str] = None
	status: Optional[str] = None  # 'running', 'completed', 'failed'


class ExperimentCreate(ExperimentBase):
	pass


class ExperimentUpdate(BaseModel):
	name: Optional[str] = None
	status: Optional[str] = None


class ExperimentRead(ExperimentBase):
	id: UUID
	created_at: Optional[datetime] = None

	class Config:
		from_attributes = True
