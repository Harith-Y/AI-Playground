from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel


class ModelTrainingBase(BaseModel):
	experiment_id: UUID
	model_type: str
	hyperparameters: Optional[Dict[str, Any]] = None


class ModelTrainingRequest(ModelTrainingBase):
	pass


class ModelTrainingResponse(BaseModel):
	model_run_id: UUID
	status: str
	created_at: datetime
