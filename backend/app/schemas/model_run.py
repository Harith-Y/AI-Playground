from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel


class ModelRunBase(BaseModel):
    experiment_id: UUID
    model_type: str
    hyperparameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    training_time: Optional[float] = None
    model_artifact_path: Optional[str] = None


class ModelRunCreate(ModelRunBase):
    pass


class ModelRunRead(ModelRunBase):
    id: UUID
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True
