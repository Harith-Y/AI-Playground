from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel


class TuningRunBase(BaseModel):
    model_run_id: UUID
    tuning_method: str
    best_params: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None  # top N results
    status: Optional[str] = None


class TuningRunCreate(TuningRunBase):
    pass


class TuningRunRead(TuningRunBase):
    id: UUID
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True
