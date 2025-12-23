from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel


class FeatureEngineeringBase(BaseModel):
    dataset_id: UUID
    operation: str
    parameters: Optional[Dict[str, Any]] = None


class FeatureEngineeringCreate(FeatureEngineeringBase):
    pass


class FeatureEngineeringRead(FeatureEngineeringBase):
    id: UUID
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True
