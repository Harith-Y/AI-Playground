from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel


class CodeArtifactBase(BaseModel):
    experiment_id: UUID
    code_type: str  # notebook, python, fastapi
    file_path: Optional[str] = None


class CodeArtifactCreate(CodeArtifactBase):
    pass


class CodeArtifactRead(CodeArtifactBase):
    id: UUID
    generated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
