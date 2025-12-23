"""
GeneratedCode database model
"""
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class GeneratedCode(Base):
    __tablename__ = "generated_code"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    code_type = Column(String, nullable=False)  # e.g., "notebook", "python", "fastapi"
    file_path = Column(String, nullable=False)  # Path to generated code file
    generated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="generated_code")

    def __repr__(self):
        return f"<GeneratedCode(id={self.id}, code_type={self.code_type})>"
