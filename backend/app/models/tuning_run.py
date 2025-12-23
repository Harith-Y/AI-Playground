"""
TuningRun database model
"""
from sqlalchemy import Column, String, DateTime, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum

from app.db.base import Base


class TuningStatus(str, enum.Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TuningRun(Base):
    __tablename__ = "tuning_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    model_run_id = Column(UUID(as_uuid=True), ForeignKey("model_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    tuning_method = Column(String, nullable=False)  # e.g., "grid_search", "random_search", "bayesian"
    best_params = Column(JSONB, nullable=True)  # Best hyperparameters found
    results = Column(JSONB, nullable=True)  # Top N results with scores
    status = Column(Enum(TuningStatus), default=TuningStatus.RUNNING, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    model_run = relationship("ModelRun", back_populates="tuning_runs")

    def __repr__(self):
        return f"<TuningRun(id={self.id}, method={self.tuning_method}, status={self.status})>"
