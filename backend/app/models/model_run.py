"""
ModelRun database model
"""
from sqlalchemy import Column, String, DateTime, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    model_type = Column(String, nullable=False)  # e.g., "linear_regression", "random_forest", etc.
    
    # Model configuration and results
    hyperparameters = Column(JSONB, nullable=True)  # {"max_depth": 10, "n_estimators": 100, ...}
    metrics = Column(JSONB, nullable=True)  # {"accuracy": 0.95, "f1_score": 0.93, ...}
    training_time = Column(Float, nullable=True)  # Training duration in seconds
    model_artifact_path = Column(String, nullable=True)  # Path to saved model file
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="model_runs")
    tuning_runs = relationship("TuningRun", back_populates="model_run", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ModelRun(id={self.id}, model_type={self.model_type}, experiment_id={self.experiment_id})>"
