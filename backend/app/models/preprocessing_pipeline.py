"""
PreprocessingPipeline database model for storing complete pipeline configurations.
"""
from sqlalchemy import Column, String, DateTime, Integer, ForeignKey, Boolean, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class PreprocessingPipeline(Base):
    """
    Database model for storing complete preprocessing pipelines.

    A pipeline encapsulates a sequence of preprocessing steps that can be
    applied to datasets. This model stores both the configuration and metadata
    about the pipeline, including fitted parameters when saved.
    """
    __tablename__ = "preprocessing_pipelines"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True, index=True)

    # Pipeline identification
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    version = Column(Integer, nullable=False, default=1)

    # Pipeline configuration
    config = Column(JSONB, nullable=False)  # Complete pipeline configuration (steps, params, etc.)
    fitted = Column(Boolean, nullable=False, default=False)

    # Storage paths for fitted pipeline
    pickle_path = Column(String, nullable=True)  # Path to pickled fitted pipeline

    # Metadata
    num_steps = Column(Integer, nullable=False, default=0)
    step_summary = Column(JSONB, nullable=True)  # Summary of steps for quick reference
    statistics = Column(JSONB, nullable=True)  # Statistics from fitting

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    fitted_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="preprocessing_pipelines")
    dataset = relationship("Dataset")

    def __repr__(self):
        return f"<PreprocessingPipeline(id={self.id}, name={self.name}, steps={self.num_steps}, fitted={self.fitted})>"
