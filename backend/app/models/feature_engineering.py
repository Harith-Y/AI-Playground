"""
FeatureEngineering database model
"""
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class FeatureEngineering(Base):
    __tablename__ = "feature_engineering"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    operation = Column(String, nullable=False)  # e.g., "pca", "feature_selection", "polynomial_features"
    parameters = Column(JSONB, nullable=True)  # {"n_components": 5, "degree": 2, ...}
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="feature_engineering")

    def __repr__(self):
        return f"<FeatureEngineering(id={self.id}, operation={self.operation})>"
