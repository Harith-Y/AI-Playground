"""
Dataset database model
"""
from sqlalchemy import Column, String, DateTime, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    
    # Dataset metadata
    rows = Column(Integer, nullable=True)
    cols = Column(Integer, nullable=True)
    dtypes = Column(JSONB, nullable=True)  # {"col1": "int64", "col2": "float64", ...}
    missing_values = Column(JSONB, nullable=True)  # {"col1": 5, "col2": 10, ...}
    
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="datasets")
    preprocessing_steps = relationship("PreprocessingStep", back_populates="dataset", cascade="all, delete-orphan")
    feature_engineering = relationship("FeatureEngineering", back_populates="dataset", cascade="all, delete-orphan")
    experiments = relationship("Experiment", back_populates="dataset", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Dataset(id={self.id}, name={self.name}, shape=({self.rows}, {self.cols}))>"
