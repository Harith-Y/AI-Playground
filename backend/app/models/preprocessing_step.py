"""
PreprocessingStep database model
"""
from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid

from app.db.base import Base


class PreprocessingStep(Base):
    __tablename__ = "preprocessing_steps"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    step_type = Column(String, nullable=False)  # e.g., "missing_value_imputation", "scaling", "encoding"
    parameters = Column(JSONB, nullable=True)  # {"strategy": "mean", "fill_value": 0, ...}
    column_name = Column(String, nullable=True)  # Column this step applies to (null for all columns)
    order = Column(Integer, nullable=False, default=0)  # Order of execution
    
    # Relationships
    dataset = relationship("Dataset", back_populates="preprocessing_steps")

    def __repr__(self):
        return f"<PreprocessingStep(id={self.id}, step_type={self.step_type}, order={self.order})>"
