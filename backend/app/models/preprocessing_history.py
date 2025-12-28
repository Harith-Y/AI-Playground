"""
PreprocessingHistory database model
"""
from sqlalchemy import Column, String, DateTime, Float, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class PreprocessingHistory(Base):
    __tablename__ = "preprocessing_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Operation details
    operation_type = Column(String, nullable=False)  # 'apply', 'preview', 'execute'
    steps_applied = Column(JSONB, nullable=False)  # Array of step details [{step_type, parameters, column_name, order}, ...]

    # Data shape tracking
    original_shape = Column(ARRAY(Integer), nullable=False)  # [rows, cols]
    transformed_shape = Column(ARRAY(Integer), nullable=False)  # [rows, cols]

    # Execution tracking
    execution_time = Column(Float, nullable=True)  # Execution time in seconds
    status = Column(String, nullable=False, default='success')  # 'success', 'failure'
    error_message = Column(String, nullable=True)  # Error message if failure

    # Additional metadata
    output_dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)  # If saved as new dataset
    statistics = Column(JSONB, nullable=True)  # Summary statistics before/after

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    # Relationships
    dataset = relationship("Dataset", foreign_keys=[dataset_id], backref="preprocessing_history")
    user = relationship("User", backref="preprocessing_operations")
    output_dataset = relationship("Dataset", foreign_keys=[output_dataset_id])

    def __repr__(self):
        return f"<PreprocessingHistory(id={self.id}, dataset_id={self.dataset_id}, operation_type={self.operation_type}, status={self.status})>"
