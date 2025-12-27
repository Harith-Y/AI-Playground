"""
Pydantic schemas for Preprocessing Pipeline API endpoints.
"""
from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


class PipelineStepConfig(BaseModel):
    """Configuration for a single preprocessing step in a pipeline."""
    class_name: str = Field(..., description="Class name of the preprocessing step")
    name: str = Field(..., description="Instance name of the step")
    params: Dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    fitted: bool = Field(default=False, description="Whether step is fitted")


class PipelineConfigBase(BaseModel):
    """Base configuration for a pipeline."""
    name: str = Field(..., description="Pipeline name")
    description: Optional[str] = Field(None, description="Pipeline description")
    steps: List[PipelineStepConfig] = Field(default_factory=list, description="List of preprocessing steps")


class PipelineCreate(BaseModel):
    """Schema for creating a new pipeline."""
    name: str = Field(..., min_length=1, max_length=255, description="Pipeline name")
    description: Optional[str] = Field(None, max_length=1000, description="Pipeline description")
    dataset_id: Optional[UUID] = Field(None, description="Optional dataset ID this pipeline is designed for")
    steps: List[PipelineStepConfig] = Field(default_factory=list, description="List of preprocessing steps")


class PipelineUpdate(BaseModel):
    """Schema for updating an existing pipeline."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Pipeline name")
    description: Optional[str] = Field(None, max_length=1000, description="Pipeline description")
    dataset_id: Optional[UUID] = Field(None, description="Dataset ID this pipeline is designed for")
    steps: Optional[List[PipelineStepConfig]] = Field(None, description="List of preprocessing steps")


class PipelineRead(BaseModel):
    """Schema for reading pipeline information."""
    id: UUID
    user_id: UUID
    dataset_id: Optional[UUID]
    name: str
    description: Optional[str]
    version: int
    fitted: bool
    num_steps: int
    step_summary: Optional[List[Dict[str, Any]]]
    created_at: datetime
    updated_at: datetime
    fitted_at: Optional[datetime]

    class Config:
        from_attributes = True


class PipelineDetail(PipelineRead):
    """Detailed pipeline schema including full configuration."""
    config: Dict[str, Any] = Field(..., description="Complete pipeline configuration")
    statistics: Optional[Dict[str, Any]] = Field(None, description="Fitting statistics")


class PipelineListResponse(BaseModel):
    """Response schema for listing pipelines."""
    total: int
    pipelines: List[PipelineRead]


class PipelineFitRequest(BaseModel):
    """Request schema for fitting a pipeline."""
    dataset_id: UUID = Field(..., description="Dataset ID to fit the pipeline on")
    save_fitted: bool = Field(default=True, description="Whether to save fitted pipeline")


class PipelineFitResponse(BaseModel):
    """Response schema for pipeline fitting."""
    success: bool
    message: str
    pipeline_id: UUID
    fitted: bool
    statistics: Optional[Dict[str, Any]] = Field(None, description="Fitting statistics")
    duration_seconds: Optional[float] = Field(None, description="Fitting duration")


class PipelineTransformRequest(BaseModel):
    """Request schema for transforming data with a pipeline."""
    dataset_id: UUID = Field(..., description="Dataset ID to transform")
    save_output: bool = Field(default=True, description="Whether to save transformed dataset")
    output_name: Optional[str] = Field(None, description="Name for output dataset")


class PipelineTransformResponse(BaseModel):
    """Response schema for pipeline transformation."""
    success: bool
    message: str
    original_shape: List[int]
    transformed_shape: List[int]
    output_dataset_id: Optional[UUID] = Field(None, description="ID of saved output dataset")
    preview: Optional[List[Dict[str, Any]]] = Field(None, description="Preview of transformed data")
    duration_seconds: Optional[float] = Field(None, description="Transformation duration")


class PipelineFitTransformRequest(BaseModel):
    """Request schema for fit-transform operation."""
    dataset_id: UUID = Field(..., description="Dataset ID to fit and transform")
    save_fitted: bool = Field(default=True, description="Whether to save fitted pipeline")
    save_output: bool = Field(default=True, description="Whether to save transformed dataset")
    output_name: Optional[str] = Field(None, description="Name for output dataset")


class PipelineFitTransformResponse(BaseModel):
    """Response schema for fit-transform operation."""
    success: bool
    message: str
    pipeline_id: UUID
    fitted: bool
    original_shape: List[int]
    transformed_shape: List[int]
    output_dataset_id: Optional[UUID] = Field(None, description="ID of saved output dataset")
    statistics: Optional[Dict[str, Any]] = Field(None, description="Fitting statistics")
    duration_seconds: Optional[float] = Field(None, description="Total duration")


class PipelineExportCodeRequest(BaseModel):
    """Request schema for exporting pipeline as code."""
    format: str = Field(default="sklearn", description="Export format (sklearn, standalone)")
    include_comments: bool = Field(default=True, description="Include explanatory comments")
    include_imports: bool = Field(default=True, description="Include import statements")


class PipelineExportCodeResponse(BaseModel):
    """Response schema for pipeline code export."""
    success: bool
    code: str = Field(..., description="Generated Python code")
    format: str
    pipeline_name: str


class PipelineCloneRequest(BaseModel):
    """Request schema for cloning a pipeline."""
    name: str = Field(..., description="Name for the cloned pipeline")
    description: Optional[str] = Field(None, description="Description for the cloned pipeline")


class PipelineCloneResponse(BaseModel):
    """Response schema for pipeline cloning."""
    success: bool
    message: str
    cloned_pipeline_id: UUID
    original_pipeline_id: UUID


class PipelineImportRequest(BaseModel):
    """Request schema for importing a pipeline from JSON."""
    config: Dict[str, Any] = Field(..., description="Pipeline configuration")
    name: str = Field(..., description="Name for the imported pipeline")
    description: Optional[str] = Field(None, description="Description for the imported pipeline")
    dataset_id: Optional[UUID] = Field(None, description="Optional dataset ID")


class PipelineImportResponse(BaseModel):
    """Response schema for pipeline import."""
    success: bool
    message: str
    pipeline_id: UUID


class PipelineStepAdd(BaseModel):
    """Schema for adding a step to a pipeline."""
    step: PipelineStepConfig = Field(..., description="Step configuration")
    position: Optional[int] = Field(None, description="Position to insert step (None = append)")


class PipelineStepRemove(BaseModel):
    """Schema for removing a step from a pipeline."""
    step_index: Optional[int] = Field(None, description="Index of step to remove")
    step_name: Optional[str] = Field(None, description="Name of step to remove")


class PipelineStepReorder(BaseModel):
    """Schema for reordering pipeline steps."""
    new_order: List[int] = Field(..., description="List of indices in new order")


class PipelineVersionInfo(BaseModel):
    """Schema for pipeline version information."""
    pipeline_id: UUID
    version: int
    created_at: datetime
    fitted: bool
    num_steps: int


class PipelineVersionListResponse(BaseModel):
    """Response schema for listing pipeline versions."""
    total: int
    versions: List[PipelineVersionInfo]


class PipelineStatistics(BaseModel):
    """Schema for pipeline statistics."""
    pipeline_id: UUID
    pipeline_name: str
    num_steps: int
    fitted: bool
    step_statistics: List[Dict[str, Any]] = Field(default_factory=list, description="Per-step statistics")
    total_fit_duration: Optional[float] = Field(None, description="Total fitting time in seconds")
    num_samples_fitted: Optional[int] = Field(None, description="Number of samples used for fitting")


class PipelineComparisonRequest(BaseModel):
    """Request schema for comparing pipelines."""
    pipeline_ids: List[UUID] = Field(..., min_items=2, max_items=5, description="Pipeline IDs to compare")


class PipelineComparisonResponse(BaseModel):
    """Response schema for pipeline comparison."""
    pipelines: List[Dict[str, Any]]
    comparison_summary: Dict[str, Any]


class PipelineValidationResponse(BaseModel):
    """Response schema for pipeline validation."""
    valid: bool
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    step_validations: List[Dict[str, Any]] = Field(default_factory=list, description="Per-step validation results")
