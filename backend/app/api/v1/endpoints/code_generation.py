"""
Code Generation Endpoints

Provides endpoints for generating production-ready Python code from ML experiments.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

from app.ml_engine.code_generation import (
    generate_preprocessing_code,
    generate_training_code,
    generate_evaluation_code,
    generate_prediction_code,
    generate_requirements,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class PreprocessingStepConfig(BaseModel):
    """Configuration for a preprocessing step."""
    type: str = Field(..., description="Step type (e.g., 'missing_value_imputation')")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    name: Optional[str] = Field(None, description="Step name")


class DatasetInfo(BaseModel):
    """Dataset information."""
    file_path: str = Field(..., description="Path to dataset file")
    file_format: str = Field(default="csv", description="File format (csv, excel, json, parquet)")
    target_column: Optional[str] = Field(None, description="Target column name")
    feature_columns: Optional[List[str]] = Field(None, description="Feature column names")


class CodeGenerationRequest(BaseModel):
    """Request for Python code generation."""
    
    # Experiment info
    experiment_name: str = Field(default="ML Experiment", description="Name of the experiment")
    experiment_id: Optional[str] = Field(None, description="Experiment UUID")
    
    # Dataset info
    dataset_info: Optional[DatasetInfo] = Field(None, description="Dataset information")
    
    # Model configuration
    model_type: str = Field(..., description="Model type (e.g., 'random_forest_classifier')")
    task_type: str = Field(..., description="Task type (classification, regression, clustering)")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    
    # Preprocessing
    preprocessing_steps: List[PreprocessingStepConfig] = Field(
        default_factory=list,
        description="Preprocessing steps"
    )
    
    # Training configuration
    test_size: float = Field(default=0.2, description="Test set size", ge=0.0, le=1.0)
    validation_size: float = Field(default=0.0, description="Validation set size", ge=0.0, le=1.0)
    random_state: int = Field(default=42, description="Random seed")
    cross_validation: bool = Field(default=False, description="Use cross-validation")
    cv_folds: int = Field(default=5, description="Number of CV folds", ge=2)
    
    # Code generation options
    output_format: str = Field(
        default="script",
        description="Output format (script, function, class, module, api)"
    )
    include_imports: bool = Field(default=True, description="Include import statements")
    include_evaluation: bool = Field(default=True, description="Include evaluation code")
    modular: bool = Field(default=False, description="Generate modular code")
    
    # Additional options
    save_model: bool = Field(default=True, description="Include model saving code")
    model_path: str = Field(default="model.pkl", description="Model save path")
    
    class Config:
        schema_extra = {
            "example": {
                "experiment_name": "Customer Churn Prediction",
                "model_type": "random_forest_classifier",
                "task_type": "classification",
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5
                },
                "preprocessing_steps": [
                    {
                        "type": "missing_value_imputation",
                        "parameters": {
                            "strategy": "mean",
                            "columns": ["age", "income"]
                        }
                    },
                    {
                        "type": "scaling",
                        "parameters": {
                            "scaler": "standard",
                            "columns": ["age", "income", "balance"]
                        }
                    }
                ],
                "dataset_info": {
                    "file_path": "data/churn.csv",
                    "file_format": "csv",
                    "target_column": "churn",
                    "feature_columns": ["age", "income", "balance", "tenure"]
                },
                "test_size": 0.2,
                "random_state": 42,
                "output_format": "module",
                "include_evaluation": True
            }
        }


class CodeGenerationResponse(BaseModel):
    """Response containing generated code."""
    
    code: str = Field(..., description="Generated Python code")
    code_type: str = Field(..., description="Type of code (preprocessing, training, etc.)")
    output_format: str = Field(..., description="Output format used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "code": "# Auto-generated Python code\nimport pandas as pd\n...",
                "code_type": "complete_pipeline",
                "output_format": "module",
                "metadata": {
                    "generated_at": "2025-12-31T12:00:00",
                    "model_type": "random_forest_classifier",
                    "lines_of_code": 150
                }
            }
        }


class ModularCodeResponse(BaseModel):
    """Response containing modular code sections."""
    
    preprocessing_code: Optional[str] = Field(None, description="Preprocessing code")
    training_code: Optional[str] = Field(None, description="Training code")
    evaluation_code: Optional[str] = Field(None, description="Evaluation code")
    prediction_code: Optional[str] = Field(None, description="Prediction code")
    requirements: Optional[Dict[str, str]] = Field(None, description="Requirements files")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")


class RequirementsGenerationRequest(BaseModel):
    """Request for requirements.txt generation."""
    
    model_type: str = Field(..., description="Model type")
    task_type: str = Field(..., description="Task type")
    preprocessing_steps: List[PreprocessingStepConfig] = Field(
        default_factory=list,
        description="Preprocessing steps"
    )
    include_evaluation: bool = Field(default=True, description="Include evaluation dependencies")
    output_format: str = Field(
        default="pip",
        description="Output format (pip, docker, conda)"
    )
    modular: bool = Field(default=False, description="Generate modular requirements")


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/python", response_model=CodeGenerationResponse)
async def generate_python_code(
    request: CodeGenerationRequest = Body(...),
) -> CodeGenerationResponse:
    """
    Generate production-ready Python code from experiment configuration.
    
    This endpoint generates complete, executable Python code including:
    - Data preprocessing
    - Model training
    - Model evaluation
    - Prediction/inference
    
    The generated code is production-ready and follows best practices.
    
    Args:
        request: Code generation configuration
    
    Returns:
        Generated Python code with metadata
    
    Raises:
        HTTPException: If code generation fails
    """
    try:
        logger.info(
            f"Generating Python code for {request.model_type}",
            extra={
                'event': 'code_generation_start',
                'model_type': request.model_type,
                'output_format': request.output_format,
                'modular': request.modular
            }
        )
        
        # Prepare configuration
        config = _prepare_config(request)
        
        # Generate code based on format
        if request.modular:
            # Generate modular code (separate files)
            code_sections = _generate_modular_code(config, request)
            
            # Combine into single response for non-modular endpoint
            code = _combine_code_sections(code_sections)
            code_type = "modular_pipeline"
            
        else:
            # Generate complete pipeline code
            code = _generate_complete_pipeline(config, request)
            code_type = "complete_pipeline"
        
        # Calculate metadata
        metadata = {
            'generated_at': datetime.utcnow().isoformat(),
            'model_type': request.model_type,
            'task_type': request.task_type,
            'output_format': request.output_format,
            'lines_of_code': len(code.split('\n')),
            'includes_preprocessing': len(request.preprocessing_steps) > 0,
            'includes_evaluation': request.include_evaluation,
            'experiment_name': request.experiment_name,
        }
        
        logger.info(
            f"Code generation completed: {metadata['lines_of_code']} lines",
            extra={
                'event': 'code_generation_complete',
                **metadata
            }
        )
        
        return CodeGenerationResponse(
            code=code,
            code_type=code_type,
            output_format=request.output_format,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(
            f"Code generation failed: {e}",
            extra={
                'event': 'code_generation_failed',
                'error': str(e),
                'model_type': request.model_type
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Code generation failed: {str(e)}"
        )


@router.post("/python/modular", response_model=ModularCodeResponse)
async def generate_modular_python_code(
    request: CodeGenerationRequest = Body(...),
    include_requirements: bool = Query(
        default=True,
        description="Include requirements.txt files"
    )
) -> ModularCodeResponse:
    """
    Generate modular Python code with separate files for each component.
    
    This endpoint generates separate code files for:
    - Preprocessing
    - Training
    - Evaluation
    - Prediction
    - Requirements (optional)
    
    Each component can be used independently or combined into a pipeline.
    
    Args:
        request: Code generation configuration
        include_requirements: Whether to include requirements files
    
    Returns:
        Modular code sections with metadata
    
    Raises:
        HTTPException: If code generation fails
    """
    try:
        logger.info(
            f"Generating modular Python code for {request.model_type}",
            extra={
                'event': 'modular_code_generation_start',
                'model_type': request.model_type,
                'include_requirements': include_requirements
            }
        )
        
        # Prepare configuration
        config = _prepare_config(request)
        
        # Generate modular code sections
        code_sections = _generate_modular_code(config, request)
        
        # Generate requirements if requested
        requirements = None
        if include_requirements:
            requirements = generate_requirements(
                config,
                modular=True,
                code_sections=code_sections
            )
        
        # Calculate metadata
        metadata = {
            'generated_at': datetime.utcnow().isoformat(),
            'model_type': request.model_type,
            'task_type': request.task_type,
            'total_lines': sum(len(code.split('\n')) for code in code_sections.values()),
            'components': list(code_sections.keys()),
            'experiment_name': request.experiment_name,
        }
        
        logger.info(
            f"Modular code generation completed: {len(code_sections)} components",
            extra={
                'event': 'modular_code_generation_complete',
                **metadata
            }
        )
        
        return ModularCodeResponse(
            preprocessing_code=code_sections.get('preprocessing'),
            training_code=code_sections.get('training'),
            evaluation_code=code_sections.get('evaluation'),
            prediction_code=code_sections.get('prediction'),
            requirements=requirements,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(
            f"Modular code generation failed: {e}",
            extra={
                'event': 'modular_code_generation_failed',
                'error': str(e)
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Modular code generation failed: {str(e)}"
        )


@router.post("/requirements", response_model=Dict[str, str])
async def generate_requirements_txt(
    request: RequirementsGenerationRequest = Body(...),
) -> Dict[str, str]:
    """
    Generate requirements.txt file(s) for the ML project.
    
    This endpoint generates minimal, modular requirements files based on
    the actual dependencies needed for the experiment.
    
    Args:
        request: Requirements generation configuration
    
    Returns:
        Dictionary mapping filename to content
    
    Raises:
        HTTPException: If requirements generation fails
    """
    try:
        logger.info(
            f"Generating requirements for {request.model_type}",
            extra={
                'event': 'requirements_generation_start',
                'model_type': request.model_type,
                'output_format': request.output_format,
                'modular': request.modular
            }
        )
        
        # Prepare configuration
        config = {
            'model_type': request.model_type,
            'task_type': request.task_type,
            'preprocessing_steps': [
                {
                    'type': step.type,
                    'parameters': step.parameters
                }
                for step in request.preprocessing_steps
            ],
            'include_evaluation': request.include_evaluation
        }
        
        # Generate requirements
        requirements = generate_requirements(
            config,
            output_format=request.output_format,
            modular=request.modular
        )
        
        logger.info(
            f"Requirements generation completed: {len(requirements)} file(s)",
            extra={
                'event': 'requirements_generation_complete',
                'files': list(requirements.keys())
            }
        )
        
        return requirements
        
    except Exception as e:
        logger.error(
            f"Requirements generation failed: {e}",
            extra={
                'event': 'requirements_generation_failed',
                'error': str(e)
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Requirements generation failed: {str(e)}"
        )


# ============================================================================
# Helper Functions
# ============================================================================

def _prepare_config(request: CodeGenerationRequest) -> Dict[str, Any]:
    """
    Prepare configuration dictionary from request.
    
    Args:
        request: Code generation request
    
    Returns:
        Configuration dictionary
    """
    config = {
        'experiment_name': request.experiment_name,
        'model_type': request.model_type,
        'task_type': request.task_type,
        'hyperparameters': request.hyperparameters,
        'test_size': request.test_size,
        'validation_size': request.validation_size,
        'random_state': request.random_state,
        'cross_validation': request.cross_validation,
        'cv_folds': request.cv_folds,
        'save_model': request.save_model,
        'model_path': request.model_path,
        'include_evaluation': request.include_evaluation,
    }
    
    # Add preprocessing steps
    if request.preprocessing_steps:
        config['preprocessing_steps'] = [
            {
                'type': step.type,
                'step_type': step.type,
                'parameters': step.parameters,
                'name': step.name or step.type.replace('_', ' ').title()
            }
            for step in request.preprocessing_steps
        ]
    else:
        config['preprocessing_steps'] = []
    
    # Add dataset info
    if request.dataset_info:
        config['dataset_info'] = {
            'file_path': request.dataset_info.file_path,
            'file_format': request.dataset_info.file_format,
        }
        if request.dataset_info.target_column:
            config['target_column'] = request.dataset_info.target_column
        if request.dataset_info.feature_columns:
            config['feature_columns'] = request.dataset_info.feature_columns
    
    # Add experiment ID if provided
    if request.experiment_id:
        config['experiment_id'] = request.experiment_id
    
    return config


def _generate_modular_code(
    config: Dict[str, Any],
    request: CodeGenerationRequest
) -> Dict[str, str]:
    """
    Generate modular code sections.
    
    Args:
        config: Configuration dictionary
        request: Code generation request
    
    Returns:
        Dictionary mapping component name to code
    """
    code_sections = {}
    
    # Generate preprocessing code
    if config.get('preprocessing_steps'):
        preprocessing_code = generate_preprocessing_code(
            config,
            output_format=request.output_format,
            include_imports=request.include_imports
        )
        code_sections['preprocessing'] = preprocessing_code
    
    # Generate training code
    training_code = generate_training_code(
        config,
        output_format=request.output_format,
        include_imports=request.include_imports,
        modular=True
    )
    code_sections['training'] = training_code
    
    # Generate evaluation code
    if request.include_evaluation:
        evaluation_code = generate_evaluation_code(
            config,
            output_format=request.output_format,
            include_imports=request.include_imports
        )
        code_sections['evaluation'] = evaluation_code
    
    # Generate prediction code
    prediction_code = generate_prediction_code(
        config,
        output_format=request.output_format,
        include_imports=request.include_imports
    )
    code_sections['prediction'] = prediction_code
    
    return code_sections


def _generate_complete_pipeline(
    config: Dict[str, Any],
    request: CodeGenerationRequest
) -> str:
    """
    Generate complete pipeline code.
    
    Args:
        config: Configuration dictionary
        request: Code generation request
    
    Returns:
        Complete pipeline code
    """
    sections = []
    
    # Header
    header = f'''"""
{config['experiment_name']} - Complete ML Pipeline

Auto-generated by AI-Playground
Generated: {datetime.utcnow().isoformat()}

Model: {config['model_type']}
Task: {config['task_type']}
"""

'''
    sections.append(header)
    
    # Generate modular sections
    code_sections = _generate_modular_code(config, request)
    
    # Combine sections
    if 'preprocessing' in code_sections:
        sections.append("# " + "=" * 78)
        sections.append("# PREPROCESSING")
        sections.append("# " + "=" * 78)
        sections.append(code_sections['preprocessing'])
        sections.append("")
    
    sections.append("# " + "=" * 78)
    sections.append("# TRAINING")
    sections.append("# " + "=" * 78)
    sections.append(code_sections['training'])
    sections.append("")
    
    if 'evaluation' in code_sections:
        sections.append("# " + "=" * 78)
        sections.append("# EVALUATION")
        sections.append("# " + "=" * 78)
        sections.append(code_sections['evaluation'])
        sections.append("")
    
    sections.append("# " + "=" * 78)
    sections.append("# PREDICTION")
    sections.append("# " + "=" * 78)
    sections.append(code_sections['prediction'])
    sections.append("")
    
    # Main execution
    main_code = '''
# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    """
    Main execution flow.
    """
    print("=" * 80)
    print(f"{config['experiment_name']}")
    print("=" * 80)
    print()
    
    # TODO: Add your data loading code here
    # df = pd.read_csv('your_data.csv')
    
    # TODO: Run preprocessing
    # df_clean = preprocess_data(df)
    
    # TODO: Split data
    # X_train, X_test, y_train, y_test = split_data(df_clean, target_column)
    
    # TODO: Train model
    # model = train_model(X_train, y_train)
    
    # TODO: Evaluate model
    # results = evaluate_model(model, X_test, y_test)
    
    # TODO: Make predictions
    # predictions = predict(model, X_new)
    
    print("\\nPipeline template generated successfully!")
    print("Uncomment and modify the TODO sections to run your pipeline.")
'''
    sections.append(main_code)
    
    return '\n'.join(sections)


def _combine_code_sections(code_sections: Dict[str, str]) -> str:
    """
    Combine modular code sections into single file.
    
    Args:
        code_sections: Dictionary of code sections
    
    Returns:
        Combined code
    """
    sections = []
    
    # Add header
    sections.append('"""')
    sections.append('Modular ML Pipeline')
    sections.append('')
    sections.append('Auto-generated by AI-Playground')
    sections.append(f'Generated: {datetime.utcnow().isoformat()}')
    sections.append('"""')
    sections.append('')
    
    # Add each section
    for name, code in code_sections.items():
        sections.append(f"# {'=' * 78}")
        sections.append(f"# {name.upper()}")
        sections.append(f"# {'=' * 78}")
        sections.append(code)
        sections.append('')
    
    return '\n'.join(sections)
