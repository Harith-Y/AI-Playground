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


@router.post("/notebook", response_model=CodeGenerationResponse)
async def generate_jupyter_notebook(
    request: CodeGenerationRequest = Body(...),
) -> CodeGenerationResponse:
    """
    Generate Jupyter Notebook (.ipynb) from experiment configuration.
    
    This endpoint generates an interactive Jupyter Notebook including:
    - Markdown cells with explanations
    - Code cells for each pipeline step
    - Visualization cells
    - Interactive exploration
    
    The generated notebook is ready to run and explore.
    
    Args:
        request: Code generation configuration
    
    Returns:
        Generated Jupyter Notebook as JSON string
    
    Raises:
        HTTPException: If notebook generation fails
    """
    try:
        logger.info(
            f"Generating Jupyter Notebook for {request.model_type}",
            extra={
                'event': 'notebook_generation_start',
                'model_type': request.model_type
            }
        )
        
        # Prepare configuration
        config = _prepare_config(request)
        
        # Generate notebook
        notebook_json = _generate_notebook(config, request)
        
        # Calculate metadata
        metadata = {
            'generated_at': datetime.utcnow().isoformat(),
            'model_type': request.model_type,
            'task_type': request.task_type,
            'format': 'jupyter_notebook',
            'cells_count': notebook_json.count('"cell_type"'),
            'experiment_name': request.experiment_name,
        }
        
        logger.info(
            f"Notebook generation completed: {metadata['cells_count']} cells",
            extra={
                'event': 'notebook_generation_complete',
                **metadata
            }
        )
        
        return CodeGenerationResponse(
            code=notebook_json,
            code_type="jupyter_notebook",
            output_format="notebook",
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(
            f"Notebook generation failed: {e}",
            extra={
                'event': 'notebook_generation_failed',
                'error': str(e)
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Notebook generation failed: {str(e)}"
        )


@router.post("/fastapi", response_model=CodeGenerationResponse)
async def generate_fastapi_service(
    request: CodeGenerationRequest = Body(...),
    include_dockerfile: bool = Query(
        default=True,
        description="Include Dockerfile for containerization"
    ),
    include_docker_compose: bool = Query(
        default=False,
        description="Include docker-compose.yml"
    )
) -> CodeGenerationResponse:
    """
    Generate FastAPI microservice for model deployment.
    
    This endpoint generates a complete FastAPI application including:
    - API endpoints for predictions
    - Model loading and caching
    - Request/response validation
    - Error handling
    - Health checks
    - API documentation
    - Dockerfile (optional)
    - docker-compose.yml (optional)
    
    The generated service is production-ready and can be deployed immediately.
    
    Args:
        request: Code generation configuration
        include_dockerfile: Whether to include Dockerfile
        include_docker_compose: Whether to include docker-compose.yml
    
    Returns:
        Generated FastAPI service code
    
    Raises:
        HTTPException: If service generation fails
    """
    try:
        logger.info(
            f"Generating FastAPI service for {request.model_type}",
            extra={
                'event': 'fastapi_generation_start',
                'model_type': request.model_type,
                'include_dockerfile': include_dockerfile,
                'include_docker_compose': include_docker_compose
            }
        )
        
        # Prepare configuration
        config = _prepare_config(request)
        
        # Generate FastAPI service
        service_code = _generate_fastapi_service(config, request)
        
        # Add Dockerfile if requested
        if include_dockerfile:
            dockerfile = _generate_dockerfile(config)
            service_code += f"\n\n# {'=' * 78}\n# DOCKERFILE\n# {'=' * 78}\n\n"
            service_code += f'"""\nSave as: Dockerfile\n\n{dockerfile}\n"""\n'
        
        # Add docker-compose if requested
        if include_docker_compose:
            docker_compose = _generate_docker_compose(config)
            service_code += f"\n\n# {'=' * 78}\n# DOCKER-COMPOSE.YML\n# {'=' * 78}\n\n"
            service_code += f'"""\nSave as: docker-compose.yml\n\n{docker_compose}\n"""\n'
        
        # Calculate metadata
        metadata = {
            'generated_at': datetime.utcnow().isoformat(),
            'model_type': request.model_type,
            'task_type': request.task_type,
            'service_type': 'fastapi_microservice',
            'lines_of_code': len(service_code.split('\n')),
            'includes_dockerfile': include_dockerfile,
            'includes_docker_compose': include_docker_compose,
            'experiment_name': request.experiment_name,
        }
        
        logger.info(
            f"FastAPI service generation completed: {metadata['lines_of_code']} lines",
            extra={
                'event': 'fastapi_generation_complete',
                **metadata
            }
        )
        
        return CodeGenerationResponse(
            code=service_code,
            code_type="fastapi_microservice",
            output_format="api",
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(
            f"FastAPI service generation failed: {e}",
            extra={
                'event': 'fastapi_generation_failed',
                'error': str(e)
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"FastAPI service generation failed: {str(e)}"
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
    
    # Step 1: Load your data
    # Example: df = pd.read_csv('your_data.csv')
    
    # Step 2: Run preprocessing
    # Example: df_clean = preprocess_data(df)
    
    # Step 3: Split data into train/test sets
    # Example: X_train, X_test, y_train, y_test = split_data(df_clean, target_column)
    
    # Step 4: Train model
    # Example: model = train_model(X_train, y_train)
    
    # Step 5: Evaluate model performance
    # Example: results = evaluate_model(model, X_test, y_test)
    
    # Step 6: Make predictions on new data
    # Example: predictions = predict(model, X_new)
    
    print("\\nPipeline template generated successfully!")
    print("Replace the example code above with your actual data paths and parameters.")
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


def _generate_notebook(
    config: Dict[str, Any],
    request: CodeGenerationRequest
) -> str:
    """
    Generate Jupyter Notebook JSON.
    
    Args:
        config: Configuration dictionary
        request: Code generation request
    
    Returns:
        Notebook JSON as string
    """
    import json
    
    cells = []
    
    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {config['experiment_name']}\n",
            "\n",
            f"**Auto-generated by AI-Playground**  \n",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "\n",
            f"- **Model**: {config['model_type']}\n",
            f"- **Task**: {config['task_type']}\n",
            f"- **Test Size**: {config['test_size']}\n",
            f"- **Random State**: {config['random_state']}\n"
        ]
    })
    
    # Imports cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Imports\n", "\n", "Import required libraries."]
    })
    
    import_code = [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import train_test_split\n",
        f"from sklearn.ensemble import RandomForestClassifier  # Adjust based on model\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# Set random seed\n",
        f"RANDOM_STATE = {config['random_state']}\n",
        "np.random.seed(RANDOM_STATE)\n",
        "\n",
        "# Configure plotting\n",
        "plt.style.use('seaborn-v0_8-darkgrid')\n",
        "sns.set_palette('husl')\n",
        "%matplotlib inline"
    ]
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": import_code
    })
    
    # Data loading cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2. Load Data\n", "\n", "Load and explore the dataset."]
    })
    
    if config.get('dataset_info'):
        data_path = config['dataset_info'].get('file_path', 'data.csv')
    else:
        data_path = 'data.csv'
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            f"# Load dataset\n",
            f"df = pd.read_csv('{data_path}')\n",
            "\n",
            "# Display basic information\n",
            "print(f'Dataset shape: {df.shape}')\n",
            "print(f'\\nColumns: {list(df.columns)}')\n",
            "print(f'\\nData types:\\n{df.dtypes}')\n",
            "\n",
            "# Display first few rows\n",
            "df.head()"
        ]
    })
    
    # EDA cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 3. Exploratory Data Analysis\n", "\n", "Explore the data distribution and relationships."]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Summary statistics\n",
            "df.describe()\n"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Check for missing values\n",
            "missing = df.isnull().sum()\n",
            "missing = missing[missing > 0].sort_values(ascending=False)\n",
            "\n",
            "if len(missing) > 0:\n",
            "    print('Missing values:')\n",
            "    print(missing)\n",
            "    \n",
            "    # Visualize missing values\n",
            "    plt.figure(figsize=(10, 6))\n",
            "    missing.plot(kind='bar')\n",
            "    plt.title('Missing Values by Column')\n",
            "    plt.ylabel('Count')\n",
            "    plt.xticks(rotation=45)\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "else:\n",
            "    print('No missing values found!')"
        ]
    })
    
    # Preprocessing cell
    if config.get('preprocessing_steps'):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Data Preprocessing\n", "\n", "Apply preprocessing steps to clean and transform the data."]
        })
        
        preprocessing_code = generate_preprocessing_code(
            config,
            output_format='function',
            include_imports=False
        )
        
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": preprocessing_code.split('\n')
        })
        
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Apply preprocessing\n",
                "df_clean = preprocess_data(df)\n",
                "print(f'Cleaned dataset shape: {df_clean.shape}')\n",
                "df_clean.head()"
            ]
        })
    
    # Feature/Target split cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 5. Prepare Features and Target\n", "\n", "Split data into features (X) and target (y)."]
    })
    
    target_col = config.get('target_column', 'target')
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            f"# Define target column\n",
            f"target_column = '{target_col}'\n",
            "\n",
            "# Separate features and target\n",
            "X = df_clean.drop(columns=[target_column])\n",
            "y = df_clean[target_column]\n",
            "\n",
            "print(f'Features shape: {X.shape}')\n",
            "print(f'Target shape: {y.shape}')\n",
            "print(f'\\nTarget distribution:\\n{y.value_counts()}')"
        ]
    })
    
    # Train/test split cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 6. Train/Test Split\n", "\n", "Split data into training and testing sets."]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            f"# Split data\n",
            f"X_train, X_test, y_train, y_test = train_test_split(\n",
            f"    X, y,\n",
            f"    test_size={config['test_size']},\n",
            f"    random_state=RANDOM_STATE,\n",
            f"    stratify=y  # For classification\n",
            f")\n",
            "\n",
            "print(f'Training set: {X_train.shape[0]} samples')\n",
            "print(f'Test set: {X_test.shape[0]} samples')"
        ]
    })
    
    # Model training cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 7. Model Training\n", "\n", f"Train {config['model_type']} model."]
    })
    
    training_code = generate_training_code(
        config,
        output_format='function',
        include_imports=False
    )
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": training_code.split('\n')
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train the model\n",
            "model = train_model(X_train, y_train)\n"
        ]
    })
    
    # Evaluation cell
    if request.include_evaluation:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 8. Model Evaluation\n", "\n", "Evaluate model performance on test set."]
        })
        
        evaluation_code = generate_evaluation_code(
            config,
            output_format='function',
            include_imports=False
        )
        
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": evaluation_code.split('\n')
        })
        
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Evaluate model\n",
                "results = evaluate_model(model, X_test, y_test)\n",
                "print('Evaluation Results:')\n",
                "for metric, value in results.items():\n",
                "    print(f'{metric}: {value:.4f}')"
            ]
        })
    
    # Prediction cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 9. Make Predictions\n", "\n", "Use the trained model to make predictions."]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Make predictions on test set\n",
            "y_pred = model.predict(X_test)\n",
            "\n",
            "# Display first few predictions\n",
            "predictions_df = pd.DataFrame({\n",
            "    'Actual': y_test.values[:10],\n",
            "    'Predicted': y_pred[:10]\n",
            "})\n",
            "predictions_df"
        ]
    })
    
    # Save model cell
    if config.get('save_model'):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 10. Save Model\n", "\n", "Save the trained model for later use."]
        })
        
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import joblib\n",
                "\n",
                f"# Save model\n",
                f"model_path = '{config.get('model_path', 'model.pkl')}'\n",
                "joblib.dump(model, model_path)\n",
                "print(f'Model saved to {model_path}')"
            ]
        })
    
    # Conclusion cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Conclusion\n",
            "\n",
            "This notebook demonstrated a complete ML pipeline:\n",
            "\n",
            "1. ✅ Data loading and exploration\n",
            "2. ✅ Data preprocessing\n",
            "3. ✅ Feature engineering\n",
            "4. ✅ Model training\n",
            "5. ✅ Model evaluation\n",
            "6. ✅ Predictions\n",
            "7. ✅ Model saving\n",
            "\n",
            "### Next Steps\n",
            "\n",
            "- Experiment with different hyperparameters\n",
            "- Try other models\n",
            "- Perform feature selection\n",
            "- Tune hyperparameters\n",
            "- Deploy the model\n",
            "\n",
            "---\n",
            "\n",
            "*Generated by AI-Playground*"
        ]
    })
    
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return json.dumps(notebook, indent=2)


def _generate_fastapi_service(
    config: Dict[str, Any],
    request: CodeGenerationRequest
) -> str:
    """
    Generate FastAPI microservice code.
    
    Args:
        config: Configuration dictionary
        request: Code generation request
    
    Returns:
        FastAPI service code
    """
    model_type = config['model_type']
    task_type = config['task_type']
    experiment_name = config['experiment_name']
    
    # Determine input/output schemas based on task type
    if task_type == 'classification':
        prediction_type = 'str'
        prediction_example = '"class_0"'
    elif task_type == 'regression':
        prediction_type = 'float'
        prediction_example = '123.45'
    else:  # clustering
        prediction_type = 'int'
        prediction_example = '0'
    
    code = f'''"""
{experiment_name} - FastAPI Microservice

Auto-generated by AI-Playground
Generated: {datetime.utcnow().isoformat()}

Model: {model_type}
Task: {task_type}

This microservice provides REST API endpoints for model predictions.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Service

```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Docker Deployment

```bash
docker build -t {experiment_name.lower().replace(" ", "-")}-api .
docker run -p 8000:8000 {experiment_name.lower().replace(" ", "-")}-api
```
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = Path("{config.get('model_path', 'model.pkl')}")
MODEL_VERSION = "1.0.0"
SERVICE_NAME = "{experiment_name}"

# ============================================================================
# Pydantic Models
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str = Field(..., description="Model version")
    timestamp: str = Field(..., description="Current timestamp")


class PredictionRequest(BaseModel):
    """Prediction request model."""
    features: List[float] = Field(
        ...,
        description="Input features for prediction",
        example=[1.0, 2.0, 3.0, 4.0]
    )
    
    @validator('features')
    def validate_features(cls, v):
        """Validate features."""
        if not v:
            raise ValueError("Features cannot be empty")
        if any(not isinstance(x, (int, float)) for x in v):
            raise ValueError("All features must be numeric")
        return v


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    instances: List[List[float]] = Field(
        ...,
        description="List of feature vectors",
        example=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )
    
    @validator('instances')
    def validate_instances(cls, v):
        """Validate instances."""
        if not v:
            raise ValueError("Instances cannot be empty")
        if not all(isinstance(inst, list) for inst in v):
            raise ValueError("Each instance must be a list")
        return v


class PredictionResponse(BaseModel):
    """Prediction response model."""
    prediction: {prediction_type} = Field(
        ...,
        description="Model prediction",
        example={prediction_example}
    )
    confidence: Optional[float] = Field(
        None,
        description="Prediction confidence (if available)",
        example=0.95
    )
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    predictions: List[{prediction_type}] = Field(
        ...,
        description="List of predictions"
    )
    confidences: Optional[List[float]] = Field(
        None,
        description="List of confidences (if available)"
    )
    count: int = Field(..., description="Number of predictions")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")


class ModelInfo(BaseModel):
    """Model information."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    type: str = Field(..., description="Model type")
    task: str = Field(..., description="Task type")
    loaded: bool = Field(..., description="Whether model is loaded")
    path: str = Field(..., description="Model file path")


# ============================================================================
# Model Management
# ============================================================================

class ModelManager:
    """Manages model loading and caching."""
    
    def __init__(self, model_path: Path):
        """Initialize model manager."""
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load model from disk."""
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {{self.model_path}}")
                raise FileNotFoundError(f"Model file not found: {{self.model_path}}")
            
            logger.info(f"Loading model from {{self.model_path}}")
            self.model = joblib.load(self.model_path)
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {{e}}")
            raise
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make prediction."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            prediction = self.model.predict(features)
            return prediction
        except Exception as e:
            logger.error(f"Prediction failed: {{e}}")
            raise
    
    def predict_proba(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Get prediction probabilities (if available)."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(features)
            return None
        except Exception as e:
            logger.error(f"Probability prediction failed: {{e}}")
            return None
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


# ============================================================================
# FastAPI Application
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title=SERVICE_NAME,
    description="ML Model Prediction API",
    version=MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
try:
    model_manager = ModelManager(MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to initialize model manager: {{e}}")
    model_manager = None


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {{
        "service": SERVICE_NAME,
        "version": MODEL_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns service health status and model information.
    """
    return HealthResponse(
        status="healthy" if model_manager and model_manager.is_loaded() else "unhealthy",
        model_loaded=model_manager.is_loaded() if model_manager else False,
        model_version=MODEL_VERSION,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Get model information.
    
    Returns details about the loaded model.
    """
    if not model_manager or not model_manager.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return ModelInfo(
        name=SERVICE_NAME,
        version=MODEL_VERSION,
        type="{model_type}",
        task="{task_type}",
        loaded=True,
        path=str(MODEL_PATH)
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a single prediction.
    
    Args:
        request: Prediction request with features
    
    Returns:
        Prediction response with result and metadata
    
    Raises:
        HTTPException: If prediction fails
    """
    if not model_manager or not model_manager.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert features to numpy array
        features = np.array([request.features])
        
        # Make prediction
        prediction = model_manager.predict(features)[0]
        
        # Get confidence if available
        confidence = None
        probabilities = model_manager.predict_proba(features)
        if probabilities is not None:
            confidence = float(np.max(probabilities[0]))
        
        # Convert prediction to appropriate type
        if "{task_type}" == "classification":
            prediction = str(prediction)
        elif "{task_type}" == "regression":
            prediction = float(prediction)
        else:  # clustering
            prediction = int(prediction)
        
        logger.info(f"Prediction made: {{prediction}}")
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_version=MODEL_VERSION,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {{e}}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {{str(e)}}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions.
    
    Args:
        request: Batch prediction request with multiple instances
    
    Returns:
        Batch prediction response with results
    
    Raises:
        HTTPException: If prediction fails
    """
    if not model_manager or not model_manager.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert instances to numpy array
        features = np.array(request.instances)
        
        # Make predictions
        predictions = model_manager.predict(features)
        
        # Get confidences if available
        confidences = None
        probabilities = model_manager.predict_proba(features)
        if probabilities is not None:
            confidences = [float(np.max(prob)) for prob in probabilities]
        
        # Convert predictions to appropriate type
        if "{task_type}" == "classification":
            predictions = [str(p) for p in predictions]
        elif "{task_type}" == "regression":
            predictions = [float(p) for p in predictions]
        else:  # clustering
            predictions = [int(p) for p in predictions]
        
        logger.info(f"Batch prediction made: {{len(predictions)}} predictions")
        
        return BatchPredictionResponse(
            predictions=predictions,
            confidences=confidences,
            count=len(predictions),
            model_version=MODEL_VERSION,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {{e}}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {{str(e)}}"
        )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
'''
    
    return code


def _generate_dockerfile(config: Dict[str, Any]) -> str:
    """
    Generate Dockerfile for FastAPI service.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Dockerfile content
    """
    experiment_name = config['experiment_name'].lower().replace(' ', '-')
    
    dockerfile = f'''# FastAPI Microservice Dockerfile
# Auto-generated by AI-Playground

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY model.pkl .

# Create non-root user
RUN useradd -m -u 1000 appuser && \\
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    return dockerfile


def _generate_docker_compose(config: Dict[str, Any]) -> str:
    """
    Generate docker-compose.yml for FastAPI service.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        docker-compose.yml content
    """
    experiment_name = config['experiment_name'].lower().replace(' ', '-')
    
    docker_compose = f'''# Docker Compose Configuration
# Auto-generated by AI-Playground

version: '3.8'

services:
  api:
    build: .
    container_name: {experiment_name}-api
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/model.pkl
      - LOG_LEVEL=info
    volumes:
      - ./model.pkl:/app/model.pkl:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - ml-network

networks:
  ml-network:
    driver: bridge
'''
    
    return docker_compose
