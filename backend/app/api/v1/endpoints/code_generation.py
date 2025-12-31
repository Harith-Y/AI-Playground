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
