"""
Prediction Code Generator

Generates Python code for making predictions with trained models.
Uses templates to create production-ready inference pipelines.

Based on: ML-TO-DO.md > ML-66
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from app.utils.logger import get_logger

logger = get_logger("prediction_generator")


class PredictionCodeGenerator:
    """
    Generator for prediction/inference code.
    
    Converts prediction configurations into executable Python code.
    
    Example:
        >>> generator = PredictionCodeGenerator()
        >>> config = {
        ...     'task_type': 'classification',
        ...     'model_path': 'model.pkl',
        ...     'include_preprocessing': True
        ... }
        >>> code = generator.generate(config)
    """
    
    def __init__(self):
        """Initialize prediction code generator."""
        logger.debug("Initialized PredictionCodeGenerator")
    
    def generate(
        self,
        prediction_config: Dict[str, Any],
        output_format: str = 'script',
        include_imports: bool = True
    ) -> str:
        """
        Generate prediction code.
        
        Args:
            prediction_config: Configuration dictionary with prediction settings
            output_format: Output format ('script', 'function', 'api', 'module')
            include_imports: Whether to include import statements
        
        Returns:
            Generated Python code as string
        """
        logger.info(f"Generating prediction code in '{output_format}' format...")
        
        # Prepare context
        context = self._prepare_context(prediction_config)
        
        # Generate based on format
        if output_format == 'script':
            code = self._generate_script(context, include_imports)
        elif output_format == 'function':
            code = self._generate_function(context)
        elif output_format == 'api':
            code = self._generate_api(context, include_imports)
        elif output_format == 'module':
            code = self._generate_module(context, include_imports)
        else:
            raise ValueError(f"Unknown output format: {output_format}")
        
        logger.info(f"Generated prediction code for task: {context['task_type']}")
        return code
    
    def _prepare_context(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare template context from configuration.
        
        Args:
            config: Prediction configuration
        
        Returns:
            Context dictionary for templates
        """
        task_type = config.get('task_type', 'classification')
        
        context = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': config.get('experiment_name', 'ML Prediction'),
            'task_type': task_type,
            'model_path': config.get('model_path', 'model.pkl'),
            'preprocessing_path': config.get('preprocessing_path', None),
            'include_preprocessing': config.get('include_preprocessing', False),
            'include_probabilities': config.get('include_probabilities', task_type == 'classification'),
            'batch_prediction': config.get('batch_prediction', False),
            'save_predictions': config.get('save_predictions', True),
            'output_path': config.get('output_path', 'predictions.csv'),
            'input_format': config.get('input_format', 'csv'),
            'feature_names': config.get('feature_names', []),
        }
        
        return context
    
    def _generate_script(self, context: Dict[str, Any], include_imports: bool) -> str:
        """Generate complete prediction script."""
        sections = []
        
        # Imports
        if include_imports:
            imports = self._generate_imports(context)
            sections.append(imports)
        
        # Load model function
        load_model = self._generate_load_model(context)
        sections.append(load_model)
        
        # Prediction function
        prediction = self._generate_prediction_code(context)
        sections.append(prediction)
        
        # Batch prediction
        if context['batch_prediction']:
            batch = self._generate_batch_prediction(context)
            sections.append(batch)
        
        # Save predictions
        if context['save_predictions']:
            save = self._generate_save_predictions(context)
            sections.append(save)
        
        # Main execution
        main = self._generate_main_execution(context)
        sections.append(main)
        
        return '\n\n'.join(sections)
    
    def _generate_function(self, context: Dict[str, Any]) -> str:
        """Generate prediction function."""
        return self._generate_prediction_code(context)
    
    def _generate_api(self, context: Dict[str, Any], include_imports: bool) -> str:
        """Generate FastAPI prediction endpoint."""
        sections = []
        
        # Imports
        if include_imports:
            imports = self._generate_api_imports(context)
            sections.append(imports)
        
        # API setup
        api_setup = self._generate_api_setup(context)
        sections.append(api_setup)
        
        # Prediction endpoint
        endpoint = self._generate_api_endpoint(context)
        sections.append(endpoint)
        
        # Health check endpoint
        health = self._generate_health_endpoint()
        sections.append(health)
        
        return '\n\n'.join(sections)
    
    def _generate_module(self, context: Dict[str, Any], include_imports: bool) -> str:
        """Generate modular prediction code."""
        sections = []
        
        # Module docstring
        docstring = f'''"""
Prediction Module - {context['experiment_name']}

Auto-generated by AI-Playground
Generated: {context['timestamp']}
Task Type: {context['task_type']}

This module can be imported and used in other scripts:
    from predict import load_model, predict, predict_batch
"""
'''
        sections.append(docstring)
        
        # Imports
        if include_imports:
            imports = self._generate_imports(context)
            sections.append(imports)
        
        # Configuration
        config = f"""
# Configuration
MODEL_PATH = '{context['model_path']}'
OUTPUT_PATH = '{context['output_path']}'
"""
        if context['preprocessing_path']:
            config += f"PREPROCESSING_PATH = '{context['preprocessing_path']}'\n"
        
        sections.append(config)
        
        # Load model function
        load_model = self._generate_load_model(context)
        sections.append(load_model)
        
        # Prediction function
        prediction = self._generate_prediction_code(context)
        sections.append(prediction)
        
        # Batch prediction
        if context['batch_prediction']:
            batch = self._generate_batch_prediction(context)
            sections.append(batch)
        
        # Save predictions
        if context['save_predictions']:
            save = self._generate_save_predictions(context)
            sections.append(save)
        
        # Main block
        main = '''
if __name__ == '__main__':
    """
    Example usage when run as a script.
    """
    print("=" * 80)
    print("Prediction Module - {experiment_name}")
    print("=" * 80)
    
    # Example: Load model and make predictions
    # model = load_model(MODEL_PATH)
    # predictions = predict(model, X_new)
    # save_predictions(predictions, OUTPUT_PATH)
    
    print("\\nTo use this module in another script:")
    print("  from predict import load_model, predict, predict_batch")
'''.format(experiment_name=context['experiment_name'])
        sections.append(main)
        
        return '\n\n'.join(sections)
    
    def _generate_imports(self, context: Dict[str, Any]) -> str:
        """Generate import statements."""
        imports = f"""# Auto-generated by AI-Playground
# Generated: {context['timestamp']}
# Experiment: {context['experiment_name']}

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')
"""
        
        if context['include_preprocessing']:
            imports += """
# Preprocessing imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
"""
        
        return imports
    
    def _generate_api_imports(self, context: Dict[str, Any]) -> str:
        """Generate imports for FastAPI."""
        imports = f"""# Auto-generated by AI-Playground
# Generated: {context['timestamp']}
# FastAPI Prediction Service

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
"""
        return imports
    
    def _generate_load_model(self, context: Dict[str, Any]) -> str:
        """Generate load model function."""
        code = f"""# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model(model_path: str = '{context['model_path']}'):
    \"\"\"
    Load trained model from file.
    
    Args:
        model_path: Path to the saved model file
    
    Returns:
        Loaded model
    \"\"\"
    print(f"Loading model from {{model_path}}...")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded successfully!")
    return model
"""
        
        if context['preprocessing_path']:
            code += f"""
def load_preprocessor(preprocessor_path: str = '{context['preprocessing_path']}'):
    \"\"\"
    Load preprocessing pipeline from file.
    
    Args:
        preprocessor_path: Path to the saved preprocessor file
    
    Returns:
        Loaded preprocessor
    \"\"\"
    print(f"Loading preprocessor from {{preprocessor_path}}...")
    
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    print(f"Preprocessor loaded successfully!")
    return preprocessor
"""
        
        return code
    
    def _generate_prediction_code(self, context: Dict[str, Any]) -> str:
        """Generate prediction function code."""
        task_type = context['task_type']
        
        if task_type == 'classification':
            return self._generate_classification_prediction(context)
        elif task_type == 'regression':
            return self._generate_regression_prediction(context)
        elif task_type == 'clustering':
            return self._generate_clustering_prediction(context)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _generate_classification_prediction(self, context: Dict[str, Any]) -> str:
        """Generate classification prediction code."""
        code = """# ============================================================================
# CLASSIFICATION PREDICTION
# ============================================================================

def predict(
    model,
    X: Union[np.ndarray, pd.DataFrame],
    return_probabilities: bool = True
) -> Dict[str, Any]:
    \"\"\"
    Make predictions using the trained classification model.
    
    Args:
        model: Trained model
        X: Input features (numpy array or pandas DataFrame)
        return_probabilities: Whether to return class probabilities
    
    Returns:
        Dictionary containing predictions and optionally probabilities
    \"\"\"
    print(f"Making predictions on {len(X)} samples...")
    
    # Get predictions
    predictions = model.predict(X)
    
    results = {
        'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
        'n_samples': len(X)
    }
    
    # Get probabilities if requested and available
    if return_probabilities and hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        results['probabilities'] = probabilities.tolist() if isinstance(probabilities, np.ndarray) else probabilities
        
        # Get predicted class names if available
        if hasattr(model, 'classes_'):
            results['classes'] = model.classes_.tolist()
    
    print(f"Predictions complete!")
    return results
"""
        
        if context['include_preprocessing']:
            code += """
def predict_with_preprocessing(
    model,
    preprocessor,
    X: Union[np.ndarray, pd.DataFrame],
    return_probabilities: bool = True
) -> Dict[str, Any]:
    \"\"\"
    Preprocess data and make predictions.
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        X: Raw input features
        return_probabilities: Whether to return class probabilities
    
    Returns:
        Dictionary containing predictions and optionally probabilities
    \"\"\"
    print("Preprocessing input data...")
    X_processed = preprocessor.transform(X)
    
    return predict(model, X_processed, return_probabilities)
"""
        
        return code
    
    def _generate_regression_prediction(self, context: Dict[str, Any]) -> str:
        """Generate regression prediction code."""
        code = """# ============================================================================
# REGRESSION PREDICTION
# ============================================================================

def predict(
    model,
    X: Union[np.ndarray, pd.DataFrame]
) -> Dict[str, Any]:
    \"\"\"
    Make predictions using the trained regression model.
    
    Args:
        model: Trained model
        X: Input features (numpy array or pandas DataFrame)
    
    Returns:
        Dictionary containing predictions
    \"\"\"
    print(f"Making predictions on {len(X)} samples...")
    
    # Get predictions
    predictions = model.predict(X)
    
    results = {
        'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
        'n_samples': len(X),
        'mean_prediction': float(np.mean(predictions)),
        'std_prediction': float(np.std(predictions)),
        'min_prediction': float(np.min(predictions)),
        'max_prediction': float(np.max(predictions))
    }
    
    print(f"Predictions complete!")
    return results
"""
        
        if context['include_preprocessing']:
            code += """
def predict_with_preprocessing(
    model,
    preprocessor,
    X: Union[np.ndarray, pd.DataFrame]
) -> Dict[str, Any]:
    \"\"\"
    Preprocess data and make predictions.
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        X: Raw input features
    
    Returns:
        Dictionary containing predictions
    \"\"\"
    print("Preprocessing input data...")
    X_processed = preprocessor.transform(X)
    
    return predict(model, X_processed)
"""
        
        return code
    
    def _generate_clustering_prediction(self, context: Dict[str, Any]) -> str:
        """Generate clustering prediction code."""
        code = """# ============================================================================
# CLUSTERING PREDICTION
# ============================================================================

def predict(
    model,
    X: Union[np.ndarray, pd.DataFrame]
) -> Dict[str, Any]:
    \"\"\"
    Assign cluster labels using the trained clustering model.
    
    Args:
        model: Trained clustering model
        X: Input features (numpy array or pandas DataFrame)
    
    Returns:
        Dictionary containing cluster assignments
    \"\"\"
    print(f"Assigning clusters for {len(X)} samples...")
    
    # Get cluster assignments
    if hasattr(model, 'predict'):
        labels = model.predict(X)
    elif hasattr(model, 'labels_'):
        # For models that don't have predict (like some clustering algorithms)
        labels = model.labels_
    else:
        raise ValueError("Model does not support prediction")
    
    # Get cluster statistics
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_distribution = dict(zip(unique_labels.tolist(), counts.tolist()))
    
    results = {
        'cluster_labels': labels.tolist() if isinstance(labels, np.ndarray) else labels,
        'n_samples': len(X),
        'n_clusters': len(unique_labels),
        'cluster_distribution': cluster_distribution
    }
    
    print(f"Cluster assignment complete!")
    return results
"""
        
        return code
    
    def _generate_batch_prediction(self, context: Dict[str, Any]) -> str:
        """Generate batch prediction function."""
        code = f"""# ============================================================================
# BATCH PREDICTION
# ============================================================================

def predict_batch(
    model,
    data_path: str,
    output_path: str = '{context['output_path']}',
    batch_size: int = 1000
) -> pd.DataFrame:
    \"\"\"
    Make predictions on a large dataset in batches.
    
    Args:
        model: Trained model
        data_path: Path to input data file
        output_path: Path to save predictions
        batch_size: Number of samples to process at once
    
    Returns:
        DataFrame with predictions
    \"\"\"
    print(f"Loading data from {{data_path}}...")
    
    # Load data based on format
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.json'):
        df = pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {{data_path}}")
    
    print(f"Loaded {{len(df)}} samples")
    
    # Process in batches
    all_predictions = []
    n_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in range(0, len(df), batch_size):
        batch_num = i // batch_size + 1
        print(f"Processing batch {{batch_num}}/{{n_batches}}...")
        
        batch = df.iloc[i:i+batch_size]
        results = predict(model, batch)
        all_predictions.extend(results['predictions'])
    
    # Add predictions to dataframe
    df['prediction'] = all_predictions
    
    # Save results
    print(f"Saving predictions to {{output_path}}...")
    df.to_csv(output_path, index=False)
    print(f"Predictions saved!")
    
    return df
"""
        return code
    
    def _generate_save_predictions(self, context: Dict[str, Any]) -> str:
        """Generate save predictions function."""
        code = f"""# ============================================================================
# SAVE PREDICTIONS
# ============================================================================

def save_predictions(
    predictions: Dict[str, Any],
    output_path: str = '{context['output_path']}',
    include_metadata: bool = True
) -> None:
    \"\"\"
    Save predictions to file.
    
    Args:
        predictions: Dictionary containing predictions
        output_path: Path to save predictions
        include_metadata: Whether to include metadata
    \"\"\"
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame({{'prediction': predictions['predictions']}})
    
    # Add probabilities if available
    if 'probabilities' in predictions:
        probs = predictions['probabilities']
        if 'classes' in predictions:
            classes = predictions['classes']
            for i, cls in enumerate(classes):
                df[f'prob_{{cls}}'] = [p[i] for p in probs]
        else:
            for i in range(len(probs[0])):
                df[f'prob_class_{{i}}'] = [p[i] for p in probs]
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {{output_path}}")
    
    # Save metadata if requested
    if include_metadata:
        metadata_path = output_path.replace('.csv', '_metadata.json')
        import json
        metadata = {{
            'n_samples': predictions['n_samples'],
            'timestamp': '{context['timestamp']}',
            'task_type': '{context['task_type']}'
        }}
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {{metadata_path}}")
"""
        return code
    
    def _generate_api_setup(self, context: Dict[str, Any]) -> str:
        """Generate FastAPI setup code."""
        code = f"""# ============================================================================
# FASTAPI SETUP
# ============================================================================

app = FastAPI(
    title="{context['experiment_name']} - Prediction API",
    description="Auto-generated prediction API",
    version="1.0.0"
)

# Load model at startup
MODEL_PATH = '{context['model_path']}'
model = None

@app.on_event("startup")
async def load_model_startup():
    \"\"\"Load model when API starts.\"\"\"
    global model
    print(f"Loading model from {{MODEL_PATH}}...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")


# Request/Response models
class PredictionRequest(BaseModel):
    \"\"\"Request model for predictions.\"\"\"
    features: List[List[float]] = Field(..., description="Input features")
    return_probabilities: bool = Field(default=True, description="Return class probabilities")
    
    class Config:
        schema_extra = {{
            "example": {{
                "features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "return_probabilities": True
            }}
        }}


class PredictionResponse(BaseModel):
    \"\"\"Response model for predictions.\"\"\"
    predictions: List
    n_samples: int
    probabilities: Optional[List[List[float]]] = None
    classes: Optional[List] = None
"""
        return code
    
    def _generate_api_endpoint(self, context: Dict[str, Any]) -> str:
        """Generate FastAPI prediction endpoint."""
        code = """# ============================================================================
# PREDICTION ENDPOINT
# ============================================================================

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    \"\"\"
    Make predictions on input data.
    
    Args:
        request: Prediction request with features
    
    Returns:
        Predictions and optionally probabilities
    \"\"\"
    try:
        # Convert features to numpy array
        X = np.array(request.features)
        
        # Make predictions
        predictions = model.predict(X)
        
        response = {
            'predictions': predictions.tolist(),
            'n_samples': len(X)
        }
        
        # Add probabilities if requested and available
        if request.return_probabilities and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            response['probabilities'] = probabilities.tolist()
            
            if hasattr(model, 'classes_'):
                response['classes'] = model.classes_.tolist()
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
"""
        return code
    
    def _generate_health_endpoint(self) -> str:
        """Generate health check endpoint."""
        code = """# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.get("/health")
async def health_check():
    \"\"\"
    Health check endpoint.
    
    Returns:
        Status of the API
    \"\"\"
    return {
        'status': 'healthy',
        'model_loaded': model is not None
    }


@app.get("/")
async def root():
    \"\"\"
    Root endpoint with API information.
    
    Returns:
        API information
    \"\"\"
    return {
        'message': 'Prediction API',
        'endpoints': {
            '/predict': 'POST - Make predictions',
            '/health': 'GET - Health check',
            '/docs': 'GET - API documentation'
        }
    }
"""
        return code
    
    def _generate_main_execution(self, context: Dict[str, Any]) -> str:
        """Generate main execution block."""
        code = f"""# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    \"\"\"
    Example usage of prediction functions.
    \"\"\"
    print("=" * 80)
    print("Prediction Script - {context['experiment_name']}")
    print("=" * 80)
    
    # Example: Load model and make predictions
    # model = load_model('{context['model_path']}')
    
    # Load new data
    # X_new = pd.read_csv('new_data.csv')
    
    # Make predictions
    # results = predict(model, X_new)
    
    # Save predictions
    # save_predictions(results, '{context['output_path']}')
    
"""
        
        if context['batch_prediction']:
            code += """    # Or use batch prediction for large datasets
    # df_with_predictions = predict_batch(model, 'large_dataset.csv', 'predictions.csv')
    
"""
        
        code += """    print("\\nPrediction script ready!")
    print("Uncomment the example code above to run predictions.")
"""
        
        return code


def generate_prediction_code(
    prediction_config: Dict[str, Any],
    output_format: str = 'script',
    include_imports: bool = True
) -> str:
    """
    Convenience function to generate prediction code.
    
    Args:
        prediction_config: Configuration dictionary with prediction settings
        output_format: Output format ('script', 'function', 'api', 'module')
        include_imports: Whether to include import statements
    
    Returns:
        Generated Python code as string
    
    Example:
        >>> config = {
        ...     'task_type': 'classification',
        ...     'model_path': 'model.pkl',
        ...     'include_probabilities': True
        ... }
        >>> code = generate_prediction_code(config)
    """
    generator = PredictionCodeGenerator()
    return generator.generate(prediction_config, output_format, include_imports)
