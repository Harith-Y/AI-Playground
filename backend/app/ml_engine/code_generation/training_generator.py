"""
Model Training Code Generator

Generates Python code for model training from experiment configurations.
Uses Jinja2 templates to create production-ready training pipelines.

Based on: ML-TO-DO.md > ML-64
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from app.ml_engine.code_generation.templates import get_template
from app.utils.logger import get_logger

logger = get_logger("training_generator")


class TrainingCodeGenerator:
    """
    Generator for model training code.
    
    Converts model training configurations into executable Python code.
    
    Example:
        >>> generator = TrainingCodeGenerator()
        >>> config = {
        ...     'model_type': 'random_forest_classifier',
        ...     'task_type': 'classification',
        ...     'hyperparameters': {'n_estimators': 100, 'max_depth': 10}
        ... }
        >>> code = generator.generate(config)
    """
    
    # Model import mappings
    MODEL_IMPORTS = {
        # Regression
        'linear_regression': 'from sklearn.linear_model import LinearRegression',
        'ridge': 'from sklearn.linear_model import Ridge',
        'lasso': 'from sklearn.linear_model import Lasso',
        'random_forest_regressor': 'from sklearn.ensemble import RandomForestRegressor',
        'gradient_boosting_regressor': 'from sklearn.ensemble import GradientBoostingRegressor',
        'svr': 'from sklearn.svm import SVR',
        
        # Classification
        'logistic_regression': 'from sklearn.linear_model import LogisticRegression',
        'random_forest_classifier': 'from sklearn.ensemble import RandomForestClassifier',
        'gradient_boosting_classifier': 'from sklearn.ensemble import GradientBoostingClassifier',
        'svc': 'from sklearn.svm import SVC',
        'decision_tree_classifier': 'from sklearn.tree import DecisionTreeClassifier',
        'knn_classifier': 'from sklearn.neighbors import KNeighborsClassifier',
        
        # Clustering
        'kmeans': 'from sklearn.cluster import KMeans',
        'dbscan': 'from sklearn.cluster import DBSCAN',
        'hierarchical': 'from sklearn.cluster import AgglomerativeClustering',
    }
    
    # Model class name mappings
    MODEL_CLASSES = {
        'linear_regression': 'LinearRegression',
        'ridge': 'Ridge',
        'lasso': 'Lasso',
        'random_forest_regressor': 'RandomForestRegressor',
        'gradient_boosting_regressor': 'GradientBoostingRegressor',
        'svr': 'SVR',
        'logistic_regression': 'LogisticRegression',
        'random_forest_classifier': 'RandomForestClassifier',
        'gradient_boosting_classifier': 'GradientBoostingClassifier',
        'svc': 'SVC',
        'decision_tree_classifier': 'DecisionTreeClassifier',
        'knn_classifier': 'KNeighborsClassifier',
        'kmeans': 'KMeans',
        'dbscan': 'DBSCAN',
        'hierarchical': 'AgglomerativeClustering',
    }
    
    def __init__(self):
        """Initialize training code generator."""
        logger.debug("Initialized TrainingCodeGenerator")
    
    def generate(
        self,
        training_config: Dict[str, Any],
        output_format: str = 'script',
        include_imports: bool = True
    ) -> str:
        """
        Generate model training code.
        
        Args:
            training_config: Configuration dictionary with model and training settings
            output_format: Output format ('script', 'function', 'class')
            include_imports: Whether to include import statements
        
        Returns:
            Generated Python code as string
        """
        logger.info(f"Generating training code in '{output_format}' format...")
        
        # Prepare context
        context = self._prepare_context(training_config)
        
        # Generate based on format
        if output_format == 'script':
            code = self._generate_script(context, include_imports)
        elif output_format == 'function':
            code = self._generate_function(context)
        elif output_format == 'class':
            code = self._generate_class(context)
        else:
            raise ValueError(f"Unknown output format: {output_format}")
        
        logger.info(f"Generated training code for model: {context['model_type']}")
        return code
    
    def _prepare_context(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare template context from configuration.
        
        Args:
            config: Training configuration
        
        Returns:
            Context dictionary for templates
        """
        model_type = config.get('model_type', 'linear_regression')
        task_type = config.get('task_type', 'regression')
        hyperparameters = config.get('hyperparameters', {})
        
        # Get model import and class name
        model_import = self.MODEL_IMPORTS.get(model_type, '')
        model_class = self.MODEL_CLASSES.get(model_type, 'LinearRegression')
        
        # Prepare context
        context = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': config.get('experiment_name', 'ML Experiment'),
            'model_type': model_type,
            'model_class': model_class,
            'model_import': model_import,
            'task_type': task_type,
            'hyperparameters': hyperparameters,
            'random_state': config.get('random_state', 42),
            'test_size': config.get('test_size', 0.2),
            'validation_size': config.get('validation_size', 0.2),
            'target_column': config.get('target_column', 'target'),
            'feature_columns': config.get('feature_columns', []),
            'cross_validation': config.get('cross_validation', False),
            'cv_folds': config.get('cv_folds', 5),
            'save_model': config.get('save_model', True),
            'model_path': config.get('model_path', 'model.pkl'),
        }
        
        return context
    
    def _generate_script(self, context: Dict[str, Any], include_imports: bool) -> str:
        """Generate complete training script."""
        sections = []
        
        # Imports
        if include_imports:
            imports = self._generate_imports(context)
            sections.append(imports)
        
        # Data splitting
        data_split = self._generate_data_split(context)
        sections.append(data_split)
        
        # Model training
        training = self._generate_training_code(context)
        sections.append(training)
        
        # Model saving
        if context['save_model']:
            saving = self._generate_model_saving(context)
            sections.append(saving)
        
        return '\n\n'.join(sections)
    
    def _generate_function(self, context: Dict[str, Any]) -> str:
        """Generate training function."""
        hyperparams_str = self._format_hyperparameters(context['hyperparameters'])
        
        code = f"""# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(X_train, y_train, X_val=None, y_val=None):
    \"\"\"
    Train {context['model_class']} model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
    
    Returns:
        Trained model
    \"\"\"
    print("Training {context['model_class']}...")
    
    # Initialize model
    {context['model_import'].split('import ')[-1]}
    model = {context['model_class']}(
        {hyperparams_str}
        random_state={context['random_state']}
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Training metrics
    train_score = model.score(X_train, y_train)
    print(f"Training score: {{train_score:.4f}}")
    
    # Validation metrics
    if X_val is not None and y_val is not None:
        val_score = model.score(X_val, y_val)
        print(f"Validation score: {{val_score:.4f}}")
    
    return model
"""
        return code
    
    def _generate_class(self, context: Dict[str, Any]) -> str:
        """Generate training class."""
        hyperparams_str = self._format_hyperparameters(context['hyperparameters'])
        
        code = f"""class ModelTrainer:
    \"\"\"
    Model training pipeline.
    
    Auto-generated from AI-Playground experiment.
    Generated: {context['timestamp']}
    Model: {context['model_class']}
    \"\"\"
    
    def __init__(self, random_state={context['random_state']}):
        \"\"\"Initialize model trainer.\"\"\"
        self.random_state = random_state
        self.model = None
        self.is_trained = False
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        \"\"\"
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        
        Returns:
            Self (for method chaining)
        \"\"\"
        print("Training {context['model_class']}...")
        
        # Initialize model
        {context['model_import'].split('import ')[-1]}
        self.model = {context['model_class']}(
            {hyperparams_str}
            random_state=self.random_state
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Training metrics
        train_score = self.model.score(X_train, y_train)
        print(f"Training score: {{train_score:.4f}}")
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            print(f"Validation score: {{val_score:.4f}}")
        
        return self
    
    def predict(self, X):
        \"\"\"
        Make predictions.
        
        Args:
            X: Features to predict on
        
        Returns:
            Predictions
        \"\"\"
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def save(self, filepath: str):
        \"\"\"
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        \"\"\"
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {{filepath}}")
    
    def load(self, filepath: str):
        \"\"\"
        Load trained model from file.
        
        Args:
            filepath: Path to load model from
        
        Returns:
            Self (for method chaining)
        \"\"\"
        import pickle
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {{filepath}}")
        return self
"""
        return code
    
    def _generate_imports(self, context: Dict[str, Any]) -> str:
        """Generate import statements."""
        imports = f"""# Auto-generated by AI-Playground
# Generated: {context['timestamp']}
# Experiment: {context['experiment_name']}

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
{context['model_import']}
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = {context['random_state']}
np.random.seed(RANDOM_STATE)
"""
        return imports
    
    def _generate_data_split(self, context: Dict[str, Any]) -> str:
        """Generate data splitting code."""
        code = f"""# ============================================================================
# DATA SPLITTING
# ============================================================================

def split_data(df: pd.DataFrame, target_column: str, test_size: float = {context['test_size']}):
    \"\"\"
    Split data into train and test sets.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        test_size: Proportion of data for testing
    
    Returns:
        X_train, X_test, y_train, y_test
    \"\"\"
    print(f"Splitting data (test_size={{test_size}})...")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        {'stratify=y,' if context['task_type'] == 'classification' else ''}
    )
    
    print(f"Training set: {{len(X_train)}} samples")
    print(f"Test set: {{len(X_test)}} samples")
    
    return X_train, X_test, y_train, y_test

# Assuming df is already loaded and preprocessed
# X_train, X_test, y_train, y_test = split_data(df, '{context['target_column']}')
"""
        return code
    
    def _generate_training_code(self, context: Dict[str, Any]) -> str:
        """Generate model training code."""
        hyperparams_str = self._format_hyperparameters(context['hyperparameters'])
        
        code = f"""# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(X_train, y_train):
    \"\"\"
    Train {context['model_class']} model.
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        Trained model
    \"\"\"
    print("Training {context['model_class']}...")
    print(f"Features: {{X_train.shape[1]}}")
    print(f"Samples: {{len(X_train)}}")
    
    # Initialize model
    model = {context['model_class']}(
        {hyperparams_str}
        random_state=RANDOM_STATE
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Training metrics
    train_score = model.score(X_train, y_train)
    print(f"Training score: {{train_score:.4f}}")
    
    return model

# Train the model
# model = train_model(X_train, y_train)
"""
        return code
    
    def _generate_model_saving(self, context: Dict[str, Any]) -> str:
        """Generate model saving code."""
        code = f"""# ============================================================================
# MODEL SAVING
# ============================================================================

def save_model(model, filepath: str = '{context['model_path']}'):
    \"\"\"
    Save trained model to file.
    
    Args:
        model: Trained model
        filepath: Path to save model
    \"\"\"
    print(f"Saving model to {{filepath}}...")
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved successfully!")

# Save the model
# save_model(model)
"""
        return code
    
    def _format_hyperparameters(self, hyperparameters: Dict[str, Any]) -> str:
        """Format hyperparameters for code generation."""
        if not hyperparameters:
            return ""
        
        lines = []
        for key, value in hyperparameters.items():
            if isinstance(value, str):
                lines.append(f"{key}='{value}',")
            else:
                lines.append(f"{key}={value},")
        
        return '\n        '.join(lines)


def generate_training_code(
    training_config: Dict[str, Any],
    output_format: str = 'script',
    include_imports: bool = True
) -> str:
    """
    Convenience function to generate training code.
    
    Args:
        training_config: Configuration dictionary with model and training settings
        output_format: Output format ('script', 'function', 'class')
        include_imports: Whether to include import statements
    
    Returns:
        Generated Python code as string
    
    Example:
        >>> config = {
        ...     'model_type': 'random_forest_classifier',
        ...     'task_type': 'classification',
        ...     'hyperparameters': {'n_estimators': 100}
        ... }
        >>> code = generate_training_code(config)
    """
    generator = TrainingCodeGenerator()
    return generator.generate(training_config, output_format, include_imports)
