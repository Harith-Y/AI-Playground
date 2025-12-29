"""
Parameter validation for model configurations.

This module provides comprehensive validation for model hyperparameters,
ensuring that configurations are valid before model creation and training.
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Enumeration of parameter types."""
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"
    LIST = "list"
    TUPLE = "tuple"
    DICT = "dict"
    CALLABLE = "callable"
    NONE = "none"


class ValidationError(Exception):
    """Exception raised when parameter validation fails."""
    pass


class ParameterSpec:
    """
    Specification for a single parameter.
    
    Defines the expected type, valid range, allowed values,
    and other constraints for a model hyperparameter.
    """
    
    def __init__(
        self,
        name: str,
        param_type: Union[ParameterType, List[ParameterType]],
        required: bool = False,
        default: Any = None,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        allowed_values: Optional[List[Any]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        custom_validator: Optional[Callable[[Any], bool]] = None,
        description: Optional[str] = None
    ):
        """
        Initialize parameter specification.
        
        Args:
            name: Parameter name
            param_type: Expected type(s) for the parameter
            required: Whether parameter is required
            default: Default value if not provided
            min_value: Minimum allowed value (for numeric types)
            max_value: Maximum allowed value (for numeric types)
            allowed_values: List of allowed values
            min_length: Minimum length (for lists/tuples/strings)
            max_length: Maximum length (for lists/tuples/strings)
            custom_validator: Custom validation function
            description: Parameter description
        """
        self.name = name
        self.param_type = param_type if isinstance(param_type, list) else [param_type]
        self.required = required
        self.default = default
        self.min_value = min_value
        self.max_value = max_value
        self.allowed_values = allowed_values
        self.min_length = min_length
        self.max_length = max_length
        self.custom_validator = custom_validator
        self.description = description
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a parameter value.
        
        Args:
            value: Value to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if None is allowed
        if value is None:
            if ParameterType.NONE in self.param_type:
                return True, None
            if not self.required and self.default is None:
                return True, None
            return False, f"Parameter '{self.name}' cannot be None"
        
        # Check type
        type_valid = False
        for ptype in self.param_type:
            if self._check_type(value, ptype):
                type_valid = True
                break
        
        if not type_valid:
            expected_types = ", ".join([pt.value for pt in self.param_type])
            return False, f"Parameter '{self.name}' must be of type {expected_types}, got {type(value).__name__}"
        
        # Check allowed values
        if self.allowed_values is not None and value not in self.allowed_values:
            return False, f"Parameter '{self.name}' must be one of {self.allowed_values}, got {value}"
        
        # Check numeric range
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                return False, f"Parameter '{self.name}' must be >= {self.min_value}, got {value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Parameter '{self.name}' must be <= {self.max_value}, got {value}"
        
        # Check length
        if hasattr(value, '__len__') and not isinstance(value, (int, float, bool)):
            length = len(value)
            if self.min_length is not None and length < self.min_length:
                return False, f"Parameter '{self.name}' must have length >= {self.min_length}, got {length}"
            if self.max_length is not None and length > self.max_length:
                return False, f"Parameter '{self.name}' must have length <= {self.max_length}, got {length}"
        
        # Custom validation
        if self.custom_validator is not None:
            try:
                if not self.custom_validator(value):
                    return False, f"Parameter '{self.name}' failed custom validation"
            except Exception as e:
                return False, f"Parameter '{self.name}' custom validation error: {str(e)}"
        
        return True, None
    
    def _check_type(self, value: Any, param_type: ParameterType) -> bool:
        """Check if value matches the parameter type."""
        if param_type == ParameterType.INT:
            return isinstance(value, int) and not isinstance(value, bool)
        elif param_type == ParameterType.FLOAT:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif param_type == ParameterType.BOOL:
            return isinstance(value, bool)
        elif param_type == ParameterType.STRING:
            return isinstance(value, str)
        elif param_type == ParameterType.LIST:
            return isinstance(value, list)
        elif param_type == ParameterType.TUPLE:
            return isinstance(value, tuple)
        elif param_type == ParameterType.DICT:
            return isinstance(value, dict)
        elif param_type == ParameterType.CALLABLE:
            return callable(value)
        elif param_type == ParameterType.NONE:
            return value is None
        return False


class ModelParameterSchema:
    """
    Schema defining all parameters for a model.
    
    Contains parameter specifications and validation logic
    for a specific model type.
    """
    
    def __init__(self, model_id: str, parameters: List[ParameterSpec]):
        """
        Initialize model parameter schema.
        
        Args:
            model_id: Model identifier
            parameters: List of parameter specifications
        """
        self.model_id = model_id
        self.parameters = {param.name: param for param in parameters}
    
    def validate(self, hyperparameters: Dict[str, Any], strict: bool = False) -> Tuple[bool, List[str]]:
        """
        Validate hyperparameters against schema.
        
        Args:
            hyperparameters: Dictionary of hyperparameters to validate
            strict: If True, reject unknown parameters
        
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        
        # Check required parameters
        for param_name, param_spec in self.parameters.items():
            if param_spec.required and param_name not in hyperparameters:
                errors.append(f"Required parameter '{param_name}' is missing")
        
        # Validate provided parameters
        for param_name, value in hyperparameters.items():
            if param_name not in self.parameters:
                if strict:
                    errors.append(f"Unknown parameter '{param_name}' for model '{self.model_id}'")
                else:
                    logger.warning(f"Unknown parameter '{param_name}' for model '{self.model_id}' (will be passed through)")
                continue
            
            param_spec = self.parameters[param_name]
            is_valid, error_msg = param_spec.validate(value)
            if not is_valid:
                errors.append(error_msg)
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default values for all parameters."""
        return {
            name: spec.default
            for name, spec in self.parameters.items()
            if spec.default is not None
        }
    
    def get_parameter_info(self, param_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific parameter."""
        if param_name not in self.parameters:
            return None
        
        spec = self.parameters[param_name]
        return {
            "name": spec.name,
            "type": [pt.value for pt in spec.param_type],
            "required": spec.required,
            "default": spec.default,
            "min_value": spec.min_value,
            "max_value": spec.max_value,
            "allowed_values": spec.allowed_values,
            "description": spec.description
        }


# Parameter schemas for each model type


# Common parameter specifications
RANDOM_STATE_SPEC = ParameterSpec(
    "random_state",
    [ParameterType.INT, ParameterType.NONE],
    required=False,
    default=None,
    min_value=0,
    description="Random seed for reproducibility"
)

N_JOBS_SPEC = ParameterSpec(
    "n_jobs",
    [ParameterType.INT, ParameterType.NONE],
    required=False,
    default=None,
    min_value=-1,
    description="Number of parallel jobs (-1 for all cores)"
)

VERBOSE_SPEC = ParameterSpec(
    "verbose",
    [ParameterType.INT, ParameterType.BOOL],
    required=False,
    default=0,
    min_value=0,
    description="Verbosity level"
)


# Classification model schemas

LOGISTIC_REGRESSION_SCHEMA = ModelParameterSchema(
    "logistic_regression",
    [
        ParameterSpec("penalty", ParameterType.STRING, default='l2',
                     allowed_values=['l1', 'l2', 'elasticnet', 'none'],
                     description="Regularization penalty"),
        ParameterSpec("C", ParameterType.FLOAT, default=1.0, min_value=0.0,
                     description="Inverse of regularization strength"),
        ParameterSpec("solver", ParameterType.STRING, default='lbfgs',
                     allowed_values=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                     description="Optimization algorithm"),
        ParameterSpec("max_iter", ParameterType.INT, default=100, min_value=1,
                     description="Maximum number of iterations"),
        ParameterSpec("tol", ParameterType.FLOAT, default=1e-4, min_value=0.0,
                     description="Tolerance for stopping criteria"),
        ParameterSpec("fit_intercept", ParameterType.BOOL, default=True,
                     description="Whether to fit intercept"),
        ParameterSpec("class_weight", [ParameterType.STRING, ParameterType.DICT, ParameterType.NONE],
                     default=None, allowed_values=['balanced', None],
                     description="Weights for classes"),
        RANDOM_STATE_SPEC,
        N_JOBS_SPEC,
        VERBOSE_SPEC
    ]
)

RANDOM_FOREST_CLASSIFIER_SCHEMA = ModelParameterSchema(
    "random_forest_classifier",
    [
        ParameterSpec("n_estimators", ParameterType.INT, default=100, min_value=1,
                     description="Number of trees in the forest"),
        ParameterSpec("criterion", ParameterType.STRING, default='gini',
                     allowed_values=['gini', 'entropy', 'log_loss'],
                     description="Function to measure split quality"),
        ParameterSpec("max_depth", [ParameterType.INT, ParameterType.NONE], default=None, min_value=1,
                     description="Maximum depth of trees"),
        ParameterSpec("min_samples_split", [ParameterType.INT, ParameterType.FLOAT], default=2, min_value=2,
                     description="Minimum samples required to split node"),
        ParameterSpec("min_samples_leaf", [ParameterType.INT, ParameterType.FLOAT], default=1, min_value=1,
                     description="Minimum samples required at leaf node"),
        ParameterSpec("max_features", [ParameterType.STRING, ParameterType.INT, ParameterType.FLOAT, ParameterType.NONE],
                     default='sqrt', allowed_values=['sqrt', 'log2', None],
                     description="Number of features to consider for best split"),
        ParameterSpec("bootstrap", ParameterType.BOOL, default=True,
                     description="Whether to use bootstrap samples"),
        ParameterSpec("oob_score", ParameterType.BOOL, default=False,
                     description="Whether to use out-of-bag samples for scoring"),
        ParameterSpec("class_weight", [ParameterType.STRING, ParameterType.DICT, ParameterType.NONE],
                     default=None, allowed_values=['balanced', 'balanced_subsample', None],
                     description="Weights for classes"),
        RANDOM_STATE_SPEC,
        N_JOBS_SPEC,
        VERBOSE_SPEC
    ]
)

SVM_CLASSIFIER_SCHEMA = ModelParameterSchema(
    "svm_classifier",
    [
        ParameterSpec("C", ParameterType.FLOAT, default=1.0, min_value=0.0,
                     description="Regularization parameter"),
        ParameterSpec("kernel", ParameterType.STRING, default='rbf',
                     allowed_values=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                     description="Kernel type"),
        ParameterSpec("degree", ParameterType.INT, default=3, min_value=0,
                     description="Degree for poly kernel"),
        ParameterSpec("gamma", [ParameterType.STRING, ParameterType.FLOAT], default='scale',
                     allowed_values=['scale', 'auto'],
                     description="Kernel coefficient"),
        ParameterSpec("coef0", ParameterType.FLOAT, default=0.0,
                     description="Independent term in kernel function"),
        ParameterSpec("shrinking", ParameterType.BOOL, default=True,
                     description="Whether to use shrinking heuristic"),
        ParameterSpec("probability", ParameterType.BOOL, default=False,
                     description="Whether to enable probability estimates"),
        ParameterSpec("tol", ParameterType.FLOAT, default=1e-3, min_value=0.0,
                     description="Tolerance for stopping criterion"),
        ParameterSpec("max_iter", ParameterType.INT, default=-1, min_value=-1,
                     description="Maximum number of iterations (-1 for no limit)"),
        ParameterSpec("class_weight", [ParameterType.STRING, ParameterType.DICT, ParameterType.NONE],
                     default=None, allowed_values=['balanced', None],
                     description="Weights for classes"),
        RANDOM_STATE_SPEC,
        VERBOSE_SPEC
    ]
)

GRADIENT_BOOSTING_CLASSIFIER_SCHEMA = ModelParameterSchema(
    "gradient_boosting_classifier",
    [
        ParameterSpec("n_estimators", ParameterType.INT, default=100, min_value=1,
                     description="Number of boosting stages"),
        ParameterSpec("learning_rate", ParameterType.FLOAT, default=0.1, min_value=0.0, max_value=1.0,
                     description="Learning rate shrinks contribution of each tree"),
        ParameterSpec("max_depth", ParameterType.INT, default=3, min_value=1,
                     description="Maximum depth of trees"),
        ParameterSpec("min_samples_split", [ParameterType.INT, ParameterType.FLOAT], default=2, min_value=2,
                     description="Minimum samples required to split node"),
        ParameterSpec("min_samples_leaf", [ParameterType.INT, ParameterType.FLOAT], default=1, min_value=1,
                     description="Minimum samples required at leaf node"),
        ParameterSpec("subsample", ParameterType.FLOAT, default=1.0, min_value=0.0, max_value=1.0,
                     description="Fraction of samples for fitting base learners"),
        ParameterSpec("loss", ParameterType.STRING, default='log_loss',
                     allowed_values=['log_loss', 'deviance', 'exponential'],
                     description="Loss function to optimize"),
        RANDOM_STATE_SPEC,
        VERBOSE_SPEC
    ]
)

KNN_CLASSIFIER_SCHEMA = ModelParameterSchema(
    "knn_classifier",
    [
        ParameterSpec("n_neighbors", ParameterType.INT, default=5, min_value=1,
                     description="Number of neighbors"),
        ParameterSpec("weights", ParameterType.STRING, default='uniform',
                     allowed_values=['uniform', 'distance'],
                     description="Weight function for prediction"),
        ParameterSpec("algorithm", ParameterType.STRING, default='auto',
                     allowed_values=['auto', 'ball_tree', 'kd_tree', 'brute'],
                     description="Algorithm to compute nearest neighbors"),
        ParameterSpec("leaf_size", ParameterType.INT, default=30, min_value=1,
                     description="Leaf size for tree algorithms"),
        ParameterSpec("p", ParameterType.INT, default=2, min_value=1,
                     description="Power parameter for Minkowski metric"),
        ParameterSpec("metric", ParameterType.STRING, default='minkowski',
                     description="Distance metric"),
        N_JOBS_SPEC
    ]
)


# Regression model schemas

LINEAR_REGRESSION_SCHEMA = ModelParameterSchema(
    "linear_regression",
    [
        ParameterSpec("fit_intercept", ParameterType.BOOL, default=True,
                     description="Whether to fit intercept"),
        ParameterSpec("copy_X", ParameterType.BOOL, default=True,
                     description="Whether to copy X"),
        N_JOBS_SPEC
    ]
)

RIDGE_REGRESSION_SCHEMA = ModelParameterSchema(
    "ridge_regression",
    [
        ParameterSpec("alpha", ParameterType.FLOAT, default=1.0, min_value=0.0,
                     description="Regularization strength"),
        ParameterSpec("fit_intercept", ParameterType.BOOL, default=True,
                     description="Whether to fit intercept"),
        ParameterSpec("copy_X", ParameterType.BOOL, default=True,
                     description="Whether to copy X"),
        ParameterSpec("max_iter", [ParameterType.INT, ParameterType.NONE], default=None, min_value=1,
                     description="Maximum number of iterations"),
        ParameterSpec("tol", ParameterType.FLOAT, default=1e-4, min_value=0.0,
                     description="Tolerance for stopping criteria"),
        ParameterSpec("solver", ParameterType.STRING, default='auto',
                     allowed_values=['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
                     description="Solver to use"),
        RANDOM_STATE_SPEC
    ]
)

LASSO_REGRESSION_SCHEMA = ModelParameterSchema(
    "lasso_regression",
    [
        ParameterSpec("alpha", ParameterType.FLOAT, default=1.0, min_value=0.0,
                     description="Regularization strength"),
        ParameterSpec("fit_intercept", ParameterType.BOOL, default=True,
                     description="Whether to fit intercept"),
        ParameterSpec("max_iter", ParameterType.INT, default=1000, min_value=1,
                     description="Maximum number of iterations"),
        ParameterSpec("tol", ParameterType.FLOAT, default=1e-4, min_value=0.0,
                     description="Tolerance for stopping criteria"),
        ParameterSpec("selection", ParameterType.STRING, default='cyclic',
                     allowed_values=['cyclic', 'random'],
                     description="Feature selection method"),
        RANDOM_STATE_SPEC
    ]
)

RANDOM_FOREST_REGRESSOR_SCHEMA = ModelParameterSchema(
    "random_forest_regressor",
    [
        ParameterSpec("n_estimators", ParameterType.INT, default=100, min_value=1,
                     description="Number of trees in the forest"),
        ParameterSpec("criterion", ParameterType.STRING, default='squared_error',
                     allowed_values=['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                     description="Function to measure split quality"),
        ParameterSpec("max_depth", [ParameterType.INT, ParameterType.NONE], default=None, min_value=1,
                     description="Maximum depth of trees"),
        ParameterSpec("min_samples_split", [ParameterType.INT, ParameterType.FLOAT], default=2, min_value=2,
                     description="Minimum samples required to split node"),
        ParameterSpec("min_samples_leaf", [ParameterType.INT, ParameterType.FLOAT], default=1, min_value=1,
                     description="Minimum samples required at leaf node"),
        ParameterSpec("max_features", [ParameterType.STRING, ParameterType.INT, ParameterType.FLOAT, ParameterType.NONE],
                     default=1.0, allowed_values=['sqrt', 'log2', None],
                     description="Number of features to consider for best split"),
        ParameterSpec("bootstrap", ParameterType.BOOL, default=True,
                     description="Whether to use bootstrap samples"),
        ParameterSpec("oob_score", ParameterType.BOOL, default=False,
                     description="Whether to use out-of-bag samples for scoring"),
        RANDOM_STATE_SPEC,
        N_JOBS_SPEC,
        VERBOSE_SPEC
    ]
)


# Clustering model schemas

KMEANS_SCHEMA = ModelParameterSchema(
    "kmeans",
    [
        ParameterSpec("n_clusters", ParameterType.INT, default=8, min_value=1,
                     description="Number of clusters"),
        ParameterSpec("init", ParameterType.STRING, default='k-means++',
                     allowed_values=['k-means++', 'random'],
                     description="Initialization method"),
        ParameterSpec("n_init", [ParameterType.INT, ParameterType.STRING], default='auto', min_value=1,
                     allowed_values=['auto'],
                     description="Number of times algorithm runs with different seeds"),
        ParameterSpec("max_iter", ParameterType.INT, default=300, min_value=1,
                     description="Maximum number of iterations"),
        ParameterSpec("tol", ParameterType.FLOAT, default=1e-4, min_value=0.0,
                     description="Tolerance for convergence"),
        ParameterSpec("algorithm", ParameterType.STRING, default='lloyd',
                     allowed_values=['lloyd', 'elkan'],
                     description="K-means algorithm variant"),
        RANDOM_STATE_SPEC,
        VERBOSE_SPEC
    ]
)

DBSCAN_SCHEMA = ModelParameterSchema(
    "dbscan",
    [
        ParameterSpec("eps", ParameterType.FLOAT, default=0.5, min_value=0.0,
                     description="Maximum distance between samples"),
        ParameterSpec("min_samples", ParameterType.INT, default=5, min_value=1,
                     description="Minimum samples in neighborhood"),
        ParameterSpec("metric", ParameterType.STRING, default='euclidean',
                     description="Distance metric"),
        ParameterSpec("algorithm", ParameterType.STRING, default='auto',
                     allowed_values=['auto', 'ball_tree', 'kd_tree', 'brute'],
                     description="Algorithm to compute nearest neighbors"),
        ParameterSpec("leaf_size", ParameterType.INT, default=30, min_value=1,
                     description="Leaf size for tree algorithms"),
        N_JOBS_SPEC
    ]
)

AGGLOMERATIVE_CLUSTERING_SCHEMA = ModelParameterSchema(
    "agglomerative_clustering",
    [
        ParameterSpec("n_clusters", [ParameterType.INT, ParameterType.NONE], default=2, min_value=1,
                     description="Number of clusters"),
        ParameterSpec("metric", ParameterType.STRING, default='euclidean',
                     description="Distance metric"),
        ParameterSpec("linkage", ParameterType.STRING, default='ward',
                     allowed_values=['ward', 'complete', 'average', 'single'],
                     description="Linkage criterion"),
        ParameterSpec("distance_threshold", [ParameterType.FLOAT, ParameterType.NONE], default=None, min_value=0.0,
                     description="Linkage distance threshold")
    ]
)

GAUSSIAN_MIXTURE_SCHEMA = ModelParameterSchema(
    "gaussian_mixture",
    [
        ParameterSpec("n_components", ParameterType.INT, default=1, min_value=1,
                     description="Number of mixture components"),
        ParameterSpec("covariance_type", ParameterType.STRING, default='full',
                     allowed_values=['full', 'tied', 'diag', 'spherical'],
                     description="Type of covariance parameters"),
        ParameterSpec("tol", ParameterType.FLOAT, default=1e-3, min_value=0.0,
                     description="Convergence threshold"),
        ParameterSpec("max_iter", ParameterType.INT, default=100, min_value=1,
                     description="Maximum number of EM iterations"),
        ParameterSpec("n_init", ParameterType.INT, default=1, min_value=1,
                     description="Number of initializations"),
        ParameterSpec("init_params", ParameterType.STRING, default='kmeans',
                     allowed_values=['kmeans', 'k-means++', 'random', 'random_from_data'],
                     description="Initialization method"),
        RANDOM_STATE_SPEC,
        VERBOSE_SPEC
    ]
)


# Registry of all model schemas
MODEL_SCHEMAS: Dict[str, ModelParameterSchema] = {
    # Classification
    "logistic_regression": LOGISTIC_REGRESSION_SCHEMA,
    "random_forest_classifier": RANDOM_FOREST_CLASSIFIER_SCHEMA,
    "svm_classifier": SVM_CLASSIFIER_SCHEMA,
    "gradient_boosting_classifier": GRADIENT_BOOSTING_CLASSIFIER_SCHEMA,
    "knn_classifier": KNN_CLASSIFIER_SCHEMA,
    
    # Regression
    "linear_regression": LINEAR_REGRESSION_SCHEMA,
    "ridge_regression": RIDGE_REGRESSION_SCHEMA,
    "lasso_regression": LASSO_REGRESSION_SCHEMA,
    "random_forest_regressor": RANDOM_FOREST_REGRESSOR_SCHEMA,
    
    # Clustering
    "kmeans": KMEANS_SCHEMA,
    "dbscan": DBSCAN_SCHEMA,
    "agglomerative_clustering": AGGLOMERATIVE_CLUSTERING_SCHEMA,
    "gaussian_mixture": GAUSSIAN_MIXTURE_SCHEMA,
}


def validate_model_config(model_id: str, hyperparameters: Dict[str, Any], strict: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate model configuration.
    
    Args:
        model_id: Model identifier
        hyperparameters: Dictionary of hyperparameters
        strict: If True, reject unknown parameters
    
    Returns:
        Tuple of (is_valid, list_of_error_messages)
    
    Example:
        >>> is_valid, errors = validate_model_config(
        ...     'random_forest_classifier',
        ...     {'n_estimators': 100, 'max_depth': 10}
        ... )
        >>> if not is_valid:
        ...     print("Validation errors:", errors)
    """
    if model_id not in MODEL_SCHEMAS:
        logger.warning(f"No validation schema found for model '{model_id}' (validation skipped)")
        return True, []
    
    schema = MODEL_SCHEMAS[model_id]
    return schema.validate(hyperparameters, strict=strict)


def get_model_defaults(model_id: str) -> Dict[str, Any]:
    """
    Get default hyperparameters for a model.
    
    Args:
        model_id: Model identifier
    
    Returns:
        Dictionary of default hyperparameters
    
    Example:
        >>> defaults = get_model_defaults('random_forest_classifier')
        >>> print(defaults)
        {'n_estimators': 100, 'criterion': 'gini', ...}
    """
    if model_id not in MODEL_SCHEMAS:
        return {}
    
    schema = MODEL_SCHEMAS[model_id]
    return schema.get_defaults()


def get_parameter_info(model_id: str, param_name: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific parameter.
    
    Args:
        model_id: Model identifier
        param_name: Parameter name
    
    Returns:
        Dictionary with parameter information, or None if not found
    
    Example:
        >>> info = get_parameter_info('random_forest_classifier', 'n_estimators')
        >>> print(info['description'])
        'Number of trees in the forest'
    """
    if model_id not in MODEL_SCHEMAS:
        return None
    
    schema = MODEL_SCHEMAS[model_id]
    return schema.get_parameter_info(param_name)


def get_available_models_with_schemas() -> List[str]:
    """
    Get list of models that have validation schemas.
    
    Returns:
        List of model IDs with schemas
    """
    return list(MODEL_SCHEMAS.keys())
