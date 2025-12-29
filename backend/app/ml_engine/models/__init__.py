"""
ML Models package.

This package provides model wrappers, base classes, and factory
for creating and managing ML models.
"""

from .base import (
    BaseModelWrapper,
    SupervisedModelWrapper,
    UnsupervisedModelWrapper,
    ModelConfig,
    TrainingMetadata
)
from .registry import ModelFactory, create_model
from .validation import (
    validate_model_config,
    get_model_defaults,
    get_parameter_info,
    get_available_models_with_schemas,
    ValidationError
)

# Import all model wrappers for convenience
from .regression import (
    LinearRegressionWrapper,
    RidgeRegressionWrapper,
    LassoRegressionWrapper,
    ElasticNetWrapper,
    DecisionTreeRegressorWrapper,
    RandomForestRegressorWrapper,
    ExtraTreesRegressorWrapper,
    GradientBoostingRegressorWrapper,
    AdaBoostRegressorWrapper,
    SVRWrapper,
    KNeighborsRegressorWrapper
)

from .classification import (
    LogisticRegressionWrapper,
    DecisionTreeClassifierWrapper,
    RandomForestClassifierWrapper,
    ExtraTreesClassifierWrapper,
    GradientBoostingClassifierWrapper,
    AdaBoostClassifierWrapper,
    SVMClassifierWrapper,
    KNeighborsClassifierWrapper,
    GaussianNBWrapper
)

from .clustering import (
    KMeansWrapper,
    DBSCANWrapper,
    AgglomerativeClusteringWrapper,
    GaussianMixtureWrapper
)

__all__ = [
    # Base classes
    "BaseModelWrapper",
    "SupervisedModelWrapper",
    "UnsupervisedModelWrapper",
    "ModelConfig",
    "TrainingMetadata",

    # Factory
    "ModelFactory",
    "create_model",
    
    # Validation
    "validate_model_config",
    "get_model_defaults",
    "get_parameter_info",
    "get_available_models_with_schemas",
    "ValidationError",

    # Regression wrappers
    "LinearRegressionWrapper",
    "RidgeRegressionWrapper",
    "LassoRegressionWrapper",
    "ElasticNetWrapper",
    "DecisionTreeRegressorWrapper",
    "RandomForestRegressorWrapper",
    "ExtraTreesRegressorWrapper",
    "GradientBoostingRegressorWrapper",
    "AdaBoostRegressorWrapper",
    "SVRWrapper",
    "KNeighborsRegressorWrapper",

    # Classification wrappers
    "LogisticRegressionWrapper",
    "DecisionTreeClassifierWrapper",
    "RandomForestClassifierWrapper",
    "ExtraTreesClassifierWrapper",
    "GradientBoostingClassifierWrapper",
    "AdaBoostClassifierWrapper",
    "SVMClassifierWrapper",
    "KNeighborsClassifierWrapper",
    "GaussianNBWrapper",

    # Clustering wrappers
    "KMeansWrapper",
    "DBSCANWrapper",
    "AgglomerativeClusteringWrapper",
    "GaussianMixtureWrapper",
]
