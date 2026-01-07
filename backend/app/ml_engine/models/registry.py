"""
Model factory and registry.

This module provides a factory pattern for creating model wrappers
and integrating with the model registry to instantiate the correct
model wrapper based on model ID.
"""

from typing import Dict, Type, Optional
from .base import BaseModelWrapper, ModelConfig
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


# Mapping from model_id to wrapper class
MODEL_WRAPPER_REGISTRY: Dict[str, Type[BaseModelWrapper]] = {
    # Regression models
    "linear_regression": LinearRegressionWrapper,
    "ridge_regression": RidgeRegressionWrapper,
    "lasso_regression": LassoRegressionWrapper,
    "elastic_net": ElasticNetWrapper,
    "decision_tree_regressor": DecisionTreeRegressorWrapper,
    "random_forest_regressor": RandomForestRegressorWrapper,
    "extra_trees_regressor": ExtraTreesRegressorWrapper,
    "gradient_boosting_regressor": GradientBoostingRegressorWrapper,
    "adaboost_regressor": AdaBoostRegressorWrapper,
    "svr": SVRWrapper,
    "knn_regressor": KNeighborsRegressorWrapper,

    # Classification models
    "logistic_regression": LogisticRegressionWrapper,
    "decision_tree_classifier": DecisionTreeClassifierWrapper,
    "random_forest_classifier": RandomForestClassifierWrapper,
    "extra_trees_classifier": ExtraTreesClassifierWrapper,
    "gradient_boosting_classifier": GradientBoostingClassifierWrapper,
    "adaboost_classifier": AdaBoostClassifierWrapper,
    "svm_classifier": SVMClassifierWrapper,
    "knn_classifier": KNeighborsClassifierWrapper,
    "gaussian_nb": GaussianNBWrapper,

    # Clustering models
    "kmeans": KMeansWrapper,
    "dbscan": DBSCANWrapper,
    "agglomerative_clustering": AgglomerativeClusteringWrapper,
    "gaussian_mixture": GaussianMixtureWrapper,
}


class ModelFactory:
    """
    Factory for creating model wrapper instances.

    This factory integrates with the model registry to instantiate
    the correct model wrapper class based on the model ID and
    configuration.
    """

    @staticmethod
    def create_model(
        model_id: str,
        config: Optional[ModelConfig] = None,
        validate: bool = True,
        strict: bool = False,
        **hyperparameters
    ) -> BaseModelWrapper:
        """
        Create a model wrapper instance.

        Args:
            model_id: Identifier for the model (e.g., 'random_forest_classifier')
            config: Optional ModelConfig object. If not provided, will be created from hyperparameters
            validate: Whether to validate hyperparameters
            strict: If True, reject unknown parameters during validation
            **hyperparameters: Hyperparameters to pass to the model (used if config is None)

        Returns:
            Instance of the appropriate model wrapper

        Raises:
            ValueError: If model_id is not found in the registry or validation fails

        Examples:
            # Create with hyperparameters
            model = ModelFactory.create_model('random_forest_classifier', n_estimators=100, max_depth=10)

            # Create with config
            config = ModelConfig(model_type='random_forest_classifier', hyperparameters={'n_estimators': 100})
            model = ModelFactory.create_model('random_forest_classifier', config=config)
            
            # Create without validation (not recommended)
            model = ModelFactory.create_model('random_forest_classifier', validate=False, n_estimators=100)
        """
        if model_id not in MODEL_WRAPPER_REGISTRY:
            available_models = ", ".join(sorted(MODEL_WRAPPER_REGISTRY.keys()))
            raise ValueError(
                f"Model '{model_id}' not found in registry. "
                f"Available models: {available_models}"
            )

        # Remove deprecated parameters for specific models
        deprecated_params = {
            'linear_regression': ['normalize'],  # Removed in sklearn 1.2+
            'ridge_regression': ['normalize'],   # Removed in sklearn 1.2+
            'lasso_regression': ['normalize'],   # Removed in sklearn 1.2+
        }
        
        # Filter out deprecated parameters
        if model_id in deprecated_params:
            for param in deprecated_params[model_id]:
                if param in hyperparameters:
                    del hyperparameters[param]

        # Create config if not provided
        if config is None:
            config = ModelConfig(
                model_type=model_id,
                hyperparameters=hyperparameters,
                validate=validate,
                strict=strict
            )

        # Get the wrapper class and instantiate it
        wrapper_class = MODEL_WRAPPER_REGISTRY[model_id]
        return wrapper_class(config)

    @staticmethod
    def get_available_models() -> Dict[str, Type[BaseModelWrapper]]:
        """
        Get all available model wrappers.

        Returns:
            Dictionary mapping model IDs to wrapper classes
        """
        return MODEL_WRAPPER_REGISTRY.copy()

    @staticmethod
    def is_model_available(model_id: str) -> bool:
        """
        Check if a model is available in the registry.

        Args:
            model_id: Identifier for the model

        Returns:
            True if model is available, False otherwise
        """
        return model_id in MODEL_WRAPPER_REGISTRY


# Convenience function for creating models
def create_model(model_id: str, **hyperparameters) -> BaseModelWrapper:
    """
    Convenience function to create a model wrapper.

    Args:
        model_id: Identifier for the model
        **hyperparameters: Hyperparameters to pass to the model

    Returns:
        Instance of the appropriate model wrapper

    Examples:
        model = create_model('random_forest_classifier', n_estimators=100, max_depth=10)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    """
    return ModelFactory.create_model(model_id, **hyperparameters)
