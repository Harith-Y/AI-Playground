"""
Regression model wrappers.

This module provides wrapper classes for scikit-learn regression models,
implementing the BaseModelWrapper interface with specific configurations
for each regression algorithm.
"""

from typing import Union, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from .base import SupervisedModelWrapper, ModelConfig


class LinearRegressionWrapper(SupervisedModelWrapper):
    """Wrapper for Linear Regression."""

    def _create_model(self) -> Any:
        return LinearRegression(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "regression"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate R² score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)


class RidgeRegressionWrapper(SupervisedModelWrapper):
    """Wrapper for Ridge Regression (L2 regularization)."""

    def _create_model(self) -> Any:
        return Ridge(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "regression"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate R² score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)


class LassoRegressionWrapper(SupervisedModelWrapper):
    """Wrapper for Lasso Regression (L1 regularization)."""

    def _create_model(self) -> Any:
        return Lasso(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "regression"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate R² score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)


class ElasticNetWrapper(SupervisedModelWrapper):
    """Wrapper for Elastic Net (L1 + L2 regularization)."""

    def _create_model(self) -> Any:
        return ElasticNet(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "regression"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate R² score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)


class DecisionTreeRegressorWrapper(SupervisedModelWrapper):
    """Wrapper for Decision Tree Regressor."""

    def _create_model(self) -> Any:
        return DecisionTreeRegressor(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "regression"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate R² score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)


class RandomForestRegressorWrapper(SupervisedModelWrapper):
    """Wrapper for Random Forest Regressor."""

    def _create_model(self) -> Any:
        return RandomForestRegressor(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "regression"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate R² score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)


class ExtraTreesRegressorWrapper(SupervisedModelWrapper):
    """Wrapper for Extra Trees Regressor."""

    def _create_model(self) -> Any:
        return ExtraTreesRegressor(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "regression"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate R² score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)


class GradientBoostingRegressorWrapper(SupervisedModelWrapper):
    """Wrapper for Gradient Boosting Regressor."""

    def _create_model(self) -> Any:
        return GradientBoostingRegressor(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "regression"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate R² score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)


class AdaBoostRegressorWrapper(SupervisedModelWrapper):
    """Wrapper for AdaBoost Regressor."""

    def _create_model(self) -> Any:
        return AdaBoostRegressor(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "regression"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate R² score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)


class SVRWrapper(SupervisedModelWrapper):
    """Wrapper for Support Vector Regression."""

    def _create_model(self) -> Any:
        return SVR(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "regression"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate R² score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)


class KNeighborsRegressorWrapper(SupervisedModelWrapper):
    """Wrapper for K-Nearest Neighbors Regressor."""

    def _create_model(self) -> Any:
        return KNeighborsRegressor(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "regression"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate R² score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)
