"""
Classification model wrappers.

This module provides wrapper classes for scikit-learn classification models,
implementing the BaseModelWrapper interface with specific configurations
for each classification algorithm.
"""

from typing import Union, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from .base import SupervisedModelWrapper, ModelConfig


class LogisticRegressionWrapper(SupervisedModelWrapper):
    """Wrapper for Logistic Regression."""

    def _create_model(self) -> Any:
        return LogisticRegression(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "classification"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate accuracy score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)


class DecisionTreeClassifierWrapper(SupervisedModelWrapper):
    """Wrapper for Decision Tree Classifier."""

    def _create_model(self) -> Any:
        return DecisionTreeClassifier(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "classification"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate accuracy score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)


class RandomForestClassifierWrapper(SupervisedModelWrapper):
    """Wrapper for Random Forest Classifier."""

    def _create_model(self) -> Any:
        return RandomForestClassifier(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "classification"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate accuracy score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)


class ExtraTreesClassifierWrapper(SupervisedModelWrapper):
    """Wrapper for Extra Trees Classifier."""

    def _create_model(self) -> Any:
        return ExtraTreesClassifier(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "classification"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate accuracy score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)


class GradientBoostingClassifierWrapper(SupervisedModelWrapper):
    """Wrapper for Gradient Boosting Classifier."""

    def _create_model(self) -> Any:
        return GradientBoostingClassifier(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "classification"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate accuracy score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)


class AdaBoostClassifierWrapper(SupervisedModelWrapper):
    """Wrapper for AdaBoost Classifier."""

    def _create_model(self) -> Any:
        return AdaBoostClassifier(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "classification"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate accuracy score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)


class SVMClassifierWrapper(SupervisedModelWrapper):
    """Wrapper for Support Vector Machine Classifier."""

    def _create_model(self) -> Any:
        return SVC(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "classification"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate accuracy score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)


class KNeighborsClassifierWrapper(SupervisedModelWrapper):
    """Wrapper for K-Nearest Neighbors Classifier."""

    def _create_model(self) -> Any:
        return KNeighborsClassifier(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "classification"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate accuracy score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)


class GaussianNBWrapper(SupervisedModelWrapper):
    """Wrapper for Gaussian Naive Bayes."""

    def _create_model(self) -> Any:
        return GaussianNB(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "classification"

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate accuracy score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        return self.model.score(X_array, y_array)
