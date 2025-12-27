"""
Clustering model wrappers.

This module provides wrapper classes for scikit-learn clustering models,
implementing the BaseModelWrapper interface with specific configurations
for each clustering algorithm.
"""

from typing import Union, Any
import pandas as pd
import numpy as np
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering
)
from sklearn.mixture import GaussianMixture

from .base import UnsupervisedModelWrapper, ModelConfig


class KMeansWrapper(UnsupervisedModelWrapper):
    """Wrapper for K-Means clustering."""

    def _create_model(self) -> Any:
        return KMeans(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "clustering"

    def get_inertia(self) -> float:
        """
        Get the inertia (within-cluster sum of squares).

        Returns:
            Inertia value

        Raises:
            RuntimeError: If model hasn't been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting inertia")
        return self.model.inertia_


class DBSCANWrapper(UnsupervisedModelWrapper):
    """Wrapper for DBSCAN (Density-Based Spatial Clustering)."""

    def _create_model(self) -> Any:
        return DBSCAN(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "clustering"

    def get_core_sample_indices(self) -> np.ndarray:
        """
        Get indices of core samples.

        Returns:
            Indices of core samples

        Raises:
            RuntimeError: If model hasn't been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting core samples")
        return self.model.core_sample_indices_


class AgglomerativeClusteringWrapper(UnsupervisedModelWrapper):
    """Wrapper for Agglomerative (Hierarchical) Clustering."""

    def _create_model(self) -> Any:
        return AgglomerativeClustering(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "clustering"

    def get_n_leaves(self) -> int:
        """
        Get the number of leaves in the hierarchical tree.

        Returns:
            Number of leaves

        Raises:
            RuntimeError: If model hasn't been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting number of leaves")
        return self.model.n_leaves_

    def get_n_connected_components(self) -> int:
        """
        Get the number of connected components.

        Returns:
            Number of connected components

        Raises:
            RuntimeError: If model hasn't been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting connected components")
        return self.model.n_connected_components_


class GaussianMixtureWrapper(UnsupervisedModelWrapper):
    """Wrapper for Gaussian Mixture Model clustering."""

    def _create_model(self) -> Any:
        return GaussianMixture(**self.config.hyperparameters)

    def get_task_type(self) -> str:
        return "clustering"

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict posterior probability of each component given the data.

        Args:
            X: Features to predict on

        Returns:
            Probability matrix

        Raises:
            RuntimeError: If model hasn't been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict_proba(X_array)

    def get_aic(self) -> float:
        """
        Get Akaike Information Criterion for the current model.

        Returns:
            AIC value

        Raises:
            RuntimeError: If model hasn't been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting AIC")
        return self.model.aic(self.model._X)

    def get_bic(self) -> float:
        """
        Get Bayesian Information Criterion for the current model.

        Returns:
            BIC value

        Raises:
            RuntimeError: If model hasn't been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting BIC")
        return self.model.bic(self.model._X)
