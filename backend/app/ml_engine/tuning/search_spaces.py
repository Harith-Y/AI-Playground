"""Default hyperparameter search spaces per model.

Provides lightweight param grids that work for grid/random/bayesian search
runners. Values are conservative to keep search sizes reasonable by default.
"""

from copy import deepcopy
from typing import Any, Dict

# Default search grids by model_id
DEFAULT_SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
    # Classification
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [200, 400, 800],
    },
    "random_forest_classifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "max_features": ["auto", "sqrt", "log2"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    },
    "svm_classifier": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
    },
    "gradient_boosting_classifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5],
        "subsample": [0.6, 0.8, 1.0],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
    "knn_classifier": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },

    # Regression
    "linear_regression": {
        "fit_intercept": [True, False],
    },
    "ridge_regression": {
        "alpha": [0.01, 0.1, 1.0, 10.0],
        "solver": ["auto", "sag", "saga", "lsqr"],
        "fit_intercept": [True, False],
    },
    "lasso_regression": {
        "alpha": [0.0005, 0.001, 0.01, 0.1, 1.0],
        "max_iter": [1000, 3000, 6000],
        "fit_intercept": [True, False],
    },
    "random_forest_regressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "max_features": ["auto", "sqrt", "log2"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    },

    # Clustering
    "kmeans": {
        "n_clusters": [2, 3, 4, 5, 8],
        "init": ["k-means++", "random"],
        "n_init": [10, 20, 40],
    },
    "dbscan": {
        "eps": [0.1, 0.3, 0.5, 0.8],
        "min_samples": [3, 5, 10],
        "metric": ["euclidean", "manhattan", "chebyshev"],
    },
    "agglomerative_clustering": {
        "n_clusters": [2, 3, 4, 5],
        "linkage": ["ward", "complete", "average", "single"],
        "metric": ["euclidean", "manhattan", "cosine"],
    },
    "gaussian_mixture": {
        "n_components": [1, 2, 3, 4, 6],
        "covariance_type": ["full", "tied", "diag", "spherical"],
        "max_iter": [100, 300, 600],
    },
}


def get_default_search_space(model_id: str) -> Dict[str, Any]:
    """Return a copy of the default search space for a model (empty if none)."""

    if model_id not in DEFAULT_SEARCH_SPACES:
        return {}
    return deepcopy(DEFAULT_SEARCH_SPACES[model_id])
