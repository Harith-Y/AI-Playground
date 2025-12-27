"""
Model Registry - Defines all available ML models with metadata.

This module serves as a central registry for all machine learning models
supported by the platform, organized by task type (regression, classification, clustering).

Based on: MODELS.md in project root
"""

from typing import Dict, List, Any, Optional
from enum import Enum


class TaskType(str, Enum):
    """ML task types."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"


class ModelCategory(str, Enum):
    """Model categories within task types."""
    LINEAR = "linear"
    TREE_BASED = "tree_based"
    BOOSTING = "boosting"
    SUPPORT_VECTOR = "support_vector"
    INSTANCE_BASED = "instance_based"
    NEURAL_NETWORK = "neural_network"
    PROBABILISTIC = "probabilistic"
    DENSITY_BASED = "density_based"
    HIERARCHICAL = "hierarchical"


class ModelInfo:
    """Model information and metadata."""

    def __init__(
        self,
        model_id: str,
        name: str,
        description: str,
        task_type: TaskType,
        category: ModelCategory,
        sklearn_class: str,
        hyperparameters: Dict[str, Any],
        default_config: Dict[str, Any],
        supports_feature_importance: bool = False,
        supports_probability: bool = False,
        supports_multiclass: bool = False,
        requires_scaling: bool = False,
        handles_missing_values: bool = False,
        tags: Optional[List[str]] = None,
    ):
        """
        Initialize model information.

        Args:
            model_id: Unique identifier for the model
            name: Display name
            description: Model description
            task_type: Type of ML task (regression, classification, clustering)
            category: Model category (linear, tree_based, etc.)
            sklearn_class: Full sklearn class path (e.g., 'sklearn.linear_model.LinearRegression')
            hyperparameters: Dictionary of hyperparameter definitions
            default_config: Default hyperparameter values
            supports_feature_importance: Whether model provides feature importance
            supports_probability: Whether model can output probabilities (classifiers)
            supports_multiclass: Whether supports multi-class classification
            requires_scaling: Whether features should be scaled
            handles_missing_values: Whether model handles NaN values natively
            tags: Additional tags for filtering/searching
        """
        self.model_id = model_id
        self.name = name
        self.description = description
        self.task_type = task_type
        self.category = category
        self.sklearn_class = sklearn_class
        self.hyperparameters = hyperparameters
        self.default_config = default_config
        self.supports_feature_importance = supports_feature_importance
        self.supports_probability = supports_probability
        self.supports_multiclass = supports_multiclass
        self.requires_scaling = requires_scaling
        self.handles_missing_values = handles_missing_values
        self.tags = tags or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "category": self.category,
            "sklearn_class": self.sklearn_class,
            "hyperparameters": self.hyperparameters,
            "default_config": self.default_config,
            "capabilities": {
                "supports_feature_importance": self.supports_feature_importance,
                "supports_probability": self.supports_probability,
                "supports_multiclass": self.supports_multiclass,
            },
            "requirements": {
                "requires_scaling": self.requires_scaling,
                "handles_missing_values": self.handles_missing_values,
            },
            "tags": self.tags
        }


# ============================================================================
# REGRESSION MODELS
# ============================================================================

REGRESSION_MODELS = [
    # LINEAR MODELS
    ModelInfo(
        model_id="linear_regression",
        name="Linear Regression",
        description="Ordinary Least Squares regression. Fast and interpretable for linear relationships.",
        task_type=TaskType.REGRESSION,
        category=ModelCategory.LINEAR,
        sklearn_class="sklearn.linear_model.LinearRegression",
        hyperparameters={
            "fit_intercept": {
                "type": "boolean",
                "default": True,
                "description": "Whether to calculate the intercept"
            }
        },
        default_config={"fit_intercept": True},
        supports_feature_importance=True,
        requires_scaling=False,
        handles_missing_values=False,
        tags=["fast", "interpretable", "baseline"],
    ),

    ModelInfo(
        model_id="ridge",
        name="Ridge Regression",
        description="Linear regression with L2 regularization. Reduces overfitting for correlated features.",
        task_type=TaskType.REGRESSION,
        category=ModelCategory.LINEAR,
        sklearn_class="sklearn.linear_model.Ridge",
        hyperparameters={
            "alpha": {
                "type": "float",
                "default": 1.0,
                "range": [0.0001, 100.0],
                "description": "Regularization strength (higher = more regularization)"
            },
            "fit_intercept": {
                "type": "boolean",
                "default": True,
                "description": "Whether to calculate the intercept"
            },
            "solver": {
                "type": "categorical",
                "default": "auto",
                "choices": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                "description": "Solver to use for computation"
            }
        },
        default_config={"alpha": 1.0, "fit_intercept": True, "solver": "auto"},
        supports_feature_importance=True,
        requires_scaling=True,
        handles_missing_values=False,
        tags=["regularized", "robust"],
    ),

    ModelInfo(
        model_id="lasso",
        name="Lasso Regression",
        description="Linear regression with L1 regularization. Performs feature selection by driving coefficients to zero.",
        task_type=TaskType.REGRESSION,
        category=ModelCategory.LINEAR,
        sklearn_class="sklearn.linear_model.Lasso",
        hyperparameters={
            "alpha": {
                "type": "float",
                "default": 1.0,
                "range": [0.0001, 100.0],
                "description": "Regularization strength"
            },
            "fit_intercept": {
                "type": "boolean",
                "default": True,
                "description": "Whether to calculate the intercept"
            },
            "max_iter": {
                "type": "integer",
                "default": 1000,
                "range": [100, 10000],
                "description": "Maximum number of iterations"
            }
        },
        default_config={"alpha": 1.0, "fit_intercept": True, "max_iter": 1000},
        supports_feature_importance=True,
        requires_scaling=True,
        handles_missing_values=False,
        tags=["regularized", "feature_selection"],
    ),

    ModelInfo(
        model_id="elastic_net",
        name="Elastic Net Regression",
        description="Linear regression with both L1 and L2 regularization. Combines benefits of Ridge and Lasso.",
        task_type=TaskType.REGRESSION,
        category=ModelCategory.LINEAR,
        sklearn_class="sklearn.linear_model.ElasticNet",
        hyperparameters={
            "alpha": {
                "type": "float",
                "default": 1.0,
                "range": [0.0001, 100.0],
                "description": "Overall regularization strength"
            },
            "l1_ratio": {
                "type": "float",
                "default": 0.5,
                "range": [0.0, 1.0],
                "description": "Mix of L1 and L2 (0=Ridge, 1=Lasso)"
            },
            "fit_intercept": {
                "type": "boolean",
                "default": True,
                "description": "Whether to calculate the intercept"
            },
            "max_iter": {
                "type": "integer",
                "default": 1000,
                "range": [100, 10000],
                "description": "Maximum iterations"
            }
        },
        default_config={"alpha": 1.0, "l1_ratio": 0.5, "fit_intercept": True, "max_iter": 1000},
        supports_feature_importance=True,
        requires_scaling=True,
        handles_missing_values=False,
        tags=["regularized", "feature_selection", "robust"],
    ),

    # TREE-BASED REGRESSION
    ModelInfo(
        model_id="decision_tree_regressor",
        name="Decision Tree Regressor",
        description="Tree-based model that learns simple decision rules. Handles non-linear relationships.",
        task_type=TaskType.REGRESSION,
        category=ModelCategory.TREE_BASED,
        sklearn_class="sklearn.tree.DecisionTreeRegressor",
        hyperparameters={
            "max_depth": {
                "type": "integer",
                "default": None,
                "range": [1, 50],
                "description": "Maximum depth of tree"
            },
            "min_samples_split": {
                "type": "integer",
                "default": 2,
                "range": [2, 20],
                "description": "Minimum samples to split a node"
            },
            "min_samples_leaf": {
                "type": "integer",
                "default": 1,
                "range": [1, 20],
                "description": "Minimum samples at leaf"
            },
            "random_state": {
                "type": "integer",
                "default": 42,
                "description": "Random seed"
            }
        },
        default_config={"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "random_state": 42},
        supports_feature_importance=True,
        requires_scaling=False,
        handles_missing_values=False,
        tags=["interpretable", "non_linear"],
    ),

    ModelInfo(
        model_id="random_forest_regressor",
        name="Random Forest Regressor",
        description="Ensemble of decision trees. Robust, handles non-linear relationships, provides feature importance.",
        task_type=TaskType.REGRESSION,
        category=ModelCategory.TREE_BASED,
        sklearn_class="sklearn.ensemble.RandomForestRegressor",
        hyperparameters={
            "n_estimators": {
                "type": "integer",
                "default": 100,
                "range": [10, 500],
                "description": "Number of trees in the forest"
            },
            "max_depth": {
                "type": "integer",
                "default": None,
                "range": [3, 50],
                "description": "Maximum depth of trees (None = unlimited)"
            },
            "min_samples_split": {
                "type": "integer",
                "default": 2,
                "range": [2, 20],
                "description": "Minimum samples required to split a node"
            },
            "min_samples_leaf": {
                "type": "integer",
                "default": 1,
                "range": [1, 20],
                "description": "Minimum samples required at a leaf node"
            },
            "max_features": {
                "type": "categorical",
                "default": "sqrt",
                "choices": ["sqrt", "log2", None],
                "description": "Number of features to consider for best split"
            },
            "random_state": {
                "type": "integer",
                "default": 42,
                "description": "Random seed for reproducibility"
            }
        },
        default_config={
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42
        },
        supports_feature_importance=True,
        requires_scaling=False,
        handles_missing_values=False,
        tags=["ensemble", "robust", "non_linear"],
    ),

    ModelInfo(
        model_id="extra_trees_regressor",
        name="Extra Trees Regressor",
        description="Similar to Random Forest but with more randomness. Often faster training.",
        task_type=TaskType.REGRESSION,
        category=ModelCategory.TREE_BASED,
        sklearn_class="sklearn.ensemble.ExtraTreesRegressor",
        hyperparameters={
            "n_estimators": {
                "type": "integer",
                "default": 100,
                "range": [10, 500],
                "description": "Number of trees"
            },
            "max_depth": {
                "type": "integer",
                "default": None,
                "range": [3, 50],
                "description": "Maximum tree depth"
            },
            "min_samples_split": {
                "type": "integer",
                "default": 2,
                "range": [2, 20],
                "description": "Minimum samples to split"
            },
            "random_state": {
                "type": "integer",
                "default": 42,
                "description": "Random seed"
            }
        },
        default_config={"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "random_state": 42},
        supports_feature_importance=True,
        requires_scaling=False,
        handles_missing_values=False,
        tags=["ensemble", "fast"],
    ),

    # BOOSTING REGRESSION
    ModelInfo(
        model_id="gradient_boosting_regressor",
        name="Gradient Boosting Regressor",
        description="Sequential ensemble that builds trees to correct previous errors. Often achieves highest accuracy.",
        task_type=TaskType.REGRESSION,
        category=ModelCategory.BOOSTING,
        sklearn_class="sklearn.ensemble.GradientBoostingRegressor",
        hyperparameters={
            "n_estimators": {
                "type": "integer",
                "default": 100,
                "range": [10, 500],
                "description": "Number of boosting stages"
            },
            "learning_rate": {
                "type": "float",
                "default": 0.1,
                "range": [0.001, 1.0],
                "description": "Shrinks contribution of each tree"
            },
            "max_depth": {
                "type": "integer",
                "default": 3,
                "range": [1, 10],
                "description": "Maximum tree depth"
            },
            "min_samples_split": {
                "type": "integer",
                "default": 2,
                "range": [2, 20],
                "description": "Minimum samples to split"
            },
            "subsample": {
                "type": "float",
                "default": 1.0,
                "range": [0.5, 1.0],
                "description": "Fraction of samples for fitting trees"
            },
            "random_state": {
                "type": "integer",
                "default": 42,
                "description": "Random seed"
            }
        },
        default_config={
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "min_samples_split": 2,
            "subsample": 1.0,
            "random_state": 42
        },
        supports_feature_importance=True,
        requires_scaling=False,
        handles_missing_values=False,
        tags=["ensemble", "powerful", "boosting"],
    ),

    ModelInfo(
        model_id="adaboost_regressor",
        name="AdaBoost Regressor",
        description="Adaptive boosting that focuses on difficult samples. Good for weak learners.",
        task_type=TaskType.REGRESSION,
        category=ModelCategory.BOOSTING,
        sklearn_class="sklearn.ensemble.AdaBoostRegressor",
        hyperparameters={
            "n_estimators": {
                "type": "integer",
                "default": 50,
                "range": [10, 200],
                "description": "Number of boosting stages"
            },
            "learning_rate": {
                "type": "float",
                "default": 1.0,
                "range": [0.01, 2.0],
                "description": "Learning rate shrinks contribution"
            },
            "loss": {
                "type": "categorical",
                "default": "linear",
                "choices": ["linear", "square", "exponential"],
                "description": "Loss function"
            },
            "random_state": {
                "type": "integer",
                "default": 42,
                "description": "Random seed"
            }
        },
        default_config={"n_estimators": 50, "learning_rate": 1.0, "loss": "linear", "random_state": 42},
        supports_feature_importance=True,
        requires_scaling=False,
        handles_missing_values=False,
        tags=["ensemble", "boosting"],
    ),

    # SUPPORT VECTOR REGRESSION
    ModelInfo(
        model_id="svr",
        name="Support Vector Regression",
        description="SVM for regression. Effective in high dimensions with different kernels.",
        task_type=TaskType.REGRESSION,
        category=ModelCategory.SUPPORT_VECTOR,
        sklearn_class="sklearn.svm.SVR",
        hyperparameters={
            "C": {
                "type": "float",
                "default": 1.0,
                "range": [0.001, 100.0],
                "description": "Regularization parameter"
            },
            "kernel": {
                "type": "categorical",
                "default": "rbf",
                "choices": ["linear", "poly", "rbf", "sigmoid"],
                "description": "Kernel type"
            },
            "gamma": {
                "type": "categorical",
                "default": "scale",
                "choices": ["scale", "auto"],
                "description": "Kernel coefficient"
            },
            "epsilon": {
                "type": "float",
                "default": 0.1,
                "range": [0.0, 1.0],
                "description": "Epsilon-tube for SVR"
            }
        },
        default_config={"C": 1.0, "kernel": "rbf", "gamma": "scale", "epsilon": 0.1},
        supports_feature_importance=False,
        requires_scaling=True,
        handles_missing_values=False,
        tags=["kernel_methods", "high_dimensional"],
    ),

    # INSTANCE-BASED
    ModelInfo(
        model_id="knn_regressor",
        name="K-Nearest Neighbors Regressor",
        description="Predicts based on k nearest training examples. Simple and effective for small datasets.",
        task_type=TaskType.REGRESSION,
        category=ModelCategory.INSTANCE_BASED,
        sklearn_class="sklearn.neighbors.KNeighborsRegressor",
        hyperparameters={
            "n_neighbors": {
                "type": "integer",
                "default": 5,
                "range": [1, 20],
                "description": "Number of neighbors"
            },
            "weights": {
                "type": "categorical",
                "default": "uniform",
                "choices": ["uniform", "distance"],
                "description": "Weight function"
            },
            "metric": {
                "type": "categorical",
                "default": "minkowski",
                "choices": ["euclidean", "manhattan", "minkowski"],
                "description": "Distance metric"
            }
        },
        default_config={"n_neighbors": 5, "weights": "uniform", "metric": "minkowski"},
        supports_feature_importance=False,
        requires_scaling=True,
        handles_missing_values=False,
        tags=["instance_based", "simple"],
    ),
]


# ============================================================================
# CLASSIFICATION MODELS
# ============================================================================

CLASSIFICATION_MODELS = [
    # LINEAR MODELS
    ModelInfo(
        model_id="logistic_regression",
        name="Logistic Regression",
        description="Linear model for binary and multi-class classification. Fast and interpretable.",
        task_type=TaskType.CLASSIFICATION,
        category=ModelCategory.LINEAR,
        sklearn_class="sklearn.linear_model.LogisticRegression",
        hyperparameters={
            "C": {
                "type": "float",
                "default": 1.0,
                "range": [0.001, 100.0],
                "description": "Inverse of regularization strength (smaller = more regularization)"
            },
            "penalty": {
                "type": "categorical",
                "default": "l2",
                "choices": ["l1", "l2", "elasticnet", None],
                "description": "Regularization penalty"
            },
            "solver": {
                "type": "categorical",
                "default": "lbfgs",
                "choices": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
                "description": "Optimization algorithm"
            },
            "max_iter": {
                "type": "integer",
                "default": 100,
                "range": [50, 1000],
                "description": "Maximum iterations for solver convergence"
            },
            "random_state": {
                "type": "integer",
                "default": 42,
                "description": "Random seed"
            }
        },
        default_config={
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 100,
            "random_state": 42
        },
        supports_feature_importance=True,
        supports_probability=True,
        supports_multiclass=True,
        requires_scaling=True,
        handles_missing_values=False,
        tags=["fast", "interpretable", "baseline"],
    ),

    # TREE-BASED CLASSIFICATION
    ModelInfo(
        model_id="decision_tree_classifier",
        name="Decision Tree Classifier",
        description="Tree-based classifier using simple decision rules. Highly interpretable.",
        task_type=TaskType.CLASSIFICATION,
        category=ModelCategory.TREE_BASED,
        sklearn_class="sklearn.tree.DecisionTreeClassifier",
        hyperparameters={
            "max_depth": {
                "type": "integer",
                "default": None,
                "range": [1, 50],
                "description": "Maximum depth of tree"
            },
            "min_samples_split": {
                "type": "integer",
                "default": 2,
                "range": [2, 20],
                "description": "Minimum samples to split"
            },
            "min_samples_leaf": {
                "type": "integer",
                "default": 1,
                "range": [1, 20],
                "description": "Minimum samples at leaf"
            },
            "criterion": {
                "type": "categorical",
                "default": "gini",
                "choices": ["gini", "entropy", "log_loss"],
                "description": "Function to measure split quality"
            },
            "random_state": {
                "type": "integer",
                "default": 42,
                "description": "Random seed"
            }
        },
        default_config={
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "gini",
            "random_state": 42
        },
        supports_feature_importance=True,
        supports_probability=True,
        supports_multiclass=True,
        requires_scaling=False,
        handles_missing_values=False,
        tags=["interpretable", "non_linear"],
    ),

    ModelInfo(
        model_id="random_forest_classifier",
        name="Random Forest Classifier",
        description="Ensemble of decision trees for classification. Handles non-linear patterns well.",
        task_type=TaskType.CLASSIFICATION,
        category=ModelCategory.TREE_BASED,
        sklearn_class="sklearn.ensemble.RandomForestClassifier",
        hyperparameters={
            "n_estimators": {
                "type": "integer",
                "default": 100,
                "range": [10, 500],
                "description": "Number of trees"
            },
            "max_depth": {
                "type": "integer",
                "default": None,
                "range": [3, 50],
                "description": "Maximum tree depth"
            },
            "min_samples_split": {
                "type": "integer",
                "default": 2,
                "range": [2, 20],
                "description": "Minimum samples to split"
            },
            "min_samples_leaf": {
                "type": "integer",
                "default": 1,
                "range": [1, 20],
                "description": "Minimum samples at leaf"
            },
            "max_features": {
                "type": "categorical",
                "default": "sqrt",
                "choices": ["sqrt", "log2", None],
                "description": "Features per split"
            },
            "class_weight": {
                "type": "categorical",
                "default": None,
                "choices": [None, "balanced", "balanced_subsample"],
                "description": "Handle imbalanced classes"
            },
            "random_state": {
                "type": "integer",
                "default": 42,
                "description": "Random seed"
            }
        },
        default_config={
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": None,
            "random_state": 42
        },
        supports_feature_importance=True,
        supports_probability=True,
        supports_multiclass=True,
        requires_scaling=False,
        handles_missing_values=False,
        tags=["ensemble", "robust", "non_linear"],
    ),

    ModelInfo(
        model_id="extra_trees_classifier",
        name="Extra Trees Classifier",
        description="Similar to Random Forest but with more randomness. Often faster training.",
        task_type=TaskType.CLASSIFICATION,
        category=ModelCategory.TREE_BASED,
        sklearn_class="sklearn.ensemble.ExtraTreesClassifier",
        hyperparameters={
            "n_estimators": {
                "type": "integer",
                "default": 100,
                "range": [10, 500],
                "description": "Number of trees"
            },
            "max_depth": {
                "type": "integer",
                "default": None,
                "range": [3, 50],
                "description": "Maximum tree depth"
            },
            "min_samples_split": {
                "type": "integer",
                "default": 2,
                "range": [2, 20],
                "description": "Minimum samples to split"
            },
            "random_state": {
                "type": "integer",
                "default": 42,
                "description": "Random seed"
            }
        },
        default_config={"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "random_state": 42},
        supports_feature_importance=True,
        supports_probability=True,
        supports_multiclass=True,
        requires_scaling=False,
        handles_missing_values=False,
        tags=["ensemble", "fast"],
    ),

    # BOOSTING CLASSIFICATION
    ModelInfo(
        model_id="gradient_boosting_classifier",
        name="Gradient Boosting Classifier",
        description="Sequential ensemble that builds trees to correct previous errors. Often highest accuracy.",
        task_type=TaskType.CLASSIFICATION,
        category=ModelCategory.BOOSTING,
        sklearn_class="sklearn.ensemble.GradientBoostingClassifier",
        hyperparameters={
            "n_estimators": {
                "type": "integer",
                "default": 100,
                "range": [10, 500],
                "description": "Number of boosting stages"
            },
            "learning_rate": {
                "type": "float",
                "default": 0.1,
                "range": [0.001, 1.0],
                "description": "Shrinks contribution of each tree"
            },
            "max_depth": {
                "type": "integer",
                "default": 3,
                "range": [1, 10],
                "description": "Maximum tree depth"
            },
            "min_samples_split": {
                "type": "integer",
                "default": 2,
                "range": [2, 20],
                "description": "Minimum samples to split"
            },
            "min_samples_leaf": {
                "type": "integer",
                "default": 1,
                "range": [1, 20],
                "description": "Minimum samples at leaf"
            },
            "subsample": {
                "type": "float",
                "default": 1.0,
                "range": [0.5, 1.0],
                "description": "Fraction of samples for fitting trees"
            },
            "random_state": {
                "type": "integer",
                "default": 42,
                "description": "Random seed"
            }
        },
        default_config={
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "subsample": 1.0,
            "random_state": 42
        },
        supports_feature_importance=True,
        supports_probability=True,
        supports_multiclass=True,
        requires_scaling=False,
        handles_missing_values=False,
        tags=["ensemble", "powerful", "boosting"],
    ),

    ModelInfo(
        model_id="adaboost_classifier",
        name="AdaBoost Classifier",
        description="Adaptive boosting that focuses on misclassified samples.",
        task_type=TaskType.CLASSIFICATION,
        category=ModelCategory.BOOSTING,
        sklearn_class="sklearn.ensemble.AdaBoostClassifier",
        hyperparameters={
            "n_estimators": {
                "type": "integer",
                "default": 50,
                "range": [10, 200],
                "description": "Number of boosting stages"
            },
            "learning_rate": {
                "type": "float",
                "default": 1.0,
                "range": [0.01, 2.0],
                "description": "Learning rate shrinks contribution"
            },
            "algorithm": {
                "type": "categorical",
                "default": "SAMME.R",
                "choices": ["SAMME", "SAMME.R"],
                "description": "Boosting algorithm"
            },
            "random_state": {
                "type": "integer",
                "default": 42,
                "description": "Random seed"
            }
        },
        default_config={"n_estimators": 50, "learning_rate": 1.0, "algorithm": "SAMME.R", "random_state": 42},
        supports_feature_importance=True,
        supports_probability=True,
        supports_multiclass=True,
        requires_scaling=False,
        handles_missing_values=False,
        tags=["ensemble", "boosting"],
    ),

    # SUPPORT VECTOR MACHINES
    ModelInfo(
        model_id="svm",
        name="Support Vector Machine",
        description="Finds optimal hyperplane for classification. Effective in high dimensions.",
        task_type=TaskType.CLASSIFICATION,
        category=ModelCategory.SUPPORT_VECTOR,
        sklearn_class="sklearn.svm.SVC",
        hyperparameters={
            "C": {
                "type": "float",
                "default": 1.0,
                "range": [0.001, 100.0],
                "description": "Regularization parameter"
            },
            "kernel": {
                "type": "categorical",
                "default": "rbf",
                "choices": ["linear", "poly", "rbf", "sigmoid"],
                "description": "Kernel type"
            },
            "gamma": {
                "type": "categorical",
                "default": "scale",
                "choices": ["scale", "auto"],
                "description": "Kernel coefficient"
            },
            "probability": {
                "type": "boolean",
                "default": True,
                "description": "Enable probability estimates (slower)"
            },
            "random_state": {
                "type": "integer",
                "default": 42,
                "description": "Random seed"
            }
        },
        default_config={
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
            "random_state": 42
        },
        supports_feature_importance=False,
        supports_probability=True,
        supports_multiclass=True,
        requires_scaling=True,
        handles_missing_values=False,
        tags=["kernel_methods", "high_dimensional"],
    ),

    # INSTANCE-BASED
    ModelInfo(
        model_id="knn_classifier",
        name="K-Nearest Neighbors Classifier",
        description="Classifies based on k nearest training examples. Simple and effective.",
        task_type=TaskType.CLASSIFICATION,
        category=ModelCategory.INSTANCE_BASED,
        sklearn_class="sklearn.neighbors.KNeighborsClassifier",
        hyperparameters={
            "n_neighbors": {
                "type": "integer",
                "default": 5,
                "range": [1, 20],
                "description": "Number of neighbors"
            },
            "weights": {
                "type": "categorical",
                "default": "uniform",
                "choices": ["uniform", "distance"],
                "description": "Weight function"
            },
            "metric": {
                "type": "categorical",
                "default": "minkowski",
                "choices": ["euclidean", "manhattan", "minkowski"],
                "description": "Distance metric"
            }
        },
        default_config={"n_neighbors": 5, "weights": "uniform", "metric": "minkowski"},
        supports_feature_importance=False,
        supports_probability=True,
        supports_multiclass=True,
        requires_scaling=True,
        handles_missing_values=False,
        tags=["instance_based", "simple"],
    ),

    # PROBABILISTIC
    ModelInfo(
        model_id="naive_bayes_gaussian",
        name="Gaussian Naive Bayes",
        description="Probabilistic classifier based on Bayes theorem. Assumes Gaussian distribution.",
        task_type=TaskType.CLASSIFICATION,
        category=ModelCategory.PROBABILISTIC,
        sklearn_class="sklearn.naive_bayes.GaussianNB",
        hyperparameters={
            "var_smoothing": {
                "type": "float",
                "default": 1e-9,
                "range": [1e-12, 1e-6],
                "description": "Portion of largest variance added to variances for stability"
            }
        },
        default_config={"var_smoothing": 1e-9},
        supports_feature_importance=False,
        supports_probability=True,
        supports_multiclass=True,
        requires_scaling=False,
        handles_missing_values=False,
        tags=["probabilistic", "fast", "simple"],
    ),
]


# ============================================================================
# CLUSTERING MODELS
# ============================================================================

CLUSTERING_MODELS = [
    ModelInfo(
        model_id="kmeans",
        name="K-Means",
        description="Partitions data into K clusters by minimizing within-cluster variance. Fast and scalable.",
        task_type=TaskType.CLUSTERING,
        category=ModelCategory.DENSITY_BASED,
        sklearn_class="sklearn.cluster.KMeans",
        hyperparameters={
            "n_clusters": {
                "type": "integer",
                "default": 3,
                "range": [2, 20],
                "description": "Number of clusters"
            },
            "init": {
                "type": "categorical",
                "default": "k-means++",
                "choices": ["k-means++", "random"],
                "description": "Initialization method"
            },
            "n_init": {
                "type": "integer",
                "default": 10,
                "range": [1, 50],
                "description": "Number of initializations"
            },
            "max_iter": {
                "type": "integer",
                "default": 300,
                "range": [100, 1000],
                "description": "Maximum iterations"
            },
            "random_state": {
                "type": "integer",
                "default": 42,
                "description": "Random seed"
            }
        },
        default_config={
            "n_clusters": 3,
            "init": "k-means++",
            "n_init": 10,
            "max_iter": 300,
            "random_state": 42
        },
        supports_feature_importance=False,
        requires_scaling=True,
        handles_missing_values=False,
        tags=["fast", "scalable"],
    ),

    ModelInfo(
        model_id="dbscan",
        name="DBSCAN",
        description="Density-based clustering. Finds arbitrary-shaped clusters and identifies outliers.",
        task_type=TaskType.CLUSTERING,
        category=ModelCategory.DENSITY_BASED,
        sklearn_class="sklearn.cluster.DBSCAN",
        hyperparameters={
            "eps": {
                "type": "float",
                "default": 0.5,
                "range": [0.1, 10.0],
                "description": "Maximum distance between samples in a neighborhood"
            },
            "min_samples": {
                "type": "integer",
                "default": 5,
                "range": [2, 50],
                "description": "Minimum samples in neighborhood to form core point"
            },
            "metric": {
                "type": "categorical",
                "default": "euclidean",
                "choices": ["euclidean", "manhattan", "cosine"],
                "description": "Distance metric"
            },
            "algorithm": {
                "type": "categorical",
                "default": "auto",
                "choices": ["auto", "ball_tree", "kd_tree", "brute"],
                "description": "Algorithm for nearest neighbors"
            }
        },
        default_config={
            "eps": 0.5,
            "min_samples": 5,
            "metric": "euclidean",
            "algorithm": "auto"
        },
        supports_feature_importance=False,
        requires_scaling=True,
        handles_missing_values=False,
        tags=["density_based", "outlier_detection"],
    ),

    ModelInfo(
        model_id="hierarchical_agglomerative",
        name="Agglomerative Clustering",
        description="Bottom-up hierarchical clustering. Builds hierarchy of clusters.",
        task_type=TaskType.CLUSTERING,
        category=ModelCategory.HIERARCHICAL,
        sklearn_class="sklearn.cluster.AgglomerativeClustering",
        hyperparameters={
            "n_clusters": {
                "type": "integer",
                "default": 2,
                "range": [2, 20],
                "description": "Number of clusters"
            },
            "linkage": {
                "type": "categorical",
                "default": "ward",
                "choices": ["ward", "complete", "average", "single"],
                "description": "Linkage criterion"
            },
            "metric": {
                "type": "categorical",
                "default": "euclidean",
                "choices": ["euclidean", "manhattan", "cosine"],
                "description": "Distance metric"
            }
        },
        default_config={"n_clusters": 2, "linkage": "ward", "metric": "euclidean"},
        supports_feature_importance=False,
        requires_scaling=True,
        handles_missing_values=False,
        tags=["hierarchical", "interpretable"],
    ),

    ModelInfo(
        model_id="gaussian_mixture",
        name="Gaussian Mixture Model",
        description="Probabilistic clustering assuming data is from mixture of Gaussian distributions.",
        task_type=TaskType.CLUSTERING,
        category=ModelCategory.PROBABILISTIC,
        sklearn_class="sklearn.mixture.GaussianMixture",
        hyperparameters={
            "n_components": {
                "type": "integer",
                "default": 2,
                "range": [2, 20],
                "description": "Number of mixture components"
            },
            "covariance_type": {
                "type": "categorical",
                "default": "full",
                "choices": ["full", "tied", "diag", "spherical"],
                "description": "Covariance matrix type"
            },
            "max_iter": {
                "type": "integer",
                "default": 100,
                "range": [50, 500],
                "description": "Maximum EM iterations"
            },
            "random_state": {
                "type": "integer",
                "default": 42,
                "description": "Random seed"
            }
        },
        default_config={"n_components": 2, "covariance_type": "full", "max_iter": 100, "random_state": 42},
        supports_feature_importance=False,
        requires_scaling=True,
        handles_missing_values=False,
        tags=["probabilistic", "soft_clustering"],
    ),
]


# ============================================================================
# MODEL REGISTRY
# ============================================================================

class ModelRegistry:
    """Central registry for all available models."""

    def __init__(self):
        """Initialize the model registry."""
        self._models = {}
        self._register_models()

    def _register_models(self):
        """Register all models."""
        all_models = REGRESSION_MODELS + CLASSIFICATION_MODELS + CLUSTERING_MODELS
        for model in all_models:
            self._models[model.model_id] = model

    def get_model(self, model_id: str) -> ModelInfo:
        """Get model by ID."""
        if model_id not in self._models:
            raise ValueError(f"Model '{model_id}' not found in registry")
        return self._models[model_id]

    def get_models_by_task(self, task_type: TaskType) -> List[ModelInfo]:
        """Get all models for a specific task type."""
        return [
            model for model in self._models.values()
            if model.task_type == task_type
        ]

    def get_models_by_category(self, category: ModelCategory) -> List[ModelInfo]:
        """Get all models in a specific category."""
        return [
            model for model in self._models.values()
            if model.category == category
        ]

    def get_all_models(self) -> Dict[str, List[ModelInfo]]:
        """Get all models grouped by task type."""
        return {
            "regression": [m for m in self._models.values() if m.task_type == TaskType.REGRESSION],
            "classification": [m for m in self._models.values() if m.task_type == TaskType.CLASSIFICATION],
            "clustering": [m for m in self._models.values() if m.task_type == TaskType.CLUSTERING],
        }

    def list_model_ids(self) -> List[str]:
        """List all model IDs."""
        return list(self._models.keys())

    def search_models(self, query: str) -> List[ModelInfo]:
        """Search models by name, description, or tags."""
        query_lower = query.lower()
        results = []
        for model in self._models.values():
            if (query_lower in model.name.lower() or
                query_lower in model.description.lower() or
                any(query_lower in tag for tag in model.tags)):
                results.append(model)
        return results


# Global registry instance
model_registry = ModelRegistry()
