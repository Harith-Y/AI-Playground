"""
Evaluation Package

Provides comprehensive model evaluation metrics and visualizations for:
- Classification models (accuracy, precision, recall, F1, AUC-ROC, etc.)
- Confusion matrices with detailed statistics
- Regression models (MAE, MSE, RMSE, R², etc.) [Coming Soon]
- Clustering models (silhouette, inertia, etc.) [Coming Soon]

Modules:
    classification_metrics: Classification evaluation metrics
    confusion_matrix: Confusion matrix computation and analysis
    metrics: General metrics utilities [Placeholder]
    visualizations: Evaluation visualizations [Placeholder]
"""

from app.ml_engine.evaluation.classification_metrics import (
    ClassificationMetricsCalculator,
    ClassificationMetrics,
    calculate_classification_metrics,
)

from app.ml_engine.evaluation.confusion_matrix import (
    ConfusionMatrixCalculator,
    ConfusionMatrixResult,
    compute_confusion_matrix,
)

__all__ = [
    # Classification metrics
    "ClassificationMetricsCalculator",
    "ClassificationMetrics",
    "calculate_classification_metrics",
    # Confusion matrix
    "ConfusionMatrixCalculator",
    "ConfusionMatrixResult",
    "compute_confusion_matrix",
]
