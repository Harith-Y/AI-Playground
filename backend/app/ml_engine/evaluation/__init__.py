"""
Evaluation Package

Provides comprehensive model evaluation metrics and visualizations for:
- Classification models (accuracy, precision, recall, F1, AUC-ROC, etc.)
- Confusion matrices with detailed statistics
- ROC curves for binary and multi-class classification
- Regression models (MAE, MSE, RMSE, R², etc.) [Coming Soon]
- Clustering models (silhouette, inertia, etc.) [Coming Soon]

Modules:
    classification_metrics: Classification evaluation metrics
    confusion_matrix: Confusion matrix computation and analysis
    roc_curve: ROC curve data generation
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

from app.ml_engine.evaluation.roc_curve import (
    ROCCurveCalculator,
    ROCCurveResult,
    MultiClassROCResult,
    compute_roc_curve,
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
    # ROC curve
    "ROCCurveCalculator",
    "ROCCurveResult",
    "MultiClassROCResult",
    "compute_roc_curve",
]
