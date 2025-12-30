"""
Evaluation Package

Provides comprehensive model evaluation metrics and visualizations for:
- Classification models (accuracy, precision, recall, F1, AUC-ROC, etc.)
- Regression models (MAE, MSE, RMSE, R², etc.) [Coming Soon]
- Clustering models (silhouette, inertia, etc.) [Coming Soon]

Modules:
    classification_metrics: Classification evaluation metrics
    metrics: General metrics utilities [Placeholder]
    visualizations: Evaluation visualizations [Placeholder]
"""

from app.ml_engine.evaluation.classification_metrics import (
    ClassificationMetricsCalculator,
    ClassificationMetrics,
    calculate_classification_metrics,
)

__all__ = [
    "ClassificationMetricsCalculator",
    "ClassificationMetrics",
    "calculate_classification_metrics",
]
