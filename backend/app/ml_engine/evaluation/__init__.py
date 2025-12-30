"""
Evaluation Package

Provides comprehensive model evaluation metrics and visualizations for:
- Classification models (accuracy, precision, recall, F1, AUC-ROC, etc.)
- Confusion matrices with detailed statistics
- ROC curves for binary and multi-class classification
- PR curves for binary and multi-class classification
- Regression models (MAE, MSE, RMSE, R², etc.) [Coming Soon]
- Clustering models (silhouette, inertia, etc.)

Modules:
    classification_metrics: Classification evaluation metrics
    confusion_matrix: Confusion matrix computation and analysis
    roc_curve: ROC curve data generation
    pr_curve: Precision-Recall curve data generation
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

from app.ml_engine.evaluation.residual_analysis import (
    ResidualAnalysisCalculator,
    ResidualAnalysisResult,
    analyze_residuals,
)

from app.ml_engine.evaluation.pr_curve import (
    PRCurveCalculator,
    PRCurveResult,
    MultiClassPRResult,
    compute_pr_curve,
)

from app.ml_engine.evaluation.actual_vs_predicted import (
    ActualVsPredictedAggregator,
    ActualVsPredictedResult,
    aggregate_actual_vs_predicted,
)

from app.ml_engine.evaluation.feature_importance import (
    FeatureImportanceCalculator,
    FeatureImportanceResult,
    calculate_feature_importance,
)

from app.ml_engine.evaluation.clustering_metrics import (
    ClusteringMetricsCalculator,
    ClusteringMetricsResult,
    calculate_clustering_metrics,
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
    # Residual analysis
    "ResidualAnalysisCalculator",
    "ResidualAnalysisResult",
    "analyze_residuals",
    # PR curve
    "PRCurveCalculator",
    "PRCurveResult",
    "MultiClassPRResult",
    "compute_pr_curve",
    # Actual vs predicted aggregation
    "ActualVsPredictedAggregator",
    "ActualVsPredictedResult",
    "aggregate_actual_vs_predicted",
    # Clustering metrics
    "ClusteringMetricsCalculator",
    "ClusteringMetricsResult",
    "calculate_clustering_metrics",
    # Feature importance
    "FeatureImportanceCalculator",
    "FeatureImportanceResult",
    "calculate_feature_importance",
]
