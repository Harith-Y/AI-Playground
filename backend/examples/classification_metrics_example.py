"""
Example: Using Classification Metrics Module

This script demonstrates how to use the classification metrics module
for evaluating binary and multi-class classification models.
"""

import numpy as np
from app.ml_engine.evaluation import calculate_classification_metrics

print("=" * 80)
print("CLASSIFICATION METRICS EXAMPLES")
print("=" * 80)

# ============================================================================
# Example 1: Binary Classification
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: Binary Classification")
print("=" * 80)

y_true_binary = [0, 1, 1, 0, 1, 0, 1, 0, 1, 1]
y_pred_binary = [0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
y_proba_binary = [
    [0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.8, 0.2], [0.1, 0.9],
    [0.4, 0.6], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]
]

metrics_binary = calculate_classification_metrics(
    y_true=y_true_binary,
    y_pred=y_pred_binary,
    y_proba=y_proba_binary,
    average='binary',
    include_advanced=True
)

print(f"\nBasic Metrics:")
print(f"  Accuracy:  {metrics_binary.accuracy:.4f}")
print(f"  Precision: {metrics_binary.precision:.4f}")
print(f"  Recall:    {metrics_binary.recall:.4f}")
print(f"  F1 Score:  {metrics_binary.f1_score:.4f}")

print(f"\nProbability-Based Metrics:")
print(f"  AUC-ROC:   {metrics_binary.auc_roc:.4f}")
print(f"  AUC-PR:    {metrics_binary.auc_pr:.4f}")
print(f"  Log Loss:  {metrics_binary.log_loss:.4f}")

print(f"\nAdvanced Metrics:")
print(f"  Balanced Accuracy: {metrics_binary.balanced_accuracy:.4f}")
print(f"  Matthews Corr:     {metrics_binary.matthews_corrcoef:.4f}")
print(f"  Cohen's Kappa:     {metrics_binary.cohen_kappa:.4f}")

print(f"\nConfusion Matrix:")
print(metrics_binary.confusion_matrix)

# ============================================================================
# Example 2: Multi-Class Classification
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Multi-Class Classification")
print("=" * 80)

y_true_multi = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred_multi = [0, 1, 2, 0, 2, 2, 1, 1, 2, 0, 1, 1]
y_proba_multi = np.array([
    [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
    [0.7, 0.2, 0.1], [0.2, 0.3, 0.5], [0.1, 0.2, 0.7],
    [0.4, 0.5, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8],
    [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.6, 0.2]
])
class_names = ['Cat', 'Dog', 'Bird']

metrics_multi = calculate_classification_metrics(
    y_true=y_true_multi,
    y_pred=y_pred_multi,
    y_proba=y_proba_multi,
    average='weighted',
    class_names=class_names,
    include_per_class=True,
    include_advanced=True
)

print(f"\nOverall Metrics (Weighted Average):")
print(f"  Accuracy:  {metrics_multi.accuracy:.4f}")
print(f"  Precision: {metrics_multi.precision:.4f}")
print(f"  Recall:    {metrics_multi.recall:.4f}")
print(f"  F1 Score:  {metrics_multi.f1_score:.4f}")
print(f"  AUC-ROC:   {metrics_multi.auc_roc:.4f}")

print(f"\nPer-Class Metrics:")
for class_name, class_metrics in metrics_multi.per_class_metrics.items():
    print(f"\n  {class_name}:")
    print(f"    Precision: {class_metrics['precision']:.4f}")
    print(f"    Recall:    {class_metrics['recall']:.4f}")
    print(f"    F1 Score:  {class_metrics['f1_score']:.4f}")
    print(f"    Support:   {class_metrics['support']}")
    if 'auc_roc' in class_metrics:
        print(f"    AUC-ROC:   {class_metrics['auc_roc']:.4f}")

print(f"\nConfusion Matrix:")
print(metrics_multi.confusion_matrix)
print(f"Classes: {class_names}")

# ============================================================================
# Example 3: Imbalanced Dataset
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Imbalanced Dataset (90% class 0, 10% class 1)")
print("=" * 80)

# Create imbalanced dataset
y_true_imbalanced = [0] * 90 + [1] * 10
y_pred_imbalanced = [0] * 85 + [1] * 5 + [0] * 10

metrics_imbalanced = calculate_classification_metrics(
    y_true=y_true_imbalanced,
    y_pred=y_pred_imbalanced,
    average='binary',
    include_advanced=True
)

print(f"\nMetrics:")
print(f"  Accuracy:          {metrics_imbalanced.accuracy:.4f} (misleading due to imbalance)")
print(f"  Balanced Accuracy: {metrics_imbalanced.balanced_accuracy:.4f} (more realistic)")
print(f"  Precision:         {metrics_imbalanced.precision:.4f}")
print(f"  Recall:            {metrics_imbalanced.recall:.4f}")
print(f"  F1 Score:          {metrics_imbalanced.f1_score:.4f}")

print(f"\nConfusion Matrix:")
print(metrics_imbalanced.confusion_matrix)
print("Note: High accuracy but low recall for minority class!")

# ============================================================================
# Example 4: Export to Dictionary (for API responses)
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Export to Dictionary (JSON-serializable)")
print("=" * 80)

metrics_dict = metrics_binary.to_dict()
print(f"\nMetrics as dictionary (first 5 keys):")
for i, (key, value) in enumerate(metrics_dict.items()):
    if i >= 5:
        break
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

print("\n" + "=" * 80)
print("Examples completed successfully!")
print("=" * 80)
