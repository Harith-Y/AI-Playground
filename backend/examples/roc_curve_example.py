"""
Example: Using ROC Curve Module

This script demonstrates how to use the ROC curve module
for binary and multi-class classification analysis.
"""

import numpy as np
from app.ml_engine.evaluation import compute_roc_curve
from app.ml_engine.evaluation.roc_curve import ROCCurveCalculator

print("=" * 80)
print("ROC CURVE EXAMPLES")
print("=" * 80)

# ============================================================================
# Example 1: Binary Classification ROC Curve
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: Binary Classification ROC Curve")
print("=" * 80)

y_true_binary = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
y_score_binary = [0.1, 0.2, 0.4, 0.6, 0.7, 0.9, 0.3, 0.8, 0.35, 0.85]

result_binary = compute_roc_curve(
    y_true=y_true_binary,
    y_score=y_score_binary
)

print(f"\nROC Curve Statistics:")
print(f"  AUC Score: {result_binary.auc_score:.4f}")
print(f"  Optimal Threshold: {result_binary.optimal_threshold:.4f}")
print(f"  Number of Samples: {result_binary.n_samples}")
print(f"  Positive Samples: {result_binary.n_positive}")
print(f"  Negative Samples: {result_binary.n_negative}")

print(f"\nROC Curve Points (first 5):")
for i in range(min(5, len(result_binary.fpr))):
    print(f"  Threshold={result_binary.thresholds[i]:.4f}: "
          f"FPR={result_binary.fpr[i]:.4f}, TPR={result_binary.tpr[i]:.4f}")

# ============================================================================
# Example 2: Perfect vs Random Classifier
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Perfect vs Random Classifier")
print("=" * 80)

# Perfect classifier
y_true_perfect = [0, 0, 0, 1, 1, 1]
y_score_perfect = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

result_perfect = compute_roc_curve(y_true_perfect, y_score_perfect)
print(f"\nPerfect Classifier:")
print(f"  AUC: {result_perfect.auc_score:.4f} (should be 1.0)")

# Random classifier
np.random.seed(42)
y_true_random = np.random.randint(0, 2, 100)
y_score_random = np.random.rand(100)

result_random = compute_roc_curve(y_true_random, y_score_random)
print(f"\nRandom Classifier:")
print(f"  AUC: {result_random.auc_score:.4f} (should be ~0.5)")

# ============================================================================
# Example 3: Multi-Class ROC Curves
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Multi-Class ROC Curves (One-vs-Rest)")
print("=" * 80)

y_true_multi = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
y_score_multi = np.array([
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8],
    [0.7, 0.2, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.2, 0.7],
    [0.6, 0.3, 0.1],
    [0.3, 0.6, 0.1],
    [0.2, 0.3, 0.5],
    [0.75, 0.15, 0.1],
    [0.15, 0.75, 0.1],
    [0.1, 0.15, 0.75]
])
class_names = ['Cat', 'Dog', 'Bird']

result_multi = compute_roc_curve(
    y_true=y_true_multi,
    y_score=y_score_multi,
    multiclass=True,
    class_names=class_names,
    average='both'
)

print(f"\nPer-Class AUC Scores:")
for class_name, class_result in result_multi.per_class.items():
    print(f"  {class_name}: {class_result.auc_score:.4f}")

print(f"\nAveraged AUC Scores:")
print(f"  Micro-Average: {result_multi.micro_average.auc_score:.4f}")
print(f"  Macro-Average: {result_multi.macro_average.auc_score:.4f}")

# ============================================================================
# Example 4: Finding Optimal Operating Points
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Finding Optimal Operating Points")
print("=" * 80)

y_true_opt = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
y_score_opt = [0.1, 0.2, 0.4, 0.6, 0.7, 0.9, 0.3, 0.8, 0.35, 0.85]

result_opt = compute_roc_curve(y_true_opt, y_score_opt)

print(f"\nOptimal Threshold (Youden's J):")
print(f"  Threshold: {result_opt.optimal_threshold:.4f}")
fpr_opt, tpr_opt = result_opt.get_point_at_threshold(result_opt.optimal_threshold)
print(f"  FPR: {fpr_opt:.4f}")
print(f"  TPR: {tpr_opt:.4f}")
print(f"  J-statistic: {tpr_opt - fpr_opt:.4f}")

# Find threshold for specific FPR
target_fpr = 0.1
threshold_at_fpr = result_opt.get_threshold_at_fpr(target_fpr)
print(f"\nThreshold for FPR <= {target_fpr}:")
print(f"  Threshold: {threshold_at_fpr:.4f}")

# Find threshold for specific TPR
target_tpr = 0.9
threshold_at_tpr = result_opt.get_threshold_at_tpr(target_tpr)
print(f"\nThreshold for TPR >= {target_tpr}:")
print(f"  Threshold: {threshold_at_tpr:.4f}")

# ============================================================================
# Example 5: Comparing Multiple Models
# ============================================================================
print("\n" + "=" * 80)
print("Example 5: Comparing Multiple Models")
print("=" * 80)

y_true_compare = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
y_scores_compare = {
    'Logistic Regression': [0.1, 0.2, 0.4, 0.6, 0.7, 0.9, 0.3, 0.8, 0.35, 0.85],
    'Random Forest': [0.15, 0.25, 0.35, 0.65, 0.75, 0.95, 0.28, 0.82, 0.32, 0.88],
    'SVM': [0.12, 0.22, 0.38, 0.62, 0.72, 0.92, 0.31, 0.81, 0.33, 0.86]
}

calculator = ROCCurveCalculator()
results_compare = calculator.compare_models(y_true_compare, y_scores_compare)

print(f"\nModel Comparison (AUC Scores):")
for model_name, model_result in sorted(
    results_compare.items(),
    key=lambda x: x[1].auc_score,
    reverse=True
):
    print(f"  {model_name}: {model_result.auc_score:.4f}")

# ============================================================================
# Example 6: DataFrame Conversion
# ============================================================================
print("\n" + "=" * 80)
print("Example 6: DataFrame Conversion")
print("=" * 80)

df = result_binary.to_dataframe()
print(f"\nROC Curve as DataFrame (first 5 rows):")
print(df.head())

# ============================================================================
# Example 7: Dictionary Export (for API responses)
# ============================================================================
print("\n" + "=" * 80)
print("Example 7: Dictionary Export (JSON-serializable)")
print("=" * 80)

result_dict = result_binary.to_dict()
print(f"\nROC Curve as Dictionary (keys):")
for key in result_dict.keys():
    if key in ['fpr', 'tpr', 'thresholds']:
        print(f"  {key}: list of {len(result_dict[key])} values")
    else:
        print(f"  {key}: {result_dict[key]}")

print("\n" + "=" * 80)
print("Examples completed successfully!")
print("=" * 80)
