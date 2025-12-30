"""
Example: Using Confusion Matrix Module

This script demonstrates how to use the confusion matrix module
for detailed classification analysis.
"""

import numpy as np
from app.ml_engine.evaluation import compute_confusion_matrix

print("=" * 80)
print("CONFUSION MATRIX EXAMPLES")
print("=" * 80)

# ============================================================================
# Example 1: Basic Binary Classification
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: Basic Binary Classification")
print("=" * 80)

y_true_binary = [0, 1, 0, 1, 0, 1, 0, 1]
y_pred_binary = [0, 1, 0, 0, 0, 1, 1, 1]

result_binary = compute_confusion_matrix(
    y_true=y_true_binary,
    y_pred=y_pred_binary,
    class_names=['Negative', 'Positive']
)

print(f"\nConfusion Matrix:")
print(result_binary.matrix)
print(f"\nClass Names: {result_binary.class_names}")

print(f"\nOverall Statistics:")
print(f"  Total Samples: {result_binary.overall_stats['total_samples']}")
print(f"  Correct: {result_binary.overall_stats['correct_predictions']}")
print(f"  Incorrect: {result_binary.overall_stats['incorrect_predictions']}")
print(f"  Accuracy: {result_binary.overall_stats['accuracy']:.4f}")
print(f"  Error Rate: {result_binary.overall_stats['error_rate']:.4f}")

print(f"\nPer-Class Statistics:")
for class_name, stats in result_binary.per_class_stats.items():
    print(f"\n  {class_name}:")
    print(f"    True Positives:  {stats['true_positives']}")
    print(f"    False Positives: {stats['false_positives']}")
    print(f"    True Negatives:  {stats['true_negatives']}")
    print(f"    False Negatives: {stats['false_negatives']}")
    print(f"    Sensitivity:     {stats['sensitivity']:.4f}")
    print(f"    Specificity:     {stats['specificity']:.4f}")
    print(f"    Precision:       {stats['precision']:.4f}")
    print(f"    F1 Score:        {stats['f1_score']:.4f}")

# ============================================================================
# Example 2: Multi-Class with Normalization
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Multi-Class Classification with Normalization")
print("=" * 80)

y_true_multi = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred_multi = [0, 1, 2, 0, 2, 2, 1, 1, 2, 0, 1, 1]
class_names = ['Cat', 'Dog', 'Bird']

result_multi = compute_confusion_matrix(
    y_true=y_true_multi,
    y_pred=y_pred_multi,
    class_names=class_names,
    normalize='true'  # Normalize over true labels (rows)
)

print(f"\nRaw Confusion Matrix:")
print(result_multi.matrix)

print(f"\nNormalized Confusion Matrix (rows sum to 1):")
print(result_multi.normalized_matrix)

print(f"\nMisclassification Matrix (errors only):")
print(result_multi.misclassification_matrix)

# ============================================================================
# Example 3: Error Analysis
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Error Analysis")
print("=" * 80)

from app.ml_engine.evaluation.confusion_matrix import ConfusionMatrixCalculator

y_true_errors = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred_errors = [0, 1, 2, 1, 2, 2, 1, 1, 2, 0, 2, 1]

calculator = ConfusionMatrixCalculator()
error_analysis = calculator.analyze_errors(
    y_true=y_true_errors,
    y_pred=y_pred_errors,
    class_names=['Cat', 'Dog', 'Bird'],
    top_n=5
)

print(f"\nTop Misclassifications:")
for i, error in enumerate(error_analysis['top_misclassifications'], 1):
    print(f"  {i}. {error['true_class']} -> {error['predicted_class']}: "
          f"{error['count']} times ({error['percentage']:.1f}%)")

print(f"\nError Rates by Class:")
for error_info in error_analysis['error_rates_by_class']:
    print(f"  {error_info['class']}: "
          f"{error_info['error_count']}/{error_info['total_samples']} "
          f"({error_info['error_rate']:.1f}%)")

print(f"\nTotal Errors: {error_analysis['total_errors']}/{error_analysis['total_samples']}")

# ============================================================================
# Example 4: Cost-Sensitive Analysis
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Cost-Sensitive Analysis")
print("=" * 80)

y_true_cost = [0, 1, 0, 1, 0, 1, 0, 1]
y_pred_cost = [0, 1, 0, 0, 0, 1, 1, 1]

# Define cost matrix
# False Negative (missing a positive) costs 10
# False Positive (false alarm) costs 1
cost_matrix = np.array([
    [0, 1],   # True 0: correct=0, predict as 1=1
    [10, 0]   # True 1: predict as 0=10, correct=0
])

cost_result = calculator.compute_cost_sensitive(
    y_true=y_true_cost,
    y_pred=y_pred_cost,
    cost_matrix=cost_matrix,
    class_names=['Negative', 'Positive']
)

print(f"\nCost Matrix:")
print(cost_matrix)

print(f"\nCost Analysis:")
print(f"  Total Cost: {cost_result['total_cost']:.2f}")
print(f"  Average Cost per Sample: {cost_result['average_cost_per_sample']:.2f}")

print(f"\nCost by True Class:")
for i, cost in enumerate(cost_result['cost_by_true_class']):
    print(f"  Class {i}: {cost:.2f}")

print(f"\nCost by Predicted Class:")
for i, cost in enumerate(cost_result['cost_by_pred_class']):
    print(f"  Class {i}: {cost:.2f}")

# ============================================================================
# Example 5: DataFrame Conversion
# ============================================================================
print("\n" + "=" * 80)
print("Example 5: DataFrame Conversion")
print("=" * 80)

result_df = compute_confusion_matrix(
    y_true=[0, 1, 2, 0, 1, 2],
    y_pred=[0, 1, 2, 1, 1, 2],
    class_names=['Cat', 'Dog', 'Bird']
)

df = result_df.to_dataframe()
print(f"\nConfusion Matrix as DataFrame:")
print(df)

# ============================================================================
# Example 6: Dictionary Export (for API responses)
# ============================================================================
print("\n" + "=" * 80)
print("Example 6: Dictionary Export (JSON-serializable)")
print("=" * 80)

result_dict = result_binary.to_dict()
print(f"\nConfusion Matrix as Dictionary (first 5 keys):")
for i, (key, value) in enumerate(result_dict.items()):
    if i >= 5:
        break
    if isinstance(value, (int, float)):
        print(f"  {key}: {value}")
    elif isinstance(value, list):
        print(f"  {key}: {value}")
    else:
        print(f"  {key}: <{type(value).__name__}>")

print("\n" + "=" * 80)
print("Examples completed successfully!")
print("=" * 80)
