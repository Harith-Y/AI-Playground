"""
Example usage of oversampling methods for handling imbalanced datasets.

This script demonstrates how to use SMOTE, BorderlineSMOTE, and ADASYN
to create synthetic samples and balance class distributions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from app.ml_engine.preprocessing.oversampling import (
    SMOTE,
    BorderlineSMOTE,
    ADASYN,
)


def create_imbalanced_dataset():
    """Create a synthetic imbalanced dataset for demonstration."""
    np.random.seed(42)

    # Minority class (class 0): 50 samples
    X_minority = np.random.randn(50, 2) - 2
    y_minority = np.zeros(50)

    # Majority class (class 1): 200 samples
    X_majority = np.random.randn(200, 2) + 2
    y_majority = np.ones(200)

    # Combine
    X = pd.DataFrame(
        np.vstack([X_minority, X_majority]),
        columns=['feature1', 'feature2']
    )
    y = pd.Series(np.hstack([y_minority, y_majority]))

    return X, y


def print_class_distribution(y, title="Class Distribution"):
    """Print class distribution statistics."""
    print(f"\n{title}")
    print("-" * 60)
    counts = y.value_counts().sort_index()
    total = len(y)

    for cls, count in counts.items():
        percentage = (count / total) * 100
        print(f"  Class {int(cls)}: {count:4d} samples ({percentage:5.1f}%)")

    print(f"  Total:     {total:4d} samples")
    print(f"  Imbalance ratio: {counts.max() / counts.min():.2f}:1")


def visualize_results(X_original, y_original, results, title="Oversampling Comparison"):
    """Visualize original and resampled datasets."""
    n_methods = len(results) + 1
    fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(12, 8))
    axes = axes.flatten()

    # Plot original data
    axes[0].scatter(
        X_original[y_original == 0]['feature1'],
        X_original[y_original == 0]['feature2'],
        c='blue', label='Class 0 (minority)', alpha=0.6, s=30, edgecolors='k', linewidths=0.5
    )
    axes[0].scatter(
        X_original[y_original == 1]['feature1'],
        X_original[y_original == 1]['feature2'],
        c='red', label='Class 1 (majority)', alpha=0.6, s=30, edgecolors='k', linewidths=0.5
    )
    axes[0].set_title(f'Original\n(0:{(y_original==0).sum()}, 1:{(y_original==1).sum()})',
                      fontweight='bold')
    axes[0].legend(loc='best', fontsize=8)
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].grid(True, alpha=0.3)

    # Plot resampled datasets
    for idx, (method_name, X_res, y_res) in enumerate(results, 1):
        # Original samples
        original_mask_0 = (y_res == 0).values[:50]  # First 50 are original minority
        original_mask_1 = (y_res == 1).values[:200]  # First 200 after minority are original majority

        # Get indices
        minority_indices = np.where(y_res == 0)[0]
        majority_indices = np.where(y_res == 1)[0]

        # Original minority
        axes[idx].scatter(
            X_res.iloc[minority_indices[:50]]['feature1'],
            X_res.iloc[minority_indices[:50]]['feature2'],
            c='blue', alpha=0.6, s=30, edgecolors='k', linewidths=0.5,
            label='Original minority'
        )

        # Synthetic minority
        if len(minority_indices) > 50:
            axes[idx].scatter(
                X_res.iloc[minority_indices[50:]]['feature1'],
                X_res.iloc[minority_indices[50:]]['feature2'],
                c='cyan', marker='s', alpha=0.4, s=20,
                label='Synthetic minority'
            )

        # Majority (unchanged)
        axes[idx].scatter(
            X_res.iloc[majority_indices[:200]]['feature1'],
            X_res.iloc[majority_indices[:200]]['feature2'],
            c='red', alpha=0.6, s=30, edgecolors='k', linewidths=0.5,
            label='Original majority'
        )

        axes[idx].set_title(f'{method_name}\n(0:{(y_res==0).sum()}, 1:{(y_res==1).sum()})',
                           fontweight='bold')
        axes[idx].legend(loc='best', fontsize=7)
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('oversampling_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to 'oversampling_comparison.png'")


def main():
    """Run the oversampling examples."""
    print("=" * 80)
    print("Oversampling Methods Example")
    print("=" * 80)

    # Create imbalanced dataset
    X, y = create_imbalanced_dataset()
    print_class_distribution(y, "Original Dataset")

    print("\n" + "=" * 80)
    print("Example 1: SMOTE (Synthetic Minority Over-sampling TEchnique)")
    print("=" * 80)

    # Standard SMOTE with auto strategy
    print("\n1a. Auto strategy (balance to majority class)")
    smote_auto = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
    X_smote_auto, y_smote_auto = smote_auto.fit_resample(X, y)
    print_class_distribution(y_smote_auto, "After SMOTE (auto)")

    # SMOTE with ratio
    print("\n1b. Ratio strategy (minority:majority = 1:2)")
    smote_ratio = SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=42)
    X_smote_ratio, y_smote_ratio = smote_ratio.fit_resample(X, y)
    print_class_distribution(y_smote_ratio, "After SMOTE (ratio=0.5)")

    # SMOTE with different k_neighbors
    print("\n1c. Different k_neighbors values")
    for k in [3, 5, 10]:
        smote_k = SMOTE(k_neighbors=k, random_state=42)
        X_k, y_k = smote_k.fit_resample(X, y)
        print(f"  k={k:2d}: Generated {len(X_k) - len(X)} synthetic samples")

    print("\n" + "=" * 80)
    print("Example 2: Borderline-SMOTE")
    print("=" * 80)

    # Borderline-SMOTE version 1
    print("\n2a. Borderline-1 (focus on borderline minority samples)")
    bsmote1 = BorderlineSMOTE(kind='borderline-1', k_neighbors=5, m_neighbors=10, random_state=42)
    X_bsmote1, y_bsmote1 = bsmote1.fit_resample(X, y)
    print_class_distribution(y_bsmote1, "After Borderline-SMOTE-1")

    # Borderline-SMOTE version 2
    print("\n2b. Borderline-2 (can use majority neighbors)")
    bsmote2 = BorderlineSMOTE(kind='borderline-2', k_neighbors=5, m_neighbors=10, random_state=42)
    X_bsmote2, y_bsmote2 = bsmote2.fit_resample(X, y)
    print_class_distribution(y_bsmote2, "After Borderline-SMOTE-2")

    print("\n" + "=" * 80)
    print("Example 3: ADASYN (Adaptive Synthetic Sampling)")
    print("=" * 80)

    # ADASYN with auto strategy
    print("\n3a. Auto strategy (adaptive density-based sampling)")
    adasyn_auto = ADASYN(k_neighbors=5, random_state=42)
    X_adasyn_auto, y_adasyn_auto = adasyn_auto.fit_resample(X, y)
    print_class_distribution(y_adasyn_auto, "After ADASYN (auto)")

    # ADASYN with ratio
    print("\n3b. Ratio strategy")
    adasyn_ratio = ADASYN(sampling_strategy=0.5, k_neighbors=5, random_state=42)
    X_adasyn_ratio, y_adasyn_ratio = adasyn_ratio.fit_resample(X, y)
    print_class_distribution(y_adasyn_ratio, "After ADASYN (ratio=0.5)")

    print("\n" + "=" * 80)
    print("Example 4: Comparing All Methods")
    print("=" * 80)

    # Collect all results for comparison
    results = [
        ("SMOTE", X_smote_auto, y_smote_auto),
        ("Borderline-1", X_bsmote1, y_bsmote1),
        ("Borderline-2", X_bsmote2, y_bsmote2),
        ("ADASYN", X_adasyn_auto, y_adasyn_auto),
    ]

    print("\nMethod Comparison:")
    print("-" * 80)
    print(f"{'Method':<20} {'Original':<12} {'Resampled':<12} {'Synthetic':<12} {'Balanced'}")
    print("-" * 80)

    original_size = len(X)
    for method_name, X_res, y_res in results:
        resampled_size = len(X_res)
        synthetic_count = resampled_size - original_size
        is_balanced = y_res.value_counts()[0] == y_res.value_counts()[1]

        print(f"{method_name:<20} {original_size:<12} {resampled_size:<12} "
              f"{synthetic_count:<12} {'Yes' if is_balanced else 'No'}")

    print("\n" + "=" * 80)
    print("Example 5: Use Case Recommendations")
    print("=" * 80)

    print("""
Use Case Recommendations:

1. SMOTE (Standard):
   ✓ General-purpose oversampling
   ✓ Simple and effective baseline
   ✓ Works well for most imbalanced datasets
   ✓ Fast and computationally efficient
   ✗ May create noisy samples in overlapping regions
   ✗ Treats all minority samples equally

2. Borderline-SMOTE:
   ✓ Focuses on decision boundary (most important region)
   ✓ Reduces noise by ignoring safe minority samples
   ✓ Version 1: Conservative (minority neighbors only)
   ✓ Version 2: Aggressive (includes majority neighbors)
   ✗ Slightly more complex than standard SMOTE
   ✗ Requires careful tuning of m_neighbors

3. ADASYN:
   ✓ Adaptive: generates more samples for difficult instances
   ✓ Density-aware: focuses on hard-to-learn regions
   ✓ Often produces best classifier performance
   ✓ Automatically adjusts to local data distribution
   ✗ Most computationally expensive
   ✗ Can generate many samples in noisy regions

When to Use What:

├─ Need quick baseline? → Use SMOTE
├─ Classes well-separated? → Use SMOTE
├─ Have overlapping classes? → Use Borderline-SMOTE or ADASYN
├─ Want to focus on boundary? → Use Borderline-SMOTE-1
├─ Want best performance? → Try ADASYN (if you have time)
└─ Production model? → Compare all three and choose best
    """)

    print("\n" + "=" * 80)
    print("Example 6: Best Practices")
    print("=" * 80)

    print("""
Best Practices for Oversampling:

1. Feature Scaling:
   - Always scale features before oversampling
   - SMOTE uses distance metrics (Euclidean)
   - Unscaled features can bias neighbor selection

2. When to Apply:
   - Apply INSIDE cross-validation folds
   - Never oversample before splitting train/test
   - Avoid data leakage

3. Evaluation:
   - Use appropriate metrics: F1-score, ROC-AUC, PR-AUC
   - Don't rely on accuracy alone
   - Test on original imbalanced test set

4. Hyperparameters:
   - k_neighbors: 3-10 (smaller k for small minority class)
   - m_neighbors (Borderline): 10-20
   - sampling_strategy: Start with 'auto', adjust if needed

5. Combination with Undersampling:
   - Can combine SMOTE + Tomek Links
   - Can combine SMOTE + Random Undersampling
   - Creates balanced dataset with clean boundaries

6. Comparison:
   - Try multiple methods
   - Use cross-validation to compare
   - Choose based on validation performance
    """)

    print("\n" + "=" * 80)
    print("Example 7: Practical Application - Fraud Detection")
    print("=" * 80)

    print("""
Scenario: Credit Card Fraud Detection
- Total transactions: 10,000
- Fraudulent (minority): 50 (0.5%)
- Normal (majority): 9,950 (99.5%)
- Imbalance ratio: 199:1

Approach 1: SMOTE Only
Before: {fraud: 50, normal: 9,950}
After:  {fraud: 9,950, normal: 9,950}
Result: Massive dataset (19,900 samples)
        Many synthetic frauds (may introduce noise)

Approach 2: SMOTE + Ratio
Strategy: 0.1 (fraud:normal = 1:10)
After:  {fraud: 995, normal: 9,950}
Result: More manageable size
        Still enough frauds to learn patterns

Approach 3: Combined (Undersampling + Oversampling)
Step 1 - Undersample normal to 1,000
After:  {fraud: 50, normal: 1,000}

Step 2 - SMOTE on fraud
After:  {fraud: 1,000, normal: 1,000}
Result: Balanced, manageable, diverse
        (Recommended approach!)

Approach 4: ADASYN + Undersampling
Step 1 - Undersample normal to 500
After:  {fraud: 50, normal: 500}

Step 2 - ADASYN (adaptive)
After:  {fraud: 500, normal: 500}
Result: Focuses on hardest fraud cases
        Best for maximizing fraud detection
    """)

    print("\n" + "=" * 80)
    print("Generating visualization...")
    print("=" * 80)

    # Generate visualization
    try:
        visualize_results(
            X, y,
            [
                ("SMOTE", X_smote_auto, y_smote_auto),
                ("Borderline-1", X_bsmote1, y_bsmote1),
                ("ADASYN", X_adasyn_auto, y_adasyn_auto),
            ]
        )
    except Exception as e:
        print(f"Could not generate visualization: {e}")
        print("(This is optional - matplotlib may not be configured for display)")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. SMOTE creates synthetic samples via interpolation")
    print("  2. Borderline-SMOTE focuses on decision boundary")
    print("  3. ADASYN adapts to local difficulty")
    print("  4. Choose method based on your specific problem")
    print("  5. Always validate with proper metrics and CV")
    print("=" * 80)


if __name__ == "__main__":
    main()
