"""
Example usage of undersampling methods for handling imbalanced datasets.

This script demonstrates how to use RandomUnderSampler, NearMissUnderSampler,
and TomekLinksRemover to address class imbalance problems.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from app.ml_engine.preprocessing.undersampling import (
    RandomUnderSampler,
    NearMissUnderSampler,
    TomekLinksRemover,
)


def create_imbalanced_dataset():
    """Create a synthetic imbalanced dataset for demonstration."""
    np.random.seed(42)

    # Minority class (class 0): 100 samples
    X_minority = np.random.randn(100, 2) - 2
    y_minority = np.zeros(100)

    # Majority class (class 1): 400 samples
    X_majority = np.random.randn(400, 2) + 2
    y_majority = np.ones(400)

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


def visualize_results(X_original, y_original, results, title="Undersampling Comparison"):
    """Visualize original and resampled datasets."""
    n_methods = len(results) + 1
    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4))

    # Plot original data
    axes[0].scatter(
        X_original[y_original == 0]['feature1'],
        X_original[y_original == 0]['feature2'],
        c='blue', label='Class 0 (minority)', alpha=0.6, s=20
    )
    axes[0].scatter(
        X_original[y_original == 1]['feature1'],
        X_original[y_original == 1]['feature2'],
        c='red', label='Class 1 (majority)', alpha=0.6, s=20
    )
    axes[0].set_title(f'Original\n(0:{(y_original==0).sum()}, 1:{(y_original==1).sum()})')
    axes[0].legend()
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')

    # Plot resampled datasets
    for idx, (method_name, X_res, y_res) in enumerate(results, 1):
        axes[idx].scatter(
            X_res[y_res == 0]['feature1'],
            X_res[y_res == 0]['feature2'],
            c='blue', label='Class 0', alpha=0.6, s=20
        )
        axes[idx].scatter(
            X_res[y_res == 1]['feature1'],
            X_res[y_res == 1]['feature2'],
            c='red', label='Class 1', alpha=0.6, s=20
        )
        axes[idx].set_title(f'{method_name}\n(0:{(y_res==0).sum()}, 1:{(y_res==1).sum()})')
        axes[idx].legend()
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('undersampling_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to 'undersampling_comparison.png'")


def main():
    """Run the undersampling examples."""
    print("=" * 80)
    print("Undersampling Methods Example")
    print("=" * 80)

    # Create imbalanced dataset
    X, y = create_imbalanced_dataset()
    print_class_distribution(y, "Original Dataset")

    print("\n" + "=" * 80)
    print("Example 1: Random Undersampling")
    print("=" * 80)

    # Random undersampling with auto strategy
    print("\n1a. Auto strategy (balance to minority class)")
    rus_auto = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_rus_auto, y_rus_auto = rus_auto.fit_resample(X, y)
    print_class_distribution(y_rus_auto, "After Random Undersampling (auto)")

    # Random undersampling with ratio
    print("\n1b. Ratio strategy (minority:majority = 1:2)")
    rus_ratio = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_rus_ratio, y_rus_ratio = rus_ratio.fit_resample(X, y)
    print_class_distribution(y_rus_ratio, "After Random Undersampling (ratio=0.5)")

    # Random undersampling with custom counts
    print("\n1c. Custom strategy (specify exact counts)")
    rus_custom = RandomUnderSampler(sampling_strategy={0: 80, 1: 150}, random_state=42)
    X_rus_custom, y_rus_custom = rus_custom.fit_resample(X, y)
    print_class_distribution(y_rus_custom, "After Random Undersampling (custom)")

    print("\n" + "=" * 80)
    print("Example 2: NearMiss Undersampling")
    print("=" * 80)

    # NearMiss Version 1
    print("\n2a. NearMiss-1 (majority samples closest to minority)")
    nm1 = NearMissUnderSampler(version=1, n_neighbors=3, random_state=42)
    X_nm1, y_nm1 = nm1.fit_resample(X, y)
    print_class_distribution(y_nm1, "After NearMiss-1")

    # NearMiss Version 2
    print("\n2b. NearMiss-2 (majority samples closest to farthest minority)")
    nm2 = NearMissUnderSampler(version=2, n_neighbors=3, random_state=42)
    X_nm2, y_nm2 = nm2.fit_resample(X, y)
    print_class_distribution(y_nm2, "After NearMiss-2")

    # NearMiss Version 3
    print("\n2c. NearMiss-3 (majority samples nearest to each minority)")
    nm3 = NearMissUnderSampler(version=3, n_neighbors=3, random_state=42)
    X_nm3, y_nm3 = nm3.fit_resample(X, y)
    print_class_distribution(y_nm3, "After NearMiss-3")

    print("\n" + "=" * 80)
    print("Example 3: Tomek Links Removal")
    print("=" * 80)

    # Tomek Links with auto strategy
    print("\n3a. Auto strategy (remove only majority class links)")
    tl_auto = TomekLinksRemover(sampling_strategy='auto')
    X_tl_auto, y_tl_auto = tl_auto.fit_resample(X, y)
    print_class_distribution(y_tl_auto, "After Tomek Links (auto)")
    print(f"  Samples removed: {len(X) - len(X_tl_auto)}")

    # Tomek Links with all strategy
    print("\n3b. All strategy (remove both samples from links)")
    tl_all = TomekLinksRemover(sampling_strategy='all')
    X_tl_all, y_tl_all = tl_all.fit_resample(X, y)
    print_class_distribution(y_tl_all, "After Tomek Links (all)")
    print(f"  Samples removed: {len(X) - len(X_tl_all)}")

    print("\n" + "=" * 80)
    print("Example 4: Combined Approach")
    print("=" * 80)
    print("\nCombining Tomek Links + Random Undersampling")
    print("(First clean boundaries, then balance)")

    # Step 1: Remove Tomek links
    tl = TomekLinksRemover(sampling_strategy='auto')
    X_cleaned, y_cleaned = tl.fit_resample(X, y)
    print_class_distribution(y_cleaned, "After Tomek Links")

    # Step 2: Random undersampling
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_final, y_final = rus.fit_resample(X_cleaned, y_cleaned)
    print_class_distribution(y_final, "After Random Undersampling")

    print("\n" + "=" * 80)
    print("Example 5: Comparing All Methods")
    print("=" * 80)

    # Collect all results for comparison
    results = [
        ("Random", X_rus_auto, y_rus_auto),
        ("NearMiss-1", X_nm1, y_nm1),
        ("NearMiss-2", X_nm2, y_nm2),
        ("Tomek Links", X_tl_auto, y_tl_auto),
        ("Combined", X_final, y_final),
    ]

    print("\nMethod Comparison:")
    print("-" * 80)
    print(f"{'Method':<20} {'Original':<12} {'Resampled':<12} {'Reduction':<12} {'Balanced'}")
    print("-" * 80)

    original_size = len(X)
    for method_name, X_res, y_res in results:
        resampled_size = len(X_res)
        reduction = ((original_size - resampled_size) / original_size) * 100
        is_balanced = y_res.value_counts()[0] == y_res.value_counts()[1]

        print(f"{method_name:<20} {original_size:<12} {resampled_size:<12} "
              f"{reduction:>6.1f}%      {'Yes' if is_balanced else 'No'}")

    print("\n" + "=" * 80)
    print("Example 6: Use Case Recommendations")
    print("=" * 80)

    print("""
Use Case Recommendations:

1. Random Undersampling:
   ✓ Fast and simple
   ✓ Good baseline approach
   ✓ Works well when you have plenty of majority class samples
   ✗ May lose important information

2. NearMiss-1:
   ✓ Focuses on decision boundary
   ✓ Selects majority samples closest to minority class
   ✓ Good for improving classifier performance
   ✗ Computationally expensive for large datasets

3. NearMiss-2:
   ✓ Alternative boundary-focused approach
   ✓ Selects samples closest to farthest minority samples
   ✗ May not always improve over NearMiss-1

4. NearMiss-3:
   ✓ Ensures even coverage around minority samples
   ✓ Good for maintaining local structure
   ✗ May keep more samples than other versions

5. Tomek Links:
   ✓ Removes noisy boundary samples
   ✓ Cleans the dataset without massive reduction
   ✓ Can be combined with other methods
   ✗ May not significantly balance classes alone

6. Combined (Tomek + Random):
   ✓ Best of both worlds
   ✓ Clean boundaries then balance
   ✓ Often produces best results
   ✗ Two-step process
    """)

    print("\n" + "=" * 80)
    print("Generating visualization...")
    print("=" * 80)

    # Generate visualization
    try:
        visualize_results(
            X, y,
            [
                ("Random", X_rus_auto, y_rus_auto),
                ("NearMiss-1", X_nm1, y_nm1),
                ("Tomek Links", X_tl_auto, y_tl_auto),
                ("Combined", X_final, y_final),
            ]
        )
    except Exception as e:
        print(f"Could not generate visualization: {e}")
        print("(This is optional - matplotlib may not be configured for display)")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
