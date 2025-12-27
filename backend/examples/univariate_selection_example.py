"""
Example usage of UnivariateSelector for feature selection.

This example demonstrates:
1. F-test (ANOVA) for classification tasks
2. F-test for regression tasks
3. Chi-square test for categorical features
4. Different selection methods (k, percentile, threshold)
5. P-value filtering for statistical significance
6. Practical comparison between methods
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from app.ml_engine.feature_selection.univariate_selector import UnivariateSelector


def example_f_classif():
    """Example: F-test for classification (ANOVA F-statistic)."""
    print("=" * 70)
    print("Example 1: F-test for Classification (ANOVA)")
    print("=" * 70)

    # Create synthetic classification dataset
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        n_repeated=0,
        n_classes=3,
        random_state=42
    )

    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(20)]
    df = pd.DataFrame(X, columns=feature_names)

    print(f"\nDataset shape: {df.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Target distribution: {np.bincount(y)}")

    # Method 1: Select top k features
    print("\n--- Method 1: Select top k=10 features ---")
    selector = UnivariateSelector(score_func='f_classif', k=10)
    X_selected = selector.fit_transform(df, y)

    print(f"Selected features: {selector.get_selected_features()}")
    print(f"Shape after selection: {X_selected.shape}")

    # View feature scores
    scores_df = selector.get_feature_scores()
    print("\nTop 5 features by F-score:")
    print(scores_df.head())

    # Statistical summary
    summary = selector.get_statistical_summary()
    print(f"\nSelection rate: {summary['selection_rate']:.2%}")
    print(f"Mean F-score: {summary['score_statistics']['mean']:.4f}")
    print(f"Max F-score: {summary['score_statistics']['max']:.4f}")

    # Method 2: Select by percentile
    print("\n--- Method 2: Select top 25% features by score ---")
    selector_pct = UnivariateSelector(score_func='f_classif', k=None, percentile=25)
    X_pct = selector_pct.fit_transform(df, y)

    print(f"Number of features selected: {len(selector_pct.get_selected_features())}")
    print(f"Selected features: {selector_pct.get_selected_features()}")

    # Method 3: Auto score function selection
    print("\n--- Method 3: Auto score function (task='classification') ---")
    selector_auto = UnivariateSelector(score_func='auto', task='classification', k=8)
    selector_auto.fit(df, y)

    print(f"Score function used: {selector_auto.score_func_used_}")
    print(f"Selected features: {selector_auto.get_selected_features()}")


def example_f_regression():
    """Example: F-test for regression tasks."""
    print("\n" + "=" * 70)
    print("Example 2: F-test for Regression")
    print("=" * 70)

    # Create synthetic regression dataset
    X, y = make_regression(
        n_samples=250,
        n_features=15,
        n_informative=6,
        noise=10.0,
        random_state=42
    )

    # Convert to DataFrame
    feature_names = [f'var_{i}' for i in range(15)]
    df = pd.DataFrame(X, columns=feature_names)

    print(f"\nDataset shape: {df.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Target mean: {y.mean():.2f}, std: {y.std():.2f}")

    # Select top features using F-regression
    print("\n--- Selecting top 8 features using F-regression ---")
    selector = UnivariateSelector(score_func='f_regression', k=8)
    X_selected = selector.fit_transform(df, y)

    print(f"Selected features: {selector.get_selected_features()}")

    # View feature scores with p-values
    scores_df = selector.get_feature_scores()
    print("\nAll features sorted by F-score:")
    print(scores_df.to_string(index=False))

    # Check statistical significance
    print("\n--- Statistical Significance Analysis ---")
    significant_features = scores_df[scores_df['significant'] == True]
    print(f"Number of statistically significant features (α=0.05): {len(significant_features)}")
    print(f"Significant features: {significant_features['feature'].tolist()}")


def example_chi2():
    """Example: Chi-square test for categorical features."""
    print("\n" + "=" * 70)
    print("Example 3: Chi-square Test for Categorical Features")
    print("=" * 70)

    # Create synthetic count data (appropriate for chi-square test)
    np.random.seed(42)
    n_samples = 200

    # Simulate count/frequency data (must be non-negative)
    df = pd.DataFrame({
        'word_count': np.random.poisson(10, n_samples),
        'unique_words': np.random.poisson(5, n_samples),
        'sentence_count': np.random.poisson(3, n_samples),
        'paragraph_count': np.random.poisson(2, n_samples),
        'char_count': np.random.poisson(50, n_samples)
    })

    # Binary classification target
    y = np.random.randint(0, 2, n_samples)

    print(f"\nDataset shape: {df.shape}")
    print(f"Target distribution: Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")
    print("\nFeature statistics:")
    print(df.describe())

    # Chi-square feature selection
    print("\n--- Selecting top 3 features using Chi-square test ---")
    selector = UnivariateSelector(score_func='chi2', k=3)
    X_selected = selector.fit_transform(df, y)

    print(f"Selected features: {selector.get_selected_features()}")

    # View chi-square scores
    scores_df = selector.get_feature_scores()
    print("\nChi-square scores and p-values:")
    print(scores_df.to_string(index=False))

    # Important note about chi-square
    print("\n⚠️  Note: Chi-square test requires non-negative feature values")
    print("   (counts, frequencies, or binary indicators)")


def example_p_value_filtering():
    """Example: Using p-value filtering for statistical significance."""
    print("\n" + "=" * 70)
    print("Example 4: P-value Filtering for Statistical Significance")
    print("=" * 70)

    # Create dataset with mix of informative and noise features
    np.random.seed(42)
    n_samples = 300

    # Informative features
    x1 = np.random.randn(n_samples)
    x2 = np.random.randn(n_samples)

    # Target depends on x1 and x2
    y = ((x1 + x2) > 0).astype(int)

    # Add noise features
    df = pd.DataFrame({
        'informative_1': x1,
        'informative_2': x2,
        'noise_1': np.random.randn(n_samples),
        'noise_2': np.random.randn(n_samples),
        'noise_3': np.random.randn(n_samples),
        'noise_4': np.random.randn(n_samples)
    })

    print(f"\nDataset: 2 informative features + 4 noise features")
    print(f"Samples: {n_samples}")

    # Without p-value filtering (just select top k)
    print("\n--- Without p-value filtering (top k=4) ---")
    selector1 = UnivariateSelector(
        score_func='f_classif',
        k=4,
        use_p_value_filter=False
    )
    selector1.fit(df, y)

    scores1 = selector1.get_feature_scores()
    print("\nAll features:")
    print(scores1.to_string(index=False))
    print(f"\nSelected (top 4): {selector1.get_selected_features()}")

    # With p-value filtering (select top k but filter by significance)
    print("\n--- With p-value filtering (top k=4, α=0.05) ---")
    selector2 = UnivariateSelector(
        score_func='f_classif',
        k=4,
        alpha=0.05,
        use_p_value_filter=True
    )
    selector2.fit(df, y)

    scores2 = selector2.get_feature_scores()
    print("\nAll features:")
    print(scores2.to_string(index=False))
    print(f"\nSelected (top 4 with p<0.05): {selector2.get_selected_features()}")
    print("\n💡 P-value filtering removes features that are not statistically significant")


def example_threshold_selection():
    """Example: Using score threshold for feature selection."""
    print("\n" + "=" * 70)
    print("Example 5: Threshold-based Selection")
    print("=" * 70)

    # Create classification dataset
    X, y = make_classification(
        n_samples=200,
        n_features=12,
        n_informative=5,
        n_redundant=3,
        random_state=42
    )

    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(12)])

    print(f"\nDataset shape: {df.shape}")

    # Use score threshold instead of k
    print("\n--- Selecting features with F-score >= 5.0 ---")
    selector = UnivariateSelector(
        score_func='f_classif',
        k=None,
        threshold=5.0
    )
    selector.fit(df, y)

    scores_df = selector.get_feature_scores()
    print("\nFeature scores:")
    print(scores_df.to_string(index=False))

    print(f"\nFeatures with score >= 5.0: {selector.get_selected_features()}")
    print(f"Number selected: {len(selector.get_selected_features())}")


def example_comparison():
    """Example: Comparing different univariate tests."""
    print("\n" + "=" * 70)
    print("Example 6: Comparing F-test vs Chi-square")
    print("=" * 70)

    # Create non-negative data (compatible with both methods)
    np.random.seed(42)
    n_samples = 250

    # Create informative feature
    informative = np.random.randint(0, 10, n_samples)
    y = (informative >= 5).astype(int)  # Binary target based on informative

    # Create dataset
    df = pd.DataFrame({
        'informative': informative,
        'noise1': np.random.randint(0, 10, n_samples),
        'noise2': np.random.randint(0, 10, n_samples),
        'noise3': np.random.randint(0, 10, n_samples)
    })

    print(f"\nDataset shape: {df.shape}")
    print(f"Target distribution: Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")

    # F-test (ANOVA)
    print("\n--- Using F-test (ANOVA) ---")
    selector_f = UnivariateSelector(score_func='f_classif', k=2)
    selector_f.fit(df, y)

    scores_f = selector_f.get_feature_scores()
    print("\nF-test scores:")
    print(scores_f[['feature', 'score', 'p_value', 'selected']].to_string(index=False))

    # Chi-square test
    print("\n--- Using Chi-square test ---")
    selector_chi = UnivariateSelector(score_func='chi2', k=2)
    selector_chi.fit(df, y)

    scores_chi = selector_chi.get_feature_scores()
    print("\nChi-square scores:")
    print(scores_chi[['feature', 'score', 'p_value', 'selected']].to_string(index=False))

    print("\n📊 Comparison:")
    print(f"F-test selected: {selector_f.get_selected_features()}")
    print(f"Chi2 selected: {selector_chi.get_selected_features()}")

    print("\n💡 Key differences:")
    print("   - F-test: Tests mean differences between groups (assumes normality)")
    print("   - Chi-square: Tests independence (for categorical/count data)")


def example_practical_workflow():
    """Example: Practical end-to-end workflow."""
    print("\n" + "=" * 70)
    print("Example 7: Practical End-to-End Workflow")
    print("=" * 70)

    # Create realistic dataset
    X, y = make_classification(
        n_samples=400,
        n_features=25,
        n_informative=10,
        n_redundant=8,
        n_repeated=0,
        random_state=42
    )

    df = pd.DataFrame(X, columns=[f'feature_{i:02d}' for i in range(25)])

    print(f"\nOriginal dataset: {df.shape}")

    # Step 1: Select features using percentile
    print("\n--- Step 1: Select top 40% features ---")
    selector = UnivariateSelector(
        score_func='f_classif',
        k=None,
        percentile=40,
        alpha=0.01,
        use_p_value_filter=True
    )

    X_selected = selector.fit_transform(df, y)
    print(f"After selection: {X_selected.shape}")

    # Step 2: Analyze results
    print("\n--- Step 2: Statistical Summary ---")
    summary = selector.get_statistical_summary()
    print(f"Score function: {summary['score_function']}")
    print(f"Total features: {summary['total_features']}")
    print(f"Selected features: {summary['selected_features']}")
    print(f"Selection rate: {summary['selection_rate']:.2%}")
    print(f"Significant features (α=0.01): {summary['significant_features_count']}")

    # Step 3: View top features
    print("\n--- Step 3: Top 10 Features by F-score ---")
    scores = selector.get_feature_scores()
    print(scores.head(10).to_string(index=False))

    # Step 4: Get support mask (for sklearn compatibility)
    print("\n--- Step 4: Get Support Mask (sklearn-compatible) ---")
    support_mask = selector.get_support(indices=False)
    support_indices = selector.get_support(indices=True)
    print(f"Boolean mask: {support_mask[:10]}... (showing first 10)")
    print(f"Feature indices: {support_indices}")

    print("\n✅ Workflow complete! Features ready for model training.")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  UNIVARIATE STATISTICAL TESTS FOR FEATURE SELECTION  ".center(68) + "║")
    print("║" + "  F-test (ANOVA) and Chi-square Examples              ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    # Run all examples
    example_f_classif()
    example_f_regression()
    example_chi2()
    example_p_value_filtering()
    example_threshold_selection()
    example_comparison()
    example_practical_workflow()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\n📚 Summary of Univariate Selection Methods:")
    print("\n1. F-test (ANOVA) - f_classif:")
    print("   - Use for: Continuous features with categorical targets")
    print("   - Tests: Mean differences between groups")
    print("   - Assumes: Normally distributed features")
    print("\n2. F-test - f_regression:")
    print("   - Use for: Continuous features with continuous targets")
    print("   - Tests: Linear dependency between features and target")
    print("   - Assumes: Linear relationships")
    print("\n3. Chi-square - chi2:")
    print("   - Use for: Categorical/count features with categorical targets")
    print("   - Tests: Independence between feature and target")
    print("   - Requires: Non-negative feature values")
    print("\n📖 Selection Methods:")
    print("   - k: Select top k features by score")
    print("   - percentile: Select top X% features by score")
    print("   - threshold: Select features with score >= threshold")
    print("   - p-value filter: Additionally filter by statistical significance")
    print("\n")


if __name__ == "__main__":
    main()
