"""
Example usage of RFESelector (Recursive Feature Elimination).

This example demonstrates:
1. Basic RFE with automatic estimator selection
2. RFE with cross-validation (RFECV) to find optimal features
3. Different base estimators (Logistic, Random Forest, SVM)
4. Feature ranking and importance analysis
5. Comparison of RFE vs univariate selection
6. Practical workflows for model improvement
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from app.ml_engine.feature_selection.rfe_selector import RFESelector


def example_basic_rfe():
    """Example: Basic RFE with automatic estimator."""
    print("=" * 70)
    print("Example 1: Basic RFE with Automatic Estimator")
    print("=" * 70)

    # Create synthetic classification dataset
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_repeated=0,
        random_state=42
    )

    feature_names = [f'feature_{i}' for i in range(20)]
    df = pd.DataFrame(X, columns=feature_names)

    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    # Basic RFE - select 10 features
    print("\n--- Selecting 10 features using RFE (auto estimator) ---")
    selector = RFESelector(
        estimator='auto',
        task='classification',
        n_features_to_select=10
    )
    X_selected = selector.fit_transform(df, y)

    print(f"Selected features: {selector.get_selected_features()}")
    print(f"Shape after selection: {X_selected.shape}")

    # View feature ranking
    ranking = selector.get_feature_ranking()
    print("\nFeature Rankings (1 = most important):")
    print(ranking.to_string(index=False))

    # Summary
    summary = selector.get_rfe_summary()
    print(f"\nEstimator used: {summary['estimator']}")
    print(f"Selection rate: {summary['selection_rate']:.2%}")


def example_rfecv():
    """Example: RFE with cross-validation to find optimal features."""
    print("\n" + "=" * 70)
    print("Example 2: RFECV - Finding Optimal Number of Features")
    print("=" * 70)

    # Create synthetic dataset
    X, y = make_classification(
        n_samples=400,
        n_features=25,
        n_informative=12,
        n_redundant=8,
        random_state=42
    )

    feature_names = [f'var_{i}' for i in range(25)]
    df = pd.DataFrame(X, columns=feature_names)

    print(f"\nOriginal dataset: {df.shape}")

    # Use RFECV to find optimal number of features
    print("\n--- Using RFECV with 5-fold cross-validation ---")
    selector = RFESelector(
        estimator='logistic',
        task='classification',
        use_cv=True,
        cv=5,
        scoring='accuracy',
        step=1
    )
    X_selected = selector.fit_transform(df, y)

    print(f"Optimal number of features: {selector.get_optimal_n_features()}")
    print(f"Selected features: {selector.get_selected_features()}")

    # Plot data
    plot_data = selector.plot_cv_scores()
    if plot_data:
        print("\n--- Cross-Validation Results ---")
        print(f"Optimal features: {plot_data['optimal_n_features']}")
        print(f"Optimal CV score: {plot_data['optimal_score']:.4f}")
        print("\nCV scores by number of features:")
        for n, score in zip(plot_data['n_features'][:10], plot_data['cv_scores'][:10]):
            print(f"  {n:2d} features: {score:.4f}")
        print("  ...")

    # Summary
    summary = selector.get_rfe_summary()
    print(f"\nCV Summary:")
    print(f"  Best score: {summary['cv_scores']['best_score']:.4f}")
    print(f"  Mean score: {summary['cv_scores']['mean_score']:.4f}")


def example_different_estimators():
    """Example: Comparing different base estimators for RFE."""
    print("\n" + "=" * 70)
    print("Example 3: Comparing Different Base Estimators")
    print("=" * 70)

    # Create dataset
    X, y = make_classification(
        n_samples=300,
        n_features=15,
        n_informative=8,
        random_state=42
    )

    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(15)])
    print(f"\nDataset shape: {df.shape}")

    estimators = ['logistic', 'random_forest', 'decision_tree']
    results = {}

    for est in estimators:
        print(f"\n--- RFE with {est.upper()} estimator ---")
        selector = RFESelector(
            estimator=est,
            task='classification',
            n_features_to_select=8
        )
        X_sel = selector.fit_transform(df, y)

        selected = selector.get_selected_features()
        results[est] = set(selected)

        print(f"Selected features: {selected}")

        # Get ranking
        ranking = selector.get_feature_ranking()
        top_5 = ranking.head(5)
        print(f"Top 5 features:")
        print(top_5[['feature', 'rank', 'selected']].to_string(index=False))

    # Compare overlap
    print("\n--- Feature Selection Overlap ---")
    all_estimators = list(results.keys())
    for i in range(len(all_estimators)):
        for j in range(i+1, len(all_estimators)):
            est1, est2 = all_estimators[i], all_estimators[j]
            overlap = results[est1] & results[est2]
            print(f"{est1} ∩ {est2}: {len(overlap)} features in common")


def example_regression_rfe():
    """Example: RFE for regression tasks."""
    print("\n" + "=" * 70)
    print("Example 4: RFE for Regression Tasks")
    print("=" * 70)

    # Create regression dataset
    X, y = make_regression(
        n_samples=250,
        n_features=20,
        n_informative=10,
        noise=10.0,
        random_state=42
    )

    df = pd.DataFrame(X, columns=[f'var_{i}' for i in range(20)])

    print(f"\nDataset shape: {df.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

    # RFE with Ridge regression
    print("\n--- RFE with Ridge Regression ---")
    selector = RFESelector(
        estimator='ridge',
        task='regression',
        n_features_to_select=10
    )
    X_selected = selector.fit_transform(df, y)

    print(f"Selected features: {selector.get_selected_features()}")

    # Feature ranking
    ranking = selector.get_feature_ranking()
    print("\nTop 10 features by importance:")
    print(ranking.head(10).to_string(index=False))


def example_custom_step_size():
    """Example: Using different step sizes for efficiency."""
    print("\n" + "=" * 70)
    print("Example 5: Custom Step Sizes for Efficiency")
    print("=" * 70)

    # Create dataset with many features
    X, y = make_classification(
        n_samples=200,
        n_features=50,
        n_informative=15,
        random_state=42
    )

    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(50)])
    print(f"\nDataset: {df.shape}")

    # Compare different step sizes
    step_sizes = [1, 5, 10]

    for step in step_sizes:
        print(f"\n--- RFE with step={step} ---")
        selector = RFESelector(
            estimator='logistic',
            task='classification',
            n_features_to_select=15,
            step=step
        )
        selector.fit(df, y)

        print(f"Selected {len(selector.get_selected_features())} features")
        print(f"Step: {step} (removes {step} feature(s) per iteration)")

    print("\n💡 Tip: Larger step sizes are faster but may miss optimal features")


def example_fractional_selection():
    """Example: Selecting a fraction of features."""
    print("\n" + "=" * 70)
    print("Example 6: Selecting Features by Fraction")
    print("=" * 70)

    # Create dataset
    X, y = make_classification(
        n_samples=300,
        n_features=30,
        n_informative=12,
        random_state=42
    )

    df = pd.DataFrame(X, columns=[f'feature_{i:02d}' for i in range(30)])
    print(f"\nDataset: {df.shape} (30 features)")

    # Select 30% of features
    print("\n--- Selecting 30% of features (0.3 fraction) ---")
    selector = RFESelector(
        estimator='random_forest',
        task='classification',
        n_features_to_select=0.3  # 30% of features
    )
    X_selected = selector.fit_transform(df, y)

    n_selected = len(selector.get_selected_features())
    print(f"Selected {n_selected} features (30% of 30 = 9 features)")
    print(f"Selected features: {selector.get_selected_features()}")


def example_model_performance_comparison():
    """Example: Comparing model performance before/after RFE."""
    print("\n" + "=" * 70)
    print("Example 7: Model Performance Before and After RFE")
    print("=" * 70)

    # Create dataset with many features
    X, y = make_classification(
        n_samples=400,
        n_features=30,
        n_informative=10,
        n_redundant=15,
        random_state=42
    )

    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(30)])
    print(f"\nDataset: {df.shape}")

    # Baseline model (all features)
    print("\n--- Baseline: All 30 features ---")
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_scores = cross_val_score(baseline_model, df, y, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {baseline_scores.mean():.4f} (+/- {baseline_scores.std():.4f})")

    # RFE model (selected features)
    print("\n--- After RFE: Selecting 10 features ---")
    selector = RFESelector(
        estimator='logistic',
        task='classification',
        n_features_to_select=10
    )
    X_selected = selector.fit_transform(df, y)

    rfe_model = LogisticRegression(max_iter=1000, random_state=42)
    rfe_scores = cross_val_score(rfe_model, X_selected, y, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {rfe_scores.mean():.4f} (+/- {rfe_scores.std():.4f})")

    print(f"\n📊 Results:")
    print(f"  Features reduced: 30 → 10 (66.7% reduction)")
    print(f"  Accuracy change: {baseline_scores.mean():.4f} → {rfe_scores.mean():.4f}")
    improvement = rfe_scores.mean() - baseline_scores.mean()
    print(f"  Improvement: {improvement:+.4f}")

    if improvement > 0:
        print("\n✅ RFE improved model performance while reducing features!")
    else:
        print("\n📉 RFE reduced features but may have slightly reduced accuracy")


def example_practical_workflow():
    """Example: Complete practical workflow with RFE."""
    print("\n" + "=" * 70)
    print("Example 8: Complete Practical Workflow")
    print("=" * 70)

    # Create realistic dataset
    X, y = make_classification(
        n_samples=500,
        n_features=25,
        n_informative=12,
        n_redundant=8,
        random_state=42
    )

    df = pd.DataFrame(X, columns=[f'feature_{i:02d}' for i in range(25)])
    print(f"\nDataset: {df.shape}")

    # Step 1: Use RFECV to find optimal number
    print("\n--- Step 1: Find optimal number of features using RFECV ---")
    selector_cv = RFESelector(
        estimator='random_forest',
        task='classification',
        use_cv=True,
        cv=5,
        scoring='accuracy',
        step=2
    )
    selector_cv.fit(df, y)

    optimal_n = selector_cv.get_optimal_n_features()
    print(f"Optimal number of features: {optimal_n}")

    # Step 2: Analyze feature rankings
    print("\n--- Step 2: Analyze feature rankings ---")
    ranking = selector_cv.get_feature_ranking()
    print("\nTop 10 features:")
    print(ranking.head(10).to_string(index=False))

    # Step 3: Get selected features for final model
    print("\n--- Step 3: Get selected features ---")
    selected_features = selector_cv.get_selected_features()
    print(f"Selected features: {selected_features}")

    # Step 4: Transform data
    print("\n--- Step 4: Transform data ---")
    X_final = selector_cv.transform(df)
    print(f"Final dataset shape: {X_final.shape}")

    # Step 5: Summary
    print("\n--- Step 5: Summary ---")
    summary = selector_cv.get_rfe_summary()
    print(f"Total features: {summary['total_features']}")
    print(f"Selected features: {summary['selected_features']}")
    print(f"Reduction: {(1 - summary['selection_rate']) * 100:.1f}%")
    print(f"Estimator: {summary['estimator']}")
    print(f"CV folds: {summary['cv_folds']}")

    if 'cv_scores' in summary:
        print(f"Optimal CV score: {summary['cv_scores']['optimal_score']:.4f}")

    print("\n✅ Ready for final model training!")


def example_feature_engineering_integration():
    """Example: Integrating RFE with feature engineering."""
    print("\n" + "=" * 70)
    print("Example 9: RFE with Feature Engineering")
    print("=" * 70)

    # Create base features
    np.random.seed(42)
    n_samples = 300

    base_features = pd.DataFrame({
        'x1': np.random.randn(n_samples),
        'x2': np.random.randn(n_samples),
        'x3': np.random.randn(n_samples),
        'x4': np.random.randn(n_samples)
    })

    # Create target
    y = ((base_features['x1'] + base_features['x2']) > 0).astype(int)

    # Add engineered features (interactions, polynomials)
    df = base_features.copy()
    df['x1_x2'] = df['x1'] * df['x2']
    df['x1_x3'] = df['x1'] * df['x3']
    df['x2_x3'] = df['x2'] * df['x3']
    df['x1_sq'] = df['x1'] ** 2
    df['x2_sq'] = df['x2'] ** 2
    df['x3_sq'] = df['x3'] ** 2

    print(f"\nOriginal features: {list(base_features.columns)}")
    print(f"After feature engineering: {df.shape[1]} features")
    print(f"Engineered features: {list(df.columns[4:])}")

    # Use RFE to select best features (original + engineered)
    print("\n--- RFE to select best features ---")
    selector = RFESelector(
        estimator='logistic',
        task='classification',
        n_features_to_select=5
    )
    X_selected = selector.fit_transform(df, y)

    selected = selector.get_selected_features()
    print(f"Selected features: {selected}")

    # Analyze what was selected
    original_selected = [f for f in selected if f in base_features.columns]
    engineered_selected = [f for f in selected if f not in base_features.columns]

    print(f"\nFrom original: {original_selected}")
    print(f"From engineered: {engineered_selected}")

    print("\n💡 RFE can help identify which engineered features are valuable!")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  RECURSIVE FEATURE ELIMINATION (RFE) EXAMPLES  ".center(68) + "║")
    print("║" + "  Feature Selection with Model-Based Ranking    ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    # Run all examples
    example_basic_rfe()
    example_rfecv()
    example_different_estimators()
    example_regression_rfe()
    example_custom_step_size()
    example_fractional_selection()
    example_model_performance_comparison()
    example_practical_workflow()
    example_feature_engineering_integration()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\n📚 Summary of RFE Concepts:")
    print("\n1. How RFE Works:")
    print("   - Trains model on all features")
    print("   - Ranks features by importance (coef_ or feature_importances_)")
    print("   - Removes least important feature(s)")
    print("   - Repeats until desired number of features reached")
    print("\n2. Base Estimators:")
    print("   - Logistic/Ridge: Fast, works well for linear relationships")
    print("   - Random Forest: Handles non-linear, feature interactions")
    print("   - SVM: Good for high-dimensional data")
    print("   - Decision Tree: Fast, interpretable")
    print("\n3. RFE vs RFECV:")
    print("   - RFE: You specify number of features to select")
    print("   - RFECV: Uses cross-validation to find optimal number")
    print("\n4. Key Parameters:")
    print("   - n_features_to_select: How many features to keep (int or fraction)")
    print("   - step: How many features to remove per iteration")
    print("   - use_cv: Whether to use cross-validation")
    print("   - cv: Number of cross-validation folds")
    print("\n5. Advantages:")
    print("   - Considers feature interactions")
    print("   - Provides feature ranking")
    print("   - Can find optimal number of features (RFECV)")
    print("\n6. Disadvantages:")
    print("   - Computationally expensive (retrains many times)")
    print("   - Slower than univariate methods")
    print("   - Results depend on choice of estimator")
    print("\n")


if __name__ == "__main__":
    main()
