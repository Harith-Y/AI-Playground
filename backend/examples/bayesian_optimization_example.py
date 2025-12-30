"""Example of Bayesian hyperparameter optimization.

Demonstrates:
- Basic Bayesian optimization with default spaces
- Custom search spaces with skopt objects
- Fallback behavior when scikit-optimize not available
- Comparison with Grid and Random Search
"""

import sys
import warnings
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from app.ml_engine.tuning import (
    run_bayesian_search,
    run_grid_search,
    run_random_search,
)


def example_basic_bayesian():
    """Basic Bayesian optimization example."""
    print("=" * 60)
    print("Example 1: Basic Bayesian Optimization")
    print("=" * 60)

    # Generate sample data
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=10, n_redundant=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define search space
    search_spaces = {
        "n_estimators": [10, 50, 100, 200, 300],
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2"],
    }

    # Run Bayesian optimization
    print("\nRunning Bayesian optimization with 30 iterations...")
    result = run_bayesian_search(
        estimator=RandomForestClassifier(random_state=42),
        X=X_train,
        y=y_train,
        search_spaces=search_spaces,
        n_iter=30,
        cv=5,
        scoring="roc_auc",
        random_state=42,
        verbose=1,
    )

    print(f"\nOptimization method used: {result.method}")
    print(f"Best cross-validation score: {result.best_score:.4f}")
    print(f"Best hyperparameters:")
    for param, value in result.best_params.items():
        print(f"  {param}: {value}")

    print(f"\nTop 5 configurations:")
    for i, config in enumerate(result.top(5), 1):
        print(f"{i}. Score: {config['mean_score']:.4f} ± {config['std_score']:.4f}")
        print(f"   Params: {config['params']}")


def example_with_skopt_spaces():
    """Example using scikit-optimize space objects."""
    print("\n" + "=" * 60)
    print("Example 2: Bayesian Optimization with skopt Spaces")
    print("=" * 60)

    try:
        from skopt.space import Real, Integer, Categorical

        X, y = make_classification(n_samples=500, n_features=20, random_state=42)

        # Define search spaces using skopt objects for finer control
        search_spaces = {
            "n_estimators": Integer(50, 300),
            "max_depth": Integer(3, 30),
            "min_samples_split": Integer(2, 20),
            "min_samples_leaf": Integer(1, 10),
            "max_features": Categorical(["sqrt", "log2"]),
        }

        print("\nRunning Bayesian optimization with skopt space objects...")
        result = run_bayesian_search(
            estimator=RandomForestClassifier(random_state=42),
            X=X,
            y=y,
            search_spaces=search_spaces,
            n_iter=25,
            cv=3,
            scoring="accuracy",
            random_state=42,
            optimizer_kwargs={"base_estimator": "GP", "acq_func": "EI"},
        )

        print(f"\nBest score: {result.best_score:.4f}")
        print(f"Best parameters: {result.best_params}")

    except ImportError:
        print("\nscikit-optimize not installed. Skipping this example.")
        print("Install with: pip install scikit-optimize")


def example_comparison():
    """Compare Grid, Random, and Bayesian search."""
    print("\n" + "=" * 60)
    print("Example 3: Comparing Search Strategies")
    print("=" * 60)

    X, y = make_classification(n_samples=300, n_features=15, random_state=42)

    # Define search space
    search_space = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
    }

    estimator = RandomForestClassifier(random_state=42)

    # Grid Search (exhaustive)
    print("\n1. Grid Search (exhaustive)...")
    grid_result = run_grid_search(
        estimator=estimator, X=X, y=y, param_grid=search_space, cv=3, scoring="accuracy"
    )
    print(f"   Total combinations tested: {grid_result.n_candidates}")
    print(f"   Best score: {grid_result.best_score:.4f}")

    # Random Search
    print("\n2. Random Search (20 iterations)...")
    random_result = run_random_search(
        estimator=estimator,
        X=X,
        y=y,
        param_distributions=search_space,
        n_iter=20,
        cv=3,
        scoring="accuracy",
        random_state=42,
    )
    print(f"   Combinations tested: {random_result.n_candidates}")
    print(f"   Best score: {random_result.best_score:.4f}")

    # Bayesian Search
    print("\n3. Bayesian Search (20 iterations)...")
    bayesian_result = run_bayesian_search(
        estimator=estimator,
        X=X,
        y=y,
        search_spaces=search_space,
        n_iter=20,
        cv=3,
        scoring="accuracy",
        random_state=42,
    )
    print(f"   Method used: {bayesian_result.method}")
    print(f"   Combinations tested: {bayesian_result.n_candidates}")
    print(f"   Best score: {bayesian_result.best_score:.4f}")

    # Summary
    print("\n" + "-" * 60)
    print("Comparison Summary:")
    print(f"  Grid Search:     {grid_result.best_score:.4f} ({grid_result.n_candidates} evaluations)")
    print(f"  Random Search:   {random_result.best_score:.4f} ({random_result.n_candidates} evaluations)")
    print(f"  Bayesian Search: {bayesian_result.best_score:.4f} ({bayesian_result.n_candidates} evaluations)")


def example_with_default_spaces():
    """Use predefined default search spaces."""
    print("\n" + "=" * 60)
    print("Example 4: Using Default Search Spaces")
    print("=" * 60)

    X, y = make_classification(n_samples=400, n_features=15, random_state=42)

    print("\nRunning Bayesian optimization with default search space...")
    result = run_bayesian_search(
        estimator=RandomForestClassifier(random_state=42),
        X=X,
        y=y,
        model_id="random_forest_classifier",  # Use default space
        n_iter=25,
        cv=5,
        scoring="f1",
        random_state=42,
    )

    print(f"\nMethod: {result.method}")
    print(f"Best score: {result.best_score:.4f}")
    print(f"Best parameters: {result.best_params}")


def example_fallback_behavior():
    """Demonstrate fallback when scikit-optimize not available."""
    print("\n" + "=" * 60)
    print("Example 5: Fallback Behavior")
    print("=" * 60)

    X, y = make_classification(n_samples=200, n_features=10, random_state=42)

    print("\nAttempting Bayesian optimization...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        result = run_bayesian_search(
            estimator=RandomForestClassifier(random_state=42),
            X=X,
            y=y,
            search_spaces={"n_estimators": [50, 100, 150], "max_depth": [5, 10, 15]},
            n_iter=10,
            cv=3,
            random_state=42,
        )

        if w and "scikit-optimize not installed" in str(w[0].message):
            print("\nWarning issued: scikit-optimize not available, using fallback")

    print(f"\nMethod used: {result.method}")
    if result.method == "random_fallback":
        print("→ Fallback to RandomizedSearchCV was triggered")
        print("→ Install scikit-optimize for true Bayesian optimization")
    else:
        print("→ Using scikit-optimize BayesSearchCV")

    print(f"\nBest score: {result.best_score:.4f}")
    print(f"Best parameters: {result.best_params}")


def main():
    """Run all examples."""
    print("\nBayesian Hyperparameter Optimization Examples")
    print("=" * 60)

    example_basic_bayesian()
    example_with_skopt_spaces()
    example_comparison()
    example_with_default_spaces()
    example_fallback_behavior()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
