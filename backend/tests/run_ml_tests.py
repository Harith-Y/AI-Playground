"""Test runner script for ML engine metrics and tuning utilities.

Runs all tests and provides a summary report.
"""

import sys
import unittest
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


def run_evaluation_tests():
    """Run all evaluation/metrics tests."""
    print("\n" + "=" * 70)
    print("RUNNING EVALUATION METRICS TESTS")
    print("=" * 70 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Load all evaluation tests
    evaluation_tests = [
        "tests.ml_engine.evaluation.test_classification_metrics",
        "tests.ml_engine.evaluation.test_confusion_matrix",
        "tests.ml_engine.evaluation.test_roc_curve",
        "tests.ml_engine.evaluation.test_pr_curve",
        "tests.ml_engine.evaluation.test_regression_metrics",
        "tests.ml_engine.evaluation.test_residual_analysis",
        "tests.ml_engine.evaluation.test_actual_vs_predicted",
        "tests.ml_engine.evaluation.test_clustering_metrics",
        "tests.ml_engine.evaluation.test_feature_importance",
        "tests.ml_engine.evaluation.test_integration",
    ]

    for test_module in evaluation_tests:
        try:
            suite.addTests(loader.loadTestsFromName(test_module))
        except Exception as e:
            print(f"Warning: Could not load {test_module}: {e}")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


def run_tuning_tests():
    """Run all tuning tests."""
    print("\n" + "=" * 70)
    print("RUNNING HYPERPARAMETER TUNING TESTS")
    print("=" * 70 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Load all tuning tests
    tuning_tests = [
        "tests.ml_engine.tuning.test_search_spaces",
        "tests.ml_engine.tuning.test_grid_search",
        "tests.ml_engine.tuning.test_random_search",
        "tests.ml_engine.tuning.test_bayesian",
        "tests.ml_engine.tuning.test_cross_validation",
        "tests.ml_engine.tuning.test_integration",
    ]

    for test_module in tuning_tests:
        try:
            suite.addTests(loader.loadTestsFromName(test_module))
        except Exception as e:
            print(f"Warning: Could not load {test_module}: {e}")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


def print_summary(eval_result, tuning_result):
    """Print test summary."""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total_tests = eval_result.testsRun + tuning_result.testsRun
    total_failures = len(eval_result.failures) + len(tuning_result.failures)
    total_errors = len(eval_result.errors) + len(tuning_result.errors)
    total_skipped = len(eval_result.skipped) + len(tuning_result.skipped)

    print(f"\nEvaluation Tests:")
    print(f"  Tests Run: {eval_result.testsRun}")
    print(f"  Failures: {len(eval_result.failures)}")
    print(f"  Errors: {len(eval_result.errors)}")
    print(f"  Skipped: {len(eval_result.skipped)}")

    print(f"\nTuning Tests:")
    print(f"  Tests Run: {tuning_result.testsRun}")
    print(f"  Failures: {len(tuning_result.failures)}")
    print(f"  Errors: {len(tuning_result.errors)}")
    print(f"  Skipped: {len(tuning_result.skipped)}")

    print(f"\nTotal:")
    print(f"  Tests Run: {total_tests}")
    print(f"  Failures: {total_failures}")
    print(f"  Errors: {total_errors}")
    print(f"  Skipped: {total_skipped}")

    success_rate = (
        (total_tests - total_failures - total_errors) / total_tests * 100
        if total_tests > 0
        else 0
    )
    print(f"  Success Rate: {success_rate:.1f}%")

    if total_failures + total_errors == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1

    return 0


def main():
    """Main test runner."""
    print("\n" + "=" * 70)
    print("ML ENGINE TEST SUITE")
    print("Testing Metrics & Tuning Utilities")
    print("=" * 70)

    # Run evaluation tests
    eval_result = run_evaluation_tests()

    # Run tuning tests
    tuning_result = run_tuning_tests()

    # Print summary
    exit_code = print_summary(eval_result, tuning_result)

    print("\n" + "=" * 70)
    print("TEST RUN COMPLETE")
    print("=" * 70 + "\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
