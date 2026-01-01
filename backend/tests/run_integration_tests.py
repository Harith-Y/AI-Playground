"""
Test Runner for ML Pipeline Integration Tests

Run all integration tests for the ML pipeline with proper configuration
and reporting.
"""

import sys
import subprocess
from pathlib import Path


def run_integration_tests():
    """Run all ML pipeline integration tests"""
    
    print("=" * 80)
    print("ML PIPELINE INTEGRATION TESTS")
    print("=" * 80)
    print()
    
    # Test categories
    test_suites = {
        "End-to-End Pipeline Tests": "tests/integration/test_ml_pipeline_end_to_end.py",
        "ML Engine Integration Tests": "tests/integration/test_ml_engine_integration.py",
        "Pipeline Orchestration Tests": "tests/integration/test_ml_pipeline_orchestration.py",
        "All Integration Tests": "tests/integration/",
    }
    
    print("Available test suites:")
    for i, (name, path) in enumerate(test_suites.items(), 1):
        print(f"  {i}. {name}")
    
    print("\nRun options:")
    print("  - Enter suite number (1-4) to run specific suite")
    print("  - Press Enter to run all tests")
    print("  - Type 'fast' to skip slow tests")
    print("  - Type 'exit' to quit")
    print()
    
    choice = input("Your choice: ").strip().lower()
    
    # Build pytest command
    cmd = ["pytest", "-v", "--tb=short", "--color=yes"]
    
    if choice == "exit":
        print("Exiting...")
        return 0
    
    elif choice == "fast":
        cmd.extend(["-m", "integration and not slow", "tests/integration/"])
        print("\n[Running] Fast integration tests (skipping slow tests)...")
    
    elif choice.isdigit():
        suite_num = int(choice)
        if 1 <= suite_num <= len(test_suites):
            suite_name, suite_path = list(test_suites.items())[suite_num - 1]
            cmd.append(suite_path)
            print(f"\n[Running] {suite_name}...")
        else:
            print("Invalid choice. Running all tests...")
            cmd.extend(["-m", "integration", "tests/integration/"])
    
    else:
        cmd.extend(["-m", "integration", "tests/integration/"])
        print("\n[Running] All integration tests...")
    
    # Add coverage if requested
    if "--cov" in sys.argv:
        cmd.extend([
            "--cov=app.ml_engine",
            "--cov=app.api.v1.endpoints",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov/integration"
        ])
        print("[Coverage] Enabled - report will be in htmlcov/integration/")
    
    print()
    print("-" * 80)
    
    # Run tests
    result = subprocess.run(cmd)
    
    print()
    print("-" * 80)
    
    if result.returncode == 0:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    
    return result.returncode


def run_specific_test_class():
    """Run a specific test class interactively"""
    
    print("=" * 80)
    print("RUN SPECIFIC TEST CLASS")
    print("=" * 80)
    print()
    
    test_classes = {
        "End-to-End": {
            "TestMLPipelineEndToEnd": "Complete ML workflow tests",
            "TestMLPipelineWithTuning": "Tests with hyperparameter tuning",
            "TestMLPipelineFeatureEngineering": "Feature engineering tests",
            "TestMLPipelineDataPreprocessing": "Data preprocessing tests",
            "TestMLPipelineModelComparison": "Model comparison tests",
            "TestMLPipelineExperimentTracking": "Experiment tracking tests",
            "TestMLPipelineCodeGeneration": "Code generation tests",
            "TestMLPipelinePerformance": "Performance tests (slow)",
        },
        "ML Engine": {
            "TestPreprocessingPipeline": "Preprocessing pipeline tests",
            "TestFeatureSelection": "Feature selection tests",
            "TestModelTraining": "Model training tests",
            "TestModelEvaluation": "Model evaluation tests",
            "TestCompleteMLWorkflow": "Complete workflow tests",
            "TestMLPipelineStressTests": "Stress tests (slow)",
        },
        "Orchestration": {
            "TestTuningOrchestration": "Tuning orchestration tests",
            "TestFeatureEngineeringOrchestration": "Feature engineering orchestration",
            "TestModelComparisonOrchestration": "Model comparison orchestration",
            "TestPipelineExportOrchestration": "Pipeline export tests",
            "TestExperimentOrchestration": "Experiment orchestration tests",
            "TestDataPipelineOrchestration": "Data pipeline tests (slow)",
        }
    }
    
    print("Test categories:")
    for i, category in enumerate(test_classes.keys(), 1):
        print(f"  {i}. {category}")
    
    print()
    category_choice = input("Select category (1-3): ").strip()
    
    try:
        category = list(test_classes.keys())[int(category_choice) - 1]
    except (ValueError, IndexError):
        print("Invalid choice")
        return 1
    
    print(f"\nTest classes in {category}:")
    classes = test_classes[category]
    for i, (class_name, description) in enumerate(classes.items(), 1):
        print(f"  {i}. {class_name}: {description}")
    
    print()
    class_choice = input("Select test class: ").strip()
    
    try:
        class_name = list(classes.keys())[int(class_choice) - 1]
    except (ValueError, IndexError):
        print("Invalid choice")
        return 1
    
    # Map category to file
    file_map = {
        "End-to-End": "tests/integration/test_ml_pipeline_end_to_end.py",
        "ML Engine": "tests/integration/test_ml_engine_integration.py",
        "Orchestration": "tests/integration/test_ml_pipeline_orchestration.py",
    }
    
    test_file = file_map[category]
    
    cmd = [
        "pytest", "-v", "--tb=short", "--color=yes",
        f"{test_file}::{class_name}"
    ]
    
    print(f"\n[Running] {class_name}...")
    print("-" * 80)
    print()
    
    result = subprocess.run(cmd)
    
    print()
    print("-" * 80)
    
    return result.returncode


def run_single_test():
    """Run a single test function"""
    
    print("=" * 80)
    print("RUN SINGLE TEST")
    print("=" * 80)
    print()
    print("Enter test path (e.g., tests/integration/test_ml_pipeline_end_to_end.py::TestMLPipelineEndToEnd::test_classification_pipeline_end_to_end)")
    print()
    
    test_path = input("Test path: ").strip()
    
    if not test_path:
        print("No test path provided")
        return 1
    
    cmd = ["pytest", "-v", "--tb=short", "--color=yes", "-s", test_path]
    
    print()
    print("[Running] Single test...")
    print("-" * 80)
    print()
    
    result = subprocess.run(cmd)
    
    print()
    print("-" * 80)
    
    return result.returncode


def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        # Command line mode
        if sys.argv[1] == "all":
            return run_integration_tests()
        elif sys.argv[1] == "class":
            return run_specific_test_class()
        elif sys.argv[1] == "single":
            return run_single_test()
        elif sys.argv[1] == "fast":
            sys.argv[1] = "fast"
            return run_integration_tests()
        else:
            print("Usage:")
            print("  python run_integration_tests.py all       # Run all tests")
            print("  python run_integration_tests.py fast      # Skip slow tests")
            print("  python run_integration_tests.py class     # Run specific test class")
            print("  python run_integration_tests.py single    # Run single test")
            print("  python run_integration_tests.py           # Interactive mode")
            return 1
    
    # Interactive mode
    while True:
        print()
        print("=" * 80)
        print("ML PIPELINE INTEGRATION TEST RUNNER")
        print("=" * 80)
        print()
        print("Options:")
        print("  1. Run all integration tests")
        print("  2. Run specific test suite")
        print("  3. Run specific test class")
        print("  4. Run single test")
        print("  5. Run fast tests (skip slow)")
        print("  6. Exit")
        print()
        
        choice = input("Your choice (1-6): ").strip()
        
        if choice == "1":
            return run_integration_tests()
        elif choice == "2":
            return run_integration_tests()
        elif choice == "3":
            return run_specific_test_class()
        elif choice == "4":
            return run_single_test()
        elif choice == "5":
            sys.argv.append("fast")
            return run_integration_tests()
        elif choice == "6":
            print("Exiting...")
            return 0
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    sys.exit(main())
