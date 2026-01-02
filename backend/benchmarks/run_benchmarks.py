"""
Comprehensive benchmarking script for AI-Playground ML pipeline.

This script:
1. Prepares benchmark datasets
2. Trains multiple models on each dataset
3. Collects performance metrics (accuracy, training time, memory usage)
4. Generates benchmark report
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml_engine.models.classification import ClassificationModel
from app.ml_engine.models.regression import RegressionModel
from app.utils.memory_manager import MemoryMonitor, memory_profiler, MemoryOptimizer
from app.utils.logger import get_logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

logger = get_logger(__name__)


class BenchmarkRunner:
    """Run benchmarks on standard datasets."""

    def __init__(self, datasets_dir: Path, output_dir: Path):
        """
        Initialize benchmark runner.

        Args:
            datasets_dir: Directory containing benchmark datasets
            output_dir: Directory to save results
        """
        self.datasets_dir = datasets_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = []
        self.memory_monitor = MemoryMonitor()

    def run_classification_benchmark(
        self,
        dataset_name: str,
        dataset_path: Path,
        target_column: str,
        model_types: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Run classification benchmarks.

        Args:
            dataset_name: Name of dataset
            dataset_path: Path to CSV file
            target_column: Target column name
            model_types: List of model types to test

        Returns:
            List of benchmark results
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Classification Benchmark: {dataset_name}")
        logger.info(f"{'=' * 60}")

        # Load data
        df = pd.read_csv(dataset_path)
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Target distribution:\n{df[target_column].value_counts()}")

        # Prepare features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        results = []

        for model_type in model_types:
            logger.info(f"\n--- Testing {model_type} ---")

            try:
                # Initialize memory monitoring
                self.memory_monitor.set_baseline()
                baseline_memory = self.memory_monitor.get_current_snapshot()

                # Train model with memory profiling
                start_time = time.time()

                with memory_profiler(f"{dataset_name} - {model_type}"):
                    model = ClassificationModel(model_type=model_type)
                    model.fit(X_train, y_train)

                training_time = time.time() - start_time

                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model.model, 'predict_proba') else None

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)

                # Handle multi-class vs binary
                n_classes = len(np.unique(y))
                avg_type = 'binary' if n_classes == 2 else 'weighted'

                precision = precision_score(y_test, y_pred, average=avg_type, zero_division=0)
                recall = recall_score(y_test, y_pred, average=avg_type, zero_division=0)
                f1 = f1_score(y_test, y_pred, average=avg_type, zero_division=0)

                # Memory metrics
                final_memory = self.memory_monitor.get_current_snapshot()
                memory_delta = self.memory_monitor.get_memory_delta()

                # Store results
                result = {
                    'dataset': dataset_name,
                    'dataset_path': str(dataset_path),
                    'task_type': 'classification',
                    'model_type': model_type,
                    'n_samples': len(df),
                    'n_features': X.shape[1],
                    'n_classes': n_classes,
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'training_time': round(training_time, 4),
                    'accuracy': round(accuracy, 4),
                    'precision': round(precision, 4),
                    'recall': round(recall, 4),
                    'f1_score': round(f1, 4),
                    'baseline_memory_mb': round(baseline_memory.rss_mb, 2),
                    'final_memory_mb': round(final_memory.rss_mb, 2),
                    'memory_delta_mb': round(memory_delta, 2),
                    'timestamp': datetime.now().isoformat()
                }

                results.append(result)

                # Log results
                logger.info(f"✓ Accuracy: {accuracy:.4f}")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")
                logger.info(f"  F1 Score: {f1:.4f}")
                logger.info(f"  Training time: {training_time:.2f}s")
                logger.info(f"  Memory delta: {memory_delta:+.2f}MB")

                # Clean up
                del model
                MemoryOptimizer.aggressive_gc()

            except Exception as e:
                logger.error(f"✗ Error training {model_type}: {e}", exc_info=True)

        return results

    def run_regression_benchmark(
        self,
        dataset_name: str,
        dataset_path: Path,
        target_column: str,
        model_types: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Run regression benchmarks.

        Args:
            dataset_name: Name of dataset
            dataset_path: Path to CSV file
            target_column: Target column name
            model_types: List of model types to test

        Returns:
            List of benchmark results
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Regression Benchmark: {dataset_name}")
        logger.info(f"{'=' * 60}")

        # Load data
        df = pd.read_csv(dataset_path)
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Target statistics:\n{df[target_column].describe()}")

        # Prepare features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        results = []

        for model_type in model_types:
            logger.info(f"\n--- Testing {model_type} ---")

            try:
                # Initialize memory monitoring
                self.memory_monitor.set_baseline()
                baseline_memory = self.memory_monitor.get_current_snapshot()

                # Train model with memory profiling
                start_time = time.time()

                with memory_profiler(f"{dataset_name} - {model_type}"):
                    model = RegressionModel(model_type=model_type)
                    model.fit(X_train, y_train)

                training_time = time.time() - start_time

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Memory metrics
                final_memory = self.memory_monitor.get_current_snapshot()
                memory_delta = self.memory_monitor.get_memory_delta()

                # Store results
                result = {
                    'dataset': dataset_name,
                    'dataset_path': str(dataset_path),
                    'task_type': 'regression',
                    'model_type': model_type,
                    'n_samples': len(df),
                    'n_features': X.shape[1],
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'training_time': round(training_time, 4),
                    'mse': round(mse, 4),
                    'rmse': round(rmse, 4),
                    'mae': round(mae, 4),
                    'r2_score': round(r2, 4),
                    'baseline_memory_mb': round(baseline_memory.rss_mb, 2),
                    'final_memory_mb': round(final_memory.rss_mb, 2),
                    'memory_delta_mb': round(memory_delta, 2),
                    'timestamp': datetime.now().isoformat()
                }

                results.append(result)

                # Log results
                logger.info(f"✓ R² Score: {r2:.4f}")
                logger.info(f"  RMSE: {rmse:.4f}")
                logger.info(f"  MAE: {mae:.4f}")
                logger.info(f"  Training time: {training_time:.2f}s")
                logger.info(f"  Memory delta: {memory_delta:+.2f}MB")

                # Clean up
                del model
                MemoryOptimizer.aggressive_gc()

            except Exception as e:
                logger.error(f"✗ Error training {model_type}: {e}", exc_info=True)

        return results

    def save_results(self):
        """Save benchmark results to files."""
        if not self.results:
            logger.warning("No results to save")
            return

        # Save as JSON
        json_path = self.output_dir / "benchmark_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\n✓ Results saved to: {json_path}")

        # Save as CSV
        csv_path = self.output_dir / "benchmark_results.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(csv_path, index=False)
        logger.info(f"✓ Results saved to: {csv_path}")

        return json_path, csv_path

    def generate_report(self):
        """Generate markdown report from results."""
        if not self.results:
            logger.warning("No results to generate report")
            return

        df = pd.DataFrame(self.results)

        # Group by task type
        classification_results = df[df['task_type'] == 'classification']
        regression_results = df[df['task_type'] == 'regression']

        report_lines = [
            "# Benchmark Results",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"**Total Benchmarks:** {len(self.results)}",
            "",
            "## Summary",
            "",
        ]

        # Classification summary
        if not classification_results.empty:
            report_lines.extend([
                "### Classification Tasks",
                "",
                f"**Datasets:** {classification_results['dataset'].nunique()}",
                f"**Models tested:** {classification_results['model_type'].nunique()}",
                f"**Total runs:** {len(classification_results)}",
                "",
                "#### Best Results by Dataset",
                ""
            ])

            for dataset in classification_results['dataset'].unique():
                dataset_results = classification_results[classification_results['dataset'] == dataset]
                best_result = dataset_results.loc[dataset_results['accuracy'].idxmax()]

                report_lines.extend([
                    f"**{dataset}**",
                    f"- Best model: {best_result['model_type']}",
                    f"- Accuracy: {best_result['accuracy']:.4f}",
                    f"- F1 Score: {best_result['f1_score']:.4f}",
                    f"- Training time: {best_result['training_time']:.2f}s",
                    f"- Memory usage: {best_result['memory_delta_mb']:+.2f}MB",
                    ""
                ])

        # Regression summary
        if not regression_results.empty:
            report_lines.extend([
                "### Regression Tasks",
                "",
                f"**Datasets:** {regression_results['dataset'].nunique()}",
                f"**Models tested:** {regression_results['model_type'].nunique()}",
                f"**Total runs:** {len(regression_results)}",
                "",
                "#### Best Results by Dataset",
                ""
            ])

            for dataset in regression_results['dataset'].unique():
                dataset_results = regression_results[regression_results['dataset'] == dataset]
                best_result = dataset_results.loc[dataset_results['r2_score'].idxmax()]

                report_lines.extend([
                    f"**{dataset}**",
                    f"- Best model: {best_result['model_type']}",
                    f"- R² Score: {best_result['r2_score']:.4f}",
                    f"- RMSE: {best_result['rmse']:.4f}",
                    f"- Training time: {best_result['training_time']:.2f}s",
                    f"- Memory usage: {best_result['memory_delta_mb']:+.2f}MB",
                    ""
                ])

        # Detailed results tables
        report_lines.extend([
            "## Detailed Results",
            "",
        ])

        if not classification_results.empty:
            report_lines.extend([
                "### Classification Results",
                "",
                "| Dataset | Model | Accuracy | Precision | Recall | F1 | Time (s) | Memory (MB) |",
                "|---------|-------|----------|-----------|--------|-------|----------|-------------|"
            ])

            for _, row in classification_results.iterrows():
                report_lines.append(
                    f"| {row['dataset']} | {row['model_type']} | "
                    f"{row['accuracy']:.4f} | {row['precision']:.4f} | "
                    f"{row['recall']:.4f} | {row['f1_score']:.4f} | "
                    f"{row['training_time']:.2f} | {row['memory_delta_mb']:+.2f} |"
                )

            report_lines.append("")

        if not regression_results.empty:
            report_lines.extend([
                "### Regression Results",
                "",
                "| Dataset | Model | R² | RMSE | MAE | Time (s) | Memory (MB) |",
                "|---------|-------|-----|------|-----|----------|-------------|"
            ])

            for _, row in regression_results.iterrows():
                report_lines.append(
                    f"| {row['dataset']} | {row['model_type']} | "
                    f"{row['r2_score']:.4f} | {row['rmse']:.4f} | "
                    f"{row['mae']:.4f} | {row['training_time']:.2f} | "
                    f"{row['memory_delta_mb']:+.2f} |"
                )

            report_lines.append("")

        # Performance insights
        report_lines.extend([
            "## Performance Insights",
            "",
            "### Training Time",
            ""
        ])

        fastest = df.loc[df['training_time'].idxmin()]
        slowest = df.loc[df['training_time'].idxmax()]

        report_lines.extend([
            f"- **Fastest:** {fastest['model_type']} on {fastest['dataset']} ({fastest['training_time']:.2f}s)",
            f"- **Slowest:** {slowest['model_type']} on {slowest['dataset']} ({slowest['training_time']:.2f}s)",
            f"- **Average:** {df['training_time'].mean():.2f}s",
            "",
            "### Memory Usage",
            ""
        ])

        lowest_mem = df.loc[df['memory_delta_mb'].idxmin()]
        highest_mem = df.loc[df['memory_delta_mb'].idxmax()]

        report_lines.extend([
            f"- **Lowest:** {lowest_mem['model_type']} on {lowest_mem['dataset']} ({lowest_mem['memory_delta_mb']:+.2f}MB)",
            f"- **Highest:** {highest_mem['model_type']} on {highest_mem['dataset']} ({highest_mem['memory_delta_mb']:+.2f}MB)",
            f"- **Average:** {df['memory_delta_mb'].mean():+.2f}MB",
            ""
        ])

        # Save report
        report_path = self.output_dir / "BENCHMARK_REPORT.md"
        report_path.write_text("\n".join(report_lines))
        logger.info(f"✓ Report generated: {report_path}")

        return report_path


def main():
    """Run all benchmarks."""
    logger.info("\n" + "=" * 70)
    logger.info("AI-Playground ML Pipeline Benchmarking")
    logger.info("=" * 70)

    # Setup paths
    benchmark_dir = Path(__file__).parent
    datasets_dir = benchmark_dir / "datasets"
    results_dir = benchmark_dir / "results"

    # Initialize benchmark runner
    runner = BenchmarkRunner(datasets_dir, results_dir)

    # Define benchmarks
    benchmarks = [
        # Classification benchmarks
        {
            'type': 'classification',
            'dataset': 'Iris',
            'path': datasets_dir / 'iris.csv',
            'target': 'species',
            'models': ['random_forest_classifier', 'logistic_regression', 'svm_classifier', 'knn_classifier']
        },
        {
            'type': 'classification',
            'dataset': 'Wine',
            'path': datasets_dir / 'wine.csv',
            'target': 'cultivar',
            'models': ['random_forest_classifier', 'gradient_boosting_classifier', 'decision_tree_classifier']
        },
        {
            'type': 'classification',
            'dataset': 'Breast Cancer',
            'path': datasets_dir / 'breast_cancer.csv',
            'target': 'diagnosis',
            'models': ['random_forest_classifier', 'logistic_regression', 'gradient_boosting_classifier']
        },
        {
            'type': 'classification',
            'dataset': 'Digits',
            'path': datasets_dir / 'digits.csv',
            'target': 'digit',
            'models': ['random_forest_classifier', 'svm_classifier', 'knn_classifier']
        },
        # Regression benchmarks
        {
            'type': 'regression',
            'dataset': 'Diabetes',
            'path': datasets_dir / 'diabetes.csv',
            'target': 'progression',
            'models': ['random_forest_regressor', 'linear_regression', 'gradient_boosting_regressor']
        },
        {
            'type': 'regression',
            'dataset': 'California Housing',
            'path': datasets_dir / 'california_housing.csv',
            'target': 'median_house_value',
            'models': ['random_forest_regressor', 'gradient_boosting_regressor', 'ridge_regression']
        }
    ]

    # Run benchmarks
    total_start = time.time()

    for benchmark in benchmarks:
        if benchmark['type'] == 'classification':
            results = runner.run_classification_benchmark(
                dataset_name=benchmark['dataset'],
                dataset_path=benchmark['path'],
                target_column=benchmark['target'],
                model_types=benchmark['models']
            )
        else:
            results = runner.run_regression_benchmark(
                dataset_name=benchmark['dataset'],
                dataset_path=benchmark['path'],
                target_column=benchmark['target'],
                model_types=benchmark['models']
            )

        runner.results.extend(results)

    total_time = time.time() - total_start

    # Save results and generate report
    logger.info("\n" + "=" * 70)
    logger.info("Saving Results")
    logger.info("=" * 70)

    runner.save_results()
    runner.generate_report()

    logger.info("\n" + "=" * 70)
    logger.info(f"✓ Benchmarking complete!")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Total benchmarks: {len(runner.results)}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
