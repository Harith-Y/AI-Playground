"""
Unit tests for classification metrics module.

Tests cover:
- Binary classification metrics
- Multi-class classification metrics
- Probability-based metrics (AUC-ROC, AUC-PR)
- Confusion matrix calculations
- Per-class metrics
- Edge cases and error handling
"""

import pytest
import numpy as np
import pandas as pd
from app.ml_engine.evaluation.classification_metrics import (
    ClassificationMetricsCalculator,
    ClassificationMetrics,
    calculate_classification_metrics
)


class TestClassificationMetricsCalculator:
    """Test suite for ClassificationMetricsCalculator."""
    
    def test_initialization(self):
        """Test calculator initialization with different parameters."""
        # Default initialization
        calc = ClassificationMetricsCalculator()
        assert calc.average == 'weighted'
        assert calc.pos_label == 1
        
        # Custom initialization
        calc = ClassificationMetricsCalculator(
            average='macro',
            pos_label=0,
            labels=[0, 1, 2]
        )
        assert calc.average == 'macro'
        assert calc.pos_label == 0
        assert calc.labels == [0, 1, 2]
    
    def test_binary_classification_perfect(self):
        """Test metrics for perfect binary classification."""
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 1, 0, 1]
        
        calc = ClassificationMetricsCalculator(average='binary')
        metrics = calc.calculate_metrics(y_true, y_pred)
        
        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.n_samples == 6
        assert metrics.n_classes == 2
    
    def test_binary_classification_with_errors(self):
        """Test metrics for binary classification with some errors."""
        y_true = [0, 1, 1, 0, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1, 1, 1, 0]
        
        calc = ClassificationMetricsCalculator(average='binary')
        metrics = calc.calculate_metrics(y_true, y_pred)
        
        # Accuracy: 6/8 = 0.75
        assert metrics.accuracy == 0.75
        
        # Precision: TP/(TP+FP) = 3/(3+1) = 0.75
        assert metrics.precision == 0.75
        
        # Recall: TP/(TP+FN) = 3/(3+1) = 0.75
        assert metrics.recall == 0.75
        
        # F1: 2 * (0.75 * 0.75) / (0.75 + 0.75) = 0.75
        assert metrics.f1_score == 0.75
        
        assert metrics.n_samples == 8
        assert metrics.n_classes == 2
    
    def test_binary_classification_with_probabilities(self):
        """Test binary classification with probability predictions."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_proba = np.array([
            [0.9, 0.1],  # Confident 0
            [0.2, 0.8],  # Confident 1
            [0.6, 0.4],  # Weak 0 (wrong)
            [0.8, 0.2],  # Confident 0
            [0.1, 0.9]   # Confident 1
        ])
        
        calc = ClassificationMetricsCalculator(average='binary')
        metrics = calc.calculate_metrics(y_true, y_pred, y_proba)
        
        assert metrics.accuracy == 0.8
        assert metrics.auc_roc is not None
        assert 0.0 <= metrics.auc_roc <= 1.0
        assert metrics.auc_pr is not None
        assert 0.0 <= metrics.auc_pr <= 1.0
        assert metrics.log_loss is not None
        assert metrics.log_loss >= 0.0
    
    def test_binary_classification_proba_1d(self):
        """Test binary classification with 1D probability array."""
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        y_proba = [0.1, 0.8, 0.4, 0.2, 0.9]  # Probabilities for class 1
        
        calc = ClassificationMetricsCalculator(average='binary')
        metrics = calc.calculate_metrics(y_true, y_pred, y_proba)
        
        assert metrics.auc_roc is not None
        assert 0.0 <= metrics.auc_roc <= 1.0
    
    def test_multiclass_classification(self):
        """Test metrics for multi-class classification."""
        y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 2, 2, 1, 1, 2]
        
        calc = ClassificationMetricsCalculator(average='weighted')
        metrics = calc.calculate_metrics(y_true, y_pred)
        
        # Accuracy: 7/9 = 0.7778
        assert abs(metrics.accuracy - 0.7778) < 0.01
        assert metrics.n_classes == 3
        assert metrics.n_samples == 9
        assert metrics.confusion_matrix is not None
        assert metrics.confusion_matrix.shape == (3, 3)
    
    def test_multiclass_with_probabilities(self):
        """Test multi-class classification with probabilities."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1])
        y_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.2, 0.5, 0.3]  # Wrong prediction
        ])
        
        calc = ClassificationMetricsCalculator(average='weighted')
        metrics = calc.calculate_metrics(y_true, y_pred, y_proba)
        
        assert metrics.auc_roc is not None
        assert 0.0 <= metrics.auc_roc <= 1.0
        assert metrics.log_loss is not None
    
    def test_per_class_metrics(self):
        """Test per-class metrics calculation."""
        y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 2, 2, 1, 1, 2]
        class_names = ['cat', 'dog', 'bird']
        
        calc = ClassificationMetricsCalculator(average='weighted')
        metrics = calc.calculate_metrics(
            y_true, y_pred,
            class_names=class_names,
            include_per_class=True
        )
        
        assert metrics.per_class_metrics is not None
        assert 'cat' in metrics.per_class_metrics
        assert 'dog' in metrics.per_class_metrics
        assert 'bird' in metrics.per_class_metrics
        
        # Check structure of per-class metrics
        for class_name in class_names:
            class_metrics = metrics.per_class_metrics[class_name]
            assert 'precision' in class_metrics
            assert 'recall' in class_metrics
            assert 'f1_score' in class_metrics
            assert 'support' in class_metrics
    
    def test_per_class_metrics_with_probabilities(self):
        """Test per-class metrics with probability predictions."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 1]
        y_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.2, 0.5, 0.3]
        ])
        class_names = ['A', 'B', 'C']
        
        calc = ClassificationMetricsCalculator(average='weighted')
        metrics = calc.calculate_metrics(
            y_true, y_pred, y_proba,
            class_names=class_names,
            include_per_class=True
        )
        
        # Check that AUC is included in per-class metrics
        for class_name in class_names:
            class_metrics = metrics.per_class_metrics[class_name]
            assert 'auc_roc' in class_metrics
    
    def test_advanced_metrics(self):
        """Test advanced metrics calculation."""
        y_true = [0, 1, 1, 0, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1, 1, 1, 0]
        
        calc = ClassificationMetricsCalculator(average='binary')
        metrics = calc.calculate_metrics(
            y_true, y_pred,
            include_advanced=True
        )
        
        assert metrics.balanced_accuracy is not None
        assert 0.0 <= metrics.balanced_accuracy <= 1.0
        assert metrics.matthews_corrcoef is not None
        assert -1.0 <= metrics.matthews_corrcoef <= 1.0
        assert metrics.cohen_kappa is not None
        assert -1.0 <= metrics.cohen_kappa <= 1.0
    
    def test_confusion_matrix_calculation(self):
        """Test confusion matrix calculation."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 1, 1, 2]
        
        calc = ClassificationMetricsCalculator()
        result = calc.calculate_confusion_matrix(y_true, y_pred)
        
        assert 'matrix' in result
        assert 'class_names' in result
        assert 'normalized' in result
        assert result['matrix'].shape == (3, 3)
        assert result['normalized'] is False
    
    def test_confusion_matrix_normalized(self):
        """Test normalized confusion matrix."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 1, 1, 2]
        
        calc = ClassificationMetricsCalculator()
        
        # Normalize over true labels (rows sum to 1)
        result = calc.calculate_confusion_matrix(y_true, y_pred, normalize='true')
        assert result['normalized'] is True
        # Check that rows sum to 1
        row_sums = result['matrix'].sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3))
    
    def test_classification_report(self):
        """Test classification report generation."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 1, 1, 2]
        class_names = ['cat', 'dog', 'bird']
        
        calc = ClassificationMetricsCalculator()
        
        # Get report as dict
        report_dict = calc.get_classification_report(
            y_true, y_pred,
            class_names=class_names,
            output_dict=True
        )
        
        assert isinstance(report_dict, dict)
        assert 'cat' in report_dict
        assert 'dog' in report_dict
        assert 'bird' in report_dict
        assert 'accuracy' in report_dict
        
        # Get report as string
        report_str = calc.get_classification_report(
            y_true, y_pred,
            class_names=class_names,
            output_dict=False
        )
        
        assert isinstance(report_str, str)
        assert 'cat' in report_str
        assert 'precision' in report_str
    
    def test_pandas_input(self):
        """Test that pandas Series/DataFrame inputs work."""
        y_true = pd.Series([0, 1, 1, 0, 1])
        y_pred = pd.Series([0, 1, 0, 0, 1])
        y_proba = pd.DataFrame({
            0: [0.9, 0.2, 0.6, 0.8, 0.1],
            1: [0.1, 0.8, 0.4, 0.2, 0.9]
        })
        
        calc = ClassificationMetricsCalculator(average='binary')
        metrics = calc.calculate_metrics(y_true, y_pred, y_proba)
        
        assert metrics.accuracy == 0.8
        assert metrics.auc_roc is not None
    
    def test_list_input(self):
        """Test that list inputs work."""
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        
        calc = ClassificationMetricsCalculator(average='binary')
        metrics = calc.calculate_metrics(y_true, y_pred)
        
        assert metrics.accuracy == 0.8
    
    def test_mismatched_lengths(self):
        """Test error handling for mismatched input lengths."""
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0]  # Different length
        
        calc = ClassificationMetricsCalculator()
        
        with pytest.raises(ValueError, match="must have same length"):
            calc.calculate_metrics(y_true, y_pred)
    
    def test_mismatched_proba_length(self):
        """Test error handling for mismatched probability length."""
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]
        y_proba = [[0.9, 0.1], [0.2, 0.8]]  # Different length
        
        calc = ClassificationMetricsCalculator()
        
        with pytest.raises(ValueError, match="must have same number of samples"):
            calc.calculate_metrics(y_true, y_pred, y_proba)
    
    def test_zero_division_handling(self):
        """Test handling of zero division in metrics."""
        # All predictions are class 0, but true labels include class 1
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 0, 0]
        
        calc = ClassificationMetricsCalculator(average='binary', zero_division=0)
        metrics = calc.calculate_metrics(y_true, y_pred)
        
        # Should not raise error
        assert metrics.precision == 0.0  # No TP for class 1
        assert metrics.recall == 0.0
    
    def test_single_class_prediction(self):
        """Test metrics when all predictions are same class."""
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 0, 0, 0, 0, 0]  # All predicted as class 0
        
        calc = ClassificationMetricsCalculator(average='binary')
        metrics = calc.calculate_metrics(y_true, y_pred)
        
        assert metrics.accuracy == 0.5  # Only class 0 correct
        assert metrics.recall == 0.0  # No class 1 predicted correctly
    
    def test_metrics_to_dict(self):
        """Test conversion of metrics to dictionary."""
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        
        calc = ClassificationMetricsCalculator(average='binary')
        metrics = calc.calculate_metrics(y_true, y_pred)
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert 'accuracy' in metrics_dict
        assert 'precision' in metrics_dict
        assert 'recall' in metrics_dict
        assert 'f1_score' in metrics_dict
        assert 'n_samples' in metrics_dict
        assert 'n_classes' in metrics_dict
        
        # Check that confusion matrix is converted to list
        if metrics_dict['confusion_matrix'] is not None:
            assert isinstance(metrics_dict['confusion_matrix'], list)
    
    def test_metrics_repr(self):
        """Test string representation of metrics."""
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        
        calc = ClassificationMetricsCalculator(average='binary')
        metrics = calc.calculate_metrics(y_true, y_pred)
        
        repr_str = repr(metrics)
        
        assert 'ClassificationMetrics' in repr_str
        assert 'accuracy' in repr_str
        assert 'precision' in repr_str
        assert 'n_samples=5' in repr_str
    
    def test_averaging_strategies(self):
        """Test different averaging strategies for multi-class."""
        y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 2, 2, 1, 1, 2]
        
        # Test macro averaging
        calc_macro = ClassificationMetricsCalculator(average='macro')
        metrics_macro = calc_macro.calculate_metrics(y_true, y_pred)
        
        # Test micro averaging
        calc_micro = ClassificationMetricsCalculator(average='micro')
        metrics_micro = calc_micro.calculate_metrics(y_true, y_pred)
        
        # Test weighted averaging
        calc_weighted = ClassificationMetricsCalculator(average='weighted')
        metrics_weighted = calc_weighted.calculate_metrics(y_true, y_pred)
        
        # All should have same accuracy
        assert metrics_macro.accuracy == metrics_micro.accuracy == metrics_weighted.accuracy
        
        # But different precision/recall/f1 due to averaging
        # (unless perfectly balanced, which this isn't)
        assert metrics_macro.precision is not None
        assert metrics_micro.precision is not None
        assert metrics_weighted.precision is not None


class TestConvenienceFunction:
    """Test suite for convenience function."""
    
    def test_calculate_classification_metrics_function(self):
        """Test the convenience function."""
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        
        metrics = calculate_classification_metrics(
            y_true, y_pred,
            average='binary'
        )
        
        assert isinstance(metrics, ClassificationMetrics)
        assert metrics.accuracy == 0.8
        assert metrics.precision is not None
        assert metrics.recall is not None
        assert metrics.f1_score is not None
    
    def test_convenience_function_with_all_params(self):
        """Test convenience function with all parameters."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 1, 1, 2]
        y_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.3, 0.5, 0.2],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])
        class_names = ['A', 'B', 'C']
        
        metrics = calculate_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            average='weighted',
            class_names=class_names,
            include_per_class=True,
            include_advanced=True
        )
        
        assert metrics.per_class_metrics is not None
        assert metrics.balanced_accuracy is not None
        assert metrics.auc_roc is not None
        assert 'A' in metrics.per_class_metrics


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        
        metrics = calculate_classification_metrics(y_true, y_pred)
        
        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
    
    def test_all_wrong_predictions(self):
        """Test with all wrong predictions (binary)."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [1, 1, 1, 0, 0, 0]
        
        metrics = calculate_classification_metrics(y_true, y_pred, average='binary')
        
        assert metrics.accuracy == 0.0
    
    def test_imbalanced_classes(self):
        """Test with highly imbalanced classes."""
        y_true = [0] * 90 + [1] * 10
        y_pred = [0] * 85 + [1] * 5 + [0] * 10  # Predict mostly class 0
        
        metrics = calculate_classification_metrics(
            y_true, y_pred,
            average='binary',
            include_advanced=True
        )
        
        # Accuracy might be high due to imbalance
        assert metrics.accuracy > 0.8
        
        # But balanced accuracy should be lower
        assert metrics.balanced_accuracy is not None
        assert metrics.balanced_accuracy < metrics.accuracy
    
    def test_empty_class_names(self):
        """Test with empty class names list."""
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        
        metrics = calculate_classification_metrics(
            y_true, y_pred,
            class_names=[],  # Empty list
            include_per_class=True
        )
        
        # Should use default names
        assert metrics.per_class_metrics is not None
        assert 'class_0' in metrics.per_class_metrics
    
    def test_large_number_of_classes(self):
        """Test with many classes."""
        n_classes = 20
        n_samples = 200
        
        np.random.seed(42)
        y_true = np.random.randint(0, n_classes, n_samples)
        y_pred = np.random.randint(0, n_classes, n_samples)
        
        metrics = calculate_classification_metrics(
            y_true, y_pred,
            average='weighted'
        )
        
        assert metrics.n_classes == n_classes
        assert metrics.n_samples == n_samples
        assert metrics.confusion_matrix.shape == (n_classes, n_classes)
