"""
Unit tests for confusion matrix module.

Tests cover:
- Basic confusion matrix computation
- Normalization modes (true, pred, all)
- Per-class statistics (TP, FP, TN, FN)
- Overall statistics
- Misclassification analysis
- Cost-sensitive analysis
- Error analysis
- DataFrame conversion
- Edge cases
"""

import pytest
import numpy as np
import pandas as pd
from app.ml_engine.evaluation.confusion_matrix import (
    ConfusionMatrixCalculator,
    ConfusionMatrixResult,
    compute_confusion_matrix
)


class TestConfusionMatrixCalculator:
    """Test suite for ConfusionMatrixCalculator."""
    
    def test_initialization(self):
        """Test calculator initialization."""
        calc = ConfusionMatrixCalculator()
        assert calc.labels is None
        assert calc.sample_weight is None
        
        calc_with_labels = ConfusionMatrixCalculator(labels=[0, 1, 2])
        assert calc_with_labels.labels == [0, 1, 2]
    
    def test_binary_confusion_matrix(self):
        """Test basic binary confusion matrix."""
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 0, 1]
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred)
        
        # Expected matrix:
        # [[3, 0],
        #  [1, 2]]
        assert result.matrix.shape == (2, 2)
        assert result.matrix[0, 0] == 3  # TN
        assert result.matrix[0, 1] == 0  # FP
        assert result.matrix[1, 0] == 1  # FN
        assert result.matrix[1, 1] == 2  # TP
        assert result.n_classes == 2
        assert result.n_samples == 6
    
    def test_multiclass_confusion_matrix(self):
        """Test multi-class confusion matrix."""
        y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 2, 2, 1, 1, 2]
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred)
        
        assert result.matrix.shape == (3, 3)
        assert result.n_classes == 3
        assert result.n_samples == 9
        # Check diagonal (correct predictions)
        assert result.matrix[0, 0] == 2  # Class 0
        assert result.matrix[1, 1] == 2  # Class 1
        assert result.matrix[2, 2] == 3  # Class 2
    
    def test_normalize_true(self):
        """Test normalization over true labels (rows)."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred, normalize='true')
        
        # Each row should sum to 1
        row_sums = result.normalized_matrix.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(2))
        
        # Check values
        assert result.normalized_matrix[0, 0] == 0.5  # 1/2
        assert result.normalized_matrix[0, 1] == 0.5  # 1/2
        assert result.normalized_matrix[1, 0] == 0.0  # 0/2
        assert result.normalized_matrix[1, 1] == 1.0  # 2/2
    
    def test_normalize_pred(self):
        """Test normalization over predicted labels (columns)."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred, normalize='pred')
        
        # Each column should sum to 1
        col_sums = result.normalized_matrix.sum(axis=0)
        np.testing.assert_array_almost_equal(col_sums, np.ones(2))
    
    def test_normalize_all(self):
        """Test normalization over all samples."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred, normalize='all')
        
        # All values should sum to 1
        total = result.normalized_matrix.sum()
        assert abs(total - 1.0) < 1e-10
    
    def test_per_class_stats_binary(self):
        """Test per-class statistics for binary classification."""
        y_true = [0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 0, 1, 1, 1]
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred, class_names=['neg', 'pos'])
        
        # Check class 0 (negative)
        stats_neg = result.per_class_stats['neg']
        assert stats_neg['true_positives'] == 3
        assert stats_neg['false_positives'] == 1
        assert stats_neg['false_negatives'] == 1
        assert stats_neg['true_negatives'] == 3
        assert stats_neg['support'] == 4
        
        # Check class 1 (positive)
        stats_pos = result.per_class_stats['pos']
        assert stats_pos['true_positives'] == 3
        assert stats_pos['false_positives'] == 1
        assert stats_pos['false_negatives'] == 1
        assert stats_pos['true_negatives'] == 3
        assert stats_pos['support'] == 4
    
    def test_per_class_stats_multiclass(self):
        """Test per-class statistics for multi-class classification."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 1, 1, 2]
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred, class_names=['A', 'B', 'C'])
        
        # Check class A
        stats_a = result.per_class_stats['A']
        assert stats_a['true_positives'] == 1
        assert stats_a['false_positives'] == 0
        assert stats_a['false_negatives'] == 1
        assert stats_a['support'] == 2
        
        # Check that all classes have required keys
        for cls_name in ['A', 'B', 'C']:
            stats = result.per_class_stats[cls_name]
            assert 'true_positives' in stats
            assert 'false_positives' in stats
            assert 'true_negatives' in stats
            assert 'false_negatives' in stats
            assert 'sensitivity' in stats
            assert 'specificity' in stats
            assert 'precision' in stats
            assert 'recall' in stats
            assert 'f1_score' in stats
            assert 'accuracy' in stats
            assert 'support' in stats
    
    def test_overall_stats(self):
        """Test overall statistics calculation."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 1, 1, 2]
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred)
        
        assert result.overall_stats['total_samples'] == 6
        assert result.overall_stats['correct_predictions'] == 5
        assert result.overall_stats['incorrect_predictions'] == 1
        assert abs(result.overall_stats['accuracy'] - 5/6) < 1e-10
        assert abs(result.overall_stats['error_rate'] - 1/6) < 1e-10
    
    def test_misclassification_matrix(self):
        """Test misclassification matrix computation."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 1, 1, 2]
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred, include_misclassification=True)
        
        # Misclassification matrix should have 0 on diagonal
        assert np.all(np.diag(result.misclassification_matrix) == 0)
        
        # Check off-diagonal elements
        assert result.misclassification_matrix[0, 1] == 1  # 0 predicted as 1
        assert result.misclassification_matrix[1, 0] == 0  # No 1 predicted as 0
    
    def test_cost_sensitive_analysis(self):
        """Test cost-sensitive confusion matrix analysis."""
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 0, 1]
        
        # Cost matrix: FN costs 10, FP costs 1
        cost_matrix = np.array([
            [0, 1],   # True 0: correct=0, predict as 1=1
            [10, 0]   # True 1: predict as 0=10, correct=0
        ])
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute_cost_sensitive(y_true, y_pred, cost_matrix)
        
        # Expected: 1 FN (cost 10) + 0 FP (cost 0) = 10
        assert result['total_cost'] == 10.0
        assert result['average_cost_per_sample'] == 10.0 / 6
        assert 'cost_matrix' in result
        assert 'cost_by_true_class' in result
        assert 'cost_by_pred_class' in result
    
    def test_error_analysis(self):
        """Test error analysis functionality."""
        y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 1, 2, 2, 1, 1, 2]
        
        calc = ConfusionMatrixCalculator()
        result = calc.analyze_errors(
            y_true, y_pred,
            class_names=['cat', 'dog', 'bird'],
            top_n=3
        )
        
        assert 'top_misclassifications' in result
        assert 'error_rates_by_class' in result
        assert 'total_errors' in result
        assert 'total_samples' in result
        
        # Check structure of top misclassifications
        if len(result['top_misclassifications']) > 0:
            top_error = result['top_misclassifications'][0]
            assert 'true_class' in top_error
            assert 'predicted_class' in top_error
            assert 'count' in top_error
            assert 'percentage' in top_error
        
        # Check error rates structure
        assert len(result['error_rates_by_class']) == 3
        for error_info in result['error_rates_by_class']:
            assert 'class' in error_info
            assert 'error_count' in error_info
            assert 'total_samples' in error_info
            assert 'error_rate' in error_info
    
    def test_to_dataframe(self):
        """Test conversion to pandas DataFrame."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 1, 1, 2]
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred, class_names=['A', 'B', 'C'])
        
        df = result.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)
        assert 'True_A' in df.index
        assert 'Pred_A' in df.columns
    
    def test_to_normalized_dataframe(self):
        """Test conversion of normalized matrix to DataFrame."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred, normalize='true')
        
        df = result.to_normalized_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred, class_names=['neg', 'pos'])
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'matrix' in result_dict
        assert 'n_classes' in result_dict
        assert 'n_samples' in result_dict
        assert 'per_class_stats' in result_dict
        
        # Check that matrix is converted to list
        assert isinstance(result_dict['matrix'], list)
    
    def test_with_class_names(self):
        """Test with custom class names."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 1, 1, 2]
        class_names = ['cat', 'dog', 'bird']
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred, class_names=class_names)
        
        assert result.class_names == class_names
        assert 'cat' in result.per_class_stats
        assert 'dog' in result.per_class_stats
        assert 'bird' in result.per_class_stats
    
    def test_without_stats(self):
        """Test computation without per-class statistics."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred, include_stats=False)
        
        assert result.per_class_stats is None
        assert result.overall_stats is not None  # Overall stats always computed
    
    def test_without_misclassification(self):
        """Test computation without misclassification analysis."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred, include_misclassification=False)
        
        assert result.misclassification_matrix is None
    
    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred)
        
        # All diagonal, no off-diagonal
        assert np.all(np.diag(result.matrix) > 0)
        assert result.misclassification_matrix.sum() == 0
        assert result.overall_stats['accuracy'] == 1.0
        assert result.overall_stats['error_rate'] == 0.0
    
    def test_all_wrong_predictions(self):
        """Test with all wrong predictions."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [1, 1, 1, 0, 0, 0]
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred)
        
        # No diagonal elements
        assert np.all(np.diag(result.matrix) == 0)
        assert result.overall_stats['accuracy'] == 0.0
        assert result.overall_stats['error_rate'] == 1.0
    
    def test_single_class(self):
        """Test with single class (edge case)."""
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred)
        
        assert result.n_classes == 1
        assert result.matrix.shape == (1, 1)
        assert result.matrix[0, 0] == 4
    
    def test_mismatched_lengths(self):
        """Test error handling for mismatched input lengths."""
        y_true = [0, 1, 0]
        y_pred = [0, 1]  # Different length
        
        calc = ConfusionMatrixCalculator()
        
        with pytest.raises(ValueError, match="must have same length"):
            calc.compute(y_true, y_pred)
    
    def test_invalid_normalization_mode(self):
        """Test error handling for invalid normalization mode."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]
        
        calc = ConfusionMatrixCalculator()
        
        with pytest.raises(ValueError, match="Invalid normalization mode"):
            calc._normalize_matrix(np.array([[1, 2], [3, 4]]), 'invalid')
    
    def test_cost_matrix_shape_mismatch(self):
        """Test error handling for cost matrix shape mismatch."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]
        cost_matrix = np.array([[0, 1, 2]])  # Wrong shape
        
        calc = ConfusionMatrixCalculator()
        
        with pytest.raises(ValueError, match="Cost matrix shape"):
            calc.compute_cost_sensitive(y_true, y_pred, cost_matrix)
    
    def test_with_labels_parameter(self):
        """Test with explicit labels parameter."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]
        labels = [0, 1, 2]  # Include class 2 even though not in data
        
        calc = ConfusionMatrixCalculator(labels=labels)
        result = calc.compute(y_true, y_pred)
        
        # Should have 3x3 matrix even though class 2 not present
        assert result.matrix.shape == (3, 3)
        assert result.n_classes == 3
    
    def test_pandas_input(self):
        """Test with pandas Series input."""
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = pd.Series([0, 1, 1, 1])
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred)
        
        assert result.matrix.shape == (2, 2)
        assert result.n_samples == 4
    
    def test_repr(self):
        """Test string representation."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]
        
        calc = ConfusionMatrixCalculator()
        result = calc.compute(y_true, y_pred)
        
        repr_str = repr(result)
        
        assert 'ConfusionMatrixResult' in repr_str
        assert 'n_classes=2' in repr_str
        assert 'n_samples=4' in repr_str


class TestConvenienceFunction:
    """Test suite for convenience function."""
    
    def test_compute_confusion_matrix_function(self):
        """Test the convenience function."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]
        
        result = compute_confusion_matrix(y_true, y_pred)
        
        assert isinstance(result, ConfusionMatrixResult)
        assert result.matrix.shape == (2, 2)
        assert result.n_samples == 4
    
    def test_convenience_function_with_all_params(self):
        """Test convenience function with all parameters."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 1, 1, 2]
        class_names = ['A', 'B', 'C']
        
        result = compute_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            normalize='true',
            include_stats=True,
            include_misclassification=True,
            labels=[0, 1, 2]
        )
        
        assert result.normalized_matrix is not None
        assert result.per_class_stats is not None
        assert result.misclassification_matrix is not None
        assert result.class_names == class_names


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_large_number_of_classes(self):
        """Test with many classes."""
        n_classes = 20
        n_samples = 200
        
        np.random.seed(42)
        y_true = np.random.randint(0, n_classes, n_samples)
        y_pred = np.random.randint(0, n_classes, n_samples)
        
        result = compute_confusion_matrix(y_true, y_pred)
        
        assert result.n_classes == n_classes
        assert result.matrix.shape == (n_classes, n_classes)
    
    def test_imbalanced_classes(self):
        """Test with highly imbalanced classes."""
        y_true = [0] * 90 + [1] * 10
        y_pred = [0] * 85 + [1] * 5 + [0] * 10
        
        result = compute_confusion_matrix(y_true, y_pred)
        
        assert result.n_samples == 100
        assert result.per_class_stats is not None
    
    def test_string_labels(self):
        """Test with string labels."""
        y_true = ['cat', 'dog', 'cat', 'dog']
        y_pred = ['cat', 'dog', 'dog', 'dog']
        
        result = compute_confusion_matrix(y_true, y_pred)
        
        assert result.matrix.shape == (2, 2)
        assert result.n_samples == 4
    
    def test_zero_division_in_stats(self):
        """Test handling of zero division in statistics."""
        # All predictions are class 0
        y_true = [0, 1, 0, 1]
        y_pred = [0, 0, 0, 0]
        
        result = compute_confusion_matrix(y_true, y_pred)
        
        # Should not raise error, should handle gracefully
        assert result.per_class_stats is not None
        # Class 1 will have 0 TP, so precision/recall should be 0
        assert result.per_class_stats['1']['precision'] == 0.0
