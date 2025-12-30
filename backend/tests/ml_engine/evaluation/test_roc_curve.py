"""
Unit tests for ROC curve module.

Tests cover:
- Binary classification ROC curves
- Multi-class ROC curves (One-vs-Rest)
- Micro and macro averaging
- AUC calculation
- Optimal threshold identification
- DataFrame conversion
- Model comparison
- Edge cases
"""

import pytest
import numpy as np
import pandas as pd
from app.ml_engine.evaluation.roc_curve import (
    ROCCurveCalculator,
    ROCCurveResult,
    MultiClassROCResult,
    compute_roc_curve
)


class TestROCCurveCalculator:
    """Test suite for ROCCurveCalculator."""
    
    def test_initialization(self):
        """Test calculator initialization."""
        calc = ROCCurveCalculator()
        assert calc.pos_label == 1
        assert calc.drop_intermediate is True
        
        calc_custom = ROCCurveCalculator(pos_label=0, drop_intermediate=False)
        assert calc_custom.pos_label == 0
        assert calc_custom.drop_intermediate is False
    
    def test_binary_roc_curve_perfect(self):
        """Test ROC curve with perfect predictions."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]
        
        calc = ROCCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        # Perfect classifier should have AUC = 1.0
        assert result.auc_score == 1.0
        assert result.n_samples == 4
        assert result.n_positive == 2
        assert result.n_negative == 2
        assert result.optimal_threshold is not None
    
    def test_binary_roc_curve_random(self):
        """Test ROC curve with random predictions."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_score = np.random.rand(100)
        
        calc = ROCCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        # Random classifier should have AUC ≈ 0.5
        assert 0.4 < result.auc_score < 0.6
        assert result.n_samples == 100
    
    def test_binary_roc_curve_good(self):
        """Test ROC curve with good but not perfect predictions."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_score = [0.1, 0.3, 0.4, 0.6, 0.8, 0.9]
        
        calc = ROCCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        # Good classifier should have AUC > 0.8
        assert result.auc_score > 0.8
        assert len(result.fpr) == len(result.tpr)
        assert len(result.fpr) == len(result.thresholds)
    
    def test_optimal_threshold_identification(self):
        """Test optimal threshold identification using Youden's J."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_score = [0.1, 0.3, 0.4, 0.6, 0.8, 0.9]
        
        calc = ROCCurveCalculator()
        result = calc.compute_binary(y_true, y_score, find_optimal_threshold=True)
        
        assert result.optimal_threshold is not None
        assert result.optimal_threshold_index is not None
        assert 0 <= result.optimal_threshold <= 1
        
        # Check that optimal point is on the curve
        fpr_opt = result.fpr[result.optimal_threshold_index]
        tpr_opt = result.tpr[result.optimal_threshold_index]
        assert 0 <= fpr_opt <= 1
        assert 0 <= tpr_opt <= 1
    
    def test_without_optimal_threshold(self):
        """Test ROC curve without optimal threshold calculation."""
        y_true = [0, 1, 0, 1]
        y_score = [0.2, 0.8, 0.3, 0.7]
        
        calc = ROCCurveCalculator()
        result = calc.compute_binary(y_true, y_score, find_optimal_threshold=False)
        
        assert result.optimal_threshold is None
        assert result.optimal_threshold_index is None
    
    def test_multiclass_roc_curve(self):
        """Test multi-class ROC curve computation."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_score = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])
        
        calc = ROCCurveCalculator()
        result = calc.compute_multiclass(y_true, y_score)
        
        assert isinstance(result, MultiClassROCResult)
        assert result.n_classes == 3
        assert len(result.per_class) == 3
        assert result.strategy == 'ovr'
    
    def test_multiclass_with_class_names(self):
        """Test multi-class ROC with custom class names."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_score = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])
        class_names = ['cat', 'dog', 'bird']
        
        calc = ROCCurveCalculator()
        result = calc.compute_multiclass(y_true, y_score, class_names=class_names)
        
        assert result.class_names == class_names
        assert 'cat' in result.per_class
        assert 'dog' in result.per_class
        assert 'bird' in result.per_class
    
    def test_multiclass_micro_average(self):
        """Test micro-averaged ROC curve."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_score = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])
        
        calc = ROCCurveCalculator()
        result = calc.compute_multiclass(y_true, y_score, average='micro')
        
        assert result.micro_average is not None
        assert result.micro_average.class_name == 'micro-average'
        assert result.micro_average.auc_score > 0
        assert result.macro_average is None
    
    def test_multiclass_macro_average(self):
        """Test macro-averaged ROC curve."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_score = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])
        
        calc = ROCCurveCalculator()
        result = calc.compute_multiclass(y_true, y_score, average='macro')
        
        assert result.macro_average is not None
        assert result.macro_average.class_name == 'macro-average'
        assert result.macro_average.auc_score > 0
        assert result.micro_average is None
    
    def test_multiclass_both_averages(self):
        """Test both micro and macro averaging."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_score = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])
        
        calc = ROCCurveCalculator()
        result = calc.compute_multiclass(y_true, y_score, average='both')
        
        assert result.micro_average is not None
        assert result.macro_average is not None
    
    def test_multiclass_no_average(self):
        """Test multi-class without averaging."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_score = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])
        
        calc = ROCCurveCalculator()
        result = calc.compute_multiclass(y_true, y_score, average=None)
        
        assert result.micro_average is None
        assert result.macro_average is None
        assert len(result.per_class) == 3
    
    def test_to_dataframe(self):
        """Test conversion to pandas DataFrame."""
        y_true = [0, 1, 0, 1]
        y_score = [0.2, 0.8, 0.3, 0.7]
        
        calc = ROCCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        df = result.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert 'fpr' in df.columns
        assert 'tpr' in df.columns
        assert 'threshold' in df.columns
        assert len(df) == len(result.fpr)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        y_true = [0, 1, 0, 1]
        y_score = [0.2, 0.8, 0.3, 0.7]
        
        calc = ROCCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'fpr' in result_dict
        assert 'tpr' in result_dict
        assert 'thresholds' in result_dict
        assert 'auc_score' in result_dict
        
        # Check that arrays are converted to lists
        assert isinstance(result_dict['fpr'], list)
        assert isinstance(result_dict['tpr'], list)
    
    def test_multiclass_to_dict(self):
        """Test multi-class result to dictionary."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_score = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])
        
        calc = ROCCurveCalculator()
        result = calc.compute_multiclass(y_true, y_score)
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'per_class' in result_dict
        assert 'micro_average' in result_dict
        assert 'macro_average' in result_dict
    
    def test_get_point_at_threshold(self):
        """Test getting FPR/TPR at specific threshold."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.6, 0.9]
        
        calc = ROCCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        fpr, tpr = result.get_point_at_threshold(0.5)
        
        assert isinstance(fpr, float)
        assert isinstance(tpr, float)
        assert 0 <= fpr <= 1
        assert 0 <= tpr <= 1
    
    def test_get_threshold_at_fpr(self):
        """Test getting threshold at target FPR."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.6, 0.9]
        
        calc = ROCCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        threshold = result.get_threshold_at_fpr(0.5)
        
        assert isinstance(threshold, float)
    
    def test_get_threshold_at_tpr(self):
        """Test getting threshold at target TPR."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.6, 0.9]
        
        calc = ROCCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        threshold = result.get_threshold_at_tpr(0.8)
        
        assert isinstance(threshold, float)
    
    def test_compare_models(self):
        """Test comparing multiple models."""
        y_true = [0, 0, 1, 1, 0, 1]
        y_scores = {
            'model_a': [0.1, 0.2, 0.6, 0.8, 0.3, 0.7],
            'model_b': [0.2, 0.3, 0.7, 0.9, 0.4, 0.8]
        }
        
        calc = ROCCurveCalculator()
        results = calc.compare_models(y_true, y_scores)
        
        assert len(results) == 2
        assert 'model_a' in results
        assert 'model_b' in results
        assert isinstance(results['model_a'], ROCCurveResult)
        assert isinstance(results['model_b'], ROCCurveResult)
    
    def test_mismatched_lengths(self):
        """Test error handling for mismatched input lengths."""
        y_true = [0, 1, 0]
        y_score = [0.2, 0.8]  # Different length
        
        calc = ROCCurveCalculator()
        
        with pytest.raises(ValueError, match="must have same length"):
            calc.compute_binary(y_true, y_score)
    
    def test_multiclass_invalid_shape(self):
        """Test error handling for invalid y_score shape."""
        y_true = [0, 1, 2]
        y_score = [0.5, 0.6, 0.7]  # Should be 2D
        
        calc = ROCCurveCalculator()
        
        with pytest.raises(ValueError, match="must be 2D array"):
            calc.compute_multiclass(y_true, y_score)
    
    def test_multiclass_mismatched_samples(self):
        """Test error handling for mismatched sample counts."""
        y_true = [0, 1, 2]
        y_score = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])  # Different length
        
        calc = ROCCurveCalculator()
        
        with pytest.raises(ValueError, match="must have same number of samples"):
            calc.compute_multiclass(y_true, y_score)
    
    def test_pandas_input(self):
        """Test with pandas Series input."""
        y_true = pd.Series([0, 1, 0, 1])
        y_score = pd.Series([0.2, 0.8, 0.3, 0.7])
        
        calc = ROCCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        assert result.n_samples == 4
        assert result.auc_score > 0
    
    def test_repr(self):
        """Test string representation."""
        y_true = [0, 1, 0, 1]
        y_score = [0.2, 0.8, 0.3, 0.7]
        
        calc = ROCCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        repr_str = repr(result)
        
        assert 'ROCCurveResult' in repr_str
        assert 'auc_score' in repr_str
        assert 'n_samples=4' in repr_str
    
    def test_multiclass_repr(self):
        """Test multi-class result string representation."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_score = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])
        
        calc = ROCCurveCalculator()
        result = calc.compute_multiclass(y_true, y_score)
        
        repr_str = repr(result)
        
        assert 'MultiClassROCResult' in repr_str
        assert 'n_classes=3' in repr_str


class TestConvenienceFunction:
    """Test suite for convenience function."""
    
    def test_compute_roc_curve_binary(self):
        """Test convenience function for binary classification."""
        y_true = [0, 1, 0, 1]
        y_score = [0.2, 0.8, 0.3, 0.7]
        
        result = compute_roc_curve(y_true, y_score)
        
        assert isinstance(result, ROCCurveResult)
        assert result.auc_score > 0
    
    def test_compute_roc_curve_multiclass(self):
        """Test convenience function for multi-class classification."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_score = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])
        
        result = compute_roc_curve(
            y_true, y_score,
            multiclass=True,
            class_names=['A', 'B', 'C']
        )
        
        assert isinstance(result, MultiClassROCResult)
        assert result.n_classes == 3


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_all_same_class(self):
        """Test with all samples from same class."""
        y_true = [1, 1, 1, 1]
        y_score = [0.5, 0.6, 0.7, 0.8]
        
        calc = ROCCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        # Should handle gracefully
        assert result.n_positive == 4
        assert result.n_negative == 0
    
    def test_large_dataset(self):
        """Test with large dataset."""
        np.random.seed(42)
        n_samples = 10000
        y_true = np.random.randint(0, 2, n_samples)
        y_score = np.random.rand(n_samples)
        
        calc = ROCCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        assert result.n_samples == n_samples
        assert 0 <= result.auc_score <= 1
    
    def test_imbalanced_classes(self):
        """Test with highly imbalanced classes."""
        y_true = [0] * 90 + [1] * 10
        y_score = np.random.rand(100)
        
        calc = ROCCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        assert result.n_positive == 10
        assert result.n_negative == 90
        assert 0 <= result.auc_score <= 1
    
    def test_perfect_separation(self):
        """Test with perfect class separation."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        
        calc = ROCCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        # Perfect separation should give AUC = 1.0
        assert result.auc_score == 1.0
    
    def test_worst_case_predictions(self):
        """Test with worst possible predictions."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_score = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]  # Inverted
        
        calc = ROCCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        # Worst case should give AUC = 0.0
        assert result.auc_score == 0.0
