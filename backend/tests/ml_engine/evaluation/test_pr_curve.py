"""
Unit tests for PR curve module.

Tests cover:
- Binary classification PR curves
- Multi-class PR curves (One-vs-Rest)
- Micro and macro averaging
- Average Precision calculation
- Optimal threshold identification (F1 maximization)
- DataFrame conversion
- Model comparison
- Edge cases
"""

import pytest
import numpy as np
import pandas as pd
from app.ml_engine.evaluation.pr_curve import (
    PRCurveCalculator,
    PRCurveResult,
    MultiClassPRResult,
    compute_pr_curve
)


class TestPRCurveCalculator:
    """Test suite for PRCurveCalculator."""
    
    def test_initialization(self):
        """Test calculator initialization."""
        calc = PRCurveCalculator()
        assert calc.pos_label == 1
        
        calc_custom = PRCurveCalculator(pos_label=0)
        assert calc_custom.pos_label == 0
    
    def test_binary_pr_curve_perfect(self):
        """Test PR curve with perfect predictions."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]
        
        calc = PRCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        # Perfect classifier should have AP = 1.0
        assert result.average_precision == 1.0
        assert result.n_samples == 4
        assert result.n_positive == 2
        assert result.n_negative == 2
        assert result.optimal_threshold is not None
        assert result.optimal_f1_score is not None
    
    def test_binary_pr_curve_good(self):
        """Test PR curve with good but not perfect predictions."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_score = [0.1, 0.3, 0.4, 0.6, 0.8, 0.9]
        
        calc = PRCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        # Good classifier should have high AP
        assert result.average_precision > 0.8
        assert len(result.precision) == len(result.recall)
        assert len(result.thresholds) == len(result.precision) - 1
    
    def test_optimal_threshold_f1(self):
        """Test optimal threshold identification (F1 maximization)."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_score = [0.1, 0.3, 0.4, 0.6, 0.8, 0.9]
        
        calc = PRCurveCalculator()
        result = calc.compute_binary(y_true, y_score, find_optimal_threshold=True)
        
        assert result.optimal_threshold is not None
        assert result.optimal_threshold_index is not None
        assert result.optimal_f1_score is not None
        assert 0 <= result.optimal_f1_score <= 1
    
    def test_without_optimal_threshold(self):
        """Test PR curve without optimal threshold calculation."""
        y_true = [0, 1, 0, 1]
        y_score = [0.2, 0.8, 0.3, 0.7]
        
        calc = PRCurveCalculator()
        result = calc.compute_binary(y_true, y_score, find_optimal_threshold=False)
        
        assert result.optimal_threshold is None
        assert result.optimal_threshold_index is None
        assert result.optimal_f1_score is None
    
    def test_baseline_precision(self):
        """Test baseline precision calculation."""
        y_true = [0, 0, 0, 1]  # 25% positive
        y_score = [0.2, 0.3, 0.4, 0.8]
        
        calc = PRCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        assert result.baseline_precision == 0.25
    
    def test_multiclass_pr_curve(self):
        """Test multi-class PR curve computation."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_score = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])
        
        calc = PRCurveCalculator()
        result = calc.compute_multiclass(y_true, y_score)
        
        assert isinstance(result, MultiClassPRResult)
        assert result.n_classes == 3
        assert len(result.per_class) == 3
        assert result.strategy == 'ovr'
    
    def test_multiclass_with_class_names(self):
        """Test multi-class PR with custom class names."""
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
        
        calc = PRCurveCalculator()
        result = calc.compute_multiclass(y_true, y_score, class_names=class_names)
        
        assert result.class_names == class_names
        assert 'cat' in result.per_class
        assert 'dog' in result.per_class
        assert 'bird' in result.per_class
    
    def test_multiclass_micro_average(self):
        """Test micro-averaged PR curve."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_score = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])
        
        calc = PRCurveCalculator()
        result = calc.compute_multiclass(y_true, y_score, average='micro')
        
        assert result.micro_average is not None
        assert result.micro_average.class_name == 'micro-average'
        assert result.micro_average.average_precision > 0
        assert result.macro_average is None
    
    def test_multiclass_macro_average(self):
        """Test macro-averaged PR curve."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_score = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])
        
        calc = PRCurveCalculator()
        result = calc.compute_multiclass(y_true, y_score, average='macro')
        
        assert result.macro_average is not None
        assert result.macro_average.class_name == 'macro-average'
        assert result.macro_average.average_precision > 0
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
        
        calc = PRCurveCalculator()
        result = calc.compute_multiclass(y_true, y_score, average='both')
        
        assert result.micro_average is not None
        assert result.macro_average is not None
    
    def test_to_dataframe(self):
        """Test conversion to pandas DataFrame."""
        y_true = [0, 1, 0, 1]
        y_score = [0.2, 0.8, 0.3, 0.7]
        
        calc = PRCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        df = result.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert 'precision' in df.columns
        assert 'recall' in df.columns
        assert 'threshold' in df.columns
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        y_true = [0, 1, 0, 1]
        y_score = [0.2, 0.8, 0.3, 0.7]
        
        calc = PRCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'precision' in result_dict
        assert 'recall' in result_dict
        assert 'thresholds' in result_dict
        assert 'average_precision' in result_dict
        
        # Check that arrays are converted to lists
        assert isinstance(result_dict['precision'], list)
        assert isinstance(result_dict['recall'], list)
    
    def test_get_point_at_threshold(self):
        """Test getting precision/recall at specific threshold."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.6, 0.9]
        
        calc = PRCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        precision, recall = result.get_point_at_threshold(0.5)
        
        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
    
    def test_get_threshold_at_recall(self):
        """Test getting threshold at target recall."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.6, 0.9]
        
        calc = PRCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        threshold = result.get_threshold_at_recall(0.8)
        
        assert isinstance(threshold, float)
    
    def test_get_threshold_at_precision(self):
        """Test getting threshold at target precision."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.6, 0.9]
        
        calc = PRCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        threshold = result.get_threshold_at_precision(0.8)
        
        assert isinstance(threshold, float)
    
    def test_compare_models(self):
        """Test comparing multiple models."""
        y_true = [0, 0, 1, 1, 0, 1]
        y_scores = {
            'model_a': [0.1, 0.2, 0.6, 0.8, 0.3, 0.7],
            'model_b': [0.2, 0.3, 0.7, 0.9, 0.4, 0.8]
        }
        
        calc = PRCurveCalculator()
        results = calc.compare_models(y_true, y_scores)
        
        assert len(results) == 2
        assert 'model_a' in results
        assert 'model_b' in results
        assert isinstance(results['model_a'], PRCurveResult)
        assert isinstance(results['model_b'], PRCurveResult)
    
    def test_mismatched_lengths(self):
        """Test error handling for mismatched input lengths."""
        y_true = [0, 1, 0]
        y_score = [0.2, 0.8]  # Different length
        
        calc = PRCurveCalculator()
        
        with pytest.raises(ValueError, match="must have same length"):
            calc.compute_binary(y_true, y_score)
    
    def test_multiclass_invalid_shape(self):
        """Test error handling for invalid y_score shape."""
        y_true = [0, 1, 2]
        y_score = [0.5, 0.6, 0.7]  # Should be 2D
        
        calc = PRCurveCalculator()
        
        with pytest.raises(ValueError, match="must be 2D array"):
            calc.compute_multiclass(y_true, y_score)
    
    def test_pandas_input(self):
        """Test with pandas Series input."""
        y_true = pd.Series([0, 1, 0, 1])
        y_score = pd.Series([0.2, 0.8, 0.3, 0.7])
        
        calc = PRCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        assert result.n_samples == 4
        assert result.average_precision > 0
    
    def test_repr(self):
        """Test string representation."""
        y_true = [0, 1, 0, 1]
        y_score = [0.2, 0.8, 0.3, 0.7]
        
        calc = PRCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        repr_str = repr(result)
        
        assert 'PRCurveResult' in repr_str
        assert 'average_precision' in repr_str
        assert 'n_samples=4' in repr_str


class TestConvenienceFunction:
    """Test suite for convenience function."""
    
    def test_compute_pr_curve_binary(self):
        """Test convenience function for binary classification."""
        y_true = [0, 1, 0, 1]
        y_score = [0.2, 0.8, 0.3, 0.7]
        
        result = compute_pr_curve(y_true, y_score)
        
        assert isinstance(result, PRCurveResult)
        assert result.average_precision > 0
    
    def test_compute_pr_curve_multiclass(self):
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
        
        result = compute_pr_curve(
            y_true, y_score,
            multiclass=True,
            class_names=['A', 'B', 'C']
        )
        
        assert isinstance(result, MultiClassPRResult)
        assert result.n_classes == 3


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_imbalanced_dataset(self):
        """Test with highly imbalanced dataset."""
        # 90% negative, 10% positive
        y_true = [0] * 90 + [1] * 10
        y_score = np.random.rand(100)
        
        calc = PRCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        assert result.n_positive == 10
        assert result.n_negative == 90
        assert result.baseline_precision == 0.1
        assert 0 <= result.average_precision <= 1
    
    def test_perfect_separation(self):
        """Test with perfect class separation."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        
        calc = PRCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        # Perfect separation should give AP = 1.0
        assert result.average_precision == 1.0
    
    def test_all_same_class(self):
        """Test with all samples from same class."""
        y_true = [1, 1, 1, 1]
        y_score = [0.5, 0.6, 0.7, 0.8]
        
        calc = PRCurveCalculator()
        result = calc.compute_binary(y_true, y_score)
        
        # Should handle gracefully
        assert result.n_positive == 4
        assert result.n_negative == 0
