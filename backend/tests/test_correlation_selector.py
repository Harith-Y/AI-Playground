"""
Tests for CorrelationSelector class.
"""

import pytest
import pandas as pd
import numpy as np
from app.ml_engine.feature_selection.correlation_selector import CorrelationSelector


class TestCorrelationSelectorInitialization:
    """Test CorrelationSelector initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        selector = CorrelationSelector()
        assert selector.params["method"] == 'multicollinearity'
        assert selector.params["threshold"] == 0.9
        assert selector.params["top_k"] is None
        assert selector.params["correlation_method"] == 'pearson'
        assert selector.params["columns"] is None
        assert not selector.fitted

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        selector = CorrelationSelector(
            method='target',
            threshold=0.5,
            top_k=10,
            correlation_method='spearman',
            columns=['a', 'b']
        )
        assert selector.params["method"] == 'target'
        assert selector.params["threshold"] == 0.5
        assert selector.params["top_k"] == 10
        assert selector.params["correlation_method"] == 'spearman'
        assert selector.params["columns"] == ['a', 'b']

    def test_error_invalid_method(self):
        """Test error with invalid method."""
        with pytest.raises(ValueError, match="method must be"):
            CorrelationSelector(method='invalid')

    def test_error_invalid_threshold(self):
        """Test error with invalid threshold."""
        with pytest.raises(ValueError, match="threshold must be between"):
            CorrelationSelector(threshold=1.5)

        with pytest.raises(ValueError, match="threshold must be between"):
            CorrelationSelector(threshold=-0.1)

    def test_error_invalid_correlation_method(self):
        """Test error with invalid correlation method."""
        with pytest.raises(ValueError, match="correlation_method must be"):
            CorrelationSelector(correlation_method='invalid')

    def test_error_invalid_top_k(self):
        """Test error with invalid top_k."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            CorrelationSelector(top_k=0)

        with pytest.raises(ValueError, match="top_k must be positive"):
            CorrelationSelector(top_k=-5)


class TestCorrelationSelectorMulticollinearity:
    """Test CorrelationSelector in multicollinearity mode."""

    def test_fit_basic(self):
        """Test basic fitting in multicollinearity mode."""
        # Create data with highly correlated features
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [1.1, 2.1, 3.1, 4.1, 5.1],  # Highly correlated with x1
            'x3': [10, 20, 30, 40, 50]  # Independent
        })

        selector = CorrelationSelector(method='multicollinearity', threshold=0.95)
        selector.fit(df)

        assert selector.fitted
        assert selector.selected_features_ is not None
        assert selector.correlation_matrix_ is not None

    def test_removes_highly_correlated_features(self):
        """Test that highly correlated features are removed."""
        # Create perfectly correlated features
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [2, 4, 6, 8, 10],  # x2 = 2*x1 (perfect correlation)
            'x3': [5, 4, 3, 2, 1]    # Negative correlation with x1
        })

        selector = CorrelationSelector(method='multicollinearity', threshold=0.9)
        selector.fit(df)

        # x2 should be dropped (correlated with x1)
        assert 'x1' in selector.selected_features_
        assert 'x2' not in selector.selected_features_
        assert 'x3' in selector.selected_features_

    def test_keeps_all_when_no_high_correlation(self):
        """Test that all features are kept when correlation is low."""
        # Create uncorrelated features
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })

        selector = CorrelationSelector(method='multicollinearity', threshold=0.9)
        selector.fit(df)

        # All features should be selected (low correlation)
        assert len(selector.selected_features_) == 3

    def test_transform_removes_features(self):
        """Test that transform removes correlated features."""
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [1.01, 2.01, 3.01, 4.01, 5.01],  # Almost identical to x1
            'x3': [10, 20, 30, 40, 50]
        })

        selector = CorrelationSelector(method='multicollinearity', threshold=0.95)
        result = selector.fit_transform(df)

        assert 'x2' not in result.columns
        assert 'x1' in result.columns
        assert 'x3' in result.columns

    def test_different_thresholds(self):
        """Test different correlation thresholds."""
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [1.5, 2.5, 3.5, 4.5, 5.5],  # r ≈ 1.0
            'x3': [2, 3, 4, 5, 6]              # r ≈ 1.0
        })

        # Strict threshold
        strict_selector = CorrelationSelector(method='multicollinearity', threshold=0.99)
        strict_selector.fit(df)

        # Lenient threshold
        lenient_selector = CorrelationSelector(method='multicollinearity', threshold=0.7)
        lenient_selector.fit(df)

        # Strict should keep more features
        assert len(strict_selector.selected_features_) >= len(lenient_selector.selected_features_)

    def test_correlation_methods(self):
        """Test different correlation methods."""
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [1, 4, 9, 16, 25],  # Quadratic relationship
            'x3': [10, 20, 30, 40, 50]
        })

        # Pearson (linear)
        pearson_selector = CorrelationSelector(
            method='multicollinearity',
            correlation_method='pearson'
        )
        pearson_selector.fit(df)

        # Spearman (monotonic)
        spearman_selector = CorrelationSelector(
            method='multicollinearity',
            correlation_method='spearman'
        )
        spearman_selector.fit(df)

        # Both should work without errors
        assert pearson_selector.fitted
        assert spearman_selector.fitted


class TestCorrelationSelectorTarget:
    """Test CorrelationSelector in target mode."""

    def test_fit_requires_target(self):
        """Test that fit requires target for target mode."""
        df = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6]})
        selector = CorrelationSelector(method='target')

        with pytest.raises(ValueError, match="y .* must be provided"):
            selector.fit(df)

    def test_selects_correlated_features(self):
        """Test selection of features correlated with target."""
        # Create features with known correlations to target
        df = pd.DataFrame({
            'high_corr': [1, 2, 3, 4, 5],
            'low_corr': [1, 1, 1, 1, 1],
            'medium_corr': [1, 2, 2, 3, 4]
        })
        y = pd.Series([2, 4, 6, 8, 10])  # Perfectly correlated with high_corr

        selector = CorrelationSelector(method='target', threshold=0.8)
        selector.fit(df, y)

        # high_corr should definitely be selected
        assert 'high_corr' in selector.selected_features_
        # low_corr should not (constant value)
        assert 'low_corr' not in selector.selected_features_

    def test_top_k_selection(self):
        """Test top_k feature selection."""
        df = pd.DataFrame({
            'f1': [1, 2, 3, 4, 5],
            'f2': [2, 4, 6, 8, 10],
            'f3': [5, 4, 3, 2, 1],
            'f4': [1, 1, 1, 1, 1],
            'f5': [3, 5, 7, 9, 11]
        })
        y = pd.Series([1, 2, 3, 4, 5])

        # Select top 2 features
        selector = CorrelationSelector(method='target', top_k=2, threshold=0.0)
        selector.fit(df, y)

        assert len(selector.selected_features_) == 2

    def test_top_k_with_threshold(self):
        """Test top_k combined with threshold."""
        df = pd.DataFrame({
            'f1': [1, 2, 3, 4, 5],      # High correlation
            'f2': [2, 4, 6, 8, 10],     # High correlation
            'f3': [5, 4, 3, 2, 1],      # Negative high correlation
            'f4': [1, 1, 2, 1, 2]       # Low correlation
        })
        y = pd.Series([1, 2, 3, 4, 5])

        # Select top 2 from features with threshold > 0.5
        selector = CorrelationSelector(method='target', top_k=2, threshold=0.5)
        selector.fit(df, y)

        # Should select at most 2 features above threshold
        assert len(selector.selected_features_) <= 2

    def test_numpy_array_target(self):
        """Test with numpy array as target."""
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [10, 20, 30, 40, 50]
        })
        y = np.array([2, 4, 6, 8, 10])

        selector = CorrelationSelector(method='target', threshold=0.8)
        selector.fit(df, y)

        assert selector.fitted
        assert len(selector.selected_features_) > 0

    def test_transform_selects_features(self):
        """Test that transform selects correct features."""
        df = pd.DataFrame({
            'keep': [1, 2, 3, 4, 5],
            'drop': [1, 1, 1, 1, 1]
        })
        y = pd.Series([1, 2, 3, 4, 5])

        selector = CorrelationSelector(method='target', threshold=0.9)
        result = selector.fit_transform(df, y)

        assert 'keep' in result.columns
        assert 'drop' not in result.columns

    def test_spearman_for_monotonic(self):
        """Test Spearman correlation for monotonic relationships."""
        df = pd.DataFrame({
            'linear': [1, 2, 3, 4, 5],
            'quadratic': [1, 4, 9, 16, 25]
        })
        y = pd.Series([1, 2, 3, 4, 5])

        # Spearman should detect monotonic relationship
        selector = CorrelationSelector(
            method='target',
            correlation_method='spearman',
            threshold=0.9
        )
        selector.fit(df, y)

        # Both should be selected (both monotonic with y)
        assert len(selector.selected_features_) == 2


class TestCorrelationSelectorUtilities:
    """Test utility methods."""

    def test_get_selected_features(self):
        """Test get_selected_features method."""
        df = pd.DataFrame({
            'x1': [1, 2, 3],
            'x2': [1, 2, 3],
            'x3': [10, 20, 30]
        })

        selector = CorrelationSelector(method='multicollinearity', threshold=0.95)
        selector.fit(df)

        features = selector.get_selected_features()
        assert isinstance(features, list)
        assert len(features) > 0

    def test_get_dropped_features(self):
        """Test get_dropped_features method."""
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [1.01, 2.01, 3.01, 4.01, 5.01]
        })

        selector = CorrelationSelector(method='multicollinearity', threshold=0.95)
        selector.fit(df)

        dropped = selector.get_dropped_features()
        assert isinstance(dropped, dict)
        if len(dropped) > 0:
            # Check that reason is provided
            for feature, reason in dropped.items():
                assert isinstance(reason, str)
                assert len(reason) > 0

    def test_get_correlation_matrix(self):
        """Test get_correlation_matrix method."""
        df = pd.DataFrame({
            'x1': [1, 2, 3],
            'x2': [4, 5, 6],
            'x3': [7, 8, 9]
        })

        selector = CorrelationSelector(method='multicollinearity')
        selector.fit(df)

        corr_matrix = selector.get_correlation_matrix()
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (3, 3)

    def test_get_correlation_matrix_none_for_target_mode(self):
        """Test get_correlation_matrix returns None for target mode."""
        df = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6]})
        y = pd.Series([1, 2, 3])

        selector = CorrelationSelector(method='target')
        selector.fit(df, y)

        assert selector.get_correlation_matrix() is None

    def test_get_target_correlations(self):
        """Test get_target_correlations method."""
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [10, 20, 30, 40, 50]
        })
        y = pd.Series([1, 2, 3, 4, 5])

        selector = CorrelationSelector(method='target')
        selector.fit(df, y)

        corr = selector.get_target_correlations()
        assert isinstance(corr, pd.Series)
        assert len(corr) == 2
        assert 'x1' in corr.index
        assert 'x2' in corr.index

    def test_get_target_correlations_none_for_multicollinearity(self):
        """Test get_target_correlations returns None for multicollinearity mode."""
        df = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6]})

        selector = CorrelationSelector(method='multicollinearity')
        selector.fit(df)

        assert selector.get_target_correlations() is None

    def test_get_feature_importance_target_mode(self):
        """Test get_feature_importance for target mode."""
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [10, 20, 30, 40, 50],
            'x3': [1, 1, 1, 1, 1]
        })
        y = pd.Series([1, 2, 3, 4, 5])

        selector = CorrelationSelector(method='target', threshold=0.8)
        selector.fit(df, y)

        importance = selector.get_feature_importance()
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'correlation' in importance.columns
        assert 'abs_correlation' in importance.columns
        assert 'selected' in importance.columns

    def test_get_feature_importance_multicollinearity_mode(self):
        """Test get_feature_importance for multicollinearity mode."""
        df = pd.DataFrame({
            'x1': [1, 2, 3],
            'x2': [1, 2, 3],
            'x3': [10, 20, 30]
        })

        selector = CorrelationSelector(method='multicollinearity', threshold=0.95)
        selector.fit(df)

        importance = selector.get_feature_importance()
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'selected' in importance.columns
        assert 'reason' in importance.columns

    def test_methods_require_fit(self):
        """Test that utility methods require fit."""
        selector = CorrelationSelector()

        with pytest.raises(RuntimeError):
            selector.get_selected_features()

        with pytest.raises(RuntimeError):
            selector.get_dropped_features()

        with pytest.raises(RuntimeError):
            selector.get_correlation_matrix()

        with pytest.raises(RuntimeError):
            selector.get_target_correlations()

        with pytest.raises(RuntimeError):
            selector.get_feature_importance()


class TestCorrelationSelectorEdgeCases:
    """Test edge cases and error handling."""

    def test_no_numeric_columns(self):
        """Test error when no numeric columns exist."""
        df = pd.DataFrame({'text': ['a', 'b', 'c']})
        selector = CorrelationSelector()

        with pytest.raises(ValueError, match="No numeric columns"):
            selector.fit(df)

    def test_single_feature(self):
        """Test with single feature."""
        df = pd.DataFrame({'x1': [1, 2, 3, 4, 5]})
        y = pd.Series([1, 2, 3, 4, 5])

        selector = CorrelationSelector(method='target')
        selector.fit(df, y)

        assert len(selector.selected_features_) == 1

    def test_all_constant_features(self):
        """Test with all constant features."""
        df = pd.DataFrame({
            'x1': [1, 1, 1, 1, 1],
            'x2': [2, 2, 2, 2, 2]
        })
        y = pd.Series([1, 2, 3, 4, 5])

        selector = CorrelationSelector(method='target', threshold=0.5)
        selector.fit(df, y)

        # All should be dropped (NaN correlation)
        assert len(selector.selected_features_) == 0

    def test_transform_missing_features(self):
        """Test error when transform data is missing features."""
        df_train = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6]})
        df_test = pd.DataFrame({'x1': [7, 8, 9]})  # Missing x2

        selector = CorrelationSelector()
        selector.fit(df_train)

        with pytest.raises(ValueError, match="Features .* not found"):
            selector.transform(df_test)

    def test_error_not_dataframe(self):
        """Test error when input is not DataFrame."""
        selector = CorrelationSelector()

        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            selector.fit(np.array([[1, 2], [3, 4]]))

    def test_transform_requires_fit(self):
        """Test that transform requires fit."""
        df = pd.DataFrame({'x1': [1, 2, 3]})
        selector = CorrelationSelector()

        with pytest.raises(RuntimeError, match="must be fitted"):
            selector.transform(df)

    def test_specific_columns(self):
        """Test with specific columns parameter."""
        df = pd.DataFrame({
            'use1': [1, 2, 3],
            'use2': [1, 2, 3],
            'ignore': [10, 20, 30]
        })

        selector = CorrelationSelector(
            method='multicollinearity',
            threshold=0.95,
            columns=['use1', 'use2']
        )
        selector.fit(df)

        # Only use1 and use2 should be considered
        all_features = selector.selected_features_ + list(selector.dropped_features_.keys())
        assert 'ignore' not in all_features


class TestCorrelationSelectorPractical:
    """Test practical use cases."""

    def test_remove_multicollinearity_for_regression(self):
        """Test removing multicollinear features for regression."""
        # Simulate features with multicollinearity
        np.random.seed(42)
        x1 = np.random.randn(100)
        x2 = x1 + np.random.randn(100) * 0.1  # Highly correlated with x1
        x3 = np.random.randn(100)  # Independent

        df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})

        selector = CorrelationSelector(method='multicollinearity', threshold=0.9)
        result = selector.fit_transform(df)

        # x2 should be removed (correlated with x1)
        assert result.shape[1] < df.shape[1]

    def test_feature_selection_for_classification(self):
        """Test feature selection for classification task."""
        # Create features with varying correlation to binary target
        np.random.seed(42)
        df = pd.DataFrame({
            'relevant1': np.random.randn(100),
            'relevant2': np.random.randn(100),
            'irrelevant': np.random.randn(100)
        })
        # Target correlated with relevant features
        y = (df['relevant1'] + df['relevant2'] > 0).astype(int)

        selector = CorrelationSelector(method='target', top_k=2)
        result = selector.fit_transform(df, y)

        assert result.shape[1] == 2

    def test_train_test_consistency(self):
        """Test consistency between train and test transforms."""
        train = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [1, 2, 3, 4, 5],
            'x3': [10, 20, 30, 40, 50]
        })
        test = pd.DataFrame({
            'x1': [6, 7],
            'x2': [6, 7],
            'x3': [60, 70]
        })

        selector = CorrelationSelector(method='multicollinearity', threshold=0.95)
        train_result = selector.fit_transform(train)
        test_result = selector.transform(test)

        # Same columns in both
        assert list(train_result.columns) == list(test_result.columns)

    def test_repr_not_fitted(self):
        """Test string representation before fitting."""
        selector = CorrelationSelector(method='target', threshold=0.7)
        repr_str = repr(selector)
        assert 'not fitted' in repr_str
        assert 'target' in repr_str
        assert '0.7' in repr_str

    def test_repr_fitted(self):
        """Test string representation after fitting."""
        df = pd.DataFrame({
            'x1': [1, 2, 3],
            'x2': [1, 2, 3],
            'x3': [10, 20, 30]
        })

        selector = CorrelationSelector(method='multicollinearity', threshold=0.9)
        selector.fit(df)

        repr_str = repr(selector)
        assert 'selected=' in repr_str
        assert 'dropped=' in repr_str
