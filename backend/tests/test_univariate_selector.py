"""
Tests for UnivariateSelector class (F-test and Chi-square).
"""

import pytest
import pandas as pd
import numpy as np
from app.ml_engine.feature_selection.univariate_selector import UnivariateSelector


class TestUnivariateSelectorInitialization:
    """Test UnivariateSelector initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        selector = UnivariateSelector()
        assert selector.params["score_func"] == 'auto'
        assert selector.params["task"] == 'classification'
        assert selector.params["k"] == 10  # Default when no selection method specified
        assert selector.params["percentile"] is None
        assert selector.params["threshold"] is None
        assert selector.params["alpha"] == 0.05
        assert selector.params["use_p_value_filter"] is False
        assert not selector.fitted

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        selector = UnivariateSelector(
            score_func='f_regression',
            task='regression',
            k=15,
            alpha=0.01,
            use_p_value_filter=True,
            columns=['a', 'b']
        )
        assert selector.params["score_func"] == 'f_regression'
        assert selector.params["task"] == 'regression'
        assert selector.params["k"] == 15
        assert selector.params["alpha"] == 0.01
        assert selector.params["use_p_value_filter"] is True
        assert selector.params["columns"] == ['a', 'b']

    def test_error_invalid_score_func(self):
        """Test error with invalid score_func."""
        with pytest.raises(ValueError, match="score_func must be"):
            UnivariateSelector(score_func='invalid')

    def test_error_invalid_task(self):
        """Test error with invalid task."""
        with pytest.raises(ValueError, match="task must be"):
            UnivariateSelector(task='invalid')

    def test_error_invalid_k(self):
        """Test error with invalid k."""
        with pytest.raises(ValueError, match="k must be positive"):
            UnivariateSelector(k=0)

        with pytest.raises(ValueError, match="k must be positive"):
            UnivariateSelector(k=-5)

    def test_error_invalid_percentile(self):
        """Test error with invalid percentile."""
        with pytest.raises(ValueError, match="percentile must be between"):
            UnivariateSelector(k=None, percentile=0)

        with pytest.raises(ValueError, match="percentile must be between"):
            UnivariateSelector(k=None, percentile=101)

    def test_error_invalid_alpha(self):
        """Test error with invalid alpha."""
        with pytest.raises(ValueError, match="alpha must be between"):
            UnivariateSelector(alpha=0)

        with pytest.raises(ValueError, match="alpha must be between"):
            UnivariateSelector(alpha=1.5)

    def test_error_conflicting_parameters(self):
        """Test error when multiple selection criteria specified."""
        with pytest.raises(ValueError, match="Only one of"):
            UnivariateSelector(k=10, percentile=50)

        with pytest.raises(ValueError, match="Only one of"):
            UnivariateSelector(k=10, threshold=0.5)

        with pytest.raises(ValueError, match="Only one of"):
            UnivariateSelector(percentile=50, threshold=0.5)


class TestUnivariateSelectorFClassif:
    """Test UnivariateSelector with F-test for classification."""

    def test_fit_f_classif(self):
        """Test basic fitting with F-test for classification."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = UnivariateSelector(score_func='f_classif', k=2)
        selector.fit(df, y)

        assert selector.fitted
        assert selector.scores_ is not None
        assert selector.p_values_ is not None
        assert len(selector.selected_features_) == 2
        assert selector.score_func_used_ == 'f_classif'

    def test_selects_informative_features_f_classif(self):
        """Test that F-test selects informative features."""
        np.random.seed(42)
        n = 200
        # Create a feature highly correlated with target
        informative = np.random.randn(n)
        y = (informative > 0).astype(int)

        df = pd.DataFrame({
            'informative': informative,
            'noise1': np.random.randn(n),
            'noise2': np.random.randn(n)
        })

        selector = UnivariateSelector(score_func='f_classif', k=1)
        selector.fit(df, y)

        # Informative feature should have the highest score
        scores = selector.get_scores()
        assert scores['informative'] == scores.max()

    def test_auto_selects_f_classif_for_classification(self):
        """Test that 'auto' selects f_classif for classification."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = UnivariateSelector(score_func='auto', task='classification', k=2)
        selector.fit(df, y)

        assert selector.score_func_used_ == 'f_classif'

    def test_top_k_selection_f_classif(self):
        """Test selecting top k features with F-test."""
        np.random.seed(42)
        df = pd.DataFrame({
            f'f{i}': np.random.randn(100) for i in range(5)
        })
        y = np.random.randint(0, 3, 100)

        selector = UnivariateSelector(score_func='f_classif', k=3)
        selector.fit(df, y)

        assert len(selector.selected_features_) == 3

    def test_multiclass_classification(self):
        """Test F-test with multiclass classification."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(150),
            'x2': np.random.randn(150),
            'x3': np.random.randn(150)
        })
        y = np.random.randint(0, 4, 150)  # 4 classes

        selector = UnivariateSelector(score_func='f_classif', k=2)
        selector.fit(df, y)

        assert selector.fitted
        assert len(selector.selected_features_) == 2


class TestUnivariateSelectorFRegression:
    """Test UnivariateSelector with F-test for regression."""

    def test_fit_f_regression(self):
        """Test fitting with F-test for regression."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        y = np.random.randn(100)

        selector = UnivariateSelector(score_func='f_regression', k=2)
        selector.fit(df, y)

        assert selector.fitted
        assert len(selector.selected_features_) == 2
        assert selector.score_func_used_ == 'f_regression'

    def test_selects_relevant_features_f_regression(self):
        """Test that F-regression selects relevant features."""
        np.random.seed(42)
        n = 200
        x_relevant = np.random.randn(n)
        y = 3 * x_relevant + np.random.randn(n) * 0.5  # Linear relationship

        df = pd.DataFrame({
            'relevant': x_relevant,
            'noise': np.random.randn(n)
        })

        selector = UnivariateSelector(score_func='f_regression', k=1)
        selector.fit(df, y)

        # Relevant feature should be selected
        assert 'relevant' in selector.selected_features_

    def test_auto_selects_f_regression_for_regression(self):
        """Test that 'auto' selects f_regression for regression."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100)
        })
        y = np.random.randn(100)

        selector = UnivariateSelector(score_func='auto', task='regression', k=2)
        selector.fit(df, y)

        assert selector.score_func_used_ == 'f_regression'


class TestUnivariateSelectorChi2:
    """Test UnivariateSelector with Chi-square test."""

    def test_fit_chi2(self):
        """Test fitting with chi-square test."""
        np.random.seed(42)
        # Chi2 requires non-negative values
        df = pd.DataFrame({
            'x1': np.random.randint(0, 10, 100),
            'x2': np.random.randint(0, 10, 100),
            'x3': np.random.randint(0, 10, 100)
        })
        y = np.random.randint(0, 2, 100)

        selector = UnivariateSelector(score_func='chi2', k=2)
        selector.fit(df, y)

        assert selector.fitted
        assert len(selector.selected_features_) == 2
        assert selector.score_func_used_ == 'chi2'

    def test_chi2_error_negative_values(self):
        """Test that chi2 raises error with negative values."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),  # Can be negative
            'x2': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = UnivariateSelector(score_func='chi2', k=1)

        with pytest.raises(ValueError, match="Chi-square test requires non-negative"):
            selector.fit(df, y)

    def test_chi2_with_counts(self):
        """Test chi2 with count data (appropriate use case)."""
        np.random.seed(42)
        n = 150
        # Simulate count data
        df = pd.DataFrame({
            'count1': np.random.poisson(5, n),
            'count2': np.random.poisson(10, n),
            'count3': np.random.poisson(3, n)
        })
        y = np.random.randint(0, 2, n)

        selector = UnivariateSelector(score_func='chi2', k=2)
        selector.fit(df, y)

        assert selector.fitted
        assert len(selector.selected_features_) == 2


class TestUnivariateSelectorSelectionMethods:
    """Test different selection methods (k, percentile, threshold)."""

    def test_percentile_selection(self):
        """Test selection by percentile."""
        np.random.seed(42)
        df = pd.DataFrame({
            f'f{i}': np.random.randn(100) for i in range(10)
        })
        y = np.random.randint(0, 2, 100)

        # Select top 30% of features
        selector = UnivariateSelector(score_func='f_classif', k=None, percentile=30)
        selector.fit(df, y)

        # Should select approximately 3 features (30% of 10)
        assert 2 <= len(selector.selected_features_) <= 4

    def test_threshold_selection(self):
        """Test selection by score threshold."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = UnivariateSelector(score_func='f_classif', k=None, threshold=1.0)
        selector.fit(df, y)

        # Only features with score >= 1.0 should be selected
        scores = selector.get_scores()
        for feature in selector.selected_features_:
            assert scores[feature] >= 1.0

    def test_p_value_filtering(self):
        """Test additional p-value filtering."""
        np.random.seed(42)
        df = pd.DataFrame({
            f'f{i}': np.random.randn(100) for i in range(10)
        })
        y = np.random.randint(0, 2, 100)

        selector = UnivariateSelector(
            score_func='f_classif',
            k=5,
            alpha=0.05,
            use_p_value_filter=True
        )
        selector.fit(df, y)

        # All selected features should have p-value < alpha
        p_values = selector.get_p_values()
        for feature in selector.selected_features_:
            assert p_values[feature] < 0.05


class TestUnivariateSelectorTransform:
    """Test transform method."""

    def test_transform_basic(self):
        """Test basic transformation."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = UnivariateSelector(score_func='f_classif', k=2)
        result = selector.fit_transform(df, y)

        assert result.shape[1] == 2
        assert all(col in df.columns for col in result.columns)

    def test_transform_preserves_data(self):
        """Test that transform preserves selected feature data."""
        np.random.seed(42)
        df = pd.DataFrame({
            'keep1': [1, 2, 3, 4, 5],
            'keep2': [10, 20, 30, 40, 50],
            'drop': [100, 200, 300, 400, 500]
        })
        y = pd.Series([0, 1, 0, 1, 0])

        selector = UnivariateSelector(score_func='f_classif', k=2)
        selector.fit(df, y)
        result = selector.transform(df)

        # Check that data is preserved for selected features
        for col in result.columns:
            pd.testing.assert_series_equal(result[col], df[col], check_names=True)

    def test_transform_requires_fit(self):
        """Test that transform requires fit."""
        df = pd.DataFrame({'x1': [1, 2, 3]})
        selector = UnivariateSelector()

        with pytest.raises(RuntimeError, match="must be fitted"):
            selector.transform(df)

    def test_transform_error_missing_features(self):
        """Test error when transform data is missing features."""
        df_train = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50)
        })
        y_train = np.random.randint(0, 2, 50)

        df_test = pd.DataFrame({'x1': np.random.randn(10)})  # Missing x2

        selector = UnivariateSelector(score_func='f_classif', k=2)
        selector.fit(df_train, y_train)

        with pytest.raises(ValueError, match="Features .* not found"):
            selector.transform(df_test)


class TestUnivariateSelectorUtilities:
    """Test utility methods."""

    def test_get_selected_features(self):
        """Test get_selected_features method."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = UnivariateSelector(score_func='f_classif', k=2)
        selector.fit(df, y)

        features = selector.get_selected_features()
        assert isinstance(features, list)
        assert len(features) == 2

    def test_get_scores(self):
        """Test get_scores method."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = UnivariateSelector(score_func='f_classif', k=1)
        selector.fit(df, y)

        scores = selector.get_scores()
        assert isinstance(scores, pd.Series)
        assert len(scores) == 2
        assert all(score >= 0 for score in scores.values)

    def test_get_p_values(self):
        """Test get_p_values method."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = UnivariateSelector(score_func='f_classif', k=1)
        selector.fit(df, y)

        p_values = selector.get_p_values()
        assert isinstance(p_values, pd.Series)
        assert len(p_values) == 2
        assert all(0 <= p <= 1 for p in p_values.values)

    def test_get_feature_scores(self):
        """Test get_feature_scores method."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = UnivariateSelector(score_func='f_classif', k=2)
        selector.fit(df, y)

        scores_df = selector.get_feature_scores()
        assert isinstance(scores_df, pd.DataFrame)
        assert 'feature' in scores_df.columns
        assert 'score' in scores_df.columns
        assert 'p_value' in scores_df.columns
        assert 'selected' in scores_df.columns
        assert 'significant' in scores_df.columns
        assert len(scores_df) == 3

        # Should be sorted by score (descending)
        assert all(
            scores_df['score'].iloc[i] >= scores_df['score'].iloc[i+1]
            for i in range(len(scores_df)-1)
        )

    def test_get_feature_importance(self):
        """Test get_feature_importance method (alias)."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = UnivariateSelector(score_func='f_classif', k=2)
        selector.fit(df, y)

        importance = selector.get_feature_importance()
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns

    def test_get_support_mask(self):
        """Test get_support method with boolean mask."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = UnivariateSelector(score_func='f_classif', k=2)
        selector.fit(df, y)

        mask = selector.get_support(indices=False)
        assert isinstance(mask, list)
        assert len(mask) == 3
        assert sum(mask) == 2  # 2 features selected

    def test_get_support_indices(self):
        """Test get_support method with indices."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = UnivariateSelector(score_func='f_classif', k=2)
        selector.fit(df, y)

        indices = selector.get_support(indices=True)
        assert isinstance(indices, list)
        assert len(indices) == 2
        assert all(0 <= idx < 3 for idx in indices)

    def test_get_statistical_summary(self):
        """Test get_statistical_summary method."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = UnivariateSelector(score_func='f_classif', k=2, alpha=0.05)
        selector.fit(df, y)

        summary = selector.get_statistical_summary()
        assert isinstance(summary, dict)
        assert 'score_function' in summary
        assert 'total_features' in summary
        assert 'selected_features' in summary
        assert 'selection_rate' in summary
        assert 'alpha' in summary
        assert 'score_statistics' in summary

        assert summary['score_function'] == 'f_classif'
        assert summary['total_features'] == 3
        assert summary['selected_features'] == 2
        assert summary['alpha'] == 0.05

    def test_methods_require_fit(self):
        """Test that utility methods require fit."""
        selector = UnivariateSelector()

        with pytest.raises(RuntimeError):
            selector.get_selected_features()

        with pytest.raises(RuntimeError):
            selector.get_scores()

        with pytest.raises(RuntimeError):
            selector.get_p_values()

        with pytest.raises(RuntimeError):
            selector.get_feature_scores()

        with pytest.raises(RuntimeError):
            selector.get_support()

        with pytest.raises(RuntimeError):
            selector.get_statistical_summary()


class TestUnivariateSelectorEdgeCases:
    """Test edge cases and error handling."""

    def test_no_numeric_columns(self):
        """Test error when no numeric columns exist."""
        df = pd.DataFrame({'text': ['a', 'b', 'c']})
        y = pd.Series([0, 1, 0])
        selector = UnivariateSelector()

        with pytest.raises(ValueError, match="No numeric columns"):
            selector.fit(df, y)

    def test_fit_requires_target(self):
        """Test that fit requires target."""
        df = pd.DataFrame({'x1': [1, 2, 3]})
        selector = UnivariateSelector()

        with pytest.raises(ValueError, match="y .* must be provided"):
            selector.fit(df, None)

    def test_single_feature(self):
        """Test with single feature."""
        np.random.seed(42)
        df = pd.DataFrame({'x1': np.random.randn(50)})
        y = np.random.randint(0, 2, 50)

        selector = UnivariateSelector(score_func='f_classif', k=1)
        selector.fit(df, y)

        assert len(selector.selected_features_) == 1

    def test_k_larger_than_features(self):
        """Test when k is larger than number of features."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50)
        })
        y = np.random.randint(0, 2, 50)

        # Request 10 features but only 2 exist
        selector = UnivariateSelector(score_func='f_classif', k=10)
        selector.fit(df, y)

        # Should select all available features
        assert len(selector.selected_features_) == 2

    def test_series_target(self):
        """Test with pandas Series as target."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50)
        })
        y = pd.Series(np.random.randint(0, 2, 50))

        selector = UnivariateSelector(score_func='f_classif', k=1)
        selector.fit(df, y)

        assert selector.fitted

    def test_error_not_dataframe(self):
        """Test error when input is not DataFrame."""
        selector = UnivariateSelector()
        y = np.array([0, 1, 0, 1])

        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            selector.fit(np.array([[1, 2], [3, 4]]), y)

    def test_specific_columns(self):
        """Test with specific columns parameter."""
        np.random.seed(42)
        df = pd.DataFrame({
            'use1': np.random.randn(100),
            'use2': np.random.randn(100),
            'ignore': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = UnivariateSelector(
            score_func='f_classif',
            k=2,
            columns=['use1', 'use2']
        )
        selector.fit(df, y)

        # Only use1 and use2 should be considered
        assert 'ignore' not in selector.scores_.index

    def test_constant_feature_handling(self):
        """Test handling of constant features (zero variance)."""
        np.random.seed(42)
        df = pd.DataFrame({
            'varying': np.random.randn(100),
            'constant': np.ones(100)  # Constant feature
        })
        y = np.random.randint(0, 2, 100)

        selector = UnivariateSelector(score_func='f_classif', k=1)
        selector.fit(df, y)

        # Should handle constant features gracefully
        assert selector.fitted


class TestUnivariateSelectorPractical:
    """Test practical use cases."""

    def test_train_test_consistency(self):
        """Test consistency between train and test transforms."""
        np.random.seed(42)
        train = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        y_train = np.random.randint(0, 2, 100)

        test = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'x3': np.random.randn(50)
        })

        selector = UnivariateSelector(score_func='f_classif', k=2)
        train_result = selector.fit_transform(train, y_train)
        test_result = selector.transform(test)

        # Same columns in both
        assert list(train_result.columns) == list(test_result.columns)

    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        np.random.seed(42)
        # 100 samples, 50 features
        df = pd.DataFrame({
            f'f{i}': np.random.randn(100) for i in range(50)
        })
        y = np.random.randint(0, 2, 100)

        selector = UnivariateSelector(score_func='f_classif', k=10)
        selector.fit(df, y)

        assert len(selector.selected_features_) == 10
        assert selector.fitted

    def test_comparison_f_classif_vs_chi2(self):
        """Test comparison between f_classif and chi2."""
        np.random.seed(42)
        # Use non-negative data for chi2 compatibility
        df = pd.DataFrame({
            'x1': np.random.randint(0, 10, 100),
            'x2': np.random.randint(0, 10, 100)
        })
        y = np.random.randint(0, 2, 100)

        selector_f = UnivariateSelector(score_func='f_classif', k=1)
        selector_f.fit(df, y)

        selector_chi2 = UnivariateSelector(score_func='chi2', k=1)
        selector_chi2.fit(df, y)

        # Both should work
        assert selector_f.fitted
        assert selector_chi2.fitted

    def test_repr_not_fitted(self):
        """Test string representation before fitting."""
        selector = UnivariateSelector(score_func='f_regression', k=5)
        repr_str = repr(selector)
        assert 'not fitted' in repr_str
        assert '5' in repr_str

    def test_repr_fitted(self):
        """Test string representation after fitting."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = UnivariateSelector(score_func='f_classif', k=2)
        selector.fit(df, y)

        repr_str = repr(selector)
        assert 'f_classif' in repr_str
        assert 'selected=2' in repr_str or 'selected=2/3' in repr_str

    def test_p_value_interpretation(self):
        """Test interpretation of p-values for feature significance."""
        np.random.seed(42)
        n = 200
        # Create strongly correlated feature
        strong_feature = np.random.randn(n)
        y = (strong_feature > 0).astype(int)

        df = pd.DataFrame({
            'strong': strong_feature,
            'weak': np.random.randn(n)
        })

        selector = UnivariateSelector(score_func='f_classif', k=2, alpha=0.05)
        selector.fit(df, y)

        p_values = selector.get_p_values()
        # Strong feature should have lower p-value
        assert p_values['strong'] < p_values['weak']
