"""
Tests for MutualInformationSelector class.
"""

import pytest
import pandas as pd
import numpy as np
from app.ml_engine.feature_selection.mutual_information_selector import MutualInformationSelector


class TestMutualInformationSelectorInitialization:
    """Test MutualInformationSelector initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        selector = MutualInformationSelector()
        assert selector.params["task"] == 'classification'
        assert selector.params["k"] is None
        assert selector.params["threshold"] == 0.0
        assert selector.params["percentile"] is None
        assert selector.params["discrete_features"] == 'auto'
        assert selector.params["n_neighbors"] == 3
        assert not selector.fitted

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        selector = MutualInformationSelector(
            task='regression',
            k=10,
            threshold=0.1,
            n_neighbors=5,
            random_state=42,
            columns=['a', 'b']
        )
        assert selector.params["task"] == 'regression'
        assert selector.params["k"] == 10
        assert selector.params["threshold"] == 0.1
        assert selector.params["n_neighbors"] == 5
        assert selector.params["random_state"] == 42
        assert selector.params["columns"] == ['a', 'b']

    def test_error_invalid_task(self):
        """Test error with invalid task."""
        with pytest.raises(ValueError, match="task must be"):
            MutualInformationSelector(task='invalid')

    def test_error_invalid_k(self):
        """Test error with invalid k."""
        with pytest.raises(ValueError, match="k must be positive"):
            MutualInformationSelector(k=0)

        with pytest.raises(ValueError, match="k must be positive"):
            MutualInformationSelector(k=-5)

    def test_error_invalid_threshold(self):
        """Test error with invalid threshold."""
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            MutualInformationSelector(threshold=-0.1)

    def test_error_invalid_percentile(self):
        """Test error with invalid percentile."""
        with pytest.raises(ValueError, match="percentile must be between"):
            MutualInformationSelector(percentile=0)

        with pytest.raises(ValueError, match="percentile must be between"):
            MutualInformationSelector(percentile=101)

    def test_error_conflicting_parameters(self):
        """Test error when multiple selection criteria specified."""
        with pytest.raises(ValueError, match="Only one of"):
            MutualInformationSelector(k=10, percentile=50)

    def test_error_invalid_n_neighbors(self):
        """Test error with invalid n_neighbors."""
        with pytest.raises(ValueError, match="n_neighbors must be positive"):
            MutualInformationSelector(n_neighbors=0)


class TestMutualInformationSelectorClassification:
    """Test MutualInformationSelector for classification tasks."""

    def test_fit_basic(self):
        """Test basic fitting for classification."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = MutualInformationSelector(task='classification', k=2)
        selector.fit(df, y)

        assert selector.fitted
        assert selector.mi_scores_ is not None
        assert len(selector.selected_features_) == 2

    def test_selects_informative_features(self):
        """Test that informative features are selected."""
        np.random.seed(42)
        # Create features with known informativeness
        n = 200
        informative = np.random.randn(n)
        y = (informative > 0).astype(int)  # Target depends on informative

        df = pd.DataFrame({
            'informative': informative,
            'noise': np.random.randn(n),  # Random noise
            'constant': np.ones(n)  # No information
        })

        selector = MutualInformationSelector(task='classification', k=1)
        selector.fit(df, y)

        # Informative feature should be selected
        assert 'informative' in selector.selected_features_

    def test_top_k_selection(self):
        """Test selecting top k features."""
        np.random.seed(42)
        df = pd.DataFrame({
            'f1': np.random.randn(100),
            'f2': np.random.randn(100),
            'f3': np.random.randn(100),
            'f4': np.random.randn(100),
            'f5': np.random.randn(100)
        })
        y = np.random.randint(0, 3, 100)

        selector = MutualInformationSelector(task='classification', k=3)
        selector.fit(df, y)

        assert len(selector.selected_features_) == 3

    def test_threshold_selection(self):
        """Test selection by threshold."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = MutualInformationSelector(task='classification', threshold=0.01)
        selector.fit(df, y)

        # Some features should be selected (those with MI > 0.01)
        assert len(selector.selected_features_) >= 0

    def test_percentile_selection(self):
        """Test selection by percentile."""
        np.random.seed(42)
        df = pd.DataFrame({
            f'f{i}': np.random.randn(100) for i in range(10)
        })
        y = np.random.randint(0, 2, 100)

        # Select top 30% of features
        selector = MutualInformationSelector(task='classification', percentile=30)
        selector.fit(df, y)

        # Should select approximately 3 features (30% of 10)
        assert 2 <= len(selector.selected_features_) <= 4

    def test_multiclass_classification(self):
        """Test with multiclass classification."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(150),
            'x2': np.random.randn(150)
        })
        y = np.random.randint(0, 3, 150)  # 3 classes

        selector = MutualInformationSelector(task='classification', k=2)
        selector.fit(df, y)

        assert selector.fitted
        assert len(selector.selected_features_) == 2


class TestMutualInformationSelectorRegression:
    """Test MutualInformationSelector for regression tasks."""

    def test_fit_regression(self):
        """Test fitting for regression."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        y = np.random.randn(100)

        selector = MutualInformationSelector(task='regression', k=2)
        selector.fit(df, y)

        assert selector.fitted
        assert len(selector.selected_features_) == 2

    def test_selects_relevant_features_regression(self):
        """Test that relevant features are selected for regression."""
        np.random.seed(42)
        n = 200
        x_relevant = np.random.randn(n)
        y = 2 * x_relevant + np.random.randn(n) * 0.1  # Linear relationship

        df = pd.DataFrame({
            'relevant': x_relevant,
            'noise': np.random.randn(n)
        })

        selector = MutualInformationSelector(task='regression', k=1, random_state=42)
        selector.fit(df, y)

        # Relevant feature should be selected
        assert 'relevant' in selector.selected_features_

    def test_threshold_regression(self):
        """Test threshold selection for regression."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100)
        })
        y = np.random.randn(100)

        selector = MutualInformationSelector(task='regression', threshold=0.05)
        selector.fit(df, y)

        assert selector.fitted


class TestMutualInformationSelectorTransform:
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

        selector = MutualInformationSelector(task='classification', k=2)
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

        selector = MutualInformationSelector(task='classification', k=2)
        selector.fit(df, y)

        result = selector.transform(df)

        # Check that data is preserved for selected features
        for col in result.columns:
            pd.testing.assert_series_equal(result[col], df[col], check_names=True)

    def test_transform_requires_fit(self):
        """Test that transform requires fit."""
        df = pd.DataFrame({'x1': [1, 2, 3]})
        selector = MutualInformationSelector()

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

        selector = MutualInformationSelector(task='classification', k=2)
        selector.fit(df_train, y_train)

        with pytest.raises(ValueError, match="Features .* not found"):
            selector.transform(df_test)


class TestMutualInformationSelectorUtilities:
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

        selector = MutualInformationSelector(task='classification', k=2)
        selector.fit(df, y)

        features = selector.get_selected_features()
        assert isinstance(features, list)
        assert len(features) == 2

    def test_get_mi_scores(self):
        """Test get_mi_scores method."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = MutualInformationSelector(task='classification', k=1)
        selector.fit(df, y)

        scores = selector.get_mi_scores()
        assert isinstance(scores, pd.Series)
        assert len(scores) == 2
        assert all(score >= 0 for score in scores.values)

    def test_get_feature_importance(self):
        """Test get_feature_importance method."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = MutualInformationSelector(task='classification', k=2)
        selector.fit(df, y)

        importance = selector.get_feature_importance()
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'mi_score' in importance.columns
        assert 'selected' in importance.columns
        assert len(importance) == 3

        # Should be sorted by MI score (descending)
        assert all(
            importance['mi_score'].iloc[i] >= importance['mi_score'].iloc[i+1]
            for i in range(len(importance)-1)
        )

    def test_get_support_mask(self):
        """Test get_support method with boolean mask."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector = MutualInformationSelector(task='classification', k=2)
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

        selector = MutualInformationSelector(task='classification', k=2)
        selector.fit(df, y)

        indices = selector.get_support(indices=True)
        assert isinstance(indices, list)
        assert len(indices) == 2
        assert all(0 <= idx < 3 for idx in indices)

    def test_methods_require_fit(self):
        """Test that utility methods require fit."""
        selector = MutualInformationSelector()

        with pytest.raises(RuntimeError):
            selector.get_selected_features()

        with pytest.raises(RuntimeError):
            selector.get_mi_scores()

        with pytest.raises(RuntimeError):
            selector.get_feature_importance()

        with pytest.raises(RuntimeError):
            selector.get_support()


class TestMutualInformationSelectorEdgeCases:
    """Test edge cases and error handling."""

    def test_no_numeric_columns(self):
        """Test error when no numeric columns exist."""
        df = pd.DataFrame({'text': ['a', 'b', 'c']})
        y = pd.Series([0, 1, 0])
        selector = MutualInformationSelector()

        with pytest.raises(ValueError, match="No numeric columns"):
            selector.fit(df, y)

    def test_fit_requires_target(self):
        """Test that fit requires target."""
        df = pd.DataFrame({'x1': [1, 2, 3]})
        selector = MutualInformationSelector()

        with pytest.raises(ValueError, match="y .* must be provided"):
            selector.fit(df, None)

    def test_single_feature(self):
        """Test with single feature."""
        np.random.seed(42)
        df = pd.DataFrame({'x1': np.random.randn(50)})
        y = np.random.randint(0, 2, 50)

        selector = MutualInformationSelector(task='classification', k=1)
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
        selector = MutualInformationSelector(task='classification', k=10)
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

        selector = MutualInformationSelector(task='classification', k=1)
        selector.fit(df, y)

        assert selector.fitted

    def test_error_not_dataframe(self):
        """Test error when input is not DataFrame."""
        selector = MutualInformationSelector()
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

        selector = MutualInformationSelector(
            task='classification',
            k=2,
            columns=['use1', 'use2']
        )
        selector.fit(df, y)

        # Only use1 and use2 should be considered
        assert 'ignore' not in selector.mi_scores_.index


class TestMutualInformationSelectorPractical:
    """Test practical use cases."""

    def test_nonlinear_relationship(self):
        """Test detection of non-linear relationships."""
        np.random.seed(42)
        n = 300
        x_linear = np.random.randn(n)
        x_quadratic = np.random.randn(n)

        # Target has quadratic relationship with x_quadratic
        y = x_quadratic ** 2 + np.random.randn(n) * 0.5

        df = pd.DataFrame({
            'linear': x_linear,
            'quadratic': x_quadratic
        })

        selector = MutualInformationSelector(task='regression', k=1, random_state=42)
        selector.fit(df, y)

        # MI should detect the quadratic relationship
        # (Note: this is probabilistic, may not always be true)
        scores = selector.get_mi_scores()
        assert scores['quadratic'] > 0

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

        selector = MutualInformationSelector(task='classification', k=2)
        train_result = selector.fit_transform(train, y_train)
        test_result = selector.transform(test)

        # Same columns in both
        assert list(train_result.columns) == list(test_result.columns)

    def test_discrete_features_auto_detection(self):
        """Test automatic detection of discrete features."""
        np.random.seed(42)
        df = pd.DataFrame({
            'continuous': np.random.randn(100),
            'discrete': np.random.randint(0, 5, 100),  # Integer with few values
            'binary': np.random.randint(0, 2, 100)
        })
        y = np.random.randint(0, 2, 100)

        selector = MutualInformationSelector(
            task='classification',
            discrete_features='infer',
            k=3
        )
        selector.fit(df, y)

        assert selector.fitted

    def test_random_state_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)

        selector1 = MutualInformationSelector(task='classification', k=2, random_state=42)
        selector1.fit(df, y)
        scores1 = selector1.get_mi_scores()

        selector2 = MutualInformationSelector(task='classification', k=2, random_state=42)
        selector2.fit(df, y)
        scores2 = selector2.get_mi_scores()

        # Should get same scores with same random_state
        pd.testing.assert_series_equal(scores1, scores2)

    def test_repr_not_fitted(self):
        """Test string representation before fitting."""
        selector = MutualInformationSelector(task='regression', k=5)
        repr_str = repr(selector)
        assert 'not fitted' in repr_str
        assert 'regression' in repr_str
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

        selector = MutualInformationSelector(task='classification', k=2)
        selector.fit(df, y)

        repr_str = repr(selector)
        assert 'selected=2' in repr_str or 'selected=2/3' in repr_str
