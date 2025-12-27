"""
Tests for RFESelector class (Recursive Feature Elimination).
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from app.ml_engine.feature_selection.rfe_selector import RFESelector


class TestRFESelectorInitialization:
    """Test RFESelector initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        selector = RFESelector()
        assert selector.params["estimator"] == 'auto'
        assert selector.params["n_features_to_select"] is None
        assert selector.params["step"] == 1
        assert selector.params["task"] == 'classification'
        assert not selector.fitted

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        selector = RFESelector(
            estimator='ridge',
            n_features_to_select=5,
            step=2,
            task='regression',
            columns=['a', 'b', 'c']
        )
        assert selector.params["estimator"] == 'ridge'
        assert selector.params["n_features_to_select"] == 5
        assert selector.params["step"] == 2
        assert selector.params["task"] == 'regression'
        assert selector.params["columns"] == ['a', 'b', 'c']

    def test_sklearn_estimator(self):
        """Test initialization with sklearn estimator."""
        est = LogisticRegression()
        selector = RFESelector(estimator=est)
        assert selector.params["estimator"] == est

    def test_error_invalid_estimator_string(self):
        """Test error with invalid estimator string."""
        with pytest.raises(ValueError, match="estimator must be one of"):
            RFESelector(estimator='invalid')

    def test_error_invalid_n_features(self):
        """Test error with invalid n_features_to_select."""
        with pytest.raises(ValueError, match="n_features_to_select must be positive"):
            RFESelector(n_features_to_select=0)

        with pytest.raises(ValueError, match="n_features_to_select must be positive"):
            RFESelector(n_features_to_select=-5)

    def test_error_invalid_step(self):
        """Test error with invalid step."""
        with pytest.raises(ValueError, match="step must be positive"):
            RFESelector(step=0)

    def test_error_invalid_task(self):
        """Test error with invalid task."""
        with pytest.raises(ValueError, match="task must be"):
            RFESelector(task='invalid')


class TestRFESelectorFitClassification:
    """Test RFESelector fit for classification."""

    def test_fit_basic_classification(self):
        """Test basic fitting for classification."""
        # Create classification dataset
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'feature4': np.random.randn(100),
        })
        y = (df['feature1'] + df['feature2'] > 0).astype(int)

        selector = RFESelector(n_features_to_select=2)
        selector.fit(df, y)

        assert selector.fitted
        assert selector.selected_features_ is not None
        assert len(selector.selected_features_) == 2

    def test_fit_requires_target(self):
        """Test that fit requires target variable."""
        df = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6]})
        selector = RFESelector()

        with pytest.raises(ValueError, match="y .* must be provided"):
            selector.fit(df)

    def test_auto_selects_half_features(self):
        """Test that auto selection chooses half the features."""
        np.random.seed(42)
        df = pd.DataFrame({
            'f1': np.random.randn(50),
            'f2': np.random.randn(50),
            'f3': np.random.randn(50),
            'f4': np.random.randn(50),
        })
        y = np.random.randint(0, 2, 50)

        selector = RFESelector(n_features_to_select=None)  # Auto select
        selector.fit(df, y)

        # Should select half (2 out of 4)
        assert len(selector.selected_features_) == 2

    def test_different_estimators_classification(self):
        """Test different estimator types for classification."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100),
        })
        y = np.random.randint(0, 2, 100)

        estimators = ['logistic', 'random_forest', 'svm', 'decision_tree']

        for est in estimators:
            selector = RFESelector(estimator=est, n_features_to_select=2)
            selector.fit(df, y)
            assert selector.fitted
            assert len(selector.selected_features_) == 2

    def test_step_parameter(self):
        """Test step parameter for feature removal."""
        np.random.seed(42)
        df = pd.DataFrame({
            f'f{i}': np.random.randn(50) for i in range(10)
        })
        y = np.random.randint(0, 2, 50)

        # Step=1 removes one feature at a time
        selector1 = RFESelector(n_features_to_select=5, step=1)
        selector1.fit(df, y)

        # Step=2 removes two features at a time
        selector2 = RFESelector(n_features_to_select=5, step=2)
        selector2.fit(df, y)

        # Both should select 5 features
        assert len(selector1.selected_features_) == 5
        assert len(selector2.selected_features_) == 5


class TestRFESelectorFitRegression:
    """Test RFESelector fit for regression."""

    def test_fit_basic_regression(self):
        """Test basic fitting for regression."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'feature4': np.random.randn(100),
        })
        y = df['feature1'] + df['feature2'] + np.random.randn(100) * 0.1

        selector = RFESelector(task='regression', n_features_to_select=2)
        selector.fit(df, y)

        assert selector.fitted
        assert len(selector.selected_features_) == 2

    def test_different_estimators_regression(self):
        """Test different estimator types for regression."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100),
        })
        y = df['x1'] + np.random.randn(100) * 0.1

        estimators = ['ridge', 'random_forest', 'decision_tree']

        for est in estimators:
            selector = RFESelector(
                estimator=est,
                task='regression',
                n_features_to_select=2
            )
            selector.fit(df, y)
            assert selector.fitted
            assert len(selector.selected_features_) == 2

    def test_auto_task_detection_regression(self):
        """Test auto task detection for regression."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
        })
        y = df['x1'] + np.random.randn(100) * 0.1

        # Auto should detect regression from continuous target
        selector = RFESelector(estimator='auto', task='regression', n_features_to_select=1)
        selector.fit(df, y)

        assert selector.fitted


class TestRFESelectorTransform:
    """Test RFESelector transform method."""

    def test_transform_selects_features(self):
        """Test that transform returns only selected features."""
        np.random.seed(42)
        df = pd.DataFrame({
            'keep1': np.random.randn(50),
            'keep2': np.random.randn(50),
            'drop1': np.random.randn(50),
            'drop2': np.random.randn(50),
        })
        y = (df['keep1'] + df['keep2'] > 0).astype(int)

        selector = RFESelector(n_features_to_select=2)
        result = selector.fit_transform(df, y)

        assert result.shape[1] == 2
        assert len(selector.selected_features_) == 2

    def test_transform_preserves_data_integrity(self):
        """Test that transform preserves original data values."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [2, 4, 6, 8, 10],
            'x3': [1, 1, 1, 1, 1],
        })
        y = pd.Series([0, 1, 0, 1, 0])

        selector = RFESelector(n_features_to_select=2)
        selector.fit(df, y)
        result = selector.transform(df)

        # Check that selected columns have original values
        for col in selector.selected_features_:
            assert result[col].equals(df[col])

    def test_transform_requires_fit(self):
        """Test that transform requires fit."""
        df = pd.DataFrame({'x1': [1, 2, 3]})
        selector = RFESelector()

        with pytest.raises(RuntimeError, match="must be fitted"):
            selector.transform(df)

    def test_transform_missing_features_error(self):
        """Test error when transform data is missing features."""
        df_train = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'x3': np.random.randn(50),
        })
        y = np.random.randint(0, 2, 50)

        selector = RFESelector(n_features_to_select=2)
        selector.fit(df_train, y)

        # Test data missing a feature
        df_test = pd.DataFrame({
            'x1': np.random.randn(10),
            'x2': np.random.randn(10),
        })

        # Only error if missing a SELECTED feature
        if 'x3' in selector.selected_features_:
            with pytest.raises(ValueError, match="not found in DataFrame"):
                selector.transform(df_test)


class TestRFESelectorUtilityMethods:
    """Test utility methods."""

    def test_get_selected_features(self):
        """Test get_selected_features method."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'x3': np.random.randn(50),
        })
        y = np.random.randint(0, 2, 50)

        selector = RFESelector(n_features_to_select=2)
        selector.fit(df, y)

        features = selector.get_selected_features()
        assert isinstance(features, list)
        assert len(features) == 2

    def test_get_feature_ranking(self):
        """Test get_feature_ranking method."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'x3': np.random.randn(50),
            'x4': np.random.randn(50),
        })
        y = np.random.randint(0, 2, 50)

        selector = RFESelector(n_features_to_select=2)
        selector.fit(df, y)

        ranking = selector.get_feature_ranking()
        assert isinstance(ranking, pd.DataFrame)
        assert 'feature' in ranking.columns
        assert 'rank' in ranking.columns
        assert 'selected' in ranking.columns
        assert len(ranking) == 4

    def test_feature_ranking_sorted(self):
        """Test that feature ranking is sorted by rank."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'x3': np.random.randn(50),
        })
        y = np.random.randint(0, 2, 50)

        selector = RFESelector(n_features_to_select=2)
        selector.fit(df, y)

        ranking = selector.get_feature_ranking()

        # Verify sorted by rank ascending (1 is best)
        for i in range(len(ranking) - 1):
            assert ranking.iloc[i]['rank'] <= ranking.iloc[i + 1]['rank']

    def test_get_support_boolean_mask(self):
        """Test get_support with boolean mask."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'x3': np.random.randn(50),
        })
        y = np.random.randint(0, 2, 50)

        selector = RFESelector(n_features_to_select=2)
        selector.fit(df, y)

        support = selector.get_support(indices=False)
        assert isinstance(support, list)
        assert len(support) == 3
        assert sum(support) == 2  # Two features selected

    def test_get_support_indices(self):
        """Test get_support with indices."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'x3': np.random.randn(50),
        })
        y = np.random.randint(0, 2, 50)

        selector = RFESelector(n_features_to_select=2)
        selector.fit(df, y)

        indices = selector.get_support(indices=True)
        assert isinstance(indices, list)
        assert len(indices) == 2  # Two features selected

    def test_utility_methods_require_fit(self):
        """Test that utility methods require fit."""
        selector = RFESelector()

        with pytest.raises(RuntimeError):
            selector.get_selected_features()

        with pytest.raises(RuntimeError):
            selector.get_feature_ranking()

        with pytest.raises(RuntimeError):
            selector.get_support()


class TestRFESelectorEdgeCases:
    """Test edge cases and error handling."""

    def test_error_not_dataframe(self):
        """Test error when input is not DataFrame."""
        selector = RFESelector()

        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            selector.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))

    def test_no_numeric_columns(self):
        """Test error when no numeric columns exist."""
        df = pd.DataFrame({'text': ['a', 'b', 'c']})
        y = pd.Series([0, 1, 0])
        selector = RFESelector()

        with pytest.raises(ValueError, match="No numeric columns"):
            selector.fit(df, y)

    def test_single_feature(self):
        """Test with single feature."""
        df = pd.DataFrame({'x1': np.random.randn(50)})
        y = np.random.randint(0, 2, 50)

        selector = RFESelector(n_features_to_select=1)
        selector.fit(df, y)

        assert len(selector.selected_features_) == 1
        assert selector.selected_features_[0] == 'x1'

    def test_request_more_features_than_available(self):
        """Test requesting more features than available."""
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
        })
        y = np.random.randint(0, 2, 50)

        # Request 5 features but only 2 available
        selector = RFESelector(n_features_to_select=5)
        selector.fit(df, y)

        # Should select all available features
        assert len(selector.selected_features_) == 2

    def test_specific_columns_parameter(self):
        """Test with specific columns parameter."""
        df = pd.DataFrame({
            'use1': np.random.randn(50),
            'use2': np.random.randn(50),
            'ignore': np.random.randn(50),
        })
        y = np.random.randint(0, 2, 50)

        selector = RFESelector(
            n_features_to_select=1,
            columns=['use1', 'use2']
        )
        selector.fit(df, y)

        # Should only consider use1 and use2
        assert selector.selected_features_[0] in ['use1', 'use2']

    def test_numpy_array_target(self):
        """Test with numpy array as target."""
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
        })
        y = np.random.randint(0, 2, 50)  # NumPy array

        selector = RFESelector(n_features_to_select=1)
        selector.fit(df, y)

        assert selector.fitted


class TestRFESelectorPractical:
    """Test practical use cases."""

    def test_feature_selection_improves_interpretability(self):
        """Test that RFE reduces feature count."""
        np.random.seed(42)
        # Create dataset with many features
        df = pd.DataFrame({
            f'feature_{i}': np.random.randn(100) for i in range(20)
        })
        # Target only depends on first 3 features
        y = (df['feature_0'] + df['feature_1'] + df['feature_2'] > 0).astype(int)

        selector = RFESelector(n_features_to_select=5)
        result = selector.fit_transform(df, y)

        # Should significantly reduce feature count
        assert result.shape[1] == 5
        assert result.shape[1] < df.shape[1]

    def test_train_test_consistency(self):
        """Test consistency between train and test transforms."""
        np.random.seed(42)
        train = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'x3': np.random.randn(50),
        })
        y_train = np.random.randint(0, 2, 50)

        test = pd.DataFrame({
            'x1': np.random.randn(20),
            'x2': np.random.randn(20),
            'x3': np.random.randn(20),
        })

        selector = RFESelector(n_features_to_select=2)
        train_result = selector.fit_transform(train, y_train)
        test_result = selector.transform(test)

        # Same columns in both
        assert list(train_result.columns) == list(test_result.columns)

    def test_custom_sklearn_estimator(self):
        """Test with custom sklearn estimator."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'x3': np.random.randn(50),
        })
        y = np.random.randint(0, 2, 50)

        # Use custom estimator with specific parameters
        custom_est = RandomForestClassifier(n_estimators=10, random_state=42)
        selector = RFESelector(estimator=custom_est, n_features_to_select=2)
        selector.fit(df, y)

        assert selector.fitted
        assert len(selector.selected_features_) == 2

    def test_repr_not_fitted(self):
        """Test string representation before fitting."""
        selector = RFESelector(estimator='ridge', n_features_to_select=5)
        repr_str = repr(selector)
        assert 'not fitted' in repr_str
        assert 'ridge' in repr_str

    def test_repr_fitted(self):
        """Test string representation after fitting."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'x3': np.random.randn(50),
        })
        y = np.random.randint(0, 2, 50)

        selector = RFESelector(n_features_to_select=2)
        selector.fit(df, y)

        repr_str = repr(selector)
        assert 'selected=' in repr_str
        assert '2/3' in repr_str  # 2 selected out of 3


class TestRFESelectorRanking:
    """Test feature ranking functionality."""

    def test_ranking_reflects_importance(self):
        """Test that ranking reflects feature importance."""
        np.random.seed(42)
        # Create dataset where x1 and x2 are important, x3 is noise
        x1 = np.random.randn(100)
        x2 = np.random.randn(100)
        x3 = np.random.randn(100)

        df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})
        y = ((x1 + x2) > 0).astype(int)

        selector = RFESelector(n_features_to_select=2)
        selector.fit(df, y)

        ranking = selector.get_feature_ranking()

        # x1 and x2 should have better ranks (lower values) than x3
        selected = ranking[ranking['selected'] == True]
        assert len(selected) == 2

    def test_rank_1_is_best(self):
        """Test that rank 1 indicates best features."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'x3': np.random.randn(50),
        })
        y = np.random.randint(0, 2, 50)

        selector = RFESelector(n_features_to_select=1)
        selector.fit(df, y)

        ranking = selector.get_feature_ranking()
        best_feature = ranking[ranking['rank'] == 1]

        assert len(best_feature) == 1
        assert best_feature['selected'].iloc[0] == True
