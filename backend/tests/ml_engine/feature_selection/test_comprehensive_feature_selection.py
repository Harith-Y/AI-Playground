"""Comprehensive tests for all feature selection modules to improve coverage."""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from app.ml_engine.feature_selection.variance_threshold import VarianceThreshold
from app.ml_engine.feature_selection.correlation_selector import CorrelationSelector
from app.ml_engine.feature_selection.mutual_information_selector import MutualInformationSelector
from app.ml_engine.feature_selection.rfe_selector import RFESelector
from app.ml_engine.feature_selection.univariate_selector import UnivariateSelector


class TestVarianceThresholdCoverage:
    """Test VarianceThreshold for coverage."""
    
    def test_basic_transform(self):
        """Test basic variance threshold transformation."""
        df = pd.DataFrame({
            'constant': [1, 1, 1, 1, 1],
            'low_var': [1, 1, 1, 1, 2],
            'high_var': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        
        selector = VarianceThreshold(threshold=0.0)
        result = selector.fit_transform(df)
        
        assert 'constant' not in result.columns
        assert 'high_var' in result.columns
        
    def test_with_columns_param(self):
        """Test variance threshold with specific columns."""
        df = pd.DataFrame({
            'col1': [1, 1, 1, 1],
            'col2': [1, 2, 3, 4],
            'col3': [5, 5, 5, 5]
        })
        
        selector = VarianceThreshold(threshold=0.0, columns=['col1', 'col3'])
        result = selector.fit_transform(df)
        
        assert 'col2' in result.columns
        
    def test_get_params(self):
        """Test get_params method."""
        selector = VarianceThreshold(threshold=0.1)
        params = selector.get_params()
        assert params['threshold'] == 0.1
        
    def test_set_params(self):
        """Test set_params method."""
        selector = VarianceThreshold(threshold=0.1)
        selector.set_params(threshold=0.2)
        assert selector.threshold == 0.2


class TestCorrelationSelectorCoverage:
    """Test CorrelationSelector for coverage."""
    
    def test_basic_transform(self):
        """Test basic correlation selector transformation."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        df['x3'] = df['x1'] + np.random.randn(100) * 0.1  # Highly correlated with x1
        
        selector = CorrelationSelector(threshold=0.8, target_column='target')
        result = selector.fit_transform(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) <= len(df.columns)
        
    def test_with_method_param(self):
        """Test with different correlation methods."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        selector = CorrelationSelector(
            threshold=0.5,
            target_column='target',
            method='spearman'
        )
        result = selector.fit_transform(df)
        
        assert isinstance(result, pd.DataFrame)
        
    def test_get_params(self):
        """Test get_params method."""
        selector = CorrelationSelector(threshold=0.9, target_column='y')
        params = selector.get_params()
        assert params['threshold'] == 0.9
        assert params['target_column'] == 'y'


class TestMutualInformationSelectorCoverage:
    """Test MutualInformationSelector for coverage."""
    
    def test_basic_transform_classification(self):
        """Test mutual information for classification."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        
        selector = MutualInformationSelector(
            k=3,
            target_column='target',
            task='classification'
        )
        result = selector.fit_transform(df)
        
        assert len(result.columns) == 4  # 3 features + target
        
    def test_basic_transform_regression(self):
        """Test mutual information for regression."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        
        selector = MutualInformationSelector(
            k=2,
            target_column='target',
            task='regression'
        )
        result = selector.fit_transform(df)
        
        assert len(result.columns) == 3  # 2 features + target
        
    def test_get_selected_features(self):
        """Test get_selected_features method."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        
        selector = MutualInformationSelector(
            k=3,
            target_column='target',
            task='classification'
        )
        selector.fit(df)
        selected = selector.get_selected_features()
        
        assert isinstance(selected, list)
        assert len(selected) == 3


class TestRFESelectorCoverage:
    """Test RFESelector for coverage."""
    
    def test_basic_transform(self):
        """Test basic RFE transformation."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        
        selector = RFESelector(
            n_features_to_select=3,
            target_column='target',
            task='classification'
        )
        result = selector.fit_transform(df)
        
        assert len(result.columns) == 4  # 3 features + target
        
    def test_with_custom_estimator(self):
        """Test RFE with custom estimator."""
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        selector = RFESelector(
            n_features_to_select=2,
            target_column='target',
            task='classification',
            estimator=estimator
        )
        result = selector.fit_transform(df)
        
        assert len(result.columns) == 3  # 2 features + target
        
    def test_get_feature_ranking(self):
        """Test get_feature_ranking method."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        
        selector = RFESelector(
            n_features_to_select=3,
            target_column='target',
            task='classification'
        )
        selector.fit(df)
        ranking = selector.get_feature_ranking()
        
        assert isinstance(ranking, dict)
        assert len(ranking) == 5


class TestUnivariateSelectorCoverage:
    """Test UnivariateSelector for coverage."""
    
    def test_basic_transform_classification(self):
        """Test univariate selector for classification."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        
        selector = UnivariateSelector(
            k=3,
            target_column='target',
            task='classification',
            score_func='f_classif'
        )
        result = selector.fit_transform(df)
        
        assert len(result.columns) == 4  # 3 features + target
        
    def test_chi2_score_function(self):
        """Test univariate selector with chi2."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            random_state=42
        )
        X = np.abs(X)  # Make features non-negative for chi2
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        
        selector = UnivariateSelector(
            k=2,
            target_column='target',
            task='classification',
            score_func='chi2'
        )
        result = selector.fit_transform(df)
        
        assert len(result.columns) == 3  # 2 features + target
        
    def test_regression_f_test(self):
        """Test univariate selector for regression."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        
        selector = UnivariateSelector(
            k=2,
            target_column='target',
            task='regression',
            score_func='f_regression'
        )
        result = selector.fit_transform(df)
        
        assert len(result.columns) == 3  # 2 features + target
        
    def test_get_scores(self):
        """Test get_scores method."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        
        selector = UnivariateSelector(
            k=3,
            target_column='target',
            task='classification'
        )
        selector.fit(df)
        scores = selector.get_scores()
        
        assert isinstance(scores, dict)
        assert len(scores) == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
