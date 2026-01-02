"""Comprehensive tests for EDA, correlation, and class distribution modules."""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

from app.ml_engine.eda_statistics import (
    EDAStatistics,
    quick_summary,
    detect_data_issues
)
from app.ml_engine.correlation_analysis import (
    CorrelationAnalyzer,
    correlation_matrix,
    correlation_heatmap_data,
    find_highly_correlated_pairs,
    detect_multicollinearity,
    rank_features_by_target
)
from app.ml_engine.class_distribution_analysis import (
    ClassDistributionAnalyzer,
    analyze_class_distribution,
    check_class_imbalance
)


class TestEDAStatistics:
    """Test EDA statistics functions for coverage."""
    
    def test_eda_statistics_init(self):
        """Test EDAStatistics initialization."""
        df = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10, 20, 30, 40, 50],
            'categorical': ['A', 'B', 'A', 'C', 'B']
        })
        
        eda = EDAStatistics(df)
        assert eda is not None
        
    def test_get_summary_statistics(self):
        """Test summary statistics calculation."""
        df = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10, 20, 30, 40, 50]
        })
        
        eda = EDAStatistics(df)
        stats = eda.get_summary_statistics()
        
        assert isinstance(stats, (dict, pd.DataFrame))
        
    def test_get_categorical_summary(self):
        """Test categorical summary."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B'] * 10
        })
        
        eda = EDAStatistics(df)
        summary = eda.get_categorical_summary()
        
        assert isinstance(summary, (dict, pd.DataFrame))
        
    def test_analyze_missing_data(self):
        """Test missing data analysis."""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [np.nan, np.nan, 3, 4, 5],
            'col3': [1, 2, 3, 4, 5]
        })
        
        eda = EDAStatistics(df)
        missing_info = eda.analyze_missing_data()
        
        assert isinstance(missing_info, (dict, pd.DataFrame))
        
    def test_detect_outliers(self):
        """Test outlier detection."""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        })
        
        eda = EDAStatistics(df)
        outliers = eda.detect_outliers()
        
        assert isinstance(outliers, (dict, pd.DataFrame, list))
        
    def test_analyze_distributions(self):
        """Test distribution analysis."""
        df = pd.DataFrame({
            'values': np.random.randn(100)
        })
        
        eda = EDAStatistics(df)
        distributions = eda.analyze_distributions()
        
        assert isinstance(distributions, (dict, pd.DataFrame))
        
    def test_get_correlation_summary(self):
        """Test correlation summary."""
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'x3': np.random.randn(50)
        })
        
        eda = EDAStatistics(df)
        corr_summary = eda.get_correlation_summary()
        
        assert isinstance(corr_summary, (dict, pd.DataFrame))
        
    def test_get_data_quality_report(self):
        """Test data quality report."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, np.nan, 5],
            'col2': ['A', 'B', 'A', 'C', 'B']
        })
        
        eda = EDAStatistics(df)
        quality = eda.get_data_quality_report()
        
        assert isinstance(quality, dict)
        
    def test_get_feature_types(self):
        """Test feature type detection."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        
        eda = EDAStatistics(df)
        types = eda.get_feature_types()
        
        assert isinstance(types, (dict, pd.DataFrame))
        
    def test_quick_summary(self):
        """Test quick summary function."""
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50)
        })
        
        summary = quick_summary(df)
        assert isinstance(summary, dict)
        
    def test_detect_data_issues(self):
        """Test data issues detection."""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [1, 1, 1, 1, 1],  # Constant
            'col3': [1, 2, 3, 4, 100]  # Outlier
        })
        
        issues = detect_data_issues(df)
        assert isinstance(issues, dict)


class TestCorrelationAnalysis:
    """Test correlation analysis functions for coverage."""
    
    def test_correlation_analyzer_init(self):
        """Test CorrelationAnalyzer initialization."""
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        
        analyzer = CorrelationAnalyzer(df)
        assert analyzer is not None
        
    def test_compute_correlation(self):
        """Test correlation matrix computation."""
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        
        analyzer = CorrelationAnalyzer(df)
        corr = analyzer.compute_correlation()
        
        assert isinstance(corr, (pd.DataFrame, dict))
        
    def test_get_correlation_pairs(self):
        """Test getting correlation pairs."""
        np.random.seed(42)
        x1 = np.random.randn(100)
        df = pd.DataFrame({
            'x1': x1,
            'x2': x1 + np.random.randn(100) * 0.1,  # Highly correlated
            'x3': np.random.randn(100)
        })
        
        analyzer = CorrelationAnalyzer(df)
        pairs = analyzer.get_correlation_pairs(threshold=0.7)
        
        assert isinstance(pairs, (list, dict, pd.DataFrame))
        
    def test_get_heatmap_data(self):
        """Test heatmap data generation."""
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50)
        })
        
        analyzer = CorrelationAnalyzer(df)
        heatmap = analyzer.get_heatmap_data()
        
        assert isinstance(heatmap, (dict, pd.DataFrame))
        
    def test_get_multicollinearity_stats(self):
        """Test multicollinearity statistics."""
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        
        analyzer = CorrelationAnalyzer(df)
        stats = analyzer.get_multicollinearity_stats()
        
        assert isinstance(stats, dict)
        
    def test_correlation_matrix_function(self):
        """Test correlation matrix helper function."""
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50)
        })
        
        corr = correlation_matrix(df)
        assert isinstance(corr, (pd.DataFrame, dict))
        
    def test_correlation_heatmap_data_function(self):
        """Test correlation heatmap data function."""
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50)
        })
        
        heatmap = correlation_heatmap_data(df)
        assert isinstance(heatmap, (dict, pd.DataFrame))
        
    def test_find_highly_correlated_pairs_function(self):
        """Test finding highly correlated pairs."""
        np.random.seed(42)
        x1 = np.random.randn(100)
        df = pd.DataFrame({
            'x1': x1,
            'x2': x1 + np.random.randn(100) * 0.1,
            'x3': np.random.randn(100)
        })
        
        pairs = find_highly_correlated_pairs(df, threshold=0.7)
        assert isinstance(pairs, (list, dict, pd.DataFrame))
        
    def test_detect_multicollinearity_function(self):
        """Test multicollinearity detection."""
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        
        result = detect_multicollinearity(df)
        assert isinstance(result, (dict, bool, pd.DataFrame))
        
    def test_rank_features_by_target_function(self):
        """Test feature ranking by target."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        
        ranking = rank_features_by_target(df, target_column='target')
        assert isinstance(ranking, (dict, list, pd.DataFrame, pd.Series))


class TestClassDistributionAnalysis:
    """Test class distribution analysis for coverage."""
    
    def test_class_distribution_analyzer_init(self):
        """Test ClassDistributionAnalyzer initialization."""
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'target': np.random.choice([0, 1, 2], 100)
        })
        
        analyzer = ClassDistributionAnalyzer(df, target_column='target')
        assert analyzer is not None
        
    def test_get_class_distribution(self):
        """Test class distribution retrieval."""
        df = pd.DataFrame({
            'target': np.random.choice([0, 1, 2], 100, p=[0.5, 0.3, 0.2])
        })
        
        analyzer = ClassDistributionAnalyzer(df, target_column='target')
        distribution = analyzer.get_class_distribution()
        
        assert isinstance(distribution, (dict, pd.DataFrame))
        
    def test_get_imbalance_ratio(self):
        """Test imbalance ratio calculation."""
        df = pd.DataFrame({
            'target': [0] * 90 + [1] * 10
        })
        
        analyzer = ClassDistributionAnalyzer(df, target_column='target')
        ratio = analyzer.get_imbalance_ratio()
        
        assert isinstance(ratio, (int, float))
        assert ratio >= 1
        
    def test_identify_minority_majority_classes(self):
        """Test minority/majority class identification."""
        df = pd.DataFrame({
            'target': [0] * 90 + [1] * 10
        })
        
        analyzer = ClassDistributionAnalyzer(df, target_column='target')
        classes = analyzer.identify_minority_majority_classes()
        
        assert isinstance(classes, dict)
        
    def test_get_imbalance_metrics(self):
        """Test imbalance metrics."""
        df = pd.DataFrame({
            'target': [0] * 80 + [1] * 20
        })
        
        analyzer = ClassDistributionAnalyzer(df, target_column='target')
        metrics = analyzer.get_imbalance_metrics()
        
        assert isinstance(metrics, dict)
        
    def test_get_resampling_recommendations(self):
        """Test resampling recommendations."""
        df = pd.DataFrame({
            'target': [0] * 90 + [1] * 10
        })
        
        analyzer = ClassDistributionAnalyzer(df, target_column='target')
        recommendations = analyzer.get_resampling_recommendations()
        
        assert isinstance(recommendations, dict)
        
    def test_get_bar_chart_data(self):
        """Test bar chart data generation."""
        df = pd.DataFrame({
            'target': np.random.choice([0, 1, 2], 100)
        })
        
        analyzer = ClassDistributionAnalyzer(df, target_column='target')
        chart_data = analyzer.get_bar_chart_data()
        
        assert isinstance(chart_data, dict)
        
    def test_get_summary_report(self):
        """Test summary report generation."""
        df = pd.DataFrame({
            'target': np.random.choice([0, 1], 100)
        })
        
        analyzer = ClassDistributionAnalyzer(df, target_column='target')
        report = analyzer.get_summary_report()
        
        assert isinstance(report, dict)
        
    def test_analyze_class_distribution_function(self):
        """Test analyze_class_distribution helper function."""
        df = pd.DataFrame({
            'target': [0] * 50 + [1] * 30 + [2] * 20
        })
        
        result = analyze_class_distribution(df, target_column='target')
        assert isinstance(result, (dict, pd.DataFrame))
        
    def test_check_class_imbalance_function(self):
        """Test check_class_imbalance helper function."""
        df = pd.DataFrame({
            'target': [0] * 90 + [1] * 10
        })
        
        is_imbalanced = check_class_imbalance(df, target_column='target')
        assert isinstance(is_imbalanced, (bool, dict))
        
    def test_balanced_classes(self):
        """Test with balanced classes."""
        df = pd.DataFrame({
            'target': [0] * 50 + [1] * 50
        })
        
        analyzer = ClassDistributionAnalyzer(df, target_column='target')
        ratio = analyzer.get_imbalance_ratio()
        
        assert ratio <= 2  # Should be balanced
        
    def test_extreme_imbalance(self):
        """Test extreme class imbalance."""
        df = pd.DataFrame({
            'target': [0] * 99 + [1]
        })
        
        analyzer = ClassDistributionAnalyzer(df, target_column='target')
        ratio = analyzer.get_imbalance_ratio()
        recommendations = analyzer.get_resampling_recommendations()
        
        assert ratio >= 10
        assert recommendations is not None


class TestEdgeCases:
    """Test edge cases for all analysis modules."""
    
    def test_single_value_column(self):
        """Test with constant column."""
        df = pd.DataFrame({
            'constant': [1] * 100,
            'varied': np.random.randn(100)
        })
        
        eda = EDAStatistics(df)
        stats = eda.get_summary_statistics()
        
        analyzer = CorrelationAnalyzer(df)
        corr = analyzer.compute_correlation()
        
        assert stats is not None
        assert corr is not None
        
    def test_all_missing_column(self):
        """Test with all missing values."""
        df = pd.DataFrame({
            'all_null': [np.nan] * 50,
            'some_null': [1, np.nan, 3, np.nan, 5] * 10
        })
        
        eda = EDAStatistics(df)
        missing_info = eda.analyze_missing_data()
        
        assert missing_info is not None
        
    def test_single_class(self):
        """Test with single class."""
        df = pd.DataFrame({
            'target': [0] * 100
        })
        
        analyzer = ClassDistributionAnalyzer(df, target_column='target')
        distribution = analyzer.get_class_distribution()
        
        assert distribution is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
