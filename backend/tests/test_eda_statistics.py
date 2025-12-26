"""
Tests for EDA Statistics module.
"""

import pytest
import pandas as pd
import numpy as np
from app.ml_engine.eda_statistics import (
    EDAStatistics,
    quick_summary,
    detect_data_issues
)


class TestEDAStatisticsInitialization:
    """Test EDAStatistics initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        eda = EDAStatistics(df)

        assert eda.df is not None
        assert len(eda._numeric_cols) == 2
        assert len(eda._categorical_cols) == 0

    def test_error_not_dataframe(self):
        """Test error when input is not DataFrame."""
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            EDAStatistics([1, 2, 3])

    def test_mixed_types(self):
        """Test with mixed column types."""
        df = pd.DataFrame({
            'numeric': [1, 2, 3],
            'categorical': ['a', 'b', 'c'],
            'numeric2': [4.0, 5.0, 6.0]
        })
        eda = EDAStatistics(df)

        assert len(eda._numeric_cols) == 2
        assert len(eda._categorical_cols) == 1


class TestSummaryStatistics:
    """Test summary statistics methods."""

    def test_get_summary_statistics_basic(self):
        """Test basic summary statistics."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })
        eda = EDAStatistics(df)
        summary = eda.get_summary_statistics()

        assert isinstance(summary, pd.DataFrame)
        assert 'mean' in summary.columns
        assert 'std' in summary.columns
        assert 'missing' in summary.columns
        assert len(summary) == 2

    def test_summary_includes_additional_metrics(self):
        """Test that summary includes skewness, kurtosis, etc."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 100]})
        eda = EDAStatistics(df)
        summary = eda.get_summary_statistics()

        assert 'skewness' in summary.columns
        assert 'kurtosis' in summary.columns
        assert 'missing_pct' in summary.columns
        assert 'unique' in summary.columns
        assert 'cv' in summary.columns

    def test_summary_specific_columns(self):
        """Test summary for specific columns."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6],
            'z': [7, 8, 9]
        })
        eda = EDAStatistics(df)
        summary = eda.get_summary_statistics(columns=['x', 'y'])

        assert len(summary) == 2
        assert 'x' in summary.index
        assert 'y' in summary.index
        assert 'z' not in summary.index

    def test_summary_with_missing_values(self):
        """Test summary with missing values."""
        df = pd.DataFrame({
            'x': [1, 2, np.nan, 4, 5],
            'y': [10, np.nan, np.nan, 40, 50]
        })
        eda = EDAStatistics(df)
        summary = eda.get_summary_statistics()

        assert summary.loc['x', 'missing'] == 1
        assert summary.loc['y', 'missing'] == 2
        assert summary.loc['x', 'missing_pct'] == 20.0
        assert summary.loc['y', 'missing_pct'] == 40.0

    def test_summary_no_numeric_columns(self):
        """Test summary when no numeric columns exist."""
        df = pd.DataFrame({'text': ['a', 'b', 'c']})
        eda = EDAStatistics(df)
        summary = eda.get_summary_statistics()

        assert len(summary) == 0


class TestCategoricalSummary:
    """Test categorical summary methods."""

    def test_get_categorical_summary_basic(self):
        """Test basic categorical summary."""
        df = pd.DataFrame({
            'category': ['a', 'b', 'a', 'c', 'a']
        })
        eda = EDAStatistics(df)
        summary = eda.get_categorical_summary()

        assert 'category' in summary
        assert 'summary' in summary['category']
        assert 'top_categories' in summary['category']

    def test_categorical_summary_stats(self):
        """Test categorical summary statistics."""
        df = pd.DataFrame({
            'color': ['red', 'blue', 'red', 'green', 'red']
        })
        eda = EDAStatistics(df)
        summary = eda.get_categorical_summary()

        stats = summary['color']['summary']
        assert stats.loc['color', 'count'] == 5
        assert stats.loc['color', 'unique'] == 3
        assert stats.loc['color', 'most_common'] == 'red'
        assert stats.loc['color', 'most_common_freq'] == 3

    def test_categorical_summary_with_missing(self):
        """Test categorical summary with missing values."""
        df = pd.DataFrame({
            'category': ['a', 'b', np.nan, 'a', np.nan]
        })
        eda = EDAStatistics(df)
        summary = eda.get_categorical_summary()

        stats = summary['category']['summary']
        assert stats.loc['category', 'missing'] == 2
        assert stats.loc['category', 'missing_pct'] == 40.0

    def test_categorical_summary_top_n(self):
        """Test categorical summary with top_n parameter."""
        df = pd.DataFrame({
            'letter': list('abcdefghijk' * 10)
        })
        eda = EDAStatistics(df)
        summary = eda.get_categorical_summary(top_n=5)

        top_cats = summary['letter']['top_categories']
        assert len(top_cats) == 5


class TestMissingDataAnalysis:
    """Test missing data analysis."""

    def test_analyze_missing_data_basic(self):
        """Test basic missing data analysis."""
        df = pd.DataFrame({
            'x': [1, 2, np.nan, 4],
            'y': [1, 2, 3, 4],
            'z': [np.nan, np.nan, 3, 4]
        })
        eda = EDAStatistics(df)
        missing = eda.analyze_missing_data()

        assert isinstance(missing, pd.DataFrame)
        assert 'column' in missing.columns
        assert 'missing_count' in missing.columns
        assert 'missing_percentage' in missing.columns

    def test_missing_data_sorted(self):
        """Test that missing data is sorted by percentage."""
        df = pd.DataFrame({
            'low': [1, 2, 3, 4, 5],
            'high': [np.nan, np.nan, np.nan, np.nan, 5],
            'medium': [1, np.nan, np.nan, 4, 5]
        })
        eda = EDAStatistics(df)
        missing = eda.analyze_missing_data()

        # Should be sorted high -> medium -> low
        assert missing.iloc[0]['column'] == 'high'
        assert missing.iloc[1]['column'] == 'medium'
        assert missing.iloc[2]['column'] == 'low'

    def test_missing_data_summary_attrs(self):
        """Test missing data summary attributes."""
        df = pd.DataFrame({
            'x': [1, np.nan, 3],
            'y': [np.nan, 2, 3]
        })
        eda = EDAStatistics(df)
        missing = eda.analyze_missing_data()

        assert 'total_cells' in missing.attrs
        assert 'total_missing' in missing.attrs
        assert 'total_missing_pct' in missing.attrs
        assert missing.attrs['total_cells'] == 6
        assert missing.attrs['total_missing'] == 2

    def test_no_missing_data(self):
        """Test analysis when no missing data exists."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        eda = EDAStatistics(df)
        missing = eda.analyze_missing_data()

        assert all(missing['missing_count'] == 0)
        assert all(missing['missing_percentage'] == 0)


class TestOutlierDetection:
    """Test outlier detection methods."""

    def test_detect_outliers_iqr(self):
        """Test IQR outlier detection."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 100]})
        eda = EDAStatistics(df)
        outliers = eda.detect_outliers(method='iqr')

        assert 'x' in outliers
        assert 'iqr_outliers' in outliers['x'].columns
        assert outliers['x']['iqr_outliers'].values[0] > 0

    def test_detect_outliers_zscore(self):
        """Test Z-score outlier detection."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 100]})
        eda = EDAStatistics(df)
        outliers = eda.detect_outliers(method='zscore')

        assert 'x' in outliers
        assert 'zscore_outliers' in outliers['x'].columns
        assert outliers['x']['zscore_outliers'].values[0] > 0

    def test_detect_outliers_both(self):
        """Test both outlier detection methods."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 100]})
        eda = EDAStatistics(df)
        outliers = eda.detect_outliers(method='both')

        assert 'x' in outliers
        assert 'iqr_outliers' in outliers['x'].columns
        assert 'zscore_outliers' in outliers['x'].columns

    def test_detect_outliers_custom_thresholds(self):
        """Test outlier detection with custom thresholds."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 10]})
        eda = EDAStatistics(df)

        strict = eda.detect_outliers(method='iqr', iqr_threshold=1.0)
        lenient = eda.detect_outliers(method='iqr', iqr_threshold=3.0)

        # Strict should find more outliers
        assert strict['x']['iqr_outliers'].values[0] >= lenient['x']['iqr_outliers'].values[0]

    def test_detect_outliers_specific_columns(self):
        """Test outlier detection for specific columns."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 100],
            'y': [1, 2, 3, 4],
            'z': [1, 2, 3, 200]
        })
        eda = EDAStatistics(df)
        outliers = eda.detect_outliers(columns=['x', 'y'])

        assert 'x' in outliers
        assert 'y' in outliers
        assert 'z' not in outliers

    def test_detect_outliers_no_outliers(self):
        """Test detection when no outliers exist."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        eda = EDAStatistics(df)
        outliers = eda.detect_outliers(method='both')

        assert outliers['x']['iqr_outliers'].values[0] == 0
        assert outliers['x']['zscore_outliers'].values[0] == 0


class TestDistributionAnalysis:
    """Test distribution analysis methods."""

    def test_analyze_distributions_basic(self):
        """Test basic distribution analysis."""
        df = pd.DataFrame({'x': np.random.randn(100)})
        eda = EDAStatistics(df)
        dist = eda.analyze_distributions()

        assert isinstance(dist, pd.DataFrame)
        assert 'skewness' in dist.columns
        assert 'kurtosis' in dist.columns
        assert 'skew_interpretation' in dist.columns

    def test_distribution_interpretations(self):
        """Test skewness and kurtosis interpretations."""
        # Symmetric distribution
        df1 = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        eda1 = EDAStatistics(df1)
        dist1 = eda1.analyze_distributions()

        assert 'Symmetric' in dist1.iloc[0]['skew_interpretation']

        # Right skewed
        df2 = pd.DataFrame({'x': [1, 1, 1, 2, 3, 100]})
        eda2 = EDAStatistics(df2)
        dist2 = eda2.analyze_distributions()

        assert 'Right' in dist2.iloc[0]['skew_interpretation']

    def test_distribution_normality_test(self):
        """Test normality test (Shapiro-Wilk)."""
        df = pd.DataFrame({'x': np.random.randn(100)})
        eda = EDAStatistics(df)
        dist = eda.analyze_distributions()

        assert 'shapiro_statistic' in dist.columns
        assert 'shapiro_p_value' in dist.columns
        assert 'is_normal' in dist.columns

    def test_distribution_small_sample(self):
        """Test distribution with small sample."""
        df = pd.DataFrame({'x': [1, 2]})
        eda = EDAStatistics(df)
        dist = eda.analyze_distributions()

        # Should handle small samples gracefully
        assert len(dist) == 0  # Too few data points


class TestCorrelationSummary:
    """Test correlation summary methods."""

    def test_get_correlation_summary_basic(self):
        """Test basic correlation summary."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]  # Perfectly correlated with x
        })
        eda = EDAStatistics(df)
        corr = eda.get_correlation_summary()

        assert 'correlation_matrix' in corr
        assert 'high_correlations' in corr
        assert isinstance(corr['correlation_matrix'], pd.DataFrame)

    def test_high_correlations_detection(self):
        """Test detection of high correlations."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10],  # Perfect correlation
            'z': [10, 20, 30, 40, 50]  # Independent scale
        })
        eda = EDAStatistics(df)
        corr = eda.get_correlation_summary(threshold=0.9)

        high_corr = corr['high_correlations']
        assert len(high_corr) > 0
        assert all('correlation' in pair for pair in high_corr)

    def test_correlation_methods(self):
        """Test different correlation methods."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [1, 4, 9, 16, 25]  # Quadratic relationship
        })
        eda = EDAStatistics(df)

        pearson = eda.get_correlation_summary(method='pearson')
        spearman = eda.get_correlation_summary(method='spearman')

        assert 'correlation_matrix' in pearson
        assert 'correlation_matrix' in spearman

    def test_correlation_threshold(self):
        """Test correlation threshold parameter."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })
        eda = EDAStatistics(df)

        high_threshold = eda.get_correlation_summary(threshold=0.99)
        low_threshold = eda.get_correlation_summary(threshold=0.5)

        # Lower threshold should find more correlations
        assert len(low_threshold['high_correlations']) >= len(high_threshold['high_correlations'])


class TestDataQualityReport:
    """Test data quality report methods."""

    def test_get_data_quality_report_basic(self):
        """Test basic data quality report."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': ['a', 'b', 'c']
        })
        eda = EDAStatistics(df)
        report = eda.get_data_quality_report()

        assert 'dataset_shape' in report
        assert 'missing_data' in report
        assert 'duplicates' in report
        assert 'column_types' in report
        assert 'data_quality_issues' in report

    def test_quality_report_shape(self):
        """Test dataset shape in quality report."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        eda = EDAStatistics(df)
        report = eda.get_data_quality_report()

        assert report['dataset_shape']['rows'] == 3
        assert report['dataset_shape']['columns'] == 2
        assert report['dataset_shape']['total_cells'] == 6

    def test_quality_report_duplicates(self):
        """Test duplicate detection in quality report."""
        df = pd.DataFrame({
            'x': [1, 2, 1, 3],
            'y': [4, 5, 4, 6]
        })
        eda = EDAStatistics(df)
        report = eda.get_data_quality_report()

        assert report['duplicates']['duplicate_rows'] == 1

    def test_quality_report_column_types(self):
        """Test column type summary in quality report."""
        df = pd.DataFrame({
            'numeric1': [1, 2, 3],
            'numeric2': [4.0, 5.0, 6.0],
            'categorical': ['a', 'b', 'c']
        })
        eda = EDAStatistics(df)
        report = eda.get_data_quality_report()

        assert report['column_types']['numeric'] == 2
        assert report['column_types']['categorical'] == 1

    def test_quality_report_issues(self):
        """Test data quality issues detection."""
        df = pd.DataFrame({
            'constant': [1, 1, 1, 1],
            'high_card': [f'cat_{i}' for i in range(100)],
            'normal': [1, 2, 3, 4]
        })
        eda = EDAStatistics(df)
        report = eda.get_data_quality_report()

        assert 'constant' in report['data_quality_issues']['constant_columns']
        assert 'high_card' in report['data_quality_issues']['high_cardinality_categorical']


class TestFeatureTypes:
    """Test feature type detection."""

    def test_get_feature_types_basic(self):
        """Test basic feature type detection."""
        df = pd.DataFrame({
            'numeric': [1, 2, 3],
            'categorical': ['a', 'b', 'c']
        })
        eda = EDAStatistics(df)
        types = eda.get_feature_types()

        assert isinstance(types, pd.DataFrame)
        assert 'column' in types.columns
        assert 'feature_type' in types.columns

    def test_binary_detection(self):
        """Test binary feature detection."""
        df = pd.DataFrame({
            'binary_num': [0, 1, 0, 1],
            'binary_cat': ['yes', 'no', 'yes', 'no']
        })
        eda = EDAStatistics(df)
        types = eda.get_feature_types()

        binary_types = types[types['column'].isin(['binary_num', 'binary_cat'])]['feature_type'].values
        assert any('Binary' in t for t in binary_types)

    def test_cardinality_detection(self):
        """Test cardinality detection for categorical."""
        df = pd.DataFrame({
            'low': ['a', 'b', 'a', 'b'],
            'medium': [f'cat_{i % 30}' for i in range(100)],
            'high': [f'cat_{i}' for i in range(100)]
        })
        eda = EDAStatistics(df)
        types = eda.get_feature_types()

        low_type = types[types['column'] == 'low']['feature_type'].values[0]
        medium_type = types[types['column'] == 'medium']['feature_type'].values[0]
        high_type = types[types['column'] == 'high']['feature_type'].values[0]

        assert 'Low' in low_type
        assert 'Medium' in medium_type
        assert 'High' in high_type


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_quick_summary(self):
        """Test quick_summary function."""
        df = pd.DataFrame({
            'x': [1, 2, 3, np.nan, 5],
            'y': ['a', 'b', 'c', 'd', 'e']
        })
        summary = quick_summary(df)

        assert 'shape' in summary
        assert 'numeric_summary' in summary
        assert 'missing_data' in summary
        assert 'quality_report' in summary

    def test_detect_data_issues(self):
        """Test detect_data_issues function."""
        df = pd.DataFrame({
            'high_missing': [np.nan] * 60 + [1] * 40,
            'constant': [1] * 100,
            'high_card': [f'cat_{i}' for i in range(100)],
            'normal': range(100)
        })
        df = pd.concat([df, df.iloc[:10]], ignore_index=True)  # Add duplicates

        issues = detect_data_issues(df)

        assert 'high_missing_rate' in issues
        assert 'constant_columns' in issues
        assert 'high_cardinality' in issues
        assert 'duplicate_rows_exist' in issues

        assert 'high_missing' in issues['high_missing_rate']
        assert 'constant' in issues['constant_columns']
        assert 'high_card' in issues['high_cardinality']
        assert issues['duplicate_rows_exist'] == True


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        eda = EDAStatistics(df)

        summary = eda.get_summary_statistics()
        assert len(summary) == 0

    def test_single_row(self):
        """Test with single row."""
        df = pd.DataFrame({'x': [1], 'y': [2]})
        eda = EDAStatistics(df)

        summary = eda.get_summary_statistics()
        assert len(summary) == 2

    def test_all_missing(self):
        """Test with all missing values."""
        df = pd.DataFrame({'x': [np.nan, np.nan, np.nan]})
        eda = EDAStatistics(df)

        missing = eda.analyze_missing_data()
        assert missing.iloc[0]['missing_percentage'] == 100.0

    def test_all_same_values(self):
        """Test with all same values."""
        df = pd.DataFrame({'x': [5, 5, 5, 5, 5]})
        eda = EDAStatistics(df)

        summary = eda.get_summary_statistics()
        assert summary.loc['x', 'std'] == 0.0
