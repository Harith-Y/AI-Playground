"""
Tests for Class Distribution Analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from app.ml_engine.class_distribution_analysis import (
    ClassDistributionAnalysis,
    analyze_class_distribution,
    check_class_imbalance,
    get_resampling_strategy,
    visualize_class_distribution_data
)


class TestClassDistributionAnalysisInitialization:
    """Test ClassDistributionAnalysis initialization."""

    def test_initialization_basic(self):
        """Test basic initialization."""
        df = pd.DataFrame({'target': [0, 1, 0, 1, 0]})
        analyzer = ClassDistributionAnalysis(df, 'target')

        assert analyzer.df is not None
        assert analyzer.target_column == 'target'
        assert analyzer.get_num_classes() == 2

    def test_error_not_dataframe(self):
        """Test error when input is not DataFrame."""
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            ClassDistributionAnalysis([1, 2, 3], 'target')

    def test_error_target_not_found(self):
        """Test error when target column not found."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        with pytest.raises(ValueError, match="not found in DataFrame"):
            ClassDistributionAnalysis(df, 'target')

    def test_error_target_all_missing(self):
        """Test error when target column is all missing."""
        df = pd.DataFrame({'target': [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError, match="contains only missing values"):
            ClassDistributionAnalysis(df, 'target')

    def test_initialization_with_missing(self):
        """Test initialization with some missing values."""
        df = pd.DataFrame({'target': [0, 1, np.nan, 0, 1]})
        analyzer = ClassDistributionAnalysis(df, 'target')

        assert analyzer.get_num_classes() == 2

    def test_multiclass_initialization(self):
        """Test initialization with multiple classes."""
        df = pd.DataFrame({'target': [0, 1, 2, 3, 0, 1, 2, 3]})
        analyzer = ClassDistributionAnalysis(df, 'target')

        assert analyzer.get_num_classes() == 4


class TestGetClassDistribution:
    """Test get_class_distribution method."""

    def test_get_class_distribution_binary(self):
        """Test class distribution for binary classification."""
        df = pd.DataFrame({'target': [0] * 30 + [1] * 70})
        analyzer = ClassDistributionAnalysis(df, 'target')
        distribution = analyzer.get_class_distribution()

        assert isinstance(distribution, pd.DataFrame)
        assert len(distribution) == 2
        assert 'class' in distribution.columns
        assert 'count' in distribution.columns
        assert 'proportion' in distribution.columns
        assert 'percentage' in distribution.columns

    def test_distribution_sorted_by_count(self):
        """Test that distribution is sorted by count descending."""
        df = pd.DataFrame({'target': [0] * 20 + [1] * 50 + [2] * 10})
        analyzer = ClassDistributionAnalysis(df, 'target')
        distribution = analyzer.get_class_distribution()

        # Should be sorted 1, 0, 2
        assert distribution.iloc[0]['class'] == 1
        assert distribution.iloc[1]['class'] == 0
        assert distribution.iloc[2]['class'] == 2

    def test_distribution_proportions(self):
        """Test that proportions sum to 1."""
        df = pd.DataFrame({'target': [0] * 30 + [1] * 70})
        analyzer = ClassDistributionAnalysis(df, 'target')
        distribution = analyzer.get_class_distribution()

        total_proportion = distribution['proportion'].sum()
        assert abs(total_proportion - 1.0) < 0.001

    def test_distribution_counts(self):
        """Test that counts are correct."""
        df = pd.DataFrame({'target': [0] * 25 + [1] * 75})
        analyzer = ClassDistributionAnalysis(df, 'target')
        distribution = analyzer.get_class_distribution()

        class_0 = distribution[distribution['class'] == 0].iloc[0]
        class_1 = distribution[distribution['class'] == 1].iloc[0]

        assert class_0['count'] == 25
        assert class_1['count'] == 75

    def test_distribution_percentage_format(self):
        """Test that percentage is formatted correctly."""
        df = pd.DataFrame({'target': [0] * 25 + [1] * 75})
        analyzer = ClassDistributionAnalysis(df, 'target')
        distribution = analyzer.get_class_distribution()

        # Check percentage string format
        assert all('%' in pct for pct in distribution['percentage'])


class TestImbalanceRatio:
    """Test imbalance ratio calculation."""

    def test_balanced_dataset(self):
        """Test imbalance ratio for balanced dataset."""
        df = pd.DataFrame({'target': [0] * 50 + [1] * 50})
        analyzer = ClassDistributionAnalysis(df, 'target')
        ratio = analyzer.get_imbalance_ratio()

        assert ratio == 1.0

    def test_moderate_imbalance(self):
        """Test imbalance ratio for moderately imbalanced dataset."""
        df = pd.DataFrame({'target': [0] * 70 + [1] * 30})
        analyzer = ClassDistributionAnalysis(df, 'target')
        ratio = analyzer.get_imbalance_ratio()

        assert abs(ratio - 2.33) < 0.01

    def test_severe_imbalance(self):
        """Test imbalance ratio for severely imbalanced dataset."""
        df = pd.DataFrame({'target': [0] * 90 + [1] * 10})
        analyzer = ClassDistributionAnalysis(df, 'target')
        ratio = analyzer.get_imbalance_ratio()

        assert ratio == 9.0

    def test_extreme_imbalance(self):
        """Test imbalance ratio for extremely imbalanced dataset."""
        df = pd.DataFrame({'target': [0] * 999 + [1] * 1})
        analyzer = ClassDistributionAnalysis(df, 'target')
        ratio = analyzer.get_imbalance_ratio()

        assert ratio == 999.0

    def test_single_class(self):
        """Test imbalance ratio with single class."""
        df = pd.DataFrame({'target': [0] * 100})
        analyzer = ClassDistributionAnalysis(df, 'target')
        ratio = analyzer.get_imbalance_ratio()

        assert ratio == 1.0


class TestMinorityMajorityClasses:
    """Test minority and majority class identification."""

    def test_identify_classes_binary(self):
        """Test minority/majority identification for binary."""
        df = pd.DataFrame({'target': [0] * 80 + [1] * 20})
        analyzer = ClassDistributionAnalysis(df, 'target')
        result = analyzer.identify_minority_majority_classes()

        assert result['minority_class'] == 1
        assert result['minority_count'] == 20
        assert result['majority_class'] == 0
        assert result['majority_count'] == 80

    def test_class_proportions(self):
        """Test that proportions are calculated correctly."""
        df = pd.DataFrame({'target': [0] * 75 + [1] * 25})
        analyzer = ClassDistributionAnalysis(df, 'target')
        result = analyzer.identify_minority_majority_classes()

        assert result['minority_proportion'] == 0.25
        assert result['majority_proportion'] == 0.75

    def test_percentage_format(self):
        """Test percentage format in result."""
        df = pd.DataFrame({'target': [0] * 60 + [1] * 40})
        analyzer = ClassDistributionAnalysis(df, 'target')
        result = analyzer.identify_minority_majority_classes()

        assert 'minority_percentage' in result
        assert 'majority_percentage' in result
        assert '%' in result['minority_percentage']
        assert '%' in result['majority_percentage']

    def test_multiclass_identification(self):
        """Test identification with multiple classes."""
        df = pd.DataFrame({'target': [0] * 10 + [1] * 50 + [2] * 5})
        analyzer = ClassDistributionAnalysis(df, 'target')
        result = analyzer.identify_minority_majority_classes()

        assert result['minority_class'] == 2
        assert result['minority_count'] == 5
        assert result['majority_class'] == 1
        assert result['majority_count'] == 50


class TestImbalanceMetrics:
    """Test imbalance metrics calculation."""

    def test_get_imbalance_metrics_basic(self):
        """Test basic imbalance metrics."""
        df = pd.DataFrame({'target': [0] * 90 + [1] * 10})
        analyzer = ClassDistributionAnalysis(df, 'target')
        metrics = analyzer.get_imbalance_metrics()

        assert 'num_classes' in metrics
        assert 'total_samples' in metrics
        assert 'imbalance_ratio' in metrics
        assert 'severity' in metrics
        assert 'is_balanced' in metrics
        assert 'gini_coefficient' in metrics
        assert 'entropy' in metrics
        assert 'requires_resampling' in metrics

    def test_metrics_balanced_dataset(self):
        """Test metrics for balanced dataset."""
        df = pd.DataFrame({'target': [0] * 50 + [1] * 50})
        analyzer = ClassDistributionAnalysis(df, 'target')
        metrics = analyzer.get_imbalance_metrics()

        assert metrics['severity'] == 'Balanced'
        assert metrics['is_balanced'] == True
        assert metrics['requires_resampling'] == False

    def test_metrics_severe_imbalance(self):
        """Test metrics for severe imbalance."""
        df = pd.DataFrame({'target': [0] * 90 + [1] * 10})
        analyzer = ClassDistributionAnalysis(df, 'target')
        metrics = analyzer.get_imbalance_metrics()

        assert metrics['severity'] == 'Severe'
        assert metrics['is_balanced'] == False
        assert metrics['requires_resampling'] == True

    def test_gini_coefficient(self):
        """Test Gini coefficient calculation."""
        # Perfectly balanced should have low Gini
        df_balanced = pd.DataFrame({'target': [0] * 50 + [1] * 50})
        analyzer_balanced = ClassDistributionAnalysis(df_balanced, 'target')
        metrics_balanced = analyzer_balanced.get_imbalance_metrics()

        # Imbalanced should have higher Gini
        df_imbalanced = pd.DataFrame({'target': [0] * 95 + [1] * 5})
        analyzer_imbalanced = ClassDistributionAnalysis(df_imbalanced, 'target')
        metrics_imbalanced = analyzer_imbalanced.get_imbalance_metrics()

        assert metrics_imbalanced['gini_coefficient'] > metrics_balanced['gini_coefficient']

    def test_entropy_calculation(self):
        """Test entropy calculation."""
        # Perfectly balanced should have highest entropy
        df_balanced = pd.DataFrame({'target': [0] * 50 + [1] * 50})
        analyzer_balanced = ClassDistributionAnalysis(df_balanced, 'target')
        metrics_balanced = analyzer_balanced.get_imbalance_metrics()

        # Imbalanced should have lower entropy
        df_imbalanced = pd.DataFrame({'target': [0] * 95 + [1] * 5})
        analyzer_imbalanced = ClassDistributionAnalysis(df_imbalanced, 'target')
        metrics_imbalanced = analyzer_imbalanced.get_imbalance_metrics()

        assert metrics_balanced['entropy'] > metrics_imbalanced['entropy']


class TestImbalanceSeverity:
    """Test imbalance severity assessment."""

    def test_severity_balanced(self):
        """Test balanced severity (ratio < 1.5)."""
        df = pd.DataFrame({'target': [0] * 55 + [1] * 45})
        analyzer = ClassDistributionAnalysis(df, 'target')
        metrics = analyzer.get_imbalance_metrics()

        assert metrics['severity'] == 'Balanced'

    def test_severity_slight(self):
        """Test slight severity (1.5 <= ratio < 3)."""
        df = pd.DataFrame({'target': [0] * 70 + [1] * 30})
        analyzer = ClassDistributionAnalysis(df, 'target')
        metrics = analyzer.get_imbalance_metrics()

        assert metrics['severity'] == 'Slight'

    def test_severity_moderate(self):
        """Test moderate severity (3 <= ratio < 10)."""
        df = pd.DataFrame({'target': [0] * 80 + [1] * 20})
        analyzer = ClassDistributionAnalysis(df, 'target')
        metrics = analyzer.get_imbalance_metrics()

        assert metrics['severity'] == 'Moderate'

    def test_severity_severe(self):
        """Test severe severity (10 <= ratio < 100)."""
        df = pd.DataFrame({'target': [0] * 95 + [1] * 5})
        analyzer = ClassDistributionAnalysis(df, 'target')
        metrics = analyzer.get_imbalance_metrics()

        assert metrics['severity'] == 'Severe'

    def test_severity_extreme(self):
        """Test extreme severity (ratio >= 100)."""
        df = pd.DataFrame({'target': [0] * 999 + [1] * 1})
        analyzer = ClassDistributionAnalysis(df, 'target')
        metrics = analyzer.get_imbalance_metrics()

        assert metrics['severity'] == 'Extreme'


class TestBarChartData:
    """Test bar chart data generation."""

    def test_get_bar_chart_data_basic(self):
        """Test basic bar chart data generation."""
        df = pd.DataFrame({'target': [0] * 70 + [1] * 30})
        analyzer = ClassDistributionAnalysis(df, 'target')
        chart_data = analyzer.get_bar_chart_data()

        assert 'labels' in chart_data
        assert 'counts' in chart_data
        assert 'proportions' in chart_data
        assert 'percentages' in chart_data
        assert 'colors' in chart_data
        assert 'chart_type' in chart_data

    def test_chart_data_structure(self):
        """Test chart data structure."""
        df = pd.DataFrame({'target': [0] * 60 + [1] * 40})
        analyzer = ClassDistributionAnalysis(df, 'target')
        chart_data = analyzer.get_bar_chart_data()

        assert len(chart_data['labels']) == 2
        assert len(chart_data['counts']) == 2
        assert len(chart_data['proportions']) == 2
        assert len(chart_data['colors']) == 2

    def test_chart_colors(self):
        """Test that colors are generated."""
        df = pd.DataFrame({'target': [0, 1, 2, 3, 4]})
        analyzer = ClassDistributionAnalysis(df, 'target')
        chart_data = analyzer.get_bar_chart_data()

        assert all(color.startswith('#') for color in chart_data['colors'])
        assert len(chart_data['colors']) == 5

    def test_chart_type(self):
        """Test chart type is set correctly."""
        df = pd.DataFrame({'target': [0] * 50 + [1] * 50})
        analyzer = ClassDistributionAnalysis(df, 'target')
        chart_data = analyzer.get_bar_chart_data()

        assert chart_data['chart_type'] == 'bar'


class TestResamplingRecommendations:
    """Test resampling recommendations."""

    def test_recommendations_balanced(self):
        """Test recommendations for balanced dataset."""
        df = pd.DataFrame({'target': [0] * 50 + [1] * 50})
        analyzer = ClassDistributionAnalysis(df, 'target')
        recommendations = analyzer.get_resampling_recommendations()

        assert recommendations['requires_action'] == False
        assert 'message' in recommendations

    def test_recommendations_moderate(self):
        """Test recommendations for moderate imbalance."""
        df = pd.DataFrame({'target': [0] * 75 + [1] * 25})
        analyzer = ClassDistributionAnalysis(df, 'target')
        recommendations = analyzer.get_resampling_recommendations()

        assert recommendations['requires_action'] == True
        assert 'SMOTE' in recommendations['recommended_strategies']

    def test_recommendations_severe(self):
        """Test recommendations for severe imbalance."""
        df = pd.DataFrame({'target': [0] * 90 + [1] * 10})
        analyzer = ClassDistributionAnalysis(df, 'target')
        recommendations = analyzer.get_resampling_recommendations()

        assert recommendations['severity'] == 'Severe'
        assert 'SMOTE' in recommendations['recommended_strategies']
        assert 'BorderlineSMOTE' in recommendations['recommended_strategies']

    def test_recommendations_extreme(self):
        """Test recommendations for extreme imbalance."""
        df = pd.DataFrame({'target': [0] * 995 + [1] * 5})
        analyzer = ClassDistributionAnalysis(df, 'target')
        recommendations = analyzer.get_resampling_recommendations()

        assert recommendations['severity'] == 'Extreme'
        assert len(recommendations['recommended_strategies']) > 0

    def test_oversampling_targets(self):
        """Test oversampling targets calculation."""
        df = pd.DataFrame({'target': [0] * 80 + [1] * 20})
        analyzer = ClassDistributionAnalysis(df, 'target')
        recommendations = analyzer.get_resampling_recommendations()

        assert 'oversampling_targets' in recommendations
        # Both classes should target majority class count
        assert recommendations['oversampling_targets']['0'] == 80
        assert recommendations['oversampling_targets']['1'] == 80

    def test_undersampling_targets(self):
        """Test undersampling targets calculation."""
        df = pd.DataFrame({'target': [0] * 80 + [1] * 20})
        analyzer = ClassDistributionAnalysis(df, 'target')
        recommendations = analyzer.get_resampling_recommendations()

        assert 'undersampling_targets' in recommendations
        # Both classes should target minority class count
        assert recommendations['undersampling_targets']['0'] == 20
        assert recommendations['undersampling_targets']['1'] == 20

    def test_strategy_details(self):
        """Test that strategy details are provided."""
        df = pd.DataFrame({'target': [0] * 85 + [1] * 15})
        analyzer = ClassDistributionAnalysis(df, 'target')
        recommendations = analyzer.get_resampling_recommendations()

        assert 'strategy_details' in recommendations
        assert isinstance(recommendations['strategy_details'], dict)


class TestMissingValueInfo:
    """Test missing value information."""

    def test_no_missing_values(self):
        """Test with no missing values."""
        df = pd.DataFrame({'target': [0] * 50 + [1] * 50})
        analyzer = ClassDistributionAnalysis(df, 'target')
        missing_info = analyzer.get_missing_value_info()

        assert missing_info['missing_samples'] == 0
        assert missing_info['missing_percentage'] == 0
        assert missing_info['has_missing'] == False

    def test_with_missing_values(self):
        """Test with missing values."""
        df = pd.DataFrame({'target': [0] * 40 + [1] * 40 + [np.nan] * 20})
        analyzer = ClassDistributionAnalysis(df, 'target')
        missing_info = analyzer.get_missing_value_info()

        assert missing_info['missing_samples'] == 20
        assert missing_info['missing_percentage'] == 20.0
        assert missing_info['has_missing'] == True
        assert missing_info['valid_samples'] == 80

    def test_missing_info_totals(self):
        """Test total samples count."""
        df = pd.DataFrame({'target': [0] * 30 + [1] * 30 + [np.nan] * 10})
        analyzer = ClassDistributionAnalysis(df, 'target')
        missing_info = analyzer.get_missing_value_info()

        assert missing_info['total_samples'] == 70


class TestClassOverlapAnalysis:
    """Test class overlap analysis."""

    def test_overlap_analysis_basic(self):
        """Test basic overlap analysis."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80],
            'target': [0, 0, 0, 0, 1, 1, 1, 1]
        })
        analyzer = ClassDistributionAnalysis(df, 'target')
        overlap = analyzer.analyze_class_overlap()

        assert overlap['analyzable'] == True
        assert 'class_statistics' in overlap

    def test_overlap_no_features(self):
        """Test overlap analysis with no numeric features."""
        df = pd.DataFrame({
            'category': ['a', 'b', 'c', 'd'],
            'target': [0, 0, 1, 1]
        })
        analyzer = ClassDistributionAnalysis(df, 'target')
        overlap = analyzer.analyze_class_overlap()

        assert overlap['analyzable'] == False

    def test_overlap_specific_features(self):
        """Test overlap analysis with specific features."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8],
            'feature3': [9, 10, 11, 12],
            'target': [0, 0, 1, 1]
        })
        analyzer = ClassDistributionAnalysis(df, 'target')
        overlap = analyzer.analyze_class_overlap(feature_columns=['feature1', 'feature2'])

        assert 'feature1' in overlap['feature_columns']
        assert 'feature2' in overlap['feature_columns']
        assert 'feature3' not in overlap['feature_columns']

    def test_overlap_class_statistics(self):
        """Test that class statistics are calculated."""
        df = pd.DataFrame({
            'feature': [1, 2, 10, 11],
            'target': [0, 0, 1, 1]
        })
        analyzer = ClassDistributionAnalysis(df, 'target')
        overlap = analyzer.analyze_class_overlap()

        stats = overlap['class_statistics']
        assert '0' in stats
        assert '1' in stats
        assert 'mean' in stats['0']
        assert 'std' in stats['0']


class TestSummaryReport:
    """Test summary report generation."""

    def test_get_summary_report_basic(self):
        """Test basic summary report."""
        df = pd.DataFrame({'target': [0] * 70 + [1] * 30})
        analyzer = ClassDistributionAnalysis(df, 'target')
        report = analyzer.get_summary_report()

        assert 'target_column' in report
        assert 'class_distribution' in report
        assert 'imbalance_metrics' in report
        assert 'minority_majority_info' in report
        assert 'recommendations' in report
        assert 'missing_value_info' in report

    def test_summary_report_completeness(self):
        """Test that summary report is comprehensive."""
        df = pd.DataFrame({'target': [0] * 80 + [1] * 20})
        analyzer = ClassDistributionAnalysis(df, 'target')
        report = analyzer.get_summary_report()

        assert report['target_column'] == 'target'
        assert isinstance(report['class_distribution'], list)
        assert isinstance(report['imbalance_metrics'], dict)
        assert isinstance(report['minority_majority_info'], dict)
        assert isinstance(report['recommendations'], dict)
        assert isinstance(report['missing_value_info'], dict)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_analyze_class_distribution_function(self):
        """Test analyze_class_distribution convenience function."""
        df = pd.DataFrame({'target': [0] * 60 + [1] * 40})
        distribution = analyze_class_distribution(df, 'target')

        assert isinstance(distribution, pd.DataFrame)
        assert len(distribution) == 2

    def test_check_class_imbalance_function(self):
        """Test check_class_imbalance convenience function."""
        df = pd.DataFrame({'target': [0] * 90 + [1] * 10})
        result = check_class_imbalance(df, 'target', threshold=3.0)

        assert 'is_imbalanced' in result
        assert 'imbalance_ratio' in result
        assert 'severity' in result
        assert 'message' in result
        assert result['is_imbalanced'] == True

    def test_check_class_imbalance_threshold(self):
        """Test threshold parameter in check_class_imbalance."""
        df = pd.DataFrame({'target': [0] * 70 + [1] * 30})

        # With low threshold
        result_low = check_class_imbalance(df, 'target', threshold=2.0)
        assert result_low['is_imbalanced'] == True

        # With high threshold
        result_high = check_class_imbalance(df, 'target', threshold=10.0)
        assert result_high['is_imbalanced'] == False

    def test_get_resampling_strategy_function(self):
        """Test get_resampling_strategy convenience function."""
        df = pd.DataFrame({'target': [0] * 85 + [1] * 15})
        strategy = get_resampling_strategy(df, 'target')

        assert 'recommended_strategies' in strategy
        assert isinstance(strategy['recommended_strategies'], list)

    def test_visualize_class_distribution_data_function(self):
        """Test visualize_class_distribution_data convenience function."""
        df = pd.DataFrame({'target': [0] * 65 + [1] * 35})
        viz_data = visualize_class_distribution_data(df, 'target')

        assert 'labels' in viz_data
        assert 'counts' in viz_data
        assert 'colors' in viz_data
        assert viz_data['chart_type'] == 'bar'


class TestEdgeCases:
    """Test edge cases."""

    def test_single_sample_per_class(self):
        """Test with single sample per class."""
        df = pd.DataFrame({'target': [0, 1]})
        analyzer = ClassDistributionAnalysis(df, 'target')

        assert analyzer.get_num_classes() == 2
        assert analyzer.get_imbalance_ratio() == 1.0

    def test_many_classes(self):
        """Test with many classes."""
        df = pd.DataFrame({'target': list(range(20)) * 5})
        analyzer = ClassDistributionAnalysis(df, 'target')

        assert analyzer.get_num_classes() == 20

    def test_string_classes(self):
        """Test with string class labels."""
        df = pd.DataFrame({'target': ['cat'] * 60 + ['dog'] * 40})
        analyzer = ClassDistributionAnalysis(df, 'target')

        assert analyzer.get_num_classes() == 2
        distribution = analyzer.get_class_distribution()
        assert 'cat' in distribution['class'].values
        assert 'dog' in distribution['class'].values

    def test_mixed_type_classes(self):
        """Test with mixed type class labels."""
        df = pd.DataFrame({'target': [0, 1, 'two', 'three']})
        analyzer = ClassDistributionAnalysis(df, 'target')

        assert analyzer.get_num_classes() == 4

    def test_large_dataset(self):
        """Test with large dataset."""
        df = pd.DataFrame({'target': [0] * 10000 + [1] * 5000})
        analyzer = ClassDistributionAnalysis(df, 'target')

        metrics = analyzer.get_imbalance_metrics()
        assert metrics['total_samples'] == 15000

    def test_extreme_class_names(self):
        """Test with unusual class names."""
        df = pd.DataFrame({'target': ['class A'] * 50 + ['class-B'] * 30 + ['Class_C'] * 20})
        analyzer = ClassDistributionAnalysis(df, 'target')

        assert analyzer.get_num_classes() == 3
        distribution = analyzer.get_class_distribution()
        assert len(distribution) == 3


class TestRobustness:
    """Test robustness and error handling."""

    def test_empty_dataframe_error(self):
        """Test that empty DataFrame after dropping NaN raises error."""
        df = pd.DataFrame({'target': [np.nan] * 5})
        with pytest.raises(ValueError, match="contains only missing values"):
            ClassDistributionAnalysis(df, 'target')

    def test_numeric_target(self):
        """Test with numeric target values."""
        df = pd.DataFrame({'target': [0, 1, 2, 3, 0, 1, 2, 3]})
        analyzer = ClassDistributionAnalysis(df, 'target')

        assert analyzer.get_num_classes() == 4

    def test_categorical_target(self):
        """Test with categorical target values."""
        df = pd.DataFrame({'target': pd.Categorical(['A', 'B', 'A', 'B'])})
        analyzer = ClassDistributionAnalysis(df, 'target')

        assert analyzer.get_num_classes() == 2

    def test_float_target(self):
        """Test with float target values."""
        df = pd.DataFrame({'target': [0.0, 1.0, 0.0, 1.0]})
        analyzer = ClassDistributionAnalysis(df, 'target')

        assert analyzer.get_num_classes() == 2

    def test_consistency_across_methods(self):
        """Test that all methods return consistent information."""
        df = pd.DataFrame({'target': [0] * 75 + [1] * 25})
        analyzer = ClassDistributionAnalysis(df, 'target')

        distribution = analyzer.get_class_distribution()
        metrics = analyzer.get_imbalance_metrics()
        minority_majority = analyzer.identify_minority_majority_classes()

        # Check consistency
        assert metrics['num_classes'] == len(distribution)
        assert metrics['total_samples'] == distribution['count'].sum()
        assert minority_majority['minority_count'] + minority_majority['majority_count'] == 100
