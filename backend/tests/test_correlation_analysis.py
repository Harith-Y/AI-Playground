"""
Tests for correlation analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from app.ml_engine.correlation_analysis import (
    CorrelationMatrix,
    correlation_matrix,
    correlation_heatmap_data,
    find_highly_correlated_pairs,
    detect_multicollinearity,
    rank_features_by_target,
)


@pytest.fixture
def sample_df():
    """Sample DataFrame with numeric columns."""
    np.random.seed(42)
    return pd.DataFrame({
        'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'b': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],  # Perfectly correlated with a
        'c': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],  # Negatively correlated with a
        'd': np.random.randn(10),  # Random
    })


@pytest.fixture
def multicollinear_df():
    """DataFrame with multicollinearity."""
    np.random.seed(42)
    x1 = np.random.randn(100)
    x2 = x1 + np.random.randn(100) * 0.1  # Highly correlated with x1
    x3 = x1 + np.random.randn(100) * 0.1  # Highly correlated with x1
    x4 = np.random.randn(100)  # Independent

    return pd.DataFrame({
        'feature1': x1,
        'feature2': x2,
        'feature3': x3,
        'feature4': x4,
    })


class TestCorrelationMatrixInitialization:
    """Test CorrelationMatrix initialization."""

    def test_init_with_valid_dataframe(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        assert corr.df.equals(sample_df)
        assert len(corr._numeric_cols) == 4

    def test_init_with_non_dataframe(self):
        with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
            CorrelationMatrix([1, 2, 3])

    def test_init_with_no_numeric_columns(self):
        df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
        with pytest.raises(ValueError, match="at least one numeric column"):
            CorrelationMatrix(df)

    def test_init_with_mixed_types(self):
        df = pd.DataFrame({
            'numeric': [1, 2, 3],
            'text': ['a', 'b', 'c'],
            'numeric2': [4.0, 5.0, 6.0]
        })
        corr = CorrelationMatrix(df)
        assert len(corr._numeric_cols) == 2
        assert 'numeric' in corr._numeric_cols
        assert 'numeric2' in corr._numeric_cols


class TestComputeCorrelation:
    """Test correlation computation."""

    def test_pearson_correlation(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        matrix = corr.compute_correlation(method='pearson')

        assert matrix.shape == (4, 4)
        assert np.allclose(matrix.loc['a', 'b'], 1.0)  # Perfect positive
        assert np.allclose(matrix.loc['a', 'c'], -1.0)  # Perfect negative
        assert corr.method_ == 'pearson'

    def test_spearman_correlation(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        matrix = corr.compute_correlation(method='spearman')

        assert matrix.shape == (4, 4)
        assert corr.method_ == 'spearman'

    def test_kendall_correlation(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        matrix = corr.compute_correlation(method='kendall')

        assert matrix.shape == (4, 4)
        assert corr.method_ == 'kendall'

    def test_specific_columns(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        matrix = corr.compute_correlation(columns=['a', 'b'])

        assert matrix.shape == (2, 2)
        assert list(matrix.columns) == ['a', 'b']

    def test_diagonal_is_one(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        matrix = corr.compute_correlation()

        for col in matrix.columns:
            assert np.allclose(matrix.loc[col, col], 1.0)

    def test_correlation_matrix_cached(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        corr.compute_correlation()
        assert corr.correlation_matrix_ is not None


class TestGetCorrelationPairs:
    """Test getting correlation pairs."""

    def test_get_pairs_above_threshold(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        corr.compute_correlation()
        pairs = corr.get_correlation_pairs(threshold=0.9)

        assert len(pairs) >= 2  # a-b and a-c
        assert any(p['feature_1'] == 'a' and p['feature_2'] == 'b' for p in pairs)

    def test_pairs_without_diagonal(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        corr.compute_correlation()
        pairs = corr.get_correlation_pairs(threshold=0.0, exclude_diagonal=True)

        # No self-correlations
        assert not any(p['feature_1'] == p['feature_2'] for p in pairs)

    def test_pairs_with_diagonal(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        corr.compute_correlation()
        pairs = corr.get_correlation_pairs(threshold=0.99, exclude_diagonal=False)

        # Should include self-correlations (1.0)
        assert any(p['feature_1'] == p['feature_2'] for p in pairs)

    def test_pairs_sorted_by_abs_correlation(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        corr.compute_correlation()
        pairs = corr.get_correlation_pairs(threshold=0.5)

        # Verify sorted descending
        for i in range(len(pairs) - 1):
            assert pairs[i]['abs_correlation'] >= pairs[i + 1]['abs_correlation']

    def test_pairs_include_strength(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        corr.compute_correlation()
        pairs = corr.get_correlation_pairs(threshold=0.5)

        assert all('strength' in p for p in pairs)
        assert all(p['strength'] in ['Very Weak', 'Weak', 'Moderate', 'Strong', 'Very Strong'] for p in pairs)

    def test_requires_computed_correlation(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        with pytest.raises(ValueError, match="Must call compute_correlation"):
            corr.get_correlation_pairs()


class TestGetCorrelationClusters:
    """Test correlation clustering."""

    def test_cluster_high_correlations(self, multicollinear_df):
        corr = CorrelationMatrix(multicollinear_df)
        corr.compute_correlation()
        clusters = corr.get_correlation_clusters(threshold=0.8)

        # Should group feature1, feature2, feature3 together
        assert len(clusters) >= 1

        # Find cluster with feature1
        cluster_with_f1 = None
        for cluster_features in clusters.values():
            if 'feature1' in cluster_features:
                cluster_with_f1 = cluster_features
                break

        assert cluster_with_f1 is not None
        assert 'feature2' in cluster_with_f1
        assert 'feature3' in cluster_with_f1

    def test_different_linkage_methods(self, multicollinear_df):
        corr = CorrelationMatrix(multicollinear_df)
        corr.compute_correlation()

        for method in ['average', 'complete', 'single']:
            clusters = corr.get_correlation_clusters(threshold=0.8, linkage_method=method)
            assert isinstance(clusters, dict)
            assert all(isinstance(v, list) for v in clusters.values())

    def test_requires_computed_correlation(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        with pytest.raises(ValueError, match="Must call compute_correlation"):
            corr.get_correlation_clusters()


class TestGetHeatmapData:
    """Test heatmap data generation."""

    def test_heatmap_data_structure(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        corr.compute_correlation()
        heatmap = corr.get_heatmap_data()

        assert 'z' in heatmap
        assert 'x' in heatmap
        assert 'y' in heatmap
        assert 'annotations' in heatmap
        assert 'colorscale' in heatmap
        assert 'type' in heatmap

    def test_heatmap_dimensions(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        corr.compute_correlation()
        heatmap = corr.get_heatmap_data()

        assert len(heatmap['z']) == 4
        assert len(heatmap['z'][0]) == 4
        assert len(heatmap['x']) == 4
        assert len(heatmap['y']) == 4

    def test_heatmap_with_annotations(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        corr.compute_correlation()
        heatmap = corr.get_heatmap_data(annotations=True, round_decimals=2)

        assert heatmap['annotations'] is not None
        assert len(heatmap['annotations']) == 4

    def test_heatmap_without_annotations(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        corr.compute_correlation()
        heatmap = corr.get_heatmap_data(annotations=False)

        assert heatmap['annotations'] is None

    def test_heatmap_range(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        corr.compute_correlation()
        heatmap = corr.get_heatmap_data()

        assert heatmap['zmin'] == -1
        assert heatmap['zmax'] == 1

    def test_requires_computed_correlation(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        with pytest.raises(ValueError, match="Must call compute_correlation"):
            corr.get_heatmap_data()


class TestGetFeatureRanking:
    """Test feature ranking by target correlation."""

    def test_rank_by_target(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        ranking = corr.get_feature_ranking(target_column='a')

        assert len(ranking) == 3  # Excludes target itself
        assert 'feature' in ranking.columns
        assert 'correlation' in ranking.columns
        assert 'abs_correlation' in ranking.columns

    def test_ranking_sorted(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        ranking = corr.get_feature_ranking(target_column='a')

        # Verify sorted by abs_correlation descending
        for i in range(len(ranking) - 1):
            assert ranking.iloc[i]['abs_correlation'] >= ranking.iloc[i + 1]['abs_correlation']

    def test_ranking_top_k(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        ranking = corr.get_feature_ranking(target_column='a', top_k=2)

        assert len(ranking) == 2

    def test_invalid_target_column(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        with pytest.raises(ValueError, match="not found"):
            corr.get_feature_ranking(target_column='nonexistent')


class TestGetMulticollinearityStats:
    """Test multicollinearity detection."""

    def test_detect_multicollinearity(self, multicollinear_df):
        corr = CorrelationMatrix(multicollinear_df)
        corr.compute_correlation()
        stats = corr.get_multicollinearity_stats(threshold=0.8)

        assert 'high_correlation_count' in stats
        assert 'high_correlation_pairs' in stats
        assert 'problematic_features' in stats
        assert 'vif_approximation' in stats
        assert 'severity' in stats

        assert stats['high_correlation_count'] > 0
        assert len(stats['problematic_features']) >= 2

    def test_no_multicollinearity(self, sample_df):
        # Create DataFrame with low correlations
        df = pd.DataFrame({
            'x': np.random.randn(50),
            'y': np.random.randn(50),
            'z': np.random.randn(50),
        })
        corr = CorrelationMatrix(df)
        corr.compute_correlation()
        stats = corr.get_multicollinearity_stats(threshold=0.95)

        assert stats['severity'] in ['None', 'Low']

    def test_vif_calculation(self, multicollinear_df):
        corr = CorrelationMatrix(multicollinear_df)
        corr.compute_correlation()
        stats = corr.get_multicollinearity_stats()

        assert 'vif_approximation' in stats
        assert all(isinstance(v, (int, float)) for v in stats['vif_approximation'].values())

    def test_requires_computed_correlation(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        with pytest.raises(ValueError, match="Must call compute_correlation"):
            corr.get_multicollinearity_stats()


class TestGetCorrelationNetwork:
    """Test correlation network generation."""

    def test_network_structure(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        corr.compute_correlation()
        network = corr.get_correlation_network(threshold=0.5)

        assert 'nodes' in network
        assert 'edges' in network
        assert 'node_count' in network
        assert 'edge_count' in network

    def test_network_nodes(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        corr.compute_correlation()
        network = corr.get_correlation_network(threshold=0.9)

        # Each node should have id and label
        assert all('id' in node for node in network['nodes'])
        assert all('label' in node for node in network['nodes'])

    def test_network_edges(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        corr.compute_correlation()
        network = corr.get_correlation_network(threshold=0.9)

        # Each edge should have source, target, weight
        assert all('source' in edge for edge in network['edges'])
        assert all('target' in edge for edge in network['edges'])
        assert all('weight' in edge for edge in network['edges'])

    def test_network_counts_match(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        corr.compute_correlation()
        network = corr.get_correlation_network(threshold=0.5)

        assert network['node_count'] == len(network['nodes'])
        assert network['edge_count'] == len(network['edges'])


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_correlation_matrix_function(self, sample_df):
        matrix = correlation_matrix(sample_df, method='pearson')
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.shape == (4, 4)

    def test_correlation_heatmap_data_function(self, sample_df):
        heatmap = correlation_heatmap_data(sample_df, method='spearman')
        assert 'z' in heatmap
        assert 'x' in heatmap
        assert 'y' in heatmap

    def test_find_highly_correlated_pairs_function(self, sample_df):
        pairs = find_highly_correlated_pairs(sample_df, threshold=0.8)
        assert isinstance(pairs, list)
        assert all('feature_1' in p for p in pairs)

    def test_detect_multicollinearity_function(self, multicollinear_df):
        stats = detect_multicollinearity(multicollinear_df, threshold=0.8)
        assert 'high_correlation_count' in stats
        assert 'severity' in stats

    def test_rank_features_by_target_function(self, sample_df):
        ranking = rank_features_by_target(sample_df, 'a', top_k=2)
        assert isinstance(ranking, pd.DataFrame)
        assert len(ranking) == 2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_column_dataframe(self):
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        corr = CorrelationMatrix(df)
        matrix = corr.compute_correlation()
        assert matrix.shape == (1, 1)
        assert matrix.iloc[0, 0] == 1.0

    def test_dataframe_with_nan(self):
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4, 5],
            'b': [2, 4, 6, 8, 10],
        })
        corr = CorrelationMatrix(df)
        matrix = corr.compute_correlation()
        assert not pd.isna(matrix.iloc[0, 1])

    def test_constant_column(self):
        df = pd.DataFrame({
            'constant': [5, 5, 5, 5, 5],
            'varying': [1, 2, 3, 4, 5],
        })
        corr = CorrelationMatrix(df)
        matrix = corr.compute_correlation()
        # Correlation with constant is NaN
        assert pd.isna(matrix.loc['constant', 'varying'])

    def test_two_column_dataframe(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [2, 4, 6]})
        corr = CorrelationMatrix(df)
        matrix = corr.compute_correlation()
        assert matrix.shape == (2, 2)

    def test_perfect_correlation(self, sample_df):
        corr = CorrelationMatrix(sample_df)
        corr.compute_correlation()
        pairs = corr.get_correlation_pairs(threshold=0.99)

        # a and b are perfectly correlated
        perfect_pair = next((p for p in pairs if
                           (p['feature_1'] == 'a' and p['feature_2'] == 'b') or
                           (p['feature_1'] == 'b' and p['feature_2'] == 'a')), None)
        assert perfect_pair is not None
        assert np.isclose(perfect_pair['abs_correlation'], 1.0)


class TestInterpretationMethods:
    """Test interpretation helper methods."""

    def test_interpret_correlation(self, sample_df):
        corr = CorrelationMatrix(sample_df)

        assert corr._interpret_correlation(0.95) == 'Very Strong'
        assert corr._interpret_correlation(0.75) == 'Strong'
        assert corr._interpret_correlation(0.55) == 'Moderate'
        assert corr._interpret_correlation(0.35) == 'Weak'
        assert corr._interpret_correlation(0.15) == 'Very Weak'

    def test_assess_multicollinearity_severity(self, sample_df):
        corr = CorrelationMatrix(sample_df)

        assert corr._assess_multicollinearity_severity(0) == 'None'
        assert corr._assess_multicollinearity_severity(1) == 'Low'
        assert corr._assess_multicollinearity_severity(3) == 'Moderate'
        assert corr._assess_multicollinearity_severity(10) == 'High'
