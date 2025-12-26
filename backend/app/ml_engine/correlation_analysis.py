"""
Correlation Analysis Module.

Provides comprehensive correlation analysis functions for exploring relationships
between features in a dataset. Includes correlation matrices, heatmap data,
cluster analysis, and feature grouping.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Literal, Union
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


class CorrelationMatrix:
    """
    Comprehensive correlation matrix analysis.

    Provides various correlation methods, cluster analysis, and utilities
    for understanding feature relationships.

    Example
    -------
    >>> corr = CorrelationMatrix(df)
    >>> matrix = corr.compute_correlation(method='pearson')
    >>> clusters = corr.get_correlation_clusters(threshold=0.8)
    >>> heatmap_data = corr.get_heatmap_data()
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize correlation matrix analyzer.

        Args:
            df: DataFrame to analyze

        Raises:
            TypeError: If df is not a pandas DataFrame
            ValueError: If df has no numeric columns
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        self.df = df.copy()
        self._numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(self._numeric_cols) == 0:
            raise ValueError("DataFrame must contain at least one numeric column")

        self.correlation_matrix_: Optional[pd.DataFrame] = None
        self.method_: Optional[str] = None

    def compute_correlation(
        self,
        method: Literal['pearson', 'spearman', 'kendall'] = 'pearson',
        columns: Optional[List[str]] = None,
        min_periods: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute correlation matrix.

        Args:
            method: Correlation method
                - 'pearson': Linear correlation (default)
                - 'spearman': Rank correlation (robust to outliers)
                - 'kendall': Tau correlation (ordinal associations)
            columns: Specific columns to analyze (None = all numeric)
            min_periods: Minimum number of observations required per pair

        Returns:
            Correlation matrix as DataFrame
        """
        if columns is None:
            columns = self._numeric_cols
        else:
            columns = [c for c in columns if c in self._numeric_cols]

        if len(columns) == 0:
            return pd.DataFrame()

        df_subset = self.df[columns]

        self.correlation_matrix_ = df_subset.corr(
            method=method,
            min_periods=min_periods
        )
        self.method_ = method

        return self.correlation_matrix_

    def get_correlation_pairs(
        self,
        threshold: float = 0.7,
        absolute: bool = True,
        exclude_diagonal: bool = True
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Get feature pairs with correlation above threshold.

        Args:
            threshold: Minimum correlation to include
            absolute: Use absolute correlation values
            exclude_diagonal: Exclude self-correlations (1.0)

        Returns:
            List of correlation pairs with metadata

        Raises:
            ValueError: If correlation matrix hasn't been computed
        """
        if self.correlation_matrix_ is None:
            raise ValueError("Must call compute_correlation() first")

        pairs = []
        n = len(self.correlation_matrix_)

        for i in range(n):
            start_j = i + 1 if exclude_diagonal else i
            for j in range(start_j, n):
                corr_val = self.correlation_matrix_.iloc[i, j]

                if pd.isna(corr_val):
                    continue

                check_val = abs(corr_val) if absolute else corr_val

                if check_val >= threshold:
                    pairs.append({
                        'feature_1': self.correlation_matrix_.columns[i],
                        'feature_2': self.correlation_matrix_.columns[j],
                        'correlation': round(corr_val, 4),
                        'abs_correlation': round(abs(corr_val), 4),
                        'strength': self._interpret_correlation(abs(corr_val))
                    })

        # Sort by absolute correlation descending
        pairs = sorted(pairs, key=lambda x: x['abs_correlation'], reverse=True)

        return pairs

    def get_correlation_clusters(
        self,
        threshold: float = 0.8,
        linkage_method: str = 'average'
    ) -> Dict[int, List[str]]:
        """
        Group features into clusters based on correlation.

        Uses hierarchical clustering to identify groups of highly
        correlated features.

        Args:
            threshold: Correlation threshold for clustering
            linkage_method: Linkage method ('average', 'complete', 'single', 'ward')

        Returns:
            Dictionary mapping cluster ID to list of feature names

        Raises:
            ValueError: If correlation matrix hasn't been computed
        """
        if self.correlation_matrix_ is None:
            raise ValueError("Must call compute_correlation() first")

        # Convert correlation to distance: distance = 1 - |correlation|
        distance_matrix = 1 - self.correlation_matrix_.abs()

        # Convert to condensed distance matrix for scipy
        condensed_dist = squareform(distance_matrix.values, checks=False)

        # Perform hierarchical clustering
        linkage_matrix = hierarchy.linkage(condensed_dist, method=linkage_method)

        # Cut tree at threshold
        distance_threshold = 1 - threshold
        cluster_labels = hierarchy.fcluster(
            linkage_matrix,
            distance_threshold,
            criterion='distance'
        )

        # Group features by cluster
        clusters = {}
        for idx, cluster_id in enumerate(cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(self.correlation_matrix_.columns[idx])

        return clusters

    def get_heatmap_data(
        self,
        annotations: bool = True,
        round_decimals: int = 2
    ) -> Dict[str, any]:
        """
        Get data formatted for heatmap visualization.

        Returns data structure compatible with Plotly/Seaborn heatmaps.

        Args:
            annotations: Include correlation values as annotations
            round_decimals: Decimal places for annotations

        Returns:
            Dictionary with heatmap data and metadata

        Raises:
            ValueError: If correlation matrix hasn't been computed
        """
        if self.correlation_matrix_ is None:
            raise ValueError("Must call compute_correlation() first")

        matrix = self.correlation_matrix_

        # Prepare annotation matrix
        if annotations:
            annot_matrix = matrix.round(round_decimals).values.tolist()
        else:
            annot_matrix = None

        return {
            'z': matrix.values.tolist(),  # 2D array of correlations
            'x': matrix.columns.tolist(),  # Column labels
            'y': matrix.index.tolist(),    # Row labels
            'annotations': annot_matrix,
            'colorscale': 'RdBu_r',  # Red-Blue reversed (red=negative, blue=positive)
            'zmin': -1,
            'zmax': 1,
            'type': 'heatmap'
        }

    def get_feature_ranking(
        self,
        target_column: str,
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Rank features by correlation with target variable.

        Args:
            target_column: Name of target variable
            top_k: Return only top k features (None = all)

        Returns:
            DataFrame with features ranked by correlation with target

        Raises:
            ValueError: If target_column not in DataFrame or correlation matrix
        """
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        if self.correlation_matrix_ is None:
            self.compute_correlation()

        if target_column not in self.correlation_matrix_.columns:
            raise ValueError(f"Target column '{target_column}' is not numeric")

        # Get correlations with target
        target_corr = self.correlation_matrix_[target_column].drop(target_column)

        # Create ranking DataFrame
        ranking = pd.DataFrame({
            'feature': target_corr.index,
            'correlation': target_corr.values,
            'abs_correlation': target_corr.abs().values,
            'strength': [self._interpret_correlation(abs(c)) for c in target_corr.values]
        })

        # Sort by absolute correlation
        ranking = ranking.sort_values('abs_correlation', ascending=False).reset_index(drop=True)

        if top_k is not None:
            ranking = ranking.head(top_k)

        return ranking

    def get_multicollinearity_stats(
        self,
        threshold: float = 0.9
    ) -> Dict[str, any]:
        """
        Identify multicollinearity issues.

        Args:
            threshold: Correlation threshold for multicollinearity warning

        Returns:
            Dictionary with multicollinearity statistics

        Raises:
            ValueError: If correlation matrix hasn't been computed
        """
        if self.correlation_matrix_ is None:
            raise ValueError("Must call compute_correlation() first")

        # Find highly correlated pairs
        high_corr_pairs = self.get_correlation_pairs(
            threshold=threshold,
            absolute=True,
            exclude_diagonal=True
        )

        # Identify features involved in multicollinearity
        problematic_features = set()
        for pair in high_corr_pairs:
            problematic_features.add(pair['feature_1'])
            problematic_features.add(pair['feature_2'])

        # Calculate VIF (Variance Inflation Factor) approximation
        # VIF ≈ 1 / (1 - R²) where R² is max correlation squared
        vif_approx = {}
        for feature in self.correlation_matrix_.columns:
            other_features = [f for f in self.correlation_matrix_.columns if f != feature]
            max_corr = self.correlation_matrix_.loc[feature, other_features].abs().max()

            if not pd.isna(max_corr):
                r_squared = max_corr ** 2
                vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
                vif_approx[feature] = round(vif, 2)

        return {
            'high_correlation_count': len(high_corr_pairs),
            'high_correlation_pairs': high_corr_pairs,
            'problematic_features': list(problematic_features),
            'vif_approximation': vif_approx,
            'severity': self._assess_multicollinearity_severity(len(high_corr_pairs))
        }

    def get_correlation_network(
        self,
        threshold: float = 0.5,
        absolute: bool = True
    ) -> Dict[str, any]:
        """
        Get network graph data for correlation visualization.

        Returns nodes (features) and edges (correlations) for network graphs.

        Args:
            threshold: Minimum correlation to include as edge
            absolute: Use absolute correlation values

        Returns:
            Dictionary with nodes and edges for network visualization
        """
        if self.correlation_matrix_ is None:
            raise ValueError("Must call compute_correlation() first")

        # Get correlation pairs above threshold
        pairs = self.get_correlation_pairs(
            threshold=threshold,
            absolute=absolute,
            exclude_diagonal=True
        )

        # Create nodes (unique features)
        features = set()
        for pair in pairs:
            features.add(pair['feature_1'])
            features.add(pair['feature_2'])

        nodes = [
            {'id': feature, 'label': feature}
            for feature in sorted(features)
        ]

        # Create edges (correlations)
        edges = [
            {
                'source': pair['feature_1'],
                'target': pair['feature_2'],
                'weight': pair['abs_correlation'],
                'correlation': pair['correlation'],
                'strength': pair['strength']
            }
            for pair in pairs
        ]

        return {
            'nodes': nodes,
            'edges': edges,
            'node_count': len(nodes),
            'edge_count': len(edges)
        }

    def _interpret_correlation(self, abs_corr: float) -> str:
        """Interpret correlation strength."""
        if abs_corr >= 0.9:
            return 'Very Strong'
        elif abs_corr >= 0.7:
            return 'Strong'
        elif abs_corr >= 0.5:
            return 'Moderate'
        elif abs_corr >= 0.3:
            return 'Weak'
        else:
            return 'Very Weak'

    def _assess_multicollinearity_severity(self, pair_count: int) -> str:
        """Assess multicollinearity severity."""
        if pair_count == 0:
            return 'None'
        elif pair_count <= 2:
            return 'Low'
        elif pair_count <= 5:
            return 'Moderate'
        else:
            return 'High'


# Convenience functions

def correlation_matrix(
    df: pd.DataFrame,
    method: Literal['pearson', 'spearman', 'kendall'] = 'pearson',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute correlation matrix (convenience function).

    Args:
        df: DataFrame to analyze
        method: Correlation method
        columns: Specific columns (None = all numeric)

    Returns:
        Correlation matrix

    Example:
        >>> matrix = correlation_matrix(df, method='spearman')
    """
    corr = CorrelationMatrix(df)
    return corr.compute_correlation(method=method, columns=columns)


def correlation_heatmap_data(
    df: pd.DataFrame,
    method: Literal['pearson', 'spearman', 'kendall'] = 'pearson',
    annotations: bool = True
) -> Dict[str, any]:
    """
    Get heatmap data for correlation matrix (convenience function).

    Args:
        df: DataFrame to analyze
        method: Correlation method
        annotations: Include value annotations

    Returns:
        Heatmap data dictionary

    Example:
        >>> heatmap_data = correlation_heatmap_data(df, method='pearson')
    """
    corr = CorrelationMatrix(df)
    corr.compute_correlation(method=method)
    return corr.get_heatmap_data(annotations=annotations)


def find_highly_correlated_pairs(
    df: pd.DataFrame,
    threshold: float = 0.7,
    method: Literal['pearson', 'spearman', 'kendall'] = 'pearson'
) -> List[Dict[str, Union[str, float]]]:
    """
    Find highly correlated feature pairs (convenience function).

    Args:
        df: DataFrame to analyze
        threshold: Minimum correlation
        method: Correlation method

    Returns:
        List of highly correlated pairs

    Example:
        >>> pairs = find_highly_correlated_pairs(df, threshold=0.8)
    """
    corr = CorrelationMatrix(df)
    corr.compute_correlation(method=method)
    return corr.get_correlation_pairs(threshold=threshold)


def detect_multicollinearity(
    df: pd.DataFrame,
    threshold: float = 0.9,
    method: Literal['pearson', 'spearman', 'kendall'] = 'pearson'
) -> Dict[str, any]:
    """
    Detect multicollinearity issues (convenience function).

    Args:
        df: DataFrame to analyze
        threshold: Correlation threshold for warning
        method: Correlation method

    Returns:
        Multicollinearity statistics

    Example:
        >>> issues = detect_multicollinearity(df, threshold=0.9)
    """
    corr = CorrelationMatrix(df)
    corr.compute_correlation(method=method)
    return corr.get_multicollinearity_stats(threshold=threshold)


def rank_features_by_target(
    df: pd.DataFrame,
    target_column: str,
    method: Literal['pearson', 'spearman', 'kendall'] = 'pearson',
    top_k: Optional[int] = None
) -> pd.DataFrame:
    """
    Rank features by correlation with target (convenience function).

    Args:
        df: DataFrame to analyze
        target_column: Target variable name
        method: Correlation method
        top_k: Return top k features (None = all)

    Returns:
        Ranked features DataFrame

    Example:
        >>> ranking = rank_features_by_target(df, 'price', top_k=10)
    """
    corr = CorrelationMatrix(df)
    corr.compute_correlation(method=method)
    return corr.get_feature_ranking(target_column, top_k=top_k)
