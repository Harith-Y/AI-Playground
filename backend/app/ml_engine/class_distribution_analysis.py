"""
Class Distribution Analysis Module.

Provides comprehensive class distribution analysis functions for classification tasks:
- Class count and proportion analysis
- Class imbalance detection and metrics
- Imbalance severity assessment
- Resampling strategy recommendations
- Visualization data preparation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Literal, Union, Tuple
from app.utils.logger import get_logger

logger = get_logger("class_distribution_analysis")


class ClassDistributionAnalysis:
    """
    Comprehensive class distribution analyzer for classification datasets.

    Analyzes class balance, detects imbalance issues, and provides recommendations
    for handling imbalanced datasets through various resampling strategies.

    Example
    -------
    >>> analyzer = ClassDistributionAnalysis(df, target_column='target')
    >>> distribution = analyzer.get_class_distribution()
    >>> imbalance = analyzer.get_imbalance_metrics()
    >>> recommendations = analyzer.get_resampling_recommendations()
    """

    def __init__(self, df: pd.DataFrame, target_column: str):
        """
        Initialize class distribution analyzer.

        Args:
            df: DataFrame containing the target column
            target_column: Name of the target/label column

        Raises:
            TypeError: If df is not a pandas DataFrame
            ValueError: If target_column not found in DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        self.df = df.copy()
        self.target_column = target_column
        self._target_data = df[target_column].copy()

        # Remove missing values for analysis
        self._target_clean = self._target_data.dropna()

        if len(self._target_clean) == 0:
            raise ValueError(f"Target column '{target_column}' contains only missing values")

        # Store class information
        self._classes = None
        self._class_counts = None
        self._class_proportions = None
        self._compute_class_info()

        logger.info(
            f"Initialized ClassDistributionAnalysis for '{target_column}' "
            f"with {self.get_num_classes()} classes"
        )

    def _compute_class_info(self) -> None:
        """Compute and cache class information."""
        value_counts = self._target_clean.value_counts()
        self._classes = value_counts.index.tolist()
        self._class_counts = value_counts.to_dict()

        total = len(self._target_clean)
        self._class_proportions = {
            cls: count / total for cls, count in self._class_counts.items()
        }

    def get_num_classes(self) -> int:
        """
        Get the number of unique classes.

        Returns:
            Number of unique classes
        """
        return len(self._classes)

    def get_class_distribution(self) -> pd.DataFrame:
        """
        Get comprehensive class distribution statistics.

        Returns:
            DataFrame with class counts, proportions, and percentages

        Example:
            >>> analyzer.get_class_distribution()
                  class  count  proportion  percentage
            0         0   1000       0.100      10.00%
            1         1   9000       0.900      90.00%
        """
        distribution = pd.DataFrame({
            'class': self._classes,
            'count': [self._class_counts[cls] for cls in self._classes],
            'proportion': [self._class_proportions[cls] for cls in self._classes],
            'percentage': [f"{self._class_proportions[cls] * 100:.2f}%" for cls in self._classes]
        })

        # Sort by count descending
        distribution = distribution.sort_values('count', ascending=False).reset_index(drop=True)

        return distribution

    def get_imbalance_ratio(self) -> float:
        """
        Calculate imbalance ratio (majority class / minority class).

        Returns:
            Imbalance ratio (higher = more imbalanced)

        Example:
            >>> analyzer.get_imbalance_ratio()
            9.0  # Majority class is 9x larger than minority
        """
        if len(self._class_counts) < 2:
            return 1.0

        counts = list(self._class_counts.values())
        max_count = max(counts)
        min_count = min(counts)

        if min_count == 0:
            return np.inf

        return round(max_count / min_count, 2)

    def identify_minority_majority_classes(self) -> Dict[str, any]:
        """
        Identify minority and majority classes.

        Returns:
            Dictionary with minority and majority class information

        Example:
            >>> analyzer.identify_minority_majority_classes()
            {
                'minority_class': 1,
                'minority_count': 1000,
                'minority_proportion': 0.10,
                'majority_class': 0,
                'majority_count': 9000,
                'majority_proportion': 0.90
            }
        """
        counts = list(self._class_counts.values())
        classes = list(self._class_counts.keys())

        min_idx = counts.index(min(counts))
        max_idx = counts.index(max(counts))

        minority_class = classes[min_idx]
        majority_class = classes[max_idx]

        return {
            'minority_class': minority_class,
            'minority_count': self._class_counts[minority_class],
            'minority_proportion': round(self._class_proportions[minority_class], 4),
            'minority_percentage': f"{self._class_proportions[minority_class] * 100:.2f}%",
            'majority_class': majority_class,
            'majority_count': self._class_counts[majority_class],
            'majority_proportion': round(self._class_proportions[majority_class], 4),
            'majority_percentage': f"{self._class_proportions[majority_class] * 100:.2f}%"
        }

    def get_imbalance_metrics(self) -> Dict[str, any]:
        """
        Get comprehensive imbalance metrics.

        Returns:
            Dictionary with various imbalance metrics and assessments

        Example:
            >>> analyzer.get_imbalance_metrics()
            {
                'imbalance_ratio': 9.0,
                'severity': 'Severe',
                'is_balanced': False,
                'gini_coefficient': 0.64,
                'entropy': 0.47
            }
        """
        imbalance_ratio = self.get_imbalance_ratio()
        severity = self._assess_imbalance_severity(imbalance_ratio)
        is_balanced = severity == 'Balanced'

        # Calculate Gini coefficient (measure of inequality)
        gini = self._calculate_gini_coefficient()

        # Calculate entropy (measure of disorder)
        entropy = self._calculate_entropy()

        return {
            'num_classes': self.get_num_classes(),
            'total_samples': len(self._target_clean),
            'imbalance_ratio': imbalance_ratio,
            'severity': severity,
            'is_balanced': is_balanced,
            'gini_coefficient': round(gini, 4),
            'entropy': round(entropy, 4),
            'requires_resampling': severity in ['Moderate', 'Severe', 'Extreme']
        }

    def _assess_imbalance_severity(self, imbalance_ratio: float) -> str:
        """
        Assess imbalance severity based on ratio.

        Args:
            imbalance_ratio: Ratio of majority to minority class

        Returns:
            Severity level as string
        """
        if imbalance_ratio == np.inf:
            return 'Extreme (Missing Class)'
        elif imbalance_ratio >= 100:
            return 'Extreme'
        elif imbalance_ratio >= 10:
            return 'Severe'
        elif imbalance_ratio >= 3:
            return 'Moderate'
        elif imbalance_ratio >= 1.5:
            return 'Slight'
        else:
            return 'Balanced'

    def _calculate_gini_coefficient(self) -> float:
        """
        Calculate Gini coefficient for class distribution.

        Gini coefficient measures inequality in distribution (0 = perfect equality, 1 = maximum inequality).

        Returns:
            Gini coefficient
        """
        proportions = sorted(self._class_proportions.values())
        n = len(proportions)

        if n == 0:
            return 0.0

        # Calculate Gini coefficient
        cumsum = np.cumsum(proportions)
        gini = (2 * np.sum((np.arange(1, n + 1) * proportions))) / (n * np.sum(proportions)) - (n + 1) / n

        return gini

    def _calculate_entropy(self) -> float:
        """
        Calculate Shannon entropy of class distribution.

        Higher entropy = more uniform distribution.
        Maximum entropy = log2(num_classes) for uniform distribution.

        Returns:
            Shannon entropy
        """
        proportions = np.array(list(self._class_proportions.values()))

        # Remove zero proportions to avoid log(0)
        proportions = proportions[proportions > 0]

        if len(proportions) == 0:
            return 0.0

        # Shannon entropy
        entropy = -np.sum(proportions * np.log2(proportions))

        return entropy

    def get_bar_chart_data(self) -> Dict[str, any]:
        """
        Get data formatted for bar chart visualization.

        Returns data structure compatible with frontend plotting libraries.

        Returns:
            Dictionary with bar chart data

        Example:
            >>> analyzer.get_bar_chart_data()
            {
                'labels': [0, 1],
                'counts': [9000, 1000],
                'proportions': [0.9, 0.1],
                'colors': ['#1f77b4', '#ff7f0e']
            }
        """
        distribution = self.get_class_distribution()

        # Generate colors (you can customize this)
        colors = self._generate_colors(len(distribution))

        return {
            'labels': distribution['class'].tolist(),
            'counts': distribution['count'].tolist(),
            'proportions': distribution['proportion'].tolist(),
            'percentages': distribution['percentage'].tolist(),
            'colors': colors,
            'chart_type': 'bar'
        }

    def _generate_colors(self, n: int) -> List[str]:
        """
        Generate distinct colors for visualization.

        Args:
            n: Number of colors to generate

        Returns:
            List of hex color codes
        """
        # Use a simple color palette (can be extended)
        base_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        if n <= len(base_colors):
            return base_colors[:n]

        # For more than 10 classes, generate gradient
        colors = base_colors.copy()
        while len(colors) < n:
            colors.append(f'#{np.random.randint(0, 0xFFFFFF):06x}')

        return colors[:n]

    def get_resampling_recommendations(self) -> Dict[str, any]:
        """
        Get recommendations for handling class imbalance.

        Provides specific resampling strategy recommendations based on
        dataset characteristics and imbalance severity.

        Returns:
            Dictionary with resampling recommendations

        Example:
            >>> analyzer.get_resampling_recommendations()
            {
                'severity': 'Severe',
                'recommended_strategies': ['SMOTE', 'BorderlineSMOTE'],
                'oversampling_targets': {...},
                'undersampling_targets': {...}
            }
        """
        metrics = self.get_imbalance_metrics()
        minority_majority = self.identify_minority_majority_classes()

        severity = metrics['severity']
        imbalance_ratio = metrics['imbalance_ratio']

        recommendations = {
            'severity': severity,
            'imbalance_ratio': imbalance_ratio,
            'requires_action': metrics['requires_resampling'],
            'recommended_strategies': [],
            'strategy_details': {},
            'oversampling_targets': None,
            'undersampling_targets': None
        }

        if not metrics['requires_resampling']:
            recommendations['message'] = "Dataset is relatively balanced. No resampling required."
            return recommendations

        # Determine recommended strategies based on severity
        if severity == 'Extreme' or severity == 'Extreme (Missing Class)':
            recommendations['recommended_strategies'] = [
                'SMOTE',
                'BorderlineSMOTE',
                'ADASYN',
                'Class Weights (Model-based)'
            ]
            recommendations['message'] = (
                "Extreme imbalance detected. Consider combining oversampling (SMOTE/ADASYN) "
                "with undersampling, or using class weights in your model."
            )

        elif severity == 'Severe':
            recommendations['recommended_strategies'] = [
                'SMOTE',
                'BorderlineSMOTE',
                'ADASYN'
            ]
            recommendations['message'] = (
                "Severe imbalance detected. SMOTE-based oversampling is recommended. "
                "Consider BorderlineSMOTE for focusing on borderline cases."
            )

        elif severity == 'Moderate':
            recommendations['recommended_strategies'] = [
                'SMOTE',
                'Random Oversampling',
                'Class Weights'
            ]
            recommendations['message'] = (
                "Moderate imbalance detected. SMOTE or random oversampling can help. "
                "Alternatively, use class weights in model training."
            )

        elif severity == 'Slight':
            recommendations['recommended_strategies'] = [
                'Class Weights',
                'Random Oversampling (minor)'
            ]
            recommendations['message'] = (
                "Slight imbalance detected. Class weights may be sufficient. "
                "Light oversampling can also be applied."
            )

        # Calculate oversampling targets (balance to majority class)
        recommendations['oversampling_targets'] = self._calculate_oversampling_targets()

        # Calculate undersampling targets (balance to minority class)
        recommendations['undersampling_targets'] = self._calculate_undersampling_targets()

        # Add strategy details
        recommendations['strategy_details'] = self._get_strategy_details()

        return recommendations

    def _calculate_oversampling_targets(self) -> Dict[str, int]:
        """
        Calculate target counts for oversampling to balance dataset.

        Returns:
            Dictionary mapping class to target count
        """
        max_count = max(self._class_counts.values())

        targets = {}
        for cls in self._classes:
            targets[str(cls)] = max_count  # Balance to majority class

        return targets

    def _calculate_undersampling_targets(self) -> Dict[str, int]:
        """
        Calculate target counts for undersampling to balance dataset.

        Returns:
            Dictionary mapping class to target count
        """
        min_count = min(self._class_counts.values())

        targets = {}
        for cls in self._classes:
            targets[str(cls)] = min_count  # Balance to minority class

        return targets

    def _get_strategy_details(self) -> Dict[str, str]:
        """
        Get detailed descriptions of resampling strategies.

        Returns:
            Dictionary mapping strategy name to description
        """
        return {
            'SMOTE': (
                "Synthetic Minority Over-sampling Technique. Generates synthetic samples "
                "by interpolating between minority class instances."
            ),
            'BorderlineSMOTE': (
                "Focuses on borderline cases near class boundaries. More effective for "
                "hard-to-classify instances."
            ),
            'ADASYN': (
                "Adaptive Synthetic Sampling. Generates more synthetic samples for "
                "harder-to-learn minority instances."
            ),
            'Random Oversampling': (
                "Randomly duplicates minority class samples. Simple but can lead to overfitting."
            ),
            'Random Undersampling': (
                "Randomly removes majority class samples. Fast but may lose information."
            ),
            'Class Weights': (
                "Assign higher weights to minority class during model training. "
                "No data modification required."
            )
        }

    def get_missing_value_info(self) -> Dict[str, any]:
        """
        Get information about missing values in target column.

        Returns:
            Dictionary with missing value statistics
        """
        total_count = len(self._target_data)
        missing_count = self._target_data.isnull().sum()
        missing_pct = (missing_count / total_count * 100) if total_count > 0 else 0

        return {
            'total_samples': total_count,
            'valid_samples': len(self._target_clean),
            'missing_samples': missing_count,
            'missing_percentage': round(missing_pct, 2),
            'has_missing': missing_count > 0
        }

    def analyze_class_overlap(
        self,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Analyze potential class overlap in feature space.

        Provides insights into how separable classes are based on features.

        Args:
            feature_columns: List of feature columns to analyze (None = all numeric)

        Returns:
            Dictionary with class overlap analysis
        """
        if feature_columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_cols if col != self.target_column]

        if len(feature_columns) == 0:
            return {
                'message': 'No numeric features available for overlap analysis',
                'analyzable': False
            }

        # Calculate feature statistics per class
        class_stats = {}
        for cls in self._classes:
            class_data = self.df[self.df[self.target_column] == cls][feature_columns]
            class_stats[str(cls)] = {
                'mean': class_data.mean().to_dict(),
                'std': class_data.std().to_dict(),
                'count': len(class_data)
            }

        return {
            'analyzable': True,
            'num_features': len(feature_columns),
            'feature_columns': feature_columns,
            'class_statistics': class_stats,
            'message': 'Class statistics calculated. Review mean/std differences to assess separability.'
        }

    def get_summary_report(self) -> Dict[str, any]:
        """
        Get comprehensive summary report of class distribution analysis.

        Returns:
            Dictionary with complete analysis summary

        Example:
            >>> report = analyzer.get_summary_report()
        """
        distribution = self.get_class_distribution()
        metrics = self.get_imbalance_metrics()
        minority_majority = self.identify_minority_majority_classes()
        recommendations = self.get_resampling_recommendations()
        missing_info = self.get_missing_value_info()

        return {
            'target_column': self.target_column,
            'class_distribution': distribution.to_dict('records'),
            'imbalance_metrics': metrics,
            'minority_majority_info': minority_majority,
            'recommendations': recommendations,
            'missing_value_info': missing_info
        }


# Convenience functions

def analyze_class_distribution(
    df: pd.DataFrame,
    target_column: str
) -> pd.DataFrame:
    """
    Get class distribution (convenience function).

    Args:
        df: DataFrame to analyze
        target_column: Name of target column

    Returns:
        DataFrame with class distribution

    Example:
        >>> distribution = analyze_class_distribution(df, 'target')
    """
    analyzer = ClassDistributionAnalysis(df, target_column)
    return analyzer.get_class_distribution()


def check_class_imbalance(
    df: pd.DataFrame,
    target_column: str,
    threshold: float = 3.0
) -> Dict[str, any]:
    """
    Check if dataset has class imbalance (convenience function).

    Args:
        df: DataFrame to analyze
        target_column: Name of target column
        threshold: Imbalance ratio threshold for warning

    Returns:
        Dictionary with imbalance check results

    Example:
        >>> result = check_class_imbalance(df, 'target', threshold=5.0)
        >>> if result['is_imbalanced']:
        ...     print(result['message'])
    """
    analyzer = ClassDistributionAnalysis(df, target_column)
    metrics = analyzer.get_imbalance_metrics()

    imbalance_ratio = metrics['imbalance_ratio']
    is_imbalanced = imbalance_ratio >= threshold

    return {
        'is_imbalanced': is_imbalanced,
        'imbalance_ratio': imbalance_ratio,
        'severity': metrics['severity'],
        'num_classes': metrics['num_classes'],
        'message': (
            f"Imbalanced dataset detected (ratio: {imbalance_ratio:.2f}). "
            f"Severity: {metrics['severity']}"
        ) if is_imbalanced else "Dataset is relatively balanced.",
        'requires_action': metrics['requires_resampling']
    }


def get_resampling_strategy(
    df: pd.DataFrame,
    target_column: str
) -> Dict[str, any]:
    """
    Get recommended resampling strategy (convenience function).

    Args:
        df: DataFrame to analyze
        target_column: Name of target column

    Returns:
        Dictionary with resampling recommendations

    Example:
        >>> strategy = get_resampling_strategy(df, 'target')
        >>> print(strategy['recommended_strategies'])
        ['SMOTE', 'BorderlineSMOTE']
    """
    analyzer = ClassDistributionAnalysis(df, target_column)
    return analyzer.get_resampling_recommendations()


def visualize_class_distribution_data(
    df: pd.DataFrame,
    target_column: str
) -> Dict[str, any]:
    """
    Get visualization data for class distribution (convenience function).

    Args:
        df: DataFrame to analyze
        target_column: Name of target column

    Returns:
        Dictionary with visualization data

    Example:
        >>> viz_data = visualize_class_distribution_data(df, 'target')
        >>> # Use viz_data with plotting library (Plotly, Matplotlib, etc.)
    """
    analyzer = ClassDistributionAnalysis(df, target_column)
    return analyzer.get_bar_chart_data()
