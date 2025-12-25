"""
Exploratory Data Analysis (EDA) Statistics Module.

Provides comprehensive statistical analysis functions for data exploration:
- Descriptive statistics (mean, median, std, etc.)
- Distribution analysis (skewness, kurtosis, normality tests)
- Missing data analysis
- Outlier detection statistics
- Data quality metrics
- Feature type detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Literal
from scipy import stats


class EDAStatistics:
    """
    Comprehensive EDA statistics calculator for DataFrames.

    Provides various statistical analyses to understand data characteristics,
    quality, and distributions.

    Example
    -------
    >>> eda = EDAStatistics(df)
    >>> summary = eda.get_summary_statistics()
    >>> missing = eda.analyze_missing_data()
    >>> outliers = eda.detect_outliers()
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize EDA statistics calculator.

        Args:
            df: DataFrame to analyze

        Raises:
            TypeError: If df is not a pandas DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        self.df = df.copy()
        self._numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    def get_summary_statistics(
        self,
        columns: Optional[List[str]] = None,
        percentiles: List[float] = [0.25, 0.5, 0.75]
    ) -> pd.DataFrame:
        """
        Get comprehensive summary statistics for numeric columns.

        Args:
            columns: Specific columns to analyze (None = all numeric)
            percentiles: Percentiles to calculate

        Returns:
            DataFrame with summary statistics
        """
        if columns is None:
            columns = self._numeric_cols
        else:
            columns = [c for c in columns if c in self._numeric_cols]

        if len(columns) == 0:
            return pd.DataFrame()

        df_subset = self.df[columns]

        # Basic statistics
        summary = df_subset.describe(percentiles=percentiles).T

        # Add additional metrics
        summary['missing'] = df_subset.isnull().sum()
        summary['missing_pct'] = (summary['missing'] / len(df_subset) * 100).round(2)
        summary['unique'] = df_subset.nunique()
        summary['skewness'] = df_subset.skew()
        summary['kurtosis'] = df_subset.kurtosis()

        # Add coefficient of variation (std/mean)
        summary['cv'] = (summary['std'] / summary['mean']).round(4)

        # Reorder columns for better readability
        col_order = ['count', 'missing', 'missing_pct', 'unique', 'mean', 'std',
                     'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis', 'cv']
        summary = summary[[c for c in col_order if c in summary.columns]]

        return summary

    def get_categorical_summary(
        self,
        columns: Optional[List[str]] = None,
        top_n: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """
        Get summary statistics for categorical columns.

        Args:
            columns: Specific columns to analyze (None = all categorical)
            top_n: Number of top categories to show

        Returns:
            Dictionary mapping column names to their statistics
        """
        if columns is None:
            columns = self._categorical_cols
        else:
            columns = [c for c in columns if c in self._categorical_cols]

        results = {}

        for col in columns:
            value_counts = self.df[col].value_counts()

            summary = pd.DataFrame({
                'count': len(self.df),
                'unique': self.df[col].nunique(),
                'missing': self.df[col].isnull().sum(),
                'missing_pct': (self.df[col].isnull().sum() / len(self.df) * 100).round(2),
                'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_common_freq': value_counts.values[0] if len(value_counts) > 0 else 0,
                'most_common_pct': (value_counts.values[0] / len(self.df) * 100).round(2) if len(value_counts) > 0 else 0
            }, index=[col])

            # Top categories
            top_categories = value_counts.head(top_n)

            results[col] = {
                'summary': summary,
                'top_categories': top_categories
            }

        return results

    def analyze_missing_data(self) -> pd.DataFrame:
        """
        Comprehensive missing data analysis.

        Returns:
            DataFrame with missing data statistics per column
        """
        missing_count = self.df.isnull().sum()
        missing_pct = (missing_count / len(self.df) * 100).round(2)

        result = pd.DataFrame({
            'column': self.df.columns,
            'missing_count': missing_count.values,
            'missing_percentage': missing_pct.values,
            'present_count': len(self.df) - missing_count.values,
            'dtype': self.df.dtypes.values
        })

        # Sort by missing percentage descending
        result = result.sort_values('missing_percentage', ascending=False).reset_index(drop=True)

        # Add overall summary
        total_cells = self.df.shape[0] * self.df.shape[1]
        total_missing = self.df.isnull().sum().sum()

        result.attrs['total_cells'] = total_cells
        result.attrs['total_missing'] = total_missing
        result.attrs['total_missing_pct'] = round(total_missing / total_cells * 100, 2)

        return result

    def detect_outliers(
        self,
        method: Literal['iqr', 'zscore', 'both'] = 'both',
        iqr_threshold: float = 1.5,
        zscore_threshold: float = 3.0,
        columns: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Detect outliers using IQR and/or Z-score methods.

        Args:
            method: Detection method ('iqr', 'zscore', or 'both')
            iqr_threshold: IQR multiplier for outlier detection
            zscore_threshold: Z-score threshold for outlier detection
            columns: Specific columns to analyze (None = all numeric)

        Returns:
            Dictionary with outlier statistics per column
        """
        if columns is None:
            columns = self._numeric_cols
        else:
            columns = [c for c in columns if c in self._numeric_cols]

        results = {}

        for col in columns:
            data = self.df[col].dropna()

            if len(data) == 0:
                continue

            stats_dict = {
                'column': col,
                'count': len(data),
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max()
            }

            # IQR method
            if method in ['iqr', 'both']:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_threshold * IQR
                upper_bound = Q3 + iqr_threshold * IQR

                iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()

                stats_dict['iqr_lower_bound'] = lower_bound
                stats_dict['iqr_upper_bound'] = upper_bound
                stats_dict['iqr_outliers'] = iqr_outliers
                stats_dict['iqr_outliers_pct'] = round(iqr_outliers / len(data) * 100, 2)

            # Z-score method
            if method in ['zscore', 'both']:
                z_scores = np.abs((data - data.mean()) / data.std())
                zscore_outliers = (z_scores > zscore_threshold).sum()

                stats_dict['zscore_threshold'] = zscore_threshold
                stats_dict['zscore_outliers'] = zscore_outliers
                stats_dict['zscore_outliers_pct'] = round(zscore_outliers / len(data) * 100, 2)

            results[col] = pd.DataFrame([stats_dict])

        return results

    def analyze_distributions(
        self,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Analyze distributions of numeric columns.

        Includes skewness, kurtosis, and normality tests.

        Args:
            columns: Specific columns to analyze (None = all numeric)

        Returns:
            DataFrame with distribution statistics
        """
        if columns is None:
            columns = self._numeric_cols
        else:
            columns = [c for c in columns if c in self._numeric_cols]

        results = []

        for col in columns:
            data = self.df[col].dropna()

            if len(data) < 3:
                continue

            # Calculate distribution metrics
            skewness = data.skew()
            kurtosis_val = data.kurtosis()

            # Shapiro-Wilk test for normality (for sample size <= 5000)
            if len(data) <= 5000:
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(data)
                except:
                    shapiro_stat, shapiro_p = np.nan, np.nan
            else:
                shapiro_stat, shapiro_p = np.nan, np.nan

            # Interpretation
            skew_interpretation = self._interpret_skewness(skewness)
            kurt_interpretation = self._interpret_kurtosis(kurtosis_val)
            is_normal = shapiro_p > 0.05 if not np.isnan(shapiro_p) else None

            results.append({
                'column': col,
                'count': len(data),
                'skewness': round(skewness, 4),
                'skew_interpretation': skew_interpretation,
                'kurtosis': round(kurtosis_val, 4),
                'kurtosis_interpretation': kurt_interpretation,
                'shapiro_statistic': round(shapiro_stat, 4) if not np.isnan(shapiro_stat) else None,
                'shapiro_p_value': round(shapiro_p, 4) if not np.isnan(shapiro_p) else None,
                'is_normal': is_normal
            })

        return pd.DataFrame(results)

    def _interpret_skewness(self, skew: float) -> str:
        """Interpret skewness value."""
        if abs(skew) < 0.5:
            return 'Approximately Symmetric'
        elif skew < -0.5:
            return 'Left Skewed (Negative)'
        else:
            return 'Right Skewed (Positive)'

    def _interpret_kurtosis(self, kurt: float) -> str:
        """Interpret kurtosis value."""
        if abs(kurt) < 0.5:
            return 'Mesokurtic (Normal-like)'
        elif kurt < -0.5:
            return 'Platykurtic (Flat)'
        else:
            return 'Leptokurtic (Heavy-tailed)'

    def get_correlation_summary(
        self,
        method: Literal['pearson', 'spearman', 'kendall'] = 'pearson',
        threshold: float = 0.7
    ) -> Dict[str, Union[pd.DataFrame, List]]:
        """
        Get correlation summary and highly correlated pairs.

        Args:
            method: Correlation method
            threshold: Threshold for identifying high correlations

        Returns:
            Dictionary with correlation matrix and highly correlated pairs
        """
        if len(self._numeric_cols) == 0:
            return {'correlation_matrix': pd.DataFrame(), 'high_correlations': []}

        corr_matrix = self.df[self._numeric_cols].corr(method=method)

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr_pairs.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': round(corr_val, 4)
                    })

        # Sort by absolute correlation
        high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)

        return {
            'correlation_matrix': corr_matrix,
            'high_correlations': high_corr_pairs
        }

    def get_data_quality_report(self) -> Dict[str, any]:
        """
        Get comprehensive data quality report.

        Returns:
            Dictionary with various data quality metrics
        """
        total_rows = len(self.df)
        total_cols = len(self.df.columns)
        total_cells = total_rows * total_cols

        # Missing data
        missing_cells = self.df.isnull().sum().sum()
        missing_pct = round(missing_cells / total_cells * 100, 2)

        # Duplicate rows
        duplicate_rows = self.df.duplicated().sum()
        duplicate_pct = round(duplicate_rows / total_rows * 100, 2)

        # Column types
        numeric_count = len(self._numeric_cols)
        categorical_count = len(self._categorical_cols)
        other_count = total_cols - numeric_count - categorical_count

        # Constant columns (zero variance)
        constant_cols = [col for col in self._numeric_cols
                        if self.df[col].nunique() <= 1]

        # High cardinality categorical columns (>50 unique values)
        high_cardinality_cols = [col for col in self._categorical_cols
                                if self.df[col].nunique() > 50]

        report = {
            'dataset_shape': {
                'rows': total_rows,
                'columns': total_cols,
                'total_cells': total_cells
            },
            'missing_data': {
                'missing_cells': missing_cells,
                'missing_percentage': missing_pct,
                'columns_with_missing': self.df.columns[self.df.isnull().any()].tolist()
            },
            'duplicates': {
                'duplicate_rows': duplicate_rows,
                'duplicate_percentage': duplicate_pct
            },
            'column_types': {
                'numeric': numeric_count,
                'categorical': categorical_count,
                'other': other_count,
                'numeric_columns': self._numeric_cols,
                'categorical_columns': self._categorical_cols
            },
            'data_quality_issues': {
                'constant_columns': constant_cols,
                'high_cardinality_categorical': high_cardinality_cols
            }
        }

        return report

    def get_feature_types(self) -> pd.DataFrame:
        """
        Detect and categorize feature types.

        Returns:
            DataFrame with feature type information
        """
        results = []

        for col in self.df.columns:
            dtype = self.df[col].dtype
            nunique = self.df[col].nunique()
            missing = self.df[col].isnull().sum()

            # Determine feature type
            if dtype in [np.int64, np.int32, np.float64, np.float32]:
                if nunique == 2:
                    feature_type = 'Binary Numeric'
                elif nunique <= 10:
                    feature_type = 'Categorical Numeric (Low Cardinality)'
                else:
                    feature_type = 'Continuous Numeric'
            elif dtype in ['object', 'category']:
                if nunique == 2:
                    feature_type = 'Binary Categorical'
                elif nunique <= 10:
                    feature_type = 'Categorical (Low Cardinality)'
                elif nunique <= 50:
                    feature_type = 'Categorical (Medium Cardinality)'
                else:
                    feature_type = 'Categorical (High Cardinality)'
            elif dtype == 'bool':
                feature_type = 'Boolean'
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                feature_type = 'Datetime'
            else:
                feature_type = 'Other'

            results.append({
                'column': col,
                'dtype': str(dtype),
                'unique_values': nunique,
                'missing_values': missing,
                'feature_type': feature_type
            })

        return pd.DataFrame(results)


# Convenience functions for quick EDA

def quick_summary(df: pd.DataFrame) -> Dict[str, any]:
    """
    Get a quick summary of the DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with summary information
    """
    eda = EDAStatistics(df)
    return {
        'shape': df.shape,
        'numeric_summary': eda.get_summary_statistics(),
        'missing_data': eda.analyze_missing_data(),
        'quality_report': eda.get_data_quality_report()
    }


def detect_data_issues(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect common data quality issues.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with lists of problematic columns
    """
    eda = EDAStatistics(df)
    quality_report = eda.get_data_quality_report()
    missing_analysis = eda.analyze_missing_data()

    issues = {
        'high_missing_rate': missing_analysis[
            missing_analysis['missing_percentage'] > 50
        ]['column'].tolist(),
        'constant_columns': quality_report['data_quality_issues']['constant_columns'],
        'high_cardinality': quality_report['data_quality_issues']['high_cardinality_categorical'],
        'duplicate_rows_exist': quality_report['duplicates']['duplicate_rows'] > 0
    }

    return issues
