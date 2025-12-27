"""
Univariate statistical tests for feature selection.

This module provides feature selection based on univariate statistical tests:
- F-test (ANOVA F-statistic) for continuous features with categorical/continuous targets
- Chi-square test for categorical features with categorical targets

These tests evaluate each feature independently with the target variable.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Literal
from sklearn.feature_selection import f_classif, f_regression, chi2, SelectKBest, SelectPercentile
from app.ml_engine.preprocessing.base import PreprocessingStep


class UnivariateSelector(PreprocessingStep):
    """
    Selects features based on univariate statistical tests.

    Univariate feature selection works by selecting the best features based on
    univariate statistical tests. Each feature is evaluated independently with
    the target variable using appropriate statistical tests.

    Statistical Tests:
    ------------------
    1. F-test (ANOVA F-statistic):
       - For classification: Computes ANOVA F-value between each feature and target
       - For regression: Computes F-statistic for linear regression
       - Assumes: Normally distributed features, homogeneity of variance
       - Use case: Continuous features with categorical or continuous targets

    2. Chi-square test:
       - Tests independence between categorical feature and target
       - Requires: Non-negative features (counts or frequencies)
       - Use case: Categorical features with categorical targets
       - Note: All feature values must be >= 0

    Key advantages:
    - Fast and simple (linear time complexity)
    - Each feature evaluated independently
    - Statistical significance through p-values
    - Well-suited for high-dimensional data
    - Interpretable results

    Limitations:
    - Cannot capture feature interactions
    - Assumes feature independence
    - Chi-square requires non-negative values

    Parameters
    ----------
    score_func : {'f_classif', 'f_regression', 'chi2', 'auto'}, default='auto'
        Univariate statistical test to use:
        - 'f_classif': ANOVA F-test for classification (continuous features)
        - 'f_regression': F-test for regression (continuous features)
        - 'chi2': Chi-square test (categorical features, requires non-negative values)
        - 'auto': Automatically select based on task parameter

    task : {'classification', 'regression'}, default='classification'
        Type of supervised learning task (used when score_func='auto'):
        - 'classification': Use f_classif or chi2
        - 'regression': Use f_regression

    k : int, optional
        Number of top features to select based on test scores
        If None, uses percentile or threshold parameter

    percentile : int, optional
        Select features with test scores in top percentile (0-100)
        Alternative to k parameter

    threshold : float, optional
        Select features with test scores above this threshold
        Used when both k and percentile are None

    alpha : float, default=0.05
        Significance level for p-value filtering
        Features with p-value > alpha are considered not significant

    use_p_value_filter : bool, default=False
        If True, additionally filter features by p-value < alpha
        Applied after k/percentile/threshold selection

    columns : list of str, optional
        Specific columns to consider for selection
        If None, uses all numeric columns

    Example
    -------
    >>> # F-test for classification (ANOVA)
    >>> selector = UnivariateSelector(score_func='f_classif', k=10)
    >>> X_selected = selector.fit_transform(X_train, y_train)
    >>>
    >>> # Chi-square test for categorical features
    >>> selector = UnivariateSelector(score_func='chi2', percentile=20)
    >>> X_selected = selector.fit_transform(X_train, y_train)
    >>>
    >>> # Auto-select score function based on task
    >>> selector = UnivariateSelector(task='regression', k=15)
    >>> X_selected = selector.fit_transform(X_train, y_train)
    >>>
    >>> # Get feature scores and p-values
    >>> scores_df = selector.get_feature_scores()
    >>> print(scores_df)
    """

    def __init__(
        self,
        score_func: Literal['f_classif', 'f_regression', 'chi2', 'auto'] = 'auto',
        task: Literal['classification', 'regression'] = 'classification',
        k: Optional[int] = None,
        percentile: Optional[int] = None,
        threshold: Optional[float] = None,
        alpha: float = 0.05,
        use_p_value_filter: bool = False,
        columns: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        """
        Initialize UnivariateSelector.

        Args:
            score_func: Statistical test to use
            task: Type of supervised learning task
            k: Number of top features to select
            percentile: Select top percentile of features
            threshold: Minimum score threshold
            alpha: Significance level for p-value filtering
            use_p_value_filter: Whether to filter by p-values
            columns: Specific columns to consider
            name: Optional custom name

        Raises:
            ValueError: If parameters are invalid or conflicting
        """
        if score_func not in ['f_classif', 'f_regression', 'chi2', 'auto']:
            raise ValueError(
                "score_func must be 'f_classif', 'f_regression', 'chi2', or 'auto'"
            )

        if task not in ['classification', 'regression']:
            raise ValueError("task must be 'classification' or 'regression'")

        if k is not None and k <= 0:
            raise ValueError("k must be positive")

        if percentile is not None and not (0 < percentile <= 100):
            raise ValueError("percentile must be between 0 and 100")

        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be between 0 and 1")

        # Check for conflicting parameters
        param_count = sum([
            k is not None,
            percentile is not None,
            threshold is not None
        ])
        if param_count > 1:
            raise ValueError(
                "Only one of k, percentile, or threshold can be specified"
            )

        # If no selection method specified, default to k=10
        if param_count == 0:
            k = 10

        super().__init__(
            name=name,
            score_func=score_func,
            task=task,
            k=k,
            percentile=percentile,
            threshold=threshold,
            alpha=alpha,
            use_p_value_filter=use_p_value_filter,
            columns=columns
        )

        # Fitted parameters
        self.selected_features_: Optional[List[str]] = None
        self.scores_: Optional[pd.Series] = None
        self.p_values_: Optional[pd.Series] = None
        self.feature_names_: Optional[List[str]] = None
        self.score_func_used_: Optional[str] = None

    def _get_score_function(self):
        """
        Get the appropriate score function based on parameters.

        Returns:
            Score function from sklearn.feature_selection
        """
        score_func = self.params["score_func"]
        task = self.params["task"]

        if score_func == 'auto':
            # Auto-select based on task
            if task == 'classification':
                return f_classif
            else:  # regression
                return f_regression
        elif score_func == 'f_classif':
            return f_classif
        elif score_func == 'f_regression':
            return f_regression
        elif score_func == 'chi2':
            return chi2
        else:
            raise ValueError(f"Unknown score_func: {score_func}")

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> "UnivariateSelector":
        """
        Compute statistical test scores and select features.

        Args:
            X: Training features (must be a pandas DataFrame)
            y: Target variable (required)

        Returns:
            Self (for method chaining)

        Raises:
            TypeError: If X is not a pandas DataFrame
            ValueError: If y is not provided or data contains invalid values
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("UnivariateSelector expects a pandas DataFrame")

        if y is None:
            raise ValueError("y (target) must be provided for univariate selection")

        k = self.params["k"]
        percentile = self.params["percentile"]
        threshold = self.params["threshold"]
        alpha = self.params["alpha"]
        use_p_value_filter = self.params["use_p_value_filter"]
        columns = self.params["columns"]

        # Determine columns to analyze
        if columns is not None:
            analyze_cols = [col for col in columns if col in X.columns]
        else:
            analyze_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(analyze_cols) == 0:
            raise ValueError("No numeric columns found to analyze")

        X_subset = X[analyze_cols]
        self.feature_names_ = analyze_cols

        # Convert y to numpy array if Series
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        # Get appropriate score function
        score_function = self._get_score_function()
        self.score_func_used_ = score_function.__name__

        # Validate data for chi2
        if self.score_func_used_ == 'chi2':
            if (X_subset < 0).any().any():
                raise ValueError(
                    "Chi-square test requires non-negative feature values. "
                    "All values must be >= 0."
                )

        # Compute statistical test scores and p-values
        try:
            scores, p_values = score_function(X_subset, y_array)
        except Exception as e:
            raise ValueError(f"Error computing {self.score_func_used_} scores: {e}")

        # Handle NaN scores (can occur with constant features)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)

        # Store scores and p-values
        self.scores_ = pd.Series(scores, index=analyze_cols)
        self.p_values_ = pd.Series(p_values, index=analyze_cols)

        # Select features based on criteria
        if k is not None:
            # Select top k features by score
            top_indices = np.argsort(scores)[-k:]
            selected = [analyze_cols[i] for i in top_indices]
        elif percentile is not None:
            # Select top percentile by score
            threshold_value = np.percentile(scores, 100 - percentile)
            selected = [
                col for col, score in zip(analyze_cols, scores)
                if score >= threshold_value
            ]
        else:
            # Use threshold (if threshold is None, default to 0)
            thresh = threshold if threshold is not None else 0
            selected = [
                col for col, score in zip(analyze_cols, scores)
                if score >= thresh
            ]

        # Additionally filter by p-value if requested
        if use_p_value_filter:
            selected = [
                col for col in selected
                if self.p_values_[col] < alpha
            ]

        self.selected_features_ = selected
        self.fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Transform data by selecting only features with high test scores.

        Args:
            X: Data to transform (must be a pandas DataFrame)

        Returns:
            DataFrame with only selected features

        Raises:
            TypeError: If X is not a pandas DataFrame
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("UnivariateSelector expects a pandas DataFrame")

        # Check for missing features
        missing_features = set(self.selected_features_) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features {missing_features} not found in DataFrame")

        return X[self.selected_features_].copy()

    def get_selected_features(self) -> List[str]:
        """
        Get list of selected feature names.

        Returns:
            List of selected feature names

        Raises:
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()
        return self.selected_features_.copy()

    def get_scores(self) -> pd.Series:
        """
        Get statistical test scores for all features.

        Returns:
            Series with test scores indexed by feature names

        Raises:
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()
        return self.scores_.copy()

    def get_p_values(self) -> pd.Series:
        """
        Get p-values for all features.

        Returns:
            Series with p-values indexed by feature names

        Raises:
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()
        return self.p_values_.copy()

    def get_feature_scores(self) -> pd.DataFrame:
        """
        Get comprehensive feature scoring summary.

        Returns:
            DataFrame with features, scores, p-values, and selection status
            sorted by score (descending)

        Raises:
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()

        df = pd.DataFrame({
            'feature': self.scores_.index,
            'score': self.scores_.values,
            'p_value': self.p_values_.values,
            'selected': [f in self.selected_features_ for f in self.scores_.index],
            'significant': self.p_values_.values < self.params["alpha"]
        })

        return df.sort_values('score', ascending=False).reset_index(drop=True)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance summary (alias for get_feature_scores).

        Returns:
            DataFrame with features, scores, p-values, and selection status

        Raises:
            RuntimeError: If selector has not been fitted
        """
        return self.get_feature_scores()

    def get_support(self, indices: bool = False) -> Union[List[bool], List[int]]:
        """
        Get mask or indices of selected features (scikit-learn compatible).

        Args:
            indices: If True, return feature indices; if False, return boolean mask

        Returns:
            Boolean mask or integer indices of selected features

        Raises:
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()

        if indices:
            # Return indices of selected features
            return [
                i for i, feat in enumerate(self.feature_names_)
                if feat in self.selected_features_
            ]
        else:
            # Return boolean mask
            return [feat in self.selected_features_ for feat in self.feature_names_]

    def get_statistical_summary(self) -> dict:
        """
        Get comprehensive statistical summary of the selection process.

        Returns:
            Dictionary with selection statistics and metadata

        Raises:
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()

        alpha = self.params["alpha"]

        significant_features = (self.p_values_ < alpha).sum()
        mean_score = self.scores_.mean()
        median_score = self.scores_.median()
        max_score = self.scores_.max()
        min_score = self.scores_.min()

        return {
            'score_function': self.score_func_used_,
            'total_features': len(self.feature_names_),
            'selected_features': len(self.selected_features_),
            'selection_rate': len(self.selected_features_) / len(self.feature_names_),
            'significant_features_count': int(significant_features),
            'alpha': alpha,
            'score_statistics': {
                'mean': float(mean_score),
                'median': float(median_score),
                'min': float(min_score),
                'max': float(max_score)
            },
            'selection_criteria': {
                'k': self.params['k'],
                'percentile': self.params['percentile'],
                'threshold': self.params['threshold'],
                'use_p_value_filter': self.params['use_p_value_filter']
            }
        }

    def __repr__(self) -> str:
        """String representation."""
        score_func = self.params["score_func"]
        k = self.params["k"]

        if self.fitted and self.selected_features_:
            n_selected = len(self.selected_features_)
            n_total = len(self.scores_)
            return (
                f"UnivariateSelector(score_func='{self.score_func_used_}', "
                f"k={k}, selected={n_selected}/{n_total})"
            )
        else:
            return f"UnivariateSelector(score_func='{score_func}', k={k}, not fitted)"
