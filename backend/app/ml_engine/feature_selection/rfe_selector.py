"""
Recursive Feature Elimination (RFE) for feature selection.

This module provides feature selection using Recursive Feature Elimination,
which recursively removes the least important features based on model coefficients
or feature importances from estimators.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Literal, Any
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from app.ml_engine.preprocessing.base import PreprocessingStep


class RFESelector(PreprocessingStep):
    """
    Recursive Feature Elimination (RFE) for feature selection.

    RFE works by recursively removing features and building a model on the remaining
    features. It ranks features by recursively considering smaller sets of features.

    The algorithm:
    1. Train a model on all features
    2. Rank features by their importance (coefficients or feature_importances_)
    3. Remove the least important feature(s)
    4. Repeat steps 1-3 until the desired number of features is reached

    Key advantages:
    - Considers feature interactions (unlike univariate methods)
    - Works with any estimator that has coef_ or feature_importances_
    - Can find optimal number of features using cross-validation (RFECV)
    - Provides feature ranking beyond just selection

    Limitations:
    - Computationally expensive (retrains model many times)
    - Can be slow for large feature sets
    - Results depend on the choice of estimator

    Parameters
    ----------
    estimator : {'logistic', 'ridge', 'random_forest', 'svm', 'decision_tree', 'auto'} or sklearn estimator, default='auto'
        The base estimator to use for feature importance:
        - 'logistic': LogisticRegression (classification)
        - 'ridge': Ridge regression (regression)
        - 'random_forest': RandomForestClassifier/Regressor
        - 'svm': SVC/SVR with linear kernel
        - 'decision_tree': DecisionTreeClassifier/Regressor
        - 'auto': Automatically select based on task
        - Custom sklearn estimator with coef_ or feature_importances_

    task : {'classification', 'regression'}, default='classification'
        Type of supervised learning task (used when estimator='auto')

    n_features_to_select : int or float, optional
        Number of features to select:
        - If int: Select exactly this many features
        - If float (0-1): Select this fraction of features
        - If None: Use cross-validation to find optimal number (RFECV)

    step : int or float, default=1
        Number of features to remove at each iteration:
        - If int >= 1: Remove this many features per iteration
        - If float (0-1): Remove this fraction of remaining features per iteration

    use_cv : bool, default=False
        If True, use RFECV (cross-validation) to find optimal number of features
        Overrides n_features_to_select parameter

    cv : int, default=5
        Number of cross-validation folds (only used if use_cv=True)

    scoring : str, optional
        Scoring metric for cross-validation (only used if use_cv=True)
        Examples: 'accuracy', 'f1', 'roc_auc', 'r2', 'neg_mean_squared_error'

    estimator_params : dict, optional
        Additional parameters to pass to the estimator

    verbose : int, default=0
        Verbosity level (0=silent, 1=progress, 2+=debug)

    random_state : int, optional
        Random seed for reproducibility

    columns : list of str, optional
        Specific columns to consider for selection
        If None, uses all numeric columns

    Example
    -------
    >>> # Basic RFE with automatic estimator selection
    >>> selector = RFESelector(n_features_to_select=10, task='classification')
    >>> X_selected = selector.fit_transform(X_train, y_train)
    >>>
    >>> # RFE with cross-validation to find optimal features
    >>> selector = RFESelector(use_cv=True, cv=5, task='classification')
    >>> X_selected = selector.fit_transform(X_train, y_train)
    >>> print(f"Optimal features: {selector.n_features_}")
    >>>
    >>> # RFE with custom estimator (Random Forest)
    >>> selector = RFESelector(
    ...     estimator='random_forest',
    ...     n_features_to_select=15,
    ...     step=2
    ... )
    >>> X_selected = selector.fit_transform(X_train, y_train)
    >>>
    >>> # Get feature rankings
    >>> rankings = selector.get_feature_ranking()
    >>> print(rankings)
    """

    def __init__(
        self,
        estimator: Union[str, Any] = 'auto',
        task: Literal['classification', 'regression'] = 'classification',
        n_features_to_select: Optional[Union[int, float]] = None,
        step: Union[int, float] = 1,
        use_cv: bool = False,
        cv: int = 5,
        scoring: Optional[str] = None,
        estimator_params: Optional[dict] = None,
        verbose: int = 0,
        random_state: Optional[int] = None,
        columns: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        """
        Initialize RFESelector.

        Args:
            estimator: Base estimator for feature importance
            task: Type of supervised learning task
            n_features_to_select: Number of features to select
            step: Number of features to remove per iteration
            use_cv: Whether to use cross-validation
            cv: Number of CV folds
            scoring: Scoring metric for CV
            estimator_params: Additional estimator parameters
            verbose: Verbosity level
            random_state: Random seed
            columns: Specific columns to consider
            name: Optional custom name

        Raises:
            ValueError: If parameters are invalid
        """
        if task not in ['classification', 'regression']:
            raise ValueError("task must be 'classification' or 'regression'")

        if isinstance(estimator, str) and estimator not in [
            'logistic', 'ridge', 'random_forest', 'svm', 'decision_tree', 'auto'
        ]:
            raise ValueError(
                "estimator must be 'logistic', 'ridge', 'random_forest', "
                "'svm', 'decision_tree', 'auto', or a sklearn estimator"
            )

        if n_features_to_select is not None:
            if isinstance(n_features_to_select, int) and n_features_to_select <= 0:
                raise ValueError("n_features_to_select must be positive")
            if isinstance(n_features_to_select, float):
                if not (0 < n_features_to_select < 1):
                    raise ValueError(
                        "n_features_to_select as float must be between 0 and 1"
                    )

        if isinstance(step, int) and step <= 0:
            raise ValueError("step must be positive")
        if isinstance(step, float) and not (0 < step < 1):
            raise ValueError("step as float must be between 0 and 1")

        if cv <= 1:
            raise ValueError("cv must be greater than 1")

        super().__init__(
            name=name,
            estimator=estimator,
            task=task,
            n_features_to_select=n_features_to_select,
            step=step,
            use_cv=use_cv,
            cv=cv,
            scoring=scoring,
            estimator_params=estimator_params or {},
            verbose=verbose,
            random_state=random_state,
            columns=columns
        )

        # Fitted parameters
        self.selected_features_: Optional[List[str]] = None
        self.ranking_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self.n_features_: Optional[int] = None
        self.rfe_: Optional[Union[RFE, RFECV]] = None
        self.estimator_: Optional[Any] = None

    def _get_estimator(self):
        """
        Get the appropriate estimator based on parameters.

        Returns:
            Sklearn estimator instance
        """
        estimator = self.params["estimator"]
        task = self.params["task"]
        random_state = self.params["random_state"]
        estimator_params = self.params["estimator_params"]

        # If already an estimator object, return it
        if not isinstance(estimator, str):
            return estimator

        # Auto-select estimator based on task
        if estimator == 'auto':
            if task == 'classification':
                estimator = 'logistic'
            else:
                estimator = 'ridge'

        # Create estimator based on string specification
        if estimator == 'logistic':
            return LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                **estimator_params
            )
        elif estimator == 'ridge':
            return Ridge(
                random_state=random_state,
                **estimator_params
            )
        elif estimator == 'random_forest':
            if task == 'classification':
                return RandomForestClassifier(
                    random_state=random_state,
                    n_estimators=100,
                    **estimator_params
                )
            else:
                return RandomForestRegressor(
                    random_state=random_state,
                    n_estimators=100,
                    **estimator_params
                )
        elif estimator == 'svm':
            if task == 'classification':
                return SVC(
                    kernel='linear',
                    random_state=random_state,
                    **estimator_params
                )
            else:
                return SVR(
                    kernel='linear',
                    **estimator_params
                )
        elif estimator == 'decision_tree':
            if task == 'classification':
                return DecisionTreeClassifier(
                    random_state=random_state,
                    **estimator_params
                )
            else:
                return DecisionTreeRegressor(
                    random_state=random_state,
                    **estimator_params
                )
        else:
            raise ValueError(f"Unknown estimator: {estimator}")

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> "RFESelector":
        """
        Fit RFE on training data to select features.

        Args:
            X: Training features (must be a pandas DataFrame)
            y: Target variable (required)

        Returns:
            Self (for method chaining)

        Raises:
            TypeError: If X is not a pandas DataFrame
            ValueError: If y is not provided
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RFESelector expects a pandas DataFrame")

        if y is None:
            raise ValueError("y (target) must be provided for RFE")

        n_features_to_select = self.params["n_features_to_select"]
        step = self.params["step"]
        use_cv = self.params["use_cv"]
        cv = self.params["cv"]
        scoring = self.params["scoring"]
        verbose = self.params["verbose"]
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

        # Get base estimator
        base_estimator = self._get_estimator()
        self.estimator_ = base_estimator

        # Convert n_features_to_select if it's a float
        if isinstance(n_features_to_select, float):
            n_features = int(n_features_to_select * len(analyze_cols))
        else:
            n_features = n_features_to_select

        # Create RFE or RFECV
        if use_cv:
            # Use cross-validation to find optimal number of features
            self.rfe_ = RFECV(
                estimator=base_estimator,
                step=step,
                cv=cv,
                scoring=scoring,
                verbose=verbose,
                n_jobs=-1  # Use all available cores
            )
        else:
            # Use regular RFE with specified number of features
            self.rfe_ = RFE(
                estimator=base_estimator,
                n_features_to_select=n_features,
                step=step,
                verbose=verbose
            )

        # Fit RFE
        self.rfe_.fit(X_subset, y_array)

        # Store results
        self.ranking_ = self.rfe_.ranking_
        self.n_features_ = self.rfe_.n_features_

        # Get selected features
        support_mask = self.rfe_.support_
        self.selected_features_ = [
            col for col, selected in zip(analyze_cols, support_mask)
            if selected
        ]

        self.fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Transform data by selecting only RFE-selected features.

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
            raise TypeError("RFESelector expects a pandas DataFrame")

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

    def get_feature_ranking(self) -> pd.DataFrame:
        """
        Get feature ranking from RFE.

        Features are ranked where rank 1 is the most important (selected first).
        All selected features have rank 1.

        Returns:
            DataFrame with features and their rankings sorted by rank

        Raises:
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()

        df = pd.DataFrame({
            'feature': self.feature_names_,
            'rank': self.ranking_,
            'selected': [f in self.selected_features_ for f in self.feature_names_]
        })

        return df.sort_values('rank').reset_index(drop=True)

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

    def get_cv_scores(self) -> Optional[np.ndarray]:
        """
        Get cross-validation scores (only available if use_cv=True).

        Returns:
            Array of CV scores for each number of features tested
            None if use_cv=False

        Raises:
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()

        if isinstance(self.rfe_, RFECV):
            return self.rfe_.cv_results_['mean_test_score']
        else:
            return None

    def get_optimal_n_features(self) -> Optional[int]:
        """
        Get optimal number of features found by cross-validation.

        Returns:
            Optimal number of features
            None if use_cv=False

        Raises:
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()

        if isinstance(self.rfe_, RFECV):
            return self.rfe_.n_features_
        else:
            return None

    def plot_cv_scores(self) -> Optional[dict]:
        """
        Get data for plotting cross-validation scores vs number of features.

        Returns:
            Dictionary with plotting data (x: n_features, y: cv_scores)
            None if use_cv=False

        Raises:
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()

        if isinstance(self.rfe_, RFECV):
            cv_scores = self.rfe_.cv_results_['mean_test_score']
            n_features_range = range(1, len(cv_scores) + 1)

            return {
                'n_features': list(n_features_range),
                'cv_scores': cv_scores.tolist(),
                'optimal_n_features': self.rfe_.n_features_,
                'optimal_score': cv_scores[self.rfe_.n_features_ - 1]
            }
        else:
            return None

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance summary (alias for get_feature_ranking).

        Returns:
            DataFrame with features and their rankings

        Raises:
            RuntimeError: If selector has not been fitted
        """
        return self.get_feature_ranking()

    def get_rfe_summary(self) -> dict:
        """
        Get comprehensive summary of RFE results.

        Returns:
            Dictionary with RFE statistics and metadata

        Raises:
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()

        summary = {
            'total_features': len(self.feature_names_),
            'selected_features': len(self.selected_features_),
            'selection_rate': len(self.selected_features_) / len(self.feature_names_),
            'n_features_selected': self.n_features_,
            'estimator': str(self.estimator_.__class__.__name__),
            'used_cv': isinstance(self.rfe_, RFECV),
            'step': self.params['step']
        }

        if isinstance(self.rfe_, RFECV):
            cv_scores = self.rfe_.cv_results_['mean_test_score']
            summary['cv_scores'] = {
                'optimal_score': float(cv_scores[self.n_features_ - 1]),
                'best_score': float(cv_scores.max()),
                'worst_score': float(cv_scores.min()),
                'mean_score': float(cv_scores.mean())
            }
            summary['cv_folds'] = self.params['cv']

        return summary

    def __repr__(self) -> str:
        """String representation."""
        estimator_name = self.params.get("estimator", "auto")
        n_features = self.params.get("n_features_to_select", "auto")

        if self.fitted and self.selected_features_:
            n_selected = len(self.selected_features_)
            n_total = len(self.feature_names_)
            cv_info = " (CV)" if isinstance(self.rfe_, RFECV) else ""
            return (
                f"RFESelector(estimator='{estimator_name}', "
                f"selected={n_selected}/{n_total}{cv_info})"
            )
        else:
            return f"RFESelector(estimator='{estimator_name}', n_features={n_features}, not fitted)"
