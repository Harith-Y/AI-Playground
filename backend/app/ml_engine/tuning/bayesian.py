"""Bayesian optimization wrapper with scikit-optimize integration."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import warnings

import numpy as np

from app.ml_engine.tuning.search_spaces import get_default_search_space


@dataclass
class BayesianSearchResult:
    """Structured Bayesian search output suitable for APIs and logging."""

    best_params: Dict[str, Any]
    best_score: float
    scoring: Optional[str]
    cv_folds: int
    n_iter: int
    n_candidates: int
    results: List[Dict[str, Any]]
    method: str  # 'bayesian' or 'random_fallback'

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["best_score"] = float(self.best_score)
        for item in data["results"]:
            item["mean_score"] = float(item["mean_score"])
            item["std_score"] = float(item["std_score"])
            item["scores"] = [float(s) for s in item["scores"]]
        return data

    def top(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return top-n results (already sorted by mean score)."""

        return self.results[:n]


def run_bayesian_search(
    estimator: Any,
    X: Any,
    y: Any,
    search_spaces: Optional[Dict[str, Any]] = None,
    model_id: Optional[str] = None,
    n_iter: int = 32,
    scoring: Optional[str] = None,
    cv: int = 5,
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    return_train_score: bool = True,
    verbose: int = 0,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> BayesianSearchResult:
    """Execute Bayesian optimization with scikit-optimize or fallback.

    Attempts to use BayesSearchCV from scikit-optimize. If not available,
    falls back to RandomizedSearchCV with a warning.

    Args:
        estimator: Sklearn-compatible estimator
        X: Training features
        y: Training targets
        search_spaces: Parameter search spaces (supports skopt spaces)
        model_id: Model ID for default search space
        n_iter: Number of optimization iterations
        scoring: Scoring metric
        cv: Number of folds or CV splitter
        n_jobs: Number of parallel jobs
        random_state: Random seed
        return_train_score: Whether to compute train scores
        verbose: Verbosity level
        optimizer_kwargs: Additional kwargs for BayesSearchCV optimizer

    Returns:
        BayesianSearchResult with optimization results
    """

    resolved_spaces = search_spaces
    if resolved_spaces is None:
        if model_id is None:
            raise ValueError("search_spaces is required when model_id is not provided")
        resolved_spaces = get_default_search_space(model_id)
        if not resolved_spaces:
            raise ValueError(f"No default search space defined for model_id='{model_id}'")

    # Try to use scikit-optimize's BayesSearchCV
    try:
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer, Categorical

        # Convert lists to skopt space objects for better optimization
        converted_spaces = {}
        for param, values in resolved_spaces.items():
            if isinstance(values, list):
                if all(isinstance(v, (int, np.integer)) for v in values):
                    converted_spaces[param] = Integer(min(values), max(values))
                elif all(isinstance(v, (float, np.floating)) for v in values):
                    converted_spaces[param] = Real(min(values), max(values), prior='log-uniform')
                else:
                    converted_spaces[param] = Categorical(values)
            else:
                # Already a skopt space object
                converted_spaces[param] = values

        optimizer_kwargs = optimizer_kwargs or {}
        search = BayesSearchCV(
            estimator=estimator,
            search_spaces=converted_spaces,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            return_train_score=return_train_score,
            verbose=verbose,
            **optimizer_kwargs,
        )
        method = "bayesian"

    except ImportError:
        warnings.warn(
            "scikit-optimize not installed. Falling back to RandomizedSearchCV. "
            "Install scikit-optimize for true Bayesian optimization: pip install scikit-optimize",
            UserWarning,
        )
        from sklearn.model_selection import RandomizedSearchCV

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=resolved_spaces,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            return_train_score=return_train_score,
            verbose=verbose,
        )
        method = "random_fallback"

    search.fit(X, y)

    cv_results = search.cv_results_
    sorted_indices = np.argsort(cv_results["mean_test_score"])[::-1]

    records: List[Dict[str, Any]] = []
    param_keys = list(resolved_spaces.keys())

    for rank, idx in enumerate(sorted_indices, start=1):
        record = {
            "rank": rank,
            "params": {key: cv_results[f"param_{key}"][idx] for key in param_keys},
            "mean_score": float(cv_results["mean_test_score"][idx]),
            "std_score": float(cv_results["std_test_score"][idx]),
            "scores": [
                float(cv_results[f"split{fold}_test_score"][idx]) for fold in range(cv)
            ],
        }
        records.append(record)

    return BayesianSearchResult(
        best_params=search.best_params_,
        best_score=float(search.best_score_),
        scoring=scoring,
        cv_folds=cv,
        n_iter=n_iter,
        n_candidates=len(sorted_indices),
        results=records,
        method=method,
    )
