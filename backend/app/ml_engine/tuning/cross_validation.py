"""Cross-validation utilities for model evaluation."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.model_selection import (
    cross_validate,
    cross_val_score,
    KFold,
    StratifiedKFold,
    GroupKFold,
    TimeSeriesSplit,
)


@dataclass
class CrossValidationResult:
    """Structured cross-validation output with scores and metrics."""

    scores: List[float]
    mean_score: float
    std_score: float
    median_score: float
    min_score: float
    max_score: float
    scoring: Optional[str]
    cv_folds: int
    fit_times: Optional[List[float]] = None
    score_times: Optional[List[float]] = None
    train_scores: Optional[List[float]] = None
    additional_metrics: Optional[Dict[str, List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["scores"] = [float(s) for s in self.scores]
        data["mean_score"] = float(self.mean_score)
        data["std_score"] = float(self.std_score)
        data["median_score"] = float(self.median_score)
        data["min_score"] = float(self.min_score)
        data["max_score"] = float(self.max_score)

        if self.fit_times:
            data["fit_times"] = [float(t) for t in self.fit_times]
            data["mean_fit_time"] = float(np.mean(self.fit_times))

        if self.score_times:
            data["score_times"] = [float(t) for t in self.score_times]
            data["mean_score_time"] = float(np.mean(self.score_times))

        if self.train_scores:
            data["train_scores"] = [float(s) for s in self.train_scores]
            data["mean_train_score"] = float(np.mean(self.train_scores))

        if self.additional_metrics:
            formatted_metrics = {}
            for metric_name, metric_scores in self.additional_metrics.items():
                formatted_metrics[metric_name] = {
                    "scores": [float(s) for s in metric_scores],
                    "mean": float(np.mean(metric_scores)),
                    "std": float(np.std(metric_scores)),
                }
            data["additional_metrics"] = formatted_metrics

        return data

    @property
    def confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """Calculate confidence interval for scores."""
        from scipy import stats

        ci = stats.t.interval(
            confidence,
            len(self.scores) - 1,
            loc=self.mean_score,
            scale=stats.sem(self.scores),
        )
        return (float(ci[0]), float(ci[1]))


def run_cross_validation(
    estimator: Any,
    X: Any,
    y: Any,
    scoring: Optional[Union[str, List[str]]] = None,
    cv: Union[int, Any] = 5,
    groups: Optional[Any] = None,
    return_train_score: bool = False,
    return_estimator: bool = False,
    n_jobs: int = -1,
    verbose: int = 0,
) -> CrossValidationResult:
    """Execute cross-validation with comprehensive metrics.

    Args:
        estimator: Sklearn-compatible estimator
        X: Training features
        y: Training targets
        scoring: Scoring metric(s). If list, first is primary.
        cv: CV splitter or number of folds (default: 5)
        groups: Group labels for GroupKFold
        return_train_score: Whether to compute train scores
        return_estimator: Whether to return fitted estimators
        n_jobs: Number of parallel jobs
        verbose: Verbosity level

    Returns:
        CrossValidationResult with scores and metadata
    """

    # Handle multiple scoring metrics
    primary_scoring = None
    if isinstance(scoring, list):
        primary_scoring = scoring[0]
        scoring_dict = scoring
    elif isinstance(scoring, str):
        primary_scoring = scoring
        scoring_dict = [scoring]
    else:
        scoring_dict = None

    cv_results = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        groups=groups,
        scoring=scoring_dict,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=return_train_score,
        return_estimator=return_estimator,
    )

    # Extract test scores
    if scoring_dict and len(scoring_dict) > 1:
        test_scores = cv_results[f"test_{primary_scoring}"]
    elif scoring_dict:
        test_scores = cv_results[f"test_{scoring_dict[0]}"]
    else:
        test_scores = cv_results["test_score"]

    # Build additional metrics dict
    additional_metrics = None
    if scoring_dict and len(scoring_dict) > 1:
        additional_metrics = {}
        for metric in scoring_dict[1:]:
            additional_metrics[metric] = cv_results[f"test_{metric}"].tolist()

    # Extract train scores if requested
    train_scores = None
    if return_train_score:
        if scoring_dict and len(scoring_dict) > 0:
            train_scores = cv_results[f"train_{scoring_dict[0]}"].tolist()
        else:
            train_scores = cv_results["train_score"].tolist()

    result = CrossValidationResult(
        scores=test_scores.tolist(),
        mean_score=float(np.mean(test_scores)),
        std_score=float(np.std(test_scores)),
        median_score=float(np.median(test_scores)),
        min_score=float(np.min(test_scores)),
        max_score=float(np.max(test_scores)),
        scoring=primary_scoring,
        cv_folds=len(test_scores),
        fit_times=cv_results["fit_time"].tolist(),
        score_times=cv_results["score_time"].tolist(),
        train_scores=train_scores,
        additional_metrics=additional_metrics,
    )

    return result


def run_simple_cross_validation(
    estimator: Any,
    X: Any,
    y: Any,
    scoring: Optional[str] = None,
    cv: int = 5,
    n_jobs: int = -1,
) -> List[float]:
    """Simplified cross-validation returning only scores."""

    scores = cross_val_score(
        estimator=estimator,
        X=X,
        y=y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
    )
    return scores.tolist()


def create_cv_splitter(
    cv_type: str = "kfold",
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Factory for creating CV splitters.

    Args:
        cv_type: Type of splitter ('kfold', 'stratified', 'group', 'timeseries')
        n_splits: Number of folds
        shuffle: Whether to shuffle data (not used for timeseries)
        random_state: Random seed
        **kwargs: Additional splitter-specific arguments

    Returns:
        Configured CV splitter
    """

    cv_type = cv_type.lower()

    if cv_type == "kfold":
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    elif cv_type == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    elif cv_type == "group":
        return GroupKFold(n_splits=n_splits)
    elif cv_type == "timeseries":
        max_train_size = kwargs.get("max_train_size", None)
        test_size = kwargs.get("test_size", None)
        gap = kwargs.get("gap", 0)
        return TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=max_train_size,
            test_size=test_size,
            gap=gap,
        )
    else:
        raise ValueError(
            f"Unknown cv_type: {cv_type}. "
            "Choose from: kfold, stratified, group, timeseries"
        )


def compare_models_cv(
    estimators: Dict[str, Any],
    X: Any,
    y: Any,
    scoring: Optional[str] = None,
    cv: int = 5,
    n_jobs: int = -1,
) -> Dict[str, CrossValidationResult]:
    """Compare multiple models using cross-validation.

    Args:
        estimators: Dict mapping model names to estimators
        X: Training features
        y: Training targets
        scoring: Scoring metric
        cv: Number of folds or CV splitter
        n_jobs: Number of parallel jobs

    Returns:
        Dict mapping model names to CrossValidationResult
    """

    results = {}
    for name, estimator in estimators.items():
        results[name] = run_cross_validation(
            estimator=estimator,
            X=X,
            y=y,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
        )
    return results
