"""Lightweight Grid Search wrapper around sklearn's GridSearchCV."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
from sklearn.model_selection import GridSearchCV

from app.ml_engine.tuning.search_spaces import get_default_search_space


ParamGrid = Union[Dict[str, Iterable[Any]], List[Dict[str, Iterable[Any]]]]


@dataclass
class GridSearchResult:
	"""Structured grid search output suitable for APIs and logging."""

	best_params: Dict[str, Any]
	best_score: float
	scoring: Optional[str]
	cv_folds: int
	n_candidates: int
	results: List[Dict[str, Any]]

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


def run_grid_search(
	estimator: Any,
	X: Any,
	y: Any,
	param_grid: Optional[ParamGrid] = None,
	model_id: Optional[str] = None,
	scoring: Optional[str] = None,
	cv: int = 5,
	n_jobs: int = -1,
	return_train_score: bool = True,
	verbose: int = 0,
) -> GridSearchResult:
	"""Execute a grid search with sane defaults and structured output.

	If `param_grid` is not provided, the function will try to load a default
	search space using `model_id`. If neither is available, a ValueError is
	raised.
	"""

	resolved_grid = param_grid
	if resolved_grid is None:
		if model_id is None:
			raise ValueError("param_grid is required when model_id is not provided")
		resolved_grid = get_default_search_space(model_id)
		if not resolved_grid:
			raise ValueError(f"No default search space defined for model_id='{model_id}'")

	search = GridSearchCV(
		estimator=estimator,
		param_grid=resolved_grid,
		cv=cv,
		scoring=scoring,
		n_jobs=n_jobs,
		return_train_score=return_train_score,
		verbose=verbose,
	)

	search.fit(X, y)

	cv_results = search.cv_results_
	sorted_indices = np.argsort(cv_results["mean_test_score"])[::-1]

	records: List[Dict[str, Any]] = []
	grid_keys = resolved_grid if isinstance(resolved_grid, dict) else resolved_grid[0]

	for rank, idx in enumerate(sorted_indices, start=1):
		record = {
			"rank": rank,
			"params": {key: cv_results[f"param_{key}"][idx] for key in grid_keys},
			"mean_score": float(cv_results["mean_test_score"][idx]),
			"std_score": float(cv_results["std_test_score"][idx]),
			"scores": [
				float(cv_results[f"split{fold}_test_score"][idx]) for fold in range(cv)
			],
		}
		records.append(record)

	return GridSearchResult(
		best_params=search.best_params_,
		best_score=float(search.best_score_),
		scoring=scoring,
		cv_folds=cv,
		n_candidates=len(sorted_indices),
		results=records,
	)
