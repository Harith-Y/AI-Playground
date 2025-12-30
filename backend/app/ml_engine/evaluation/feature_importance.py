"""
Feature importance calculators for model interpretability.

Supports native model importances (feature_importances_ / coef_),
permutation importance, and a SHAP-friendly hook (optional dependency).
Implements ML-54/ML-55.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.inspection import permutation_importance

from app.utils.logger import get_logger

logger = get_logger("feature_importance")

ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass
class FeatureImportanceResult:
    """Container for feature importance outputs."""

    importances: Dict[str, float]
    method: str
    feature_names: List[str]
    n_features: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable mapping."""

        result = asdict(self)
        # Ensure plain floats for JSON
        result["importances"] = {k: float(v) for k, v in self.importances.items()}
        return result

    def to_ranked_list(self) -> List[Dict[str, Union[str, float, int]]]:
        """Return a sorted list of feature importances with ranks."""

        ranked = sorted(self.importances.items(), key=lambda kv: kv[1], reverse=True)
        return [
            {"feature": name, "importance": float(value), "rank": idx + 1}
            for idx, (name, value) in enumerate(ranked)
        ]


class FeatureImportanceCalculator:
    """Calculate feature importance using native, permutation, or SHAP methods."""

    def __init__(
        self,
        scoring: Optional[str] = None,
        n_repeats: int = 5,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        use_abs_coefficients: bool = True,
    ) -> None:
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.use_abs_coefficients = use_abs_coefficients
        logger.debug(
            "Initialized FeatureImportanceCalculator with scoring=%s, n_repeats=%s, random_state=%s, n_jobs=%s",
            scoring,
            n_repeats,
            random_state,
            n_jobs,
        )

    def calculate(
        self,
        estimator: Any,
        X: Union[np.ndarray, "pd.DataFrame"],
        y: Optional[ArrayLike] = None,
        feature_names: Optional[Sequence[str]] = None,
        method: str = "auto",
    ) -> FeatureImportanceResult:
        """Compute feature importances.

        Args:
            estimator: Fitted estimator.
            X: Training features (NumPy array or DataFrame).
            y: Targets (required for permutation importance or SHAP).
            feature_names: Optional names; inferred from DataFrame columns if not provided.
            method: 'auto', 'model', 'permutation', or 'shap'.

        Returns:
            FeatureImportanceResult with importance mapping and metadata.
        """

        X_array, names = self._prepare_features(X, feature_names)
        method = (method or "auto").lower()

        if method not in {"auto", "model", "permutation", "shap"}:
            raise ValueError("method must be one of: auto, model, permutation, shap")

        logger.info("Calculating feature importance using method=%s", method)

        if method in {"auto", "model"}:
            native = self._from_model_attributes(estimator, names)
            if native is not None:
                return native
            if method == "model":
                raise ValueError("Estimator does not expose feature_importances_ or coef_")
            # Fall back to permutation if auto
            return self._permutation_importance(estimator, X_array, y, names)

        if method == "permutation":
            return self._permutation_importance(estimator, X_array, y, names)

        # method == "shap"
        return self._shap_importance(estimator, X_array, y, names)

    def _prepare_features(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        feature_names: Optional[Sequence[str]],
    ) -> tuple[np.ndarray, List[str]]:
        try:
            import pandas as pd  # type: ignore
        except ImportError:  # pragma: no cover - pandas present in project deps
            pd = None  # type: ignore

        if pd is not None and isinstance(X, pd.DataFrame):
            names = list(feature_names) if feature_names is not None else list(X.columns)
            X_array = X.values
        else:
            X_array = np.asarray(X)
            n_features = X_array.shape[1] if X_array.ndim == 2 else X_array.size
            names = list(feature_names) if feature_names is not None else [f"feature_{i}" for i in range(n_features)]

        if X_array.ndim != 2:
            raise ValueError("X must be a 2D array or DataFrame")

        if len(names) != X_array.shape[1]:
            raise ValueError(
                f"feature_names length ({len(names)}) must match number of columns in X ({X_array.shape[1]})"
            )

        if not np.isfinite(X_array).all():
            raise ValueError("X must contain only finite values")

        return X_array, names

    def _validate_y(self, y: Optional[ArrayLike], n_samples: int) -> np.ndarray:
        if y is None:
            raise ValueError("y is required for permutation or SHAP importance")
        y_array = np.asarray(y)
        if y_array.shape[0] != n_samples:
            raise ValueError(
                f"y length ({y_array.shape[0]}) must match number of rows in X ({n_samples})"
            )
        if y_array.ndim != 1:
            y_array = y_array.reshape(-1)
        if not np.isfinite(y_array).all():
            raise ValueError("y must contain only finite values")
        return y_array

    def _from_model_attributes(
        self,
        estimator: Any,
        feature_names: List[str],
    ) -> Optional[FeatureImportanceResult]:
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_, dtype=float)
            if values.shape[0] != len(feature_names):
                raise ValueError(
                    "feature_importances_ length does not match provided feature names"
                )
            logger.debug("Using feature_importances_ from estimator")
            return self._build_result(values, feature_names, "feature_importances_", {})

        if hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_, dtype=float)
            if coef.ndim > 1:
                values = np.mean(np.abs(coef), axis=0) if self.use_abs_coefficients else np.mean(coef, axis=0)
            else:
                values = np.abs(coef) if self.use_abs_coefficients else coef
            if values.shape[0] != len(feature_names):
                raise ValueError("coef_ length does not match provided feature names")
            logger.debug("Using coef_ from estimator")
            return self._build_result(values, feature_names, "coef_", {})

        return None

    def _permutation_importance(
        self,
        estimator: Any,
        X_array: np.ndarray,
        y: Optional[ArrayLike],
        feature_names: List[str],
    ) -> FeatureImportanceResult:
        y_array = self._validate_y(y, X_array.shape[0])
        logger.debug(
            "Running permutation_importance with n_repeats=%s, random_state=%s", self.n_repeats, self.random_state
        )
        perm = permutation_importance(
            estimator,
            X_array,
            y_array,
            scoring=self.scoring,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        metadata = {
            "n_repeats": self.n_repeats,
            "random_state": self.random_state,
            "importances_std": perm.importances_std.tolist(),
        }
        return self._build_result(perm.importances_mean, feature_names, "permutation", metadata)

    def _shap_importance(
        self,
        estimator: Any,
        X_array: np.ndarray,
        y: Optional[ArrayLike],
        feature_names: List[str],
    ) -> FeatureImportanceResult:
        try:
            import shap  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "shap is required for SHAP-based feature importance. Install shap to use this method."
            ) from exc

        self._validate_y(y, X_array.shape[0])  # keep parity with permutation requirements

        logger.debug("Calculating SHAP values for feature importance")
        explainer = shap.Explainer(estimator, X_array)
        shap_values = explainer(X_array)
        # Aggregate mean absolute SHAP values across samples and outputs
        values = np.mean(np.abs(shap_values.values), axis=0)
        while values.ndim > 1:
            values = np.mean(values, axis=0)

        metadata = {"shap_type": explainer.__class__.__name__}
        return self._build_result(values, feature_names, "shap", metadata)

    def _build_result(
        self,
        values: np.ndarray,
        feature_names: List[str],
        method: str,
        metadata: Dict[str, Any],
    ) -> FeatureImportanceResult:
        values = np.asarray(values, dtype=float)
        if values.shape[0] != len(feature_names):
            raise ValueError("Importance values length must match feature names")

        importances = {name: float(val) for name, val in zip(feature_names, values)}
        return FeatureImportanceResult(
            importances=importances,
            method=method,
            feature_names=list(feature_names),
            n_features=len(feature_names),
            metadata=metadata,
        )


def calculate_feature_importance(
    estimator: Any,
    X: Union[np.ndarray, "pd.DataFrame"],
    y: Optional[ArrayLike] = None,
    feature_names: Optional[Sequence[str]] = None,
    method: str = "auto",
    scoring: Optional[str] = None,
    n_repeats: int = 5,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
) -> FeatureImportanceResult:
    """Convenience function to compute feature importances."""

    calculator = FeatureImportanceCalculator(
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return calculator.calculate(
        estimator=estimator,
        X=X,
        y=y,
        feature_names=feature_names,
        method=method,
    )
