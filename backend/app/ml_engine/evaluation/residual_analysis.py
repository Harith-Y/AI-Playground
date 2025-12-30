"""
Residual analysis utilities for regression models.

Provides reusable helpers to compute residuals, summary statistics,
standardized residuals, outlier flags, and normality diagnostics.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from scipy import stats

from app.utils.logger import get_logger

logger = get_logger("residual_analysis")

ArrayLike = Union[List[float], np.ndarray]


@dataclass
class ResidualAnalysisResult:
    """Container for residual analysis outputs."""

    residuals: np.ndarray
    predicted: np.ndarray
    actual: np.ndarray
    standardized_residuals: Optional[np.ndarray]
    mean_error: float
    median_error: float
    std_error: float
    mae: float
    mse: float
    rmse: float
    mape: Optional[float]
    min_error: float
    max_error: float
    quantiles: Dict[str, float]
    skewness: Optional[float]
    kurtosis: Optional[float]
    outlier_indices: List[int]
    outlier_threshold: Optional[float]
    n_samples: int
    correlation_abs_residuals_predicted: Optional[float]
    normality_test: Optional[Dict[str, float]]

    def to_dict(self) -> Dict:
        """Convert result to JSON-serializable dictionary."""

        result = asdict(self)
        result["residuals"] = self.residuals.tolist()
        result["predicted"] = self.predicted.tolist()
        result["actual"] = self.actual.tolist()

        if self.standardized_residuals is not None:
            result["standardized_residuals"] = self.standardized_residuals.tolist()

        return result

    @property
    def residual_series(self) -> Dict[str, List[float]]:
        """Lightweight dict useful for plotting."""

        return {
            "predicted": self.predicted.tolist(),
            "actual": self.actual.tolist(),
            "residual": self.residuals.tolist(),
        }


class ResidualAnalysisCalculator:
    """Residual analysis helper for regression outputs."""

    def __init__(
        self,
        zscore_threshold: float = 3.0,
        ddof: int = 1,
        include_standardized: bool = True,
    ) -> None:
        self.zscore_threshold = zscore_threshold
        self.ddof = ddof
        self.include_standardized = include_standardized
        logger.debug(
            "Initialized ResidualAnalysisCalculator with zscore_threshold=%s, ddof=%s",
            zscore_threshold,
            ddof,
        )

    def analyze(
        self,
        actual: ArrayLike,
        predicted: ArrayLike,
        include_standardized: Optional[bool] = None,
        compute_normality: bool = True,
    ) -> ResidualAnalysisResult:
        """Compute residual diagnostics for regression predictions."""

        include_standardized = (
            self.include_standardized
            if include_standardized is None
            else include_standardized
        )

        actual_arr, predicted_arr = self._validate_inputs(actual, predicted)
        residuals = actual_arr - predicted_arr

        logger.info("Calculating residual analysis for %s samples", len(residuals))

        stats_result = self._compute_basic_stats(residuals, actual_arr)
        standardized = (
            self._standardize_residuals(residuals, stats_result["std_error"])
            if include_standardized
            else None
        )
        outlier_indices = (
            self._detect_outliers(standardized, stats_result["std_error"])
            if include_standardized
            else self._detect_outliers(None, stats_result["std_error"], residuals)
        )
        correlation_abs = self._correlate_abs_residuals_predicted(residuals, predicted_arr)
        normality = self._run_normality_test(residuals) if compute_normality else None

        result = ResidualAnalysisResult(
            residuals=residuals,
            predicted=predicted_arr,
            actual=actual_arr,
            standardized_residuals=standardized,
            mean_error=stats_result["mean_error"],
            median_error=stats_result["median_error"],
            std_error=stats_result["std_error"],
            mae=stats_result["mae"],
            mse=stats_result["mse"],
            rmse=stats_result["rmse"],
            mape=stats_result["mape"],
            min_error=stats_result["min_error"],
            max_error=stats_result["max_error"],
            quantiles=stats_result["quantiles"],
            skewness=stats_result["skewness"],
            kurtosis=stats_result["kurtosis"],
            outlier_indices=outlier_indices,
            outlier_threshold=self.zscore_threshold if include_standardized else None,
            n_samples=len(residuals),
            correlation_abs_residuals_predicted=correlation_abs,
            normality_test=normality,
        )

        logger.info("Residual analysis complete: MAE=%.4f, RMSE=%.4f", result.mae, result.rmse)
        return result

    def _validate_inputs(
        self, actual: ArrayLike, predicted: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        actual_arr = np.asarray(actual, dtype=float)
        predicted_arr = np.asarray(predicted, dtype=float)

        if actual_arr.shape != predicted_arr.shape:
            raise ValueError(
                f"actual and predicted must have the same shape. Got {actual_arr.shape} and {predicted_arr.shape}"
            )

        if actual_arr.ndim != 1:
            raise ValueError("actual and predicted must be 1D arrays or lists")

        if actual_arr.size == 0:
            raise ValueError("actual and predicted cannot be empty")

        if not np.all(np.isfinite(actual_arr)) or not np.all(np.isfinite(predicted_arr)):
            raise ValueError("actual and predicted must contain only finite values")

        return actual_arr, predicted_arr

    def _compute_basic_stats(self, residuals: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
        n = len(residuals)
        mean_error = float(residuals.mean())
        median_error = float(np.median(residuals))
        std_error = float(
            residuals.std(ddof=self.ddof if n > self.ddof else 0)
        )
        mae = float(np.mean(np.abs(residuals)))
        mse = float(np.mean(np.square(residuals)))
        rmse = float(np.sqrt(mse))

        quantiles = {
            "q05": float(np.percentile(residuals, 5)),
            "q25": float(np.percentile(residuals, 25)),
            "q50": float(np.percentile(residuals, 50)),
            "q75": float(np.percentile(residuals, 75)),
            "q95": float(np.percentile(residuals, 95)),
        }

        mape = self._safe_mape(residuals, actual)

        skewness = float(stats.skew(residuals, bias=False)) if n > 2 else None
        kurtosis = float(stats.kurtosis(residuals, bias=False, fisher=True)) if n > 3 else None

        return {
            "mean_error": mean_error,
            "median_error": median_error,
            "std_error": std_error,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "min_error": float(residuals.min()),
            "max_error": float(residuals.max()),
            "quantiles": quantiles,
            "skewness": skewness,
            "kurtosis": kurtosis,
        }

    def _standardize_residuals(self, residuals: np.ndarray, std_error: float) -> Optional[np.ndarray]:
        if std_error == 0.0:
            return None
        return (residuals - residuals.mean()) / std_error

    def _detect_outliers(
        self,
        standardized_residuals: Optional[np.ndarray],
        std_error: float,
        residuals: Optional[np.ndarray] = None,
    ) -> List[int]:
        if standardized_residuals is not None:
            mask = np.abs(standardized_residuals) > self.zscore_threshold
            return np.where(mask)[0].tolist()

        if residuals is None or std_error == 0.0:
            return []

        threshold = self.zscore_threshold * std_error
        mask = np.abs(residuals) > threshold
        return np.where(mask)[0].tolist()

    def _correlate_abs_residuals_predicted(
        self, residuals: np.ndarray, predicted: np.ndarray
    ) -> Optional[float]:
        if residuals.size < 2:
            return None
        if np.all(predicted == predicted[0]):
            return None

        abs_res = np.abs(residuals)
        corr_matrix = np.corrcoef(abs_res, predicted)
        corr_value = float(corr_matrix[0, 1])
        return corr_value

    def _run_normality_test(self, residuals: np.ndarray) -> Optional[Dict[str, float]]:
        if residuals.size < 3:
            return None
        try:
            stat, p_value = stats.shapiro(residuals)
            return {"statistic": float(stat), "p_value": float(p_value)}
        except Exception as exc:
            logger.debug("Normality test failed: %s", exc)
            return None

    def _safe_mape(self, residuals: np.ndarray, actual: np.ndarray) -> Optional[float]:
        non_zero_mask = actual != 0
        if not np.any(non_zero_mask):
            return None
        return float(np.mean(np.abs(residuals[non_zero_mask] / actual[non_zero_mask])) * 100)


def analyze_residuals(
    actual: ArrayLike,
    predicted: ArrayLike,
    zscore_threshold: float = 3.0,
    ddof: int = 1,
    include_standardized: bool = True,
    compute_normality: bool = True,
) -> ResidualAnalysisResult:
    """Convenience function to run residual analysis in one call."""

    calculator = ResidualAnalysisCalculator(
        zscore_threshold=zscore_threshold,
        ddof=ddof,
        include_standardized=include_standardized,
    )
    return calculator.analyze(
        actual=actual,
        predicted=predicted,
        include_standardized=include_standardized,
        compute_normality=compute_normality,
    )
