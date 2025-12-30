"""
Actual vs Predicted aggregation utilities for regression outputs.

Summarizes paired actual/predicted values for scatter plotting,
basic error statistics, simple correlation diagnostics, and a
best-fit line for quick visualization. Implements ML-52.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

from app.utils.logger import get_logger

logger = get_logger("actual_vs_predicted")

ArrayLike = Union[List[float], np.ndarray]


@dataclass
class ActualVsPredictedResult:
    """Container for actual vs predicted aggregation."""

    actual: np.ndarray
    predicted: np.ndarray
    residuals: np.ndarray
    mae: float
    mse: float
    rmse: float
    mape: Optional[float]
    mean_error: float
    median_error: float
    r2: Optional[float]
    pearson_r: Optional[float]
    spearman_rho: Optional[float]
    best_fit: Optional[Dict[str, float]]
    n_samples: int
    min_actual: float
    max_actual: float
    min_predicted: float
    max_predicted: float

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""

        result = asdict(self)
        result["actual"] = self.actual.tolist()
        result["predicted"] = self.predicted.tolist()
        result["residuals"] = self.residuals.tolist()
        return result

    @property
    def series(self) -> Dict[str, List[float]]:
        """Return paired series for plotting."""

        return {
            "actual": self.actual.tolist(),
            "predicted": self.predicted.tolist(),
            "residual": self.residuals.tolist(),
        }


class ActualVsPredictedAggregator:
    """Aggregates paired actual/predicted values with diagnostics."""

    def __init__(self, ddof: int = 1) -> None:
        self.ddof = ddof
        logger.debug("Initialized ActualVsPredictedAggregator with ddof=%s", ddof)

    def aggregate(
        self,
        actual: ArrayLike,
        predicted: ArrayLike,
        compute_rank_corr: bool = True,
    ) -> ActualVsPredictedResult:
        """Aggregate actual vs predicted values with error statistics."""

        actual_arr, predicted_arr = self._validate_inputs(actual, predicted)
        residuals = actual_arr - predicted_arr

        logger.info("Aggregating actual vs predicted for %s samples", len(residuals))

        mae = float(np.mean(np.abs(residuals)))
        mse = float(np.mean(np.square(residuals)))
        rmse = float(np.sqrt(mse))
        mean_error = float(residuals.mean())
        median_error = float(np.median(residuals))
        mape = self._safe_mape(residuals, actual_arr)

        r2 = self._safe_r2(actual_arr, predicted_arr)
        pearson_r = self._pearson(actual_arr, predicted_arr)
        spearman_rho = self._spearman(actual_arr, predicted_arr) if compute_rank_corr else None
        best_fit = self._best_fit_line(predicted_arr, actual_arr)

        result = ActualVsPredictedResult(
            actual=actual_arr,
            predicted=predicted_arr,
            residuals=residuals,
            mae=mae,
            mse=mse,
            rmse=rmse,
            mape=mape,
            mean_error=mean_error,
            median_error=median_error,
            r2=r2,
            pearson_r=pearson_r,
            spearman_rho=spearman_rho,
            best_fit=best_fit,
            n_samples=len(residuals),
            min_actual=float(actual_arr.min()),
            max_actual=float(actual_arr.max()),
            min_predicted=float(predicted_arr.min()),
            max_predicted=float(predicted_arr.max()),
        )

        logger.info(
            "Aggregation complete: MAE=%.4f, RMSE=%.4f, R2=%s",
            result.mae,
            result.rmse,
            f"{result.r2:.4f}" if result.r2 is not None else "N/A",
        )
        return result

    def _validate_inputs(self, actual: ArrayLike, predicted: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
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

    def _best_fit_line(self, x: np.ndarray, y: np.ndarray) -> Optional[Dict[str, float]]:
        # Fit y = slope * x + intercept
        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom == 0:
            return None
        slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
        intercept = float(y_mean - slope * x_mean)
        return {"slope": slope, "intercept": intercept}

    def _pearson(self, actual: np.ndarray, predicted: np.ndarray) -> Optional[float]:
        if actual.size < 2:
            return None
        if np.all(actual == actual[0]) or np.all(predicted == predicted[0]):
            return None
        try:
            corr, _ = stats.pearsonr(predicted, actual)
            return float(corr)
        except Exception as exc:
            logger.debug("Pearson correlation failed: %s", exc)
            return None

    def _spearman(self, actual: np.ndarray, predicted: np.ndarray) -> Optional[float]:
        if actual.size < 2:
            return None
        try:
            corr, _ = stats.spearmanr(predicted, actual)
            return float(corr)
        except Exception as exc:
            logger.debug("Spearman correlation failed: %s", exc)
            return None

    def _safe_r2(self, actual: np.ndarray, predicted: np.ndarray) -> Optional[float]:
        try:
            return float(r2_score(actual, predicted))
        except Exception as exc:
            logger.debug("R2 computation failed: %s", exc)
            return None

    def _safe_mape(self, residuals: np.ndarray, actual: np.ndarray) -> Optional[float]:
        non_zero_mask = actual != 0
        if not np.any(non_zero_mask):
            return None
        return float(np.mean(np.abs(residuals[non_zero_mask] / actual[non_zero_mask])) * 100)


def aggregate_actual_vs_predicted(
    actual: ArrayLike,
    predicted: ArrayLike,
    compute_rank_corr: bool = True,
) -> ActualVsPredictedResult:
    """Convenience function to aggregate actual vs predicted data."""

    aggregator = ActualVsPredictedAggregator()
    return aggregator.aggregate(
        actual=actual,
        predicted=predicted,
        compute_rank_corr=compute_rank_corr,
    )
