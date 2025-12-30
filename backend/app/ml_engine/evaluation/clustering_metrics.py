"""
Clustering metrics utilities.

Provides silhouette, Calinski-Harabasz, Davies-Bouldin scores, inertia
calculation, and cluster size summaries for unsupervised models.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Union

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from app.utils.logger import get_logger

logger = get_logger("clustering_metrics")

ArrayLike = Union[np.ndarray, Iterable[float]]


@dataclass
class ClusteringMetricsResult:
    """Container for clustering evaluation metrics."""

    silhouette: Optional[float]
    calinski_harabasz: Optional[float]
    davies_bouldin: Optional[float]
    inertia: Optional[float]
    n_clusters: int
    n_samples: int
    cluster_sizes: Dict[str, int]
    noise_points: int

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""

        return asdict(self)


class ClusteringMetricsCalculator:
    """Calculator for clustering metrics."""

    def __init__(
        self,
        silhouette_metric: str = "euclidean",
        sample_size: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.silhouette_metric = silhouette_metric
        self.sample_size = sample_size
        self.random_state = random_state
        logger.debug(
            "Initialized ClusteringMetricsCalculator metric=%s sample_size=%s",
            silhouette_metric,
            sample_size,
        )

    def calculate(
        self,
        X: np.ndarray,
        labels: ArrayLike,
        model: Optional[object] = None,
    ) -> ClusteringMetricsResult:
        """Calculate clustering metrics.

        Args:
            X: Feature matrix (n_samples, n_features)
            labels: Cluster labels for each sample
            model: Optional fitted clustering model (used for inertia if available)
        """

        X_arr = self._validate_X(X)
        labels_arr = self._validate_labels(labels, X_arr.shape[0])

        unique_labels = np.unique(labels_arr)
        n_clusters = int(np.sum(unique_labels != -1)) if -1 in unique_labels else len(unique_labels)
        n_samples = X_arr.shape[0]

        logger.info(
            "Calculating clustering metrics: n_samples=%s, n_clusters=%s",
            n_samples,
            n_clusters,
        )

        silhouette = self._safe_silhouette(X_arr, labels_arr)
        ch_score = self._safe_calinski_harabasz(X_arr, labels_arr)
        db_score = self._safe_davies_bouldin(X_arr, labels_arr)
        inertia = self._safe_inertia(model, X_arr, labels_arr)

        cluster_sizes, noise_points = self._cluster_size_summary(labels_arr)

        result = ClusteringMetricsResult(
            silhouette=silhouette,
            calinski_harabasz=ch_score,
            davies_bouldin=db_score,
            inertia=inertia,
            n_clusters=n_clusters,
            n_samples=n_samples,
            cluster_sizes=cluster_sizes,
            noise_points=noise_points,
        )

        logger.info(
            "Clustering metrics done: silhouette=%s, CH=%s, DB=%s, inertia=%s",
            f"{silhouette:.4f}" if silhouette is not None else "N/A",
            f"{ch_score:.2f}" if ch_score is not None else "N/A",
            f"{db_score:.2f}" if db_score is not None else "N/A",
            f"{inertia:.2f}" if inertia is not None else "N/A",
        )
        return result

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array-like of shape (n_samples, n_features)")
        if X_arr.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("X must contain only finite values")
        return X_arr

    def _validate_labels(self, labels: ArrayLike, n_samples: int) -> np.ndarray:
        labels_arr = np.asarray(labels)
        if labels_arr.shape[0] != n_samples:
            raise ValueError(
                f"labels length ({labels_arr.shape[0]}) must match n_samples ({n_samples})"
            )
        if labels_arr.ndim != 1:
            raise ValueError("labels must be 1D")
        return labels_arr

    def _safe_silhouette(self, X: np.ndarray, labels: np.ndarray) -> Optional[float]:
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return None
        if len(unique_labels) == 1 and unique_labels[0] == -1:
            return None
        try:
            return float(
                silhouette_score(
                    X,
                    labels,
                    metric=self.silhouette_metric,
                    sample_size=self.sample_size,
                    random_state=self.random_state,
                )
            )
        except Exception as exc:
            logger.debug("Silhouette score failed: %s", exc)
            return None

    def _safe_calinski_harabasz(self, X: np.ndarray, labels: np.ndarray) -> Optional[float]:
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return None
        try:
            return float(calinski_harabasz_score(X, labels))
        except Exception as exc:
            logger.debug("Calinski-Harabasz failed: %s", exc)
            return None

    def _safe_davies_bouldin(self, X: np.ndarray, labels: np.ndarray) -> Optional[float]:
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return None
        try:
            return float(davies_bouldin_score(X, labels))
        except Exception as exc:
            logger.debug("Davies-Bouldin failed: %s", exc)
            return None

    def _safe_inertia(
        self, model: Optional[object], X: np.ndarray, labels: np.ndarray
    ) -> Optional[float]:
        if model is not None and hasattr(model, "inertia_"):
            try:
                return float(model.inertia_)
            except Exception as exc:
                logger.debug("Model inertia retrieval failed: %s", exc)

        # Manual inertia (sum of squared distances to cluster centroids)
        unique_labels = np.unique(labels)
        if len(unique_labels) == 0:
            return None

        try:
            inertia = 0.0
            for label in unique_labels:
                mask = labels == label
                if not np.any(mask):
                    continue
                centroid = X[mask].mean(axis=0)
                inertia += float(np.sum((X[mask] - centroid) ** 2))
            return inertia
        except Exception as exc:
            logger.debug("Manual inertia computation failed: %s", exc)
            return None

    def _cluster_size_summary(self, labels: np.ndarray) -> tuple[Dict[str, int], int]:
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {str(int(lbl)): int(cnt) for lbl, cnt in zip(unique, counts) if lbl != -1}
        noise_points = int(counts[unique == -1][0]) if -1 in unique else 0
        return cluster_sizes, noise_points


def calculate_clustering_metrics(
    X: np.ndarray,
    labels: ArrayLike,
    model: Optional[object] = None,
    silhouette_metric: str = "euclidean",
    sample_size: Optional[int] = None,
    random_state: Optional[int] = None,
) -> ClusteringMetricsResult:
    """Convenience function to compute clustering metrics."""

    calculator = ClusteringMetricsCalculator(
        silhouette_metric=silhouette_metric,
        sample_size=sample_size,
        random_state=random_state,
    )
    return calculator.calculate(X=X, labels=labels, model=model)
