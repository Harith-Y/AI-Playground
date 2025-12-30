"""Tests for clustering metrics utilities."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from app.ml_engine.evaluation.clustering_metrics import (
    ClusteringMetricsCalculator,
    calculate_clustering_metrics,
)


def test_clustering_metrics_basic():
    X, labels_true = make_blobs(n_samples=50, centers=2, cluster_std=0.60, random_state=42)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
    labels = kmeans.labels_

    result = calculate_clustering_metrics(X, labels, model=kmeans)

    assert result.n_clusters == 2
    assert result.n_samples == 50
    assert result.silhouette is not None and result.silhouette > 0.5
    assert result.calinski_harabasz is not None
    assert result.davies_bouldin is not None
    assert result.inertia is not None
    assert result.cluster_sizes == {"0": 25, "1": 25}
    assert result.noise_points == 0


def test_handles_noise_label():
    X, _ = make_blobs(n_samples=20, centers=2, cluster_std=0.5, random_state=0)
    labels = np.array([0] * 10 + [1] * 8 + [-1] * 2)

    calc = ClusteringMetricsCalculator()
    result = calc.calculate(X, labels)

    assert result.noise_points == 2
    assert result.cluster_sizes == {"0": 10, "1": 8}


def test_sample_size_and_random_state():
    X, labels = make_blobs(n_samples=30, centers=3, cluster_std=0.4, random_state=5)

    calc = ClusteringMetricsCalculator(sample_size=20, random_state=123)
    result = calc.calculate(X, labels)

    assert result.silhouette is not None
    # sample_size still returns a score in range [-1, 1]
    assert -1.0 <= result.silhouette <= 1.0


def test_validation_errors_for_shape_mismatch():
    X, labels = make_blobs(n_samples=10, centers=2, random_state=1)
    calc = ClusteringMetricsCalculator()

    with pytest.raises(ValueError):
        calc.calculate(X[:5], labels)


def test_inertia_manual_computation_without_model():
    X, labels = make_blobs(n_samples=15, centers=3, cluster_std=0.3, random_state=7)

    result = calculate_clustering_metrics(X, labels, model=None)

    assert result.inertia is not None
    assert result.inertia > 0
