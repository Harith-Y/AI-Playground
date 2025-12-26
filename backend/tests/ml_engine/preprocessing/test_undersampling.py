"""
Tests for undersampling preprocessing methods.
"""

import pytest
import pandas as pd
import numpy as np
from app.ml_engine.preprocessing.undersampling import (
    RandomUnderSampler,
    NearMissUnderSampler,
    TomekLinksRemover,
)


class TestRandomUnderSampler:
    """Tests for RandomUnderSampler class."""

    def test_initialization_default(self):
        """Test default initialization."""
        sampler = RandomUnderSampler()
        assert sampler.sampling_strategy == 'auto'
        assert sampler.random_state is None
        assert sampler.replacement is False

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        sampler = RandomUnderSampler(
            sampling_strategy='majority',
            random_state=42,
            replacement=True
        )
        assert sampler.sampling_strategy == 'majority'
        assert sampler.random_state == 42
        assert sampler.replacement is True

    def test_fit_resample_auto_strategy(self):
        """Test fit_resample with auto strategy balances classes."""
        # Create imbalanced dataset
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(100), np.random.randn(300) + 2]),
            'feature2': np.concatenate([np.random.randn(100) * 0.5, np.random.randn(300) * 0.5 + 1]),
        })
        y = pd.Series([0] * 100 + [1] * 300)

        sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Check that classes are balanced
        assert y_resampled.value_counts()[0] == y_resampled.value_counts()[1]
        assert y_resampled.value_counts()[0] == 100  # Balanced to minority class

        # Check that data types are preserved
        assert isinstance(X_resampled, pd.DataFrame)
        assert isinstance(y_resampled, pd.Series)

        # Check that all minority samples are kept
        assert y_resampled.value_counts()[0] == 100

    def test_fit_resample_majority_strategy(self):
        """Test fit_resample with majority strategy."""
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(100), np.random.randn(300)]),
        })
        y = pd.Series([0] * 100 + [1] * 300)

        sampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Check that minority class is unchanged
        assert y_resampled.value_counts()[0] == 100
        # Check that majority class is sampled to minority count
        assert y_resampled.value_counts()[1] == 100

    def test_fit_resample_ratio_strategy(self):
        """Test fit_resample with ratio strategy."""
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(100), np.random.randn(300)]),
        })
        y = pd.Series([0] * 100 + [1] * 300)

        # Use ratio of 0.5 (minority:majority = 1:2)
        sampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Check ratio
        minority_count = y_resampled.value_counts()[0]
        majority_count = y_resampled.value_counts()[1]
        assert minority_count == 100
        assert majority_count == 200  # 100 / 0.5 = 200

    def test_fit_resample_dict_strategy(self):
        """Test fit_resample with dictionary strategy."""
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(100), np.random.randn(300)]),
        })
        y = pd.Series([0] * 100 + [1] * 300)

        # Specify exact counts for each class
        sampler = RandomUnderSampler(sampling_strategy={0: 80, 1: 150}, random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Check exact counts
        assert y_resampled.value_counts()[0] == 80
        assert y_resampled.value_counts()[1] == 150

    def test_fit_resample_multiclass(self):
        """Test fit_resample with multi-class dataset."""
        X = pd.DataFrame({
            'feature1': np.concatenate([
                np.random.randn(50),
                np.random.randn(200),
                np.random.randn(100)
            ]),
        })
        y = pd.Series([0] * 50 + [1] * 200 + [2] * 100)

        sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # All classes should be balanced to minority (50)
        assert y_resampled.value_counts()[0] == 50
        assert y_resampled.value_counts()[1] == 50
        assert y_resampled.value_counts()[2] == 50

    def test_fit_resample_with_replacement(self):
        """Test fit_resample with replacement enabled."""
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(100), np.random.randn(300)]),
        })
        y = pd.Series([0] * 100 + [1] * 300)

        sampler = RandomUnderSampler(
            sampling_strategy='auto',
            random_state=42,
            replacement=True
        )
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Should still balance classes
        assert y_resampled.value_counts()[0] == y_resampled.value_counts()[1]

    def test_random_state_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(100), np.random.randn(300)]),
        })
        y = pd.Series([0] * 100 + [1] * 300)

        sampler1 = RandomUnderSampler(random_state=42)
        X_res1, y_res1 = sampler1.fit_resample(X, y)

        sampler2 = RandomUnderSampler(random_state=42)
        X_res2, y_res2 = sampler2.fit_resample(X, y)

        # Results should be identical
        pd.testing.assert_frame_equal(X_res1, X_res2)
        pd.testing.assert_series_equal(y_res1, y_res2)

    def test_fit_resample_numpy_arrays(self):
        """Test fit_resample with numpy arrays."""
        X = np.concatenate([np.random.randn(100, 2), np.random.randn(300, 2)])
        y = np.array([0] * 100 + [1] * 300)

        sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Check that output is numpy arrays
        assert isinstance(X_resampled, np.ndarray)
        assert isinstance(y_resampled, np.ndarray)

        # Check balancing
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert counts[0] == counts[1]

    def test_fit_resample_preserves_index(self):
        """Test that fit_resample maintains pandas index integrity."""
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(100), np.random.randn(300)]),
        }, index=range(400))
        y = pd.Series([0] * 100 + [1] * 300, index=range(400))

        sampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Check that X and y have matching indices
        pd.testing.assert_index_equal(X_resampled.index, y_resampled.index)

    def test_get_params(self):
        """Test get_params method."""
        sampler = RandomUnderSampler(
            sampling_strategy='majority',
            random_state=42,
            replacement=True
        )
        params = sampler.get_params()
        assert params['sampling_strategy'] == 'majority'
        assert params['random_state'] == 42
        assert params['replacement'] is True


class TestNearMissUnderSampler:
    """Tests for NearMissUnderSampler class."""

    def test_initialization_default(self):
        """Test default initialization."""
        sampler = NearMissUnderSampler()
        assert sampler.version == 1
        assert sampler.n_neighbors == 3
        assert sampler.sampling_strategy == 'auto'
        assert sampler.n_jobs == 1

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        sampler = NearMissUnderSampler(
            version=2,
            n_neighbors=5,
            sampling_strategy='majority',
            n_jobs=2
        )
        assert sampler.version == 2
        assert sampler.n_neighbors == 5
        assert sampler.sampling_strategy == 'majority'
        assert sampler.n_jobs == 2

    def test_version_1_undersampling(self):
        """Test NearMiss version 1."""
        # Create linearly separable imbalanced data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([
                np.random.randn(50) - 2,  # Class 0 (minority)
                np.random.randn(200) + 2   # Class 1 (majority)
            ]),
            'feature2': np.concatenate([
                np.random.randn(50) - 2,
                np.random.randn(200) + 2
            ]),
        })
        y = pd.Series([0] * 50 + [1] * 200)

        sampler = NearMissUnderSampler(version=1, n_neighbors=3, random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Check balancing
        assert y_resampled.value_counts()[0] == y_resampled.value_counts()[1]
        assert y_resampled.value_counts()[0] == 50

        # Check data types
        assert isinstance(X_resampled, pd.DataFrame)
        assert isinstance(y_resampled, pd.Series)

    def test_version_2_undersampling(self):
        """Test NearMiss version 2."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([
                np.random.randn(50) - 2,
                np.random.randn(200) + 2
            ]),
            'feature2': np.concatenate([
                np.random.randn(50) - 2,
                np.random.randn(200) + 2
            ]),
        })
        y = pd.Series([0] * 50 + [1] * 200)

        sampler = NearMissUnderSampler(version=2, n_neighbors=3, random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Check balancing
        assert y_resampled.value_counts()[0] == y_resampled.value_counts()[1]
        assert y_resampled.value_counts()[0] == 50

    def test_version_3_undersampling(self):
        """Test NearMiss version 3."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([
                np.random.randn(50) - 2,
                np.random.randn(200) + 2
            ]),
            'feature2': np.concatenate([
                np.random.randn(50) - 2,
                np.random.randn(200) + 2
            ]),
        })
        y = pd.Series([0] * 50 + [1] * 200)

        sampler = NearMissUnderSampler(version=3, n_neighbors=3, random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Version 3 may not perfectly balance but should reduce majority
        assert y_resampled.value_counts()[1] < 200
        assert y_resampled.value_counts()[0] == 50  # Minority unchanged

    def test_invalid_version(self):
        """Test that invalid version raises error."""
        with pytest.raises(ValueError, match="version must be 1, 2, or 3"):
            sampler = NearMissUnderSampler(version=4)

    def test_fit_resample_numpy_arrays(self):
        """Test fit_resample with numpy arrays."""
        np.random.seed(42)
        X = np.concatenate([
            np.random.randn(50, 2) - 2,
            np.random.randn(200, 2) + 2
        ])
        y = np.array([0] * 50 + [1] * 200)

        sampler = NearMissUnderSampler(version=1, random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Check output types
        assert isinstance(X_resampled, np.ndarray)
        assert isinstance(y_resampled, np.ndarray)

        # Check balancing
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert counts[0] == counts[1]

    def test_multiclass_raises_error(self):
        """Test that multi-class raises appropriate error."""
        X = pd.DataFrame({
            'feature1': np.random.randn(300),
        })
        y = pd.Series([0] * 100 + [1] * 100 + [2] * 100)

        sampler = NearMissUnderSampler(version=1)

        # NearMiss typically only works for binary classification
        with pytest.raises(ValueError):
            sampler.fit_resample(X, y)

    def test_n_neighbors_affects_results(self):
        """Test that different n_neighbors values produce different results."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([
                np.random.randn(50) - 2,
                np.random.randn(200) + 2
            ]),
            'feature2': np.concatenate([
                np.random.randn(50) - 2,
                np.random.randn(200) + 2
            ]),
        })
        y = pd.Series([0] * 50 + [1] * 200)

        sampler1 = NearMissUnderSampler(version=1, n_neighbors=3, random_state=42)
        X_res1, y_res1 = sampler1.fit_resample(X, y)

        sampler2 = NearMissUnderSampler(version=1, n_neighbors=5, random_state=42)
        X_res2, y_res2 = sampler2.fit_resample(X, y)

        # Different n_neighbors should produce different samples
        # (though both should be balanced)
        assert not X_res1.equals(X_res2)

    def test_get_params(self):
        """Test get_params method."""
        sampler = NearMissUnderSampler(
            version=2,
            n_neighbors=5,
            sampling_strategy='majority'
        )
        params = sampler.get_params()
        assert params['version'] == 2
        assert params['n_neighbors'] == 5
        assert params['sampling_strategy'] == 'majority'


class TestTomekLinksRemover:
    """Tests for TomekLinksRemover class."""

    def test_initialization_default(self):
        """Test default initialization."""
        remover = TomekLinksRemover()
        assert remover.sampling_strategy == 'auto'
        assert remover.n_jobs == 1

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        remover = TomekLinksRemover(
            sampling_strategy='all',
            n_jobs=2
        )
        assert remover.sampling_strategy == 'all'
        assert remover.n_jobs == 2

    def test_fit_resample_removes_tomek_links(self):
        """Test that Tomek links are removed."""
        # Create dataset with clear Tomek links on boundary
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([
                np.random.randn(100) - 2,    # Class 0 cluster
                [0],                          # Boundary sample from class 0
                np.random.randn(200) + 2,    # Class 1 cluster
                [-0.1]                        # Boundary sample from class 1
            ]),
            'feature2': np.concatenate([
                np.random.randn(100) - 2,
                [0],
                np.random.randn(200) + 2,
                [0.1]
            ]),
        })
        y = pd.Series([0] * 100 + [0] + [1] * 200 + [1])

        remover = TomekLinksRemover(sampling_strategy='auto')
        X_resampled, y_resampled = remover.fit_resample(X, y)

        # Some samples should be removed
        assert len(X_resampled) < len(X)

        # Check data types
        assert isinstance(X_resampled, pd.DataFrame)
        assert isinstance(y_resampled, pd.Series)

    def test_fit_resample_auto_strategy(self):
        """Test auto strategy removes only majority class links."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([
                np.random.randn(50) - 2,
                np.random.randn(200) + 2
            ]),
            'feature2': np.concatenate([
                np.random.randn(50) - 2,
                np.random.randn(200) + 2
            ]),
        })
        y = pd.Series([0] * 50 + [1] * 200)

        remover = TomekLinksRemover(sampling_strategy='auto')
        X_resampled, y_resampled = remover.fit_resample(X, y)

        # Minority class count should remain unchanged (auto strategy)
        assert y_resampled.value_counts()[0] == 50
        # Majority class may be reduced
        assert y_resampled.value_counts()[1] <= 200

    def test_fit_resample_all_strategy(self):
        """Test all strategy removes both samples from Tomek links."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([
                np.random.randn(100) - 2,
                np.random.randn(200) + 2
            ]),
            'feature2': np.concatenate([
                np.random.randn(100) - 2,
                np.random.randn(200) + 2
            ]),
        })
        y = pd.Series([0] * 100 + [1] * 200)

        remover = TomekLinksRemover(sampling_strategy='all')
        X_resampled, y_resampled = remover.fit_resample(X, y)

        # Both minority and majority may be reduced
        assert y_resampled.value_counts()[0] <= 100
        assert y_resampled.value_counts()[1] <= 200

    def test_fit_resample_majority_strategy(self):
        """Test majority strategy removes only majority class links."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([
                np.random.randn(100) - 2,
                np.random.randn(200) + 2
            ]),
            'feature2': np.concatenate([
                np.random.randn(100) - 2,
                np.random.randn(200) + 2
            ]),
        })
        y = pd.Series([0] * 100 + [1] * 200)

        remover = TomekLinksRemover(sampling_strategy='majority')
        X_resampled, y_resampled = remover.fit_resample(X, y)

        # Minority should be unchanged
        assert y_resampled.value_counts()[0] == 100
        # Majority may be reduced
        assert y_resampled.value_counts()[1] <= 200

    def test_no_tomek_links_no_removal(self):
        """Test that well-separated data has no Tomek links removed."""
        # Create well-separated clusters
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([
                np.random.randn(100) - 10,  # Far left
                np.random.randn(200) + 10   # Far right
            ]),
            'feature2': np.concatenate([
                np.random.randn(100),
                np.random.randn(200)
            ]),
        })
        y = pd.Series([0] * 100 + [1] * 200)

        remover = TomekLinksRemover(sampling_strategy='all')
        X_resampled, y_resampled = remover.fit_resample(X, y)

        # With well-separated data, few or no samples should be removed
        # (allowing for some randomness in cluster generation)
        assert len(X_resampled) >= len(X) * 0.95

    def test_fit_resample_numpy_arrays(self):
        """Test fit_resample with numpy arrays."""
        np.random.seed(42)
        X = np.concatenate([
            np.random.randn(100, 2) - 2,
            np.random.randn(200, 2) + 2
        ])
        y = np.array([0] * 100 + [1] * 200)

        remover = TomekLinksRemover()
        X_resampled, y_resampled = remover.fit_resample(X, y)

        # Check output types
        assert isinstance(X_resampled, np.ndarray)
        assert isinstance(y_resampled, np.ndarray)

        # Some samples may be removed
        assert len(X_resampled) <= len(X)

    def test_multiclass_support(self):
        """Test that multi-class datasets work."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([
                np.random.randn(50) - 3,
                np.random.randn(100),
                np.random.randn(100) + 3
            ]),
            'feature2': np.concatenate([
                np.random.randn(50) - 3,
                np.random.randn(100),
                np.random.randn(100) + 3
            ]),
        })
        y = pd.Series([0] * 50 + [1] * 100 + [2] * 100)

        remover = TomekLinksRemover()
        X_resampled, y_resampled = remover.fit_resample(X, y)

        # Should work without error
        assert len(X_resampled) <= len(X)
        # All three classes should still be present
        assert len(y_resampled.unique()) == 3

    def test_get_params(self):
        """Test get_params method."""
        remover = TomekLinksRemover(
            sampling_strategy='all',
            n_jobs=2
        )
        params = remover.get_params()
        assert params['sampling_strategy'] == 'all'
        assert params['n_jobs'] == 2


class TestEdgeCases:
    """Tests for edge cases across all undersampling methods."""

    def test_already_balanced_data_random(self):
        """Test RandomUnderSampler with already balanced data."""
        X = pd.DataFrame({
            'feature1': np.random.randn(200),
        })
        y = pd.Series([0] * 100 + [1] * 100)

        sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Should remain balanced
        assert y_resampled.value_counts()[0] == 100
        assert y_resampled.value_counts()[1] == 100

    def test_single_sample_minority_class(self):
        """Test behavior with very small minority class."""
        X = pd.DataFrame({
            'feature1': np.random.randn(101),
        })
        y = pd.Series([0] * 1 + [1] * 100)

        sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Should balance to minority (1 sample each)
        assert y_resampled.value_counts()[0] == 1
        assert y_resampled.value_counts()[1] == 1

    def test_empty_dataframe_handling(self):
        """Test error handling for empty DataFrame."""
        X = pd.DataFrame()
        y = pd.Series(dtype=int)

        sampler = RandomUnderSampler()

        # Should raise an error or return empty
        with pytest.raises((ValueError, IndexError)):
            sampler.fit_resample(X, y)

    def test_single_class_handling(self):
        """Test error handling for single class in target."""
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
        })
        y = pd.Series([0] * 100)

        sampler = RandomUnderSampler()

        # Should raise error as there's nothing to balance
        with pytest.raises((ValueError, KeyError)):
            sampler.fit_resample(X, y)
