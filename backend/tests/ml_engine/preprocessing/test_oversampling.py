"""
Tests for oversampling preprocessing methods.
"""

import pytest
import pandas as pd
import numpy as np
from app.ml_engine.preprocessing.oversampling import (
    SMOTE,
    BorderlineSMOTE,
    ADASYN,
)


class TestSMOTE:
    """Tests for SMOTE class."""

    def test_initialization_default(self):
        """Test default initialization."""
        smote = SMOTE()
        assert smote.sampling_strategy == 'auto'
        assert smote.k_neighbors == 5
        assert smote.random_state is None
        assert smote.n_jobs == 1

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        smote = SMOTE(
            sampling_strategy='minority',
            k_neighbors=3,
            random_state=42,
            n_jobs=2
        )
        assert smote.sampling_strategy == 'minority'
        assert smote.k_neighbors == 3
        assert smote.random_state == 42
        assert smote.n_jobs == 2

    def test_fit_resample_auto_strategy(self):
        """Test fit_resample with auto strategy balances classes."""
        # Create imbalanced dataset
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(50) - 2, np.random.randn(200) + 2]),
            'feature2': np.concatenate([np.random.randn(50) * 0.5, np.random.randn(200) * 0.5 + 1]),
        })
        y = pd.Series([0] * 50 + [1] * 200)

        smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Check that classes are balanced
        assert y_resampled.value_counts()[0] == y_resampled.value_counts()[1]
        assert y_resampled.value_counts()[0] == 200  # Balanced to majority class

        # Check that data types are preserved
        assert isinstance(X_resampled, pd.DataFrame)
        assert isinstance(y_resampled, pd.Series)

        # Check that synthetic samples were generated
        assert len(X_resampled) > len(X)
        assert len(X_resampled) == 400  # 200 + 200

    def test_fit_resample_ratio_strategy(self):
        """Test fit_resample with ratio strategy."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(50), np.random.randn(200)]),
        })
        y = pd.Series([0] * 50 + [1] * 200)

        # Use ratio of 0.5 (minority:majority = 1:2)
        smote = SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Check ratio
        minority_count = y_resampled.value_counts()[0]
        majority_count = y_resampled.value_counts()[1]
        assert minority_count == 100  # 200 * 0.5 = 100
        assert majority_count == 200

    def test_fit_resample_dict_strategy(self):
        """Test fit_resample with dictionary strategy."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(50), np.random.randn(200)]),
        })
        y = pd.Series([0] * 50 + [1] * 200)

        # Specify exact counts for each class
        smote = SMOTE(sampling_strategy={0: 150, 1: 200}, k_neighbors=5, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Check exact counts
        assert y_resampled.value_counts()[0] == 150
        assert y_resampled.value_counts()[1] == 200

    def test_synthetic_samples_are_interpolated(self):
        """Test that synthetic samples are between original samples."""
        np.random.seed(42)
        # Create well-separated clusters
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(50) - 5, np.random.randn(200) + 5]),
            'feature2': np.concatenate([np.random.randn(50) - 5, np.random.randn(200) + 5]),
        })
        y = pd.Series([0] * 50 + [1] * 200)

        smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Get synthetic minority samples (those beyond original 50)
        minority_mask = y_resampled == 0
        X_minority_resampled = X_resampled[minority_mask]

        # Synthetic samples should be in similar range as original minority samples
        original_minority = X[y == 0]

        # Check that synthetic samples are roughly in the same region
        assert X_minority_resampled['feature1'].min() >= original_minority['feature1'].min() - 2
        assert X_minority_resampled['feature1'].max() <= original_minority['feature1'].max() + 2

    def test_random_state_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(50), np.random.randn(200)]),
        })
        y = pd.Series([0] * 50 + [1] * 200)

        smote1 = SMOTE(random_state=42)
        X_res1, y_res1 = smote1.fit_resample(X, y)

        smote2 = SMOTE(random_state=42)
        X_res2, y_res2 = smote2.fit_resample(X, y)

        # Results should be identical
        pd.testing.assert_frame_equal(X_res1, X_res2)
        pd.testing.assert_series_equal(y_res1, y_res2)

    def test_fit_resample_numpy_arrays(self):
        """Test fit_resample with numpy arrays."""
        np.random.seed(42)
        X = np.concatenate([np.random.randn(50, 2) - 2, np.random.randn(200, 2) + 2])
        y = np.array([0] * 50 + [1] * 200)

        smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Check that output is numpy arrays
        assert isinstance(X_resampled, np.ndarray)
        assert isinstance(y_resampled, np.ndarray)

        # Check balancing
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert counts[0] == counts[1]

    def test_small_minority_class(self):
        """Test SMOTE with very small minority class."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(5), np.random.randn(100)]),
            'feature2': np.concatenate([np.random.randn(5), np.random.randn(100)]),
        })
        y = pd.Series([0] * 5 + [1] * 100)

        # k_neighbors will be adjusted automatically
        smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Should still generate synthetic samples
        assert y_resampled.value_counts()[0] == 100
        assert len(X_resampled) > len(X)

    def test_multiclass_oversampling(self):
        """Test SMOTE with multi-class dataset."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([
                np.random.randn(30) - 3,
                np.random.randn(100),
                np.random.randn(200) + 3
            ]),
        })
        y = pd.Series([0] * 30 + [1] * 100 + [2] * 200)

        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # All classes should be balanced to majority (200)
        assert y_resampled.value_counts()[0] == 200
        assert y_resampled.value_counts()[1] == 200
        assert y_resampled.value_counts()[2] == 200

    def test_get_params(self):
        """Test get_params method."""
        smote = SMOTE(
            sampling_strategy='minority',
            k_neighbors=3,
            random_state=42
        )
        params = smote.get_params()
        assert params['sampling_strategy'] == 'minority'
        assert params['k_neighbors'] == 3
        assert params['random_state'] == 42


class TestBorderlineSMOTE:
    """Tests for BorderlineSMOTE class."""

    def test_initialization_default(self):
        """Test default initialization."""
        bsmote = BorderlineSMOTE()
        assert bsmote.sampling_strategy == 'auto'
        assert bsmote.k_neighbors == 5
        assert bsmote.m_neighbors == 10
        assert bsmote.kind == 'borderline-1'
        assert bsmote.random_state is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        bsmote = BorderlineSMOTE(
            kind='borderline-2',
            k_neighbors=3,
            m_neighbors=8,
            random_state=42
        )
        assert bsmote.kind == 'borderline-2'
        assert bsmote.k_neighbors == 3
        assert bsmote.m_neighbors == 8
        assert bsmote.random_state == 42

    def test_invalid_kind_raises_error(self):
        """Test that invalid kind raises error."""
        with pytest.raises(ValueError, match="kind must be"):
            BorderlineSMOTE(kind='invalid')

    def test_borderline1_oversampling(self):
        """Test Borderline-SMOTE version 1."""
        np.random.seed(42)
        # Create data with clear boundary region
        X = pd.DataFrame({
            'feature1': np.concatenate([
                np.random.randn(50) - 3,  # Minority far from boundary
                np.random.randn(150) + 3   # Majority far from boundary
            ]),
            'feature2': np.concatenate([
                np.random.randn(50) - 3,
                np.random.randn(150) + 3
            ]),
        })
        y = pd.Series([0] * 50 + [1] * 150)

        bsmote = BorderlineSMOTE(kind='borderline-1', k_neighbors=5, random_state=42)
        X_resampled, y_resampled = bsmote.fit_resample(X, y)

        # Check balancing
        assert y_resampled.value_counts()[0] == y_resampled.value_counts()[1]
        assert len(X_resampled) > len(X)

    def test_borderline2_oversampling(self):
        """Test Borderline-SMOTE version 2."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([
                np.random.randn(50) - 3,
                np.random.randn(150) + 3
            ]),
            'feature2': np.concatenate([
                np.random.randn(50) - 3,
                np.random.randn(150) + 3
            ]),
        })
        y = pd.Series([0] * 50 + [1] * 150)

        bsmote = BorderlineSMOTE(kind='borderline-2', k_neighbors=5, random_state=42)
        X_resampled, y_resampled = bsmote.fit_resample(X, y)

        # Check balancing
        assert y_resampled.value_counts()[0] == y_resampled.value_counts()[1]
        assert len(X_resampled) > len(X)

    def test_borderline_finds_boundary_samples(self):
        """Test that borderline detection works."""
        np.random.seed(42)
        # Create data with mixed boundary region
        minority_safe = np.random.randn(30, 2) - 5  # Far from boundary
        minority_border = np.random.randn(20, 2)     # Near boundary
        majority = np.random.randn(200, 2) + 2       # Majority side

        X = pd.DataFrame(
            np.vstack([minority_safe, minority_border, majority]),
            columns=['f1', 'f2']
        )
        y = pd.Series([0] * 50 + [1] * 200)

        bsmote = BorderlineSMOTE(kind='borderline-1', m_neighbors=10, random_state=42)
        X_resampled, y_resampled = bsmote.fit_resample(X, y)

        # Should generate synthetic samples
        assert len(X_resampled) > len(X)
        assert y_resampled.value_counts()[0] == 200

    def test_random_state_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(50), np.random.randn(200)]),
        })
        y = pd.Series([0] * 50 + [1] * 200)

        bsmote1 = BorderlineSMOTE(random_state=42)
        X_res1, y_res1 = bsmote1.fit_resample(X, y)

        bsmote2 = BorderlineSMOTE(random_state=42)
        X_res2, y_res2 = bsmote2.fit_resample(X, y)

        # Results should be identical
        pd.testing.assert_frame_equal(X_res1, X_res2)
        pd.testing.assert_series_equal(y_res1, y_res2)

    def test_fit_resample_numpy_arrays(self):
        """Test fit_resample with numpy arrays."""
        np.random.seed(42)
        X = np.concatenate([np.random.randn(50, 2) - 2, np.random.randn(200, 2) + 2])
        y = np.array([0] * 50 + [1] * 200)

        bsmote = BorderlineSMOTE(random_state=42)
        X_resampled, y_resampled = bsmote.fit_resample(X, y)

        # Check output types
        assert isinstance(X_resampled, np.ndarray)
        assert isinstance(y_resampled, np.ndarray)

        # Check balancing
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert counts[0] == counts[1]

    def test_get_params(self):
        """Test get_params method."""
        bsmote = BorderlineSMOTE(
            kind='borderline-2',
            k_neighbors=3,
            m_neighbors=8
        )
        params = bsmote.get_params()
        assert params['kind'] == 'borderline-2'
        assert params['k_neighbors'] == 3
        assert params['m_neighbors'] == 8


class TestADASYN:
    """Tests for ADASYN class."""

    def test_initialization_default(self):
        """Test default initialization."""
        adasyn = ADASYN()
        assert adasyn.sampling_strategy == 'auto'
        assert adasyn.k_neighbors == 5
        assert adasyn.random_state is None
        assert adasyn.n_jobs == 1

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        adasyn = ADASYN(
            sampling_strategy='minority',
            k_neighbors=3,
            random_state=42,
            n_jobs=2
        )
        assert adasyn.sampling_strategy == 'minority'
        assert adasyn.k_neighbors == 3
        assert adasyn.random_state == 42
        assert adasyn.n_jobs == 2

    def test_fit_resample_auto_strategy(self):
        """Test ADASYN with auto strategy."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(50) - 2, np.random.randn(200) + 2]),
            'feature2': np.concatenate([np.random.randn(50) * 0.5, np.random.randn(200) * 0.5 + 1]),
        })
        y = pd.Series([0] * 50 + [1] * 200)

        adasyn = ADASYN(sampling_strategy='auto', k_neighbors=5, random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)

        # Check that classes are balanced
        assert y_resampled.value_counts()[0] == y_resampled.value_counts()[1]
        assert y_resampled.value_counts()[0] == 200

        # Check that data types are preserved
        assert isinstance(X_resampled, pd.DataFrame)
        assert isinstance(y_resampled, pd.Series)

        # Check that synthetic samples were generated
        assert len(X_resampled) > len(X)

    def test_adaptive_sampling_weights(self):
        """Test that ADASYN generates more samples for difficult instances."""
        np.random.seed(42)
        # Create data where some minority samples are surrounded by majority
        X = pd.DataFrame({
            'feature1': np.concatenate([
                np.random.randn(40) - 5,      # Safe minority
                np.random.randn(10) + 2,      # Difficult minority (in majority region)
                np.random.randn(200) + 2      # Majority
            ]),
            'feature2': np.concatenate([
                np.random.randn(40) - 5,
                np.random.randn(10) + 2,
                np.random.randn(200) + 2
            ]),
        })
        y = pd.Series([0] * 50 + [1] * 200)

        adasyn = ADASYN(k_neighbors=5, random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)

        # Should generate synthetic samples adaptively
        assert len(X_resampled) > len(X)
        assert y_resampled.value_counts()[0] == 200

    def test_density_distribution_calculation(self):
        """Test that density distribution is calculated correctly."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(50), np.random.randn(200)]),
        })
        y = pd.Series([0] * 50 + [1] * 200)

        adasyn = ADASYN(k_neighbors=5, random_state=42)
        adasyn.fit(X, y)

        # Get minority samples
        X_array = X.values
        y_array = y.values
        X_minority = X_array[y_array == 0]

        # Calculate densities
        densities = adasyn._calculate_density_distribution(X_array, y_array, X_minority)

        # Densities should be between 0 and 1
        assert np.all(densities >= 0)
        assert np.all(densities <= 1)
        assert len(densities) == len(X_minority)

    def test_random_state_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(50), np.random.randn(200)]),
        })
        y = pd.Series([0] * 50 + [1] * 200)

        adasyn1 = ADASYN(random_state=42)
        X_res1, y_res1 = adasyn1.fit_resample(X, y)

        adasyn2 = ADASYN(random_state=42)
        X_res2, y_res2 = adasyn2.fit_resample(X, y)

        # Results should be identical
        pd.testing.assert_frame_equal(X_res1, X_res2)
        pd.testing.assert_series_equal(y_res1, y_res2)

    def test_fit_resample_numpy_arrays(self):
        """Test fit_resample with numpy arrays."""
        np.random.seed(42)
        X = np.concatenate([np.random.randn(50, 2) - 2, np.random.randn(200, 2) + 2])
        y = np.array([0] * 50 + [1] * 200)

        adasyn = ADASYN(random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)

        # Check output types
        assert isinstance(X_resampled, np.ndarray)
        assert isinstance(y_resampled, np.ndarray)

        # Check balancing
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert counts[0] == counts[1]

    def test_ratio_strategy(self):
        """Test ADASYN with ratio strategy."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(50), np.random.randn(200)]),
        })
        y = pd.Series([0] * 50 + [1] * 200)

        adasyn = ADASYN(sampling_strategy=0.5, random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)

        # Check ratio
        minority_count = y_resampled.value_counts()[0]
        majority_count = y_resampled.value_counts()[1]
        assert minority_count == 100  # 200 * 0.5
        assert majority_count == 200

    def test_get_params(self):
        """Test get_params method."""
        adasyn = ADASYN(
            sampling_strategy='minority',
            k_neighbors=3,
            random_state=42
        )
        params = adasyn.get_params()
        assert params['sampling_strategy'] == 'minority'
        assert params['k_neighbors'] == 3
        assert params['random_state'] == 42


class TestEdgeCases:
    """Tests for edge cases across all oversampling methods."""

    def test_already_balanced_data_smote(self):
        """Test SMOTE with already balanced data."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(200),
        })
        y = pd.Series([0] * 100 + [1] * 100)

        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Should remain balanced (no oversampling needed)
        assert y_resampled.value_counts()[0] == 100
        assert y_resampled.value_counts()[1] == 100
        assert len(X_resampled) == len(X)

    def test_very_small_minority_class(self):
        """Test with extremely small minority class."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(2), np.random.randn(100)]),
            'feature2': np.concatenate([np.random.randn(2), np.random.randn(100)]),
        })
        y = pd.Series([0] * 2 + [1] * 100)

        # k_neighbors will be automatically adjusted to 1
        smote = SMOTE(k_neighbors=5, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Should still work and balance
        assert y_resampled.value_counts()[0] == 100
        assert len(X_resampled) > len(X)

    def test_no_oversampling_needed(self):
        """Test when minority is already larger than target."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
        })
        y = pd.Series([0] * 60 + [1] * 40)

        # Target minority to be 30 (less than current 40)
        smote = SMOTE(sampling_strategy={0: 60, 1: 30}, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Should not generate synthetic samples for class 1
        assert y_resampled.value_counts()[1] == 40  # Unchanged

    def test_comparison_smote_vs_borderline_vs_adasyn(self):
        """Compare all three methods on same dataset."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(50) - 2, np.random.randn(200) + 2]),
            'feature2': np.concatenate([np.random.randn(50) - 2, np.random.randn(200) + 2]),
        })
        y = pd.Series([0] * 50 + [1] * 200)

        # Test all methods
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)

        bsmote = BorderlineSMOTE(random_state=42)
        X_bsmote, y_bsmote = bsmote.fit_resample(X, y)

        adasyn = ADASYN(random_state=42)
        X_adasyn, y_adasyn = adasyn.fit_resample(X, y)

        # All should balance the classes
        assert y_smote.value_counts()[0] == 200
        assert y_bsmote.value_counts()[0] == 200
        assert y_adasyn.value_counts()[0] == 200

        # All should generate different synthetic samples
        # (cannot directly compare due to randomness, but sizes should match)
        assert len(X_smote) == len(X_bsmote) == len(X_adasyn)
