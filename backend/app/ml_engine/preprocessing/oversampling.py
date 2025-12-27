"""
Oversampling methods for handling imbalanced datasets.

This module provides implementations of various oversampling techniques that
create synthetic samples to balance class distributions, particularly useful
when the minority class has too few samples.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.neighbors import NearestNeighbors
from app.ml_engine.preprocessing.base import PreprocessingStep
from app.utils.logger import get_logger

logger = get_logger("oversampling")


class SMOTE(PreprocessingStep):
    """
    Synthetic Minority Over-sampling TEchnique (SMOTE).

    SMOTE creates synthetic samples by interpolating between existing minority
    class samples and their nearest neighbors. This helps prevent overfitting
    compared to simple duplication.

    Algorithm:
    1. For each minority sample x:
       - Find k nearest neighbors from the same class
       - Randomly select one neighbor x_nn
       - Generate synthetic sample along the line: x_new = x + λ(x_nn - x)
       - Where λ is random [0, 1]

    Parameters:
        sampling_strategy (str, float, dict): How to determine target class distribution
            - 'auto': Oversample to match majority class count
            - 'minority': Oversample only minority class to majority count
            - float: Desired ratio minority/majority (e.g., 0.5 = 1:2)
            - dict: {class: n_samples} exact counts
        k_neighbors (int): Number of nearest neighbors to use (default: 5)
        random_state (int, optional): Random seed for reproducibility
        n_jobs (int): Number of parallel jobs for KNN (default: 1)

    Example:
        >>> smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
        >>> X_resampled, y_resampled = smote.fit_resample(X, y)

    References:
        Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
    """

    def __init__(
        self,
        sampling_strategy: Union[str, float, Dict] = 'auto',
        k_neighbors: int = 5,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        **params
    ):
        """Initialize SMOTE oversampler."""
        super().__init__(**params)
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.rng = np.random.RandomState(random_state)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'SMOTE':
        """
        Fit the SMOTE oversampler.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            self
        """
        # Convert to numpy for processing
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # Get class distribution
        classes, counts = np.unique(y_array, return_counts=True)
        self.class_counts_ = dict(zip(classes, counts))

        # Determine target distribution
        self.target_counts_ = self._calculate_target_counts(self.class_counts_)

        logger.info(f"Original class distribution: {self.class_counts_}")
        logger.info(f"Target class distribution: {self.target_counts_}")

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Tuple:
        """
        Apply SMOTE oversampling.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        # Store original types
        is_dataframe = isinstance(X, pd.DataFrame)
        is_series = isinstance(y, pd.Series)

        # Convert to numpy
        X_array = X.values if is_dataframe else X
        y_array = y.values if is_series else y

        # Collect resampled data
        X_resampled_list = []
        y_resampled_list = []

        # Process each class
        for class_label in self.target_counts_:
            # Get samples for this class
            class_mask = y_array == class_label
            X_class = X_array[class_mask]

            original_count = self.class_counts_[class_label]
            target_count = self.target_counts_[class_label]

            # Add original samples
            X_resampled_list.append(X_class)
            y_resampled_list.append(np.full(original_count, class_label))

            # Generate synthetic samples if needed
            if target_count > original_count:
                n_synthetic = target_count - original_count
                X_synthetic = self._generate_samples(X_class, n_synthetic)

                X_resampled_list.append(X_synthetic)
                y_resampled_list.append(np.full(n_synthetic, class_label))

                logger.info(f"Generated {n_synthetic} synthetic samples for class {class_label}")

        # Combine all samples
        X_resampled = np.vstack(X_resampled_list)
        y_resampled = np.hstack(y_resampled_list)

        # Shuffle
        shuffle_idx = self.rng.permutation(len(X_resampled))
        X_resampled = X_resampled[shuffle_idx]
        y_resampled = y_resampled[shuffle_idx]

        # Convert back to original types
        if is_dataframe:
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        if is_series:
            y_resampled = pd.Series(y_resampled, name=y.name)

        logger.info(f"Resampled class distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
        logger.info(f"Generated {len(X_resampled) - len(X)} total synthetic samples")

        return X_resampled, y_resampled

    def _generate_samples(self, X_class: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Generate synthetic samples using SMOTE algorithm.

        Args:
            X_class: Samples from minority class
            n_samples: Number of synthetic samples to generate

        Returns:
            Array of synthetic samples
        """
        n_features = X_class.shape[1]

        # Handle edge case: very few samples
        if len(X_class) <= self.k_neighbors:
            k = len(X_class) - 1 if len(X_class) > 1 else 1
            logger.warning(f"Class has only {len(X_class)} samples, using k={k} neighbors")
        else:
            k = self.k_neighbors

        # Fit KNN to find nearest neighbors
        knn = NearestNeighbors(n_neighbors=k + 1, n_jobs=self.n_jobs)  # +1 because first neighbor is itself
        knn.fit(X_class)

        # Find nearest neighbors for all samples
        distances, indices = knn.kneighbors(X_class)

        # Generate synthetic samples
        synthetic_samples = np.zeros((n_samples, n_features))

        for i in range(n_samples):
            # Randomly select a sample from minority class
            sample_idx = self.rng.randint(0, len(X_class))
            sample = X_class[sample_idx]

            # Select a random neighbor (excluding itself at index 0)
            neighbor_idx = self.rng.randint(1, k + 1)
            neighbor = X_class[indices[sample_idx, neighbor_idx]]

            # Generate synthetic sample via interpolation
            # lambda (gap) is random value between 0 and 1
            gap = self.rng.random()
            synthetic_samples[i] = sample + gap * (neighbor - sample)

        return synthetic_samples

    def _calculate_target_counts(self, class_counts: Dict) -> Dict:
        """Calculate target counts for each class based on strategy."""
        if self.sampling_strategy == 'auto' or self.sampling_strategy == 'minority':
            # Oversample all classes to majority class count
            max_count = max(class_counts.values())
            return {cls: max_count for cls in class_counts}

        elif isinstance(self.sampling_strategy, float):
            # Use ratio: minority/majority = sampling_strategy
            majority_class = max(class_counts, key=class_counts.get)
            minority_class = min(class_counts, key=class_counts.get)
            majority_count = class_counts[majority_class]
            minority_count = class_counts[minority_class]

            # Calculate target minority count based on ratio
            target_minority = int(majority_count * self.sampling_strategy)

            target_counts = class_counts.copy()
            target_counts[minority_class] = target_minority
            return target_counts

        elif isinstance(self.sampling_strategy, dict):
            # Use exact counts specified
            return self.sampling_strategy

        else:
            raise ValueError(f"Invalid sampling_strategy: {self.sampling_strategy}")

    def fit_resample(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Tuple:
        """Fit and resample in one step."""
        return self.fit(X, y).transform(X, y)


class BorderlineSMOTE(PreprocessingStep):
    """
    Borderline-SMOTE variant that focuses on samples near decision boundary.

    This variant only generates synthetic samples from minority class instances
    that are close to the decision boundary (borderline samples), as these are
    more likely to be misclassified.

    Algorithm:
    1. Identify borderline samples: minority samples where m/2 < neighbors_from_majority < m
    2. Apply SMOTE only to these borderline samples

    This reduces overfitting by not oversampling safe minority samples (far from boundary).

    Parameters:
        sampling_strategy (str, float, dict): Target distribution (same as SMOTE)
        k_neighbors (int): Number of nearest neighbors for SMOTE (default: 5)
        m_neighbors (int): Number of neighbors to check for borderline (default: 10)
        kind (str): 'borderline-1' or 'borderline-2'
            - borderline-1: Generate samples between minority samples only
            - borderline-2: Generate samples using majority neighbors too
        random_state (int, optional): Random seed
        n_jobs (int): Number of parallel jobs

    Example:
        >>> bsmote = BorderlineSMOTE(kind='borderline-1', k_neighbors=5, random_state=42)
        >>> X_resampled, y_resampled = bsmote.fit_resample(X, y)

    References:
        Han, H., et al. (2005). "Borderline-SMOTE: A New Over-Sampling Method in
        Imbalanced Data Sets Learning"
    """

    def __init__(
        self,
        sampling_strategy: Union[str, float, Dict] = 'auto',
        k_neighbors: int = 5,
        m_neighbors: int = 10,
        kind: str = 'borderline-1',
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        **params
    ):
        """Initialize Borderline-SMOTE oversampler."""
        super().__init__(**params)

        if kind not in ['borderline-1', 'borderline-2']:
            raise ValueError("kind must be 'borderline-1' or 'borderline-2'")

        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.kind = kind
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.rng = np.random.RandomState(random_state)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'BorderlineSMOTE':
        """Fit the Borderline-SMOTE oversampler."""
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        classes, counts = np.unique(y_array, return_counts=True)
        self.class_counts_ = dict(zip(classes, counts))
        self.target_counts_ = self._calculate_target_counts(self.class_counts_)

        logger.info(f"Borderline-SMOTE ({self.kind}): Original distribution {self.class_counts_}")
        logger.info(f"Target distribution: {self.target_counts_}")

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Tuple:
        """Apply Borderline-SMOTE oversampling."""
        is_dataframe = isinstance(X, pd.DataFrame)
        is_series = isinstance(y, pd.Series)

        X_array = X.values if is_dataframe else X
        y_array = y.values if is_series else y

        # Identify minority and majority classes
        minority_class = min(self.class_counts_, key=self.class_counts_.get)
        majority_class = max(self.class_counts_, key=self.class_counts_.get)

        # Get minority and majority samples
        minority_mask = y_array == minority_class
        X_minority = X_array[minority_mask]
        X_majority = X_array[~minority_mask]

        # Find borderline samples
        borderline_indices = self._find_borderline_samples(X_minority, X_majority)

        if len(borderline_indices) == 0:
            logger.warning("No borderline samples found, using all minority samples")
            borderline_indices = np.arange(len(X_minority))

        X_borderline = X_minority[borderline_indices]

        logger.info(f"Identified {len(borderline_indices)} borderline samples out of {len(X_minority)} minority samples")

        # Calculate how many synthetic samples to generate
        target_count = self.target_counts_[minority_class]
        current_count = self.class_counts_[minority_class]
        n_synthetic = target_count - current_count

        # Generate synthetic samples from borderline samples
        if n_synthetic > 0:
            X_synthetic = self._generate_samples(X_borderline, X_minority, X_majority, n_synthetic)

            # Combine original and synthetic
            X_resampled = np.vstack([X_array, X_synthetic])
            y_resampled = np.hstack([y_array, np.full(n_synthetic, minority_class)])

            # Shuffle
            shuffle_idx = self.rng.permutation(len(X_resampled))
            X_resampled = X_resampled[shuffle_idx]
            y_resampled = y_resampled[shuffle_idx]
        else:
            X_resampled, y_resampled = X_array, y_array

        # Convert back to original types
        if is_dataframe:
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        if is_series:
            y_resampled = pd.Series(y_resampled, name=y.name)

        logger.info(f"Generated {n_synthetic} synthetic samples using {self.kind}")

        return X_resampled, y_resampled

    def _find_borderline_samples(self, X_minority: np.ndarray, X_majority: np.ndarray) -> np.ndarray:
        """
        Find borderline minority samples.

        Borderline samples are those where at least half of their m nearest
        neighbors belong to the majority class.
        """
        # Combine minority and majority for neighbor search
        X_combined = np.vstack([X_minority, X_majority])
        y_combined = np.hstack([
            np.zeros(len(X_minority)),  # 0 for minority
            np.ones(len(X_majority))     # 1 for majority
        ])

        # Find m nearest neighbors for each minority sample
        knn = NearestNeighbors(n_neighbors=self.m_neighbors + 1, n_jobs=self.n_jobs)
        knn.fit(X_combined)

        distances, indices = knn.kneighbors(X_minority)

        borderline_indices = []

        for i, neighbor_indices in enumerate(indices):
            # Exclude self (first neighbor)
            neighbor_indices = neighbor_indices[1:]

            # Count how many neighbors are from majority class
            majority_neighbors = np.sum(y_combined[neighbor_indices] == 1)

            # Borderline if at least half neighbors are majority
            # but not all (otherwise it's in majority region)
            if self.m_neighbors / 2 <= majority_neighbors < self.m_neighbors:
                borderline_indices.append(i)

        return np.array(borderline_indices)

    def _generate_samples(
        self,
        X_borderline: np.ndarray,
        X_minority: np.ndarray,
        X_majority: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """Generate synthetic samples from borderline samples."""
        n_features = X_borderline.shape[1]

        if len(X_borderline) == 0:
            # Fallback to regular minority samples
            X_borderline = X_minority

        # Adjust k if needed
        k = min(self.k_neighbors, len(X_borderline) - 1) if len(X_borderline) > 1 else 1

        if self.kind == 'borderline-1':
            # Generate samples between borderline minority samples
            knn = NearestNeighbors(n_neighbors=k + 1, n_jobs=self.n_jobs)
            knn.fit(X_borderline)

            synthetic_samples = np.zeros((n_samples, n_features))

            for i in range(n_samples):
                sample_idx = self.rng.randint(0, len(X_borderline))
                sample = X_borderline[sample_idx]

                distances, indices = knn.kneighbors([sample])
                neighbor_idx = self.rng.randint(1, k + 1)
                neighbor = X_borderline[indices[0, neighbor_idx]]

                gap = self.rng.random()
                synthetic_samples[i] = sample + gap * (neighbor - sample)

        else:  # borderline-2
            # Generate samples between borderline minority and ANY neighbors (including majority)
            X_all = np.vstack([X_minority, X_majority])

            knn = NearestNeighbors(n_neighbors=k + 1, n_jobs=self.n_jobs)
            knn.fit(X_all)

            synthetic_samples = np.zeros((n_samples, n_features))

            for i in range(n_samples):
                sample_idx = self.rng.randint(0, len(X_borderline))
                sample = X_borderline[sample_idx]

                distances, indices = knn.kneighbors([sample])
                neighbor_idx = self.rng.randint(1, k + 1)
                neighbor = X_all[indices[0, neighbor_idx]]

                gap = self.rng.random()
                synthetic_samples[i] = sample + gap * (neighbor - sample)

        return synthetic_samples

    def _calculate_target_counts(self, class_counts: Dict) -> Dict:
        """Calculate target counts (same logic as SMOTE)."""
        if self.sampling_strategy == 'auto' or self.sampling_strategy == 'minority':
            max_count = max(class_counts.values())
            return {cls: max_count for cls in class_counts}
        elif isinstance(self.sampling_strategy, float):
            majority_class = max(class_counts, key=class_counts.get)
            minority_class = min(class_counts, key=class_counts.get)
            majority_count = class_counts[majority_class]
            target_minority = int(majority_count * self.sampling_strategy)
            target_counts = class_counts.copy()
            target_counts[minority_class] = target_minority
            return target_counts
        elif isinstance(self.sampling_strategy, dict):
            return self.sampling_strategy
        else:
            raise ValueError(f"Invalid sampling_strategy: {self.sampling_strategy}")

    def fit_resample(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Tuple:
        """Fit and resample in one step."""
        return self.fit(X, y).transform(X, y)


class ADASYN(PreprocessingStep):
    """
    Adaptive Synthetic Sampling (ADASYN).

    ADASYN adaptively generates different numbers of synthetic samples for each
    minority instance based on its difficulty (density of nearby majority samples).
    Instances that are harder to learn get more synthetic samples.

    Algorithm:
    1. Calculate density distribution: for each minority sample, count nearby majority samples
    2. Normalize to get sampling weights (harder samples get higher weights)
    3. Generate synthetic samples proportional to weights

    This focuses on the most difficult regions of the feature space.

    Parameters:
        sampling_strategy (str, float, dict): Target distribution
        k_neighbors (int): Number of nearest neighbors (default: 5)
        random_state (int, optional): Random seed
        n_jobs (int): Number of parallel jobs

    Example:
        >>> adasyn = ADASYN(k_neighbors=5, random_state=42)
        >>> X_resampled, y_resampled = adasyn.fit_resample(X, y)

    References:
        He, H., et al. (2008). "ADASYN: Adaptive Synthetic Sampling Approach for
        Imbalanced Learning"
    """

    def __init__(
        self,
        sampling_strategy: Union[str, float, Dict] = 'auto',
        k_neighbors: int = 5,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        **params
    ):
        """Initialize ADASYN oversampler."""
        super().__init__(**params)
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.rng = np.random.RandomState(random_state)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'ADASYN':
        """Fit the ADASYN oversampler."""
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        classes, counts = np.unique(y_array, return_counts=True)
        self.class_counts_ = dict(zip(classes, counts))
        self.target_counts_ = self._calculate_target_counts(self.class_counts_)

        logger.info(f"ADASYN: Original distribution {self.class_counts_}")
        logger.info(f"Target distribution: {self.target_counts_}")

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Tuple:
        """Apply ADASYN oversampling."""
        is_dataframe = isinstance(X, pd.DataFrame)
        is_series = isinstance(y, pd.Series)

        X_array = X.values if is_dataframe else X
        y_array = y.values if is_series else y

        # Identify minority class
        minority_class = min(self.class_counts_, key=self.class_counts_.get)

        # Get minority samples
        minority_mask = y_array == minority_class
        X_minority = X_array[minority_mask]

        # Calculate how many synthetic samples needed
        target_count = self.target_counts_[minority_class]
        current_count = self.class_counts_[minority_class]
        n_synthetic = target_count - current_count

        if n_synthetic <= 0:
            logger.info("No oversampling needed")
            if is_dataframe:
                X_array = pd.DataFrame(X_array, columns=X.columns)
            if is_series:
                y_array = pd.Series(y_array, name=y.name)
            return X_array, y_array

        # Calculate density distribution (weights for each minority sample)
        weights = self._calculate_density_distribution(X_array, y_array, X_minority)

        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            # If all weights are 0, use uniform distribution
            weights = np.ones(len(X_minority)) / len(X_minority)

        # Calculate number of synthetic samples for each minority instance
        samples_per_instance = (weights * n_synthetic).astype(int)

        # Adjust to ensure we generate exactly n_synthetic samples
        diff = n_synthetic - samples_per_instance.sum()
        if diff > 0:
            # Add remaining samples to instances with highest weights
            top_indices = np.argsort(weights)[-diff:]
            samples_per_instance[top_indices] += 1

        logger.info(f"Generating {n_synthetic} synthetic samples with adaptive weights")

        # Generate synthetic samples
        X_synthetic_list = []

        # Fit KNN for minority class
        k = min(self.k_neighbors, len(X_minority) - 1) if len(X_minority) > 1 else 1
        knn = NearestNeighbors(n_neighbors=k + 1, n_jobs=self.n_jobs)
        knn.fit(X_minority)

        for i, n_samples_i in enumerate(samples_per_instance):
            if n_samples_i == 0:
                continue

            sample = X_minority[i]

            # Find neighbors
            distances, indices = knn.kneighbors([sample])

            # Generate synthetic samples for this instance
            for _ in range(n_samples_i):
                # Select random neighbor (excluding itself)
                neighbor_idx = self.rng.randint(1, k + 1)
                neighbor = X_minority[indices[0, neighbor_idx]]

                # Interpolate
                gap = self.rng.random()
                synthetic = sample + gap * (neighbor - sample)
                X_synthetic_list.append(synthetic)

        if len(X_synthetic_list) > 0:
            X_synthetic = np.vstack(X_synthetic_list)

            # Combine with original data
            X_resampled = np.vstack([X_array, X_synthetic])
            y_resampled = np.hstack([y_array, np.full(len(X_synthetic), minority_class)])

            # Shuffle
            shuffle_idx = self.rng.permutation(len(X_resampled))
            X_resampled = X_resampled[shuffle_idx]
            y_resampled = y_resampled[shuffle_idx]
        else:
            X_resampled, y_resampled = X_array, y_array

        # Convert back to original types
        if is_dataframe:
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        if is_series:
            y_resampled = pd.Series(y_resampled, name=y.name)

        logger.info(f"ADASYN generated {len(X_synthetic_list)} synthetic samples")

        return X_resampled, y_resampled

    def _calculate_density_distribution(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_minority: np.ndarray
    ) -> np.ndarray:
        """
        Calculate density distribution for each minority sample.

        Density = ratio of majority class neighbors to all neighbors.
        Higher density means more difficult (more majority neighbors).
        """
        # Fit KNN on all samples
        k = min(self.k_neighbors, len(X) - 1)
        knn = NearestNeighbors(n_neighbors=k + 1, n_jobs=self.n_jobs)
        knn.fit(X)

        # Get minority class label
        minority_class = min(self.class_counts_, key=self.class_counts_.get)

        # Calculate density for each minority sample
        densities = np.zeros(len(X_minority))

        for i, sample in enumerate(X_minority):
            # Find neighbors
            distances, indices = knn.kneighbors([sample])

            # Exclude self (first neighbor)
            neighbor_indices = indices[0, 1:]
            neighbor_labels = y[neighbor_indices]

            # Count majority class neighbors
            majority_neighbors = np.sum(neighbor_labels != minority_class)

            # Density = ratio of majority neighbors
            densities[i] = majority_neighbors / k

        return densities

    def _calculate_target_counts(self, class_counts: Dict) -> Dict:
        """Calculate target counts (same logic as SMOTE)."""
        if self.sampling_strategy == 'auto' or self.sampling_strategy == 'minority':
            max_count = max(class_counts.values())
            return {cls: max_count for cls in class_counts}
        elif isinstance(self.sampling_strategy, float):
            majority_class = max(class_counts, key=class_counts.get)
            minority_class = min(class_counts, key=class_counts.get)
            majority_count = class_counts[majority_class]
            target_minority = int(majority_count * self.sampling_strategy)
            target_counts = class_counts.copy()
            target_counts[minority_class] = target_minority
            return target_counts
        elif isinstance(self.sampling_strategy, dict):
            return self.sampling_strategy
        else:
            raise ValueError(f"Invalid sampling_strategy: {self.sampling_strategy}")

    def fit_resample(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Tuple:
        """Fit and resample in one step."""
        return self.fit(X, y).transform(X, y)
