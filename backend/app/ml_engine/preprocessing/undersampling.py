"""
Undersampling methods for handling imbalanced datasets.

This module provides various undersampling techniques to reduce the number
of samples in the majority class(es) to balance class distributions.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from app.ml_engine.preprocessing.base import PreprocessingStep
from app.utils.logger import get_logger

logger = get_logger("undersampling")


class RandomUnderSampler(PreprocessingStep):
    """
    Random undersampling of the majority class.

    Randomly removes samples from the majority class(es) to achieve
    a desired class distribution.

    Parameters:
        sampling_strategy: Target ratio or dict of class ratios
            - 'auto': Balance all classes to minority class count
            - 'majority': Undersample only the majority class
            - float: Ratio of minority/majority (e.g., 0.5 = 1:2 ratio)
            - dict: {class_label: n_samples} for each class
        random_state: Random seed for reproducibility
        replacement: Whether to sample with replacement (default: False)

    Example:
        >>> sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        >>> X_resampled, y_resampled = sampler.fit_resample(X, y)
    """

    def __init__(
        self,
        sampling_strategy: Union[str, float, Dict] = 'auto',
        random_state: Optional[int] = None,
        replacement: bool = False,
        **params
    ):
        super().__init__(name="RandomUnderSampler", **params)
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.replacement = replacement
        self.sampling_strategy_ = None
        self.class_counts_ = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "RandomUnderSampler":
        """
        Fit the undersampler on the data.

        Args:
            X: Features
            y: Target labels

        Returns:
            Self
        """
        if y is None:
            raise ValueError("Target labels y are required for undersampling")

        # Convert to numpy for processing
        y_array = y.values if isinstance(y, pd.Series) else y

        # Count class distribution
        self.class_counts_ = Counter(y_array)
        logger.info(f"Original class distribution: {dict(self.class_counts_)}")

        # Calculate sampling strategy
        self.sampling_strategy_ = self._calculate_sampling_strategy(self.class_counts_)
        logger.info(f"Target class distribution: {self.sampling_strategy_}")

        self.fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Tuple:
        """
        Transform the dataset by undersampling.

        Args:
            X: Features
            y: Target labels

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        self._check_fitted()

        if y is None:
            raise ValueError("Target labels y are required for undersampling")

        # Set random seed
        np.random.seed(self.random_state)

        # Convert inputs
        is_dataframe = isinstance(X, pd.DataFrame)
        X_array = X.values if is_dataframe else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # Get indices for each class
        indices_to_keep = []

        for class_label, target_count in self.sampling_strategy_.items():
            class_indices = np.where(y_array == class_label)[0]
            current_count = len(class_indices)

            if current_count <= target_count:
                # Keep all samples if already at or below target
                selected_indices = class_indices
            else:
                # Randomly sample to reach target count
                selected_indices = np.random.choice(
                    class_indices,
                    size=target_count,
                    replace=self.replacement
                )

            indices_to_keep.extend(selected_indices)

        # Sort indices to maintain order
        indices_to_keep = np.array(sorted(indices_to_keep))

        # Resample
        X_resampled = X_array[indices_to_keep]
        y_resampled = y_array[indices_to_keep]

        # Convert back to original format
        if is_dataframe:
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled = pd.Series(y_resampled, name=y.name if hasattr(y, 'name') else None)

        new_counts = Counter(y_resampled)
        logger.info(f"Resampled class distribution: {dict(new_counts)}")
        logger.info(f"Removed {len(y_array) - len(y_resampled)} samples")

        return X_resampled, y_resampled

    def fit_resample(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Tuple:
        """
        Fit and resample in one step.

        Args:
            X: Features
            y: Target labels

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        return self.fit(X, y).transform(X, y)

    def _calculate_sampling_strategy(self, class_counts: Counter) -> Dict:
        """Calculate target number of samples for each class."""
        if isinstance(self.sampling_strategy, dict):
            return self.sampling_strategy

        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        minority_count = min(counts)
        majority_count = max(counts)

        if self.sampling_strategy == 'auto':
            # Balance all classes to minority class count
            return {cls: minority_count for cls in classes}

        elif self.sampling_strategy == 'majority':
            # Only undersample the majority class
            majority_class = classes[counts.index(majority_count)]
            strategy = {cls: count for cls, count in class_counts.items()}
            strategy[majority_class] = minority_count
            return strategy

        elif isinstance(self.sampling_strategy, (int, float)):
            # Use ratio
            ratio = float(self.sampling_strategy)
            target_majority = int(minority_count / ratio)

            strategy = {}
            for cls, count in class_counts.items():
                if count == majority_count:
                    strategy[cls] = target_majority
                else:
                    strategy[cls] = count
            return strategy

        else:
            raise ValueError(f"Invalid sampling_strategy: {self.sampling_strategy}")


class NearMissUnderSampler(PreprocessingStep):
    """
    NearMiss undersampling algorithm.

    Selects samples from the majority class based on their distance
    to minority class samples. Three versions available:

    - Version 1: Select majority samples with smallest average distance to k nearest minority samples
    - Version 2: Select majority samples with smallest average distance to k farthest minority samples
    - Version 3: Select majority samples with largest distance to k nearest minority samples

    Parameters:
        version: NearMiss version (1, 2, or 3)
        n_neighbors: Number of nearest neighbors to consider
        sampling_strategy: Same as RandomUnderSampler
        n_jobs: Number of parallel jobs (-1 for all cores)

    Example:
        >>> sampler = NearMissUnderSampler(version=1, n_neighbors=3)
        >>> X_resampled, y_resampled = sampler.fit_resample(X, y)
    """

    def __init__(
        self,
        version: int = 1,
        n_neighbors: int = 3,
        sampling_strategy: Union[str, float, Dict] = 'auto',
        n_jobs: int = 1,
        **params
    ):
        super().__init__(name=f"NearMiss-{version}", **params)
        if version not in [1, 2, 3]:
            raise ValueError("version must be 1, 2, or 3")

        self.version = version
        self.n_neighbors = n_neighbors
        self.sampling_strategy = sampling_strategy
        self.n_jobs = n_jobs
        self.sampling_strategy_ = None
        self.class_counts_ = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "NearMissUnderSampler":
        """Fit the NearMiss undersampler."""
        if y is None:
            raise ValueError("Target labels y are required")

        y_array = y.values if isinstance(y, pd.Series) else y
        self.class_counts_ = Counter(y_array)

        # Use RandomUnderSampler's strategy calculation
        random_sampler = RandomUnderSampler(sampling_strategy=self.sampling_strategy)
        random_sampler.class_counts_ = self.class_counts_
        self.sampling_strategy_ = random_sampler._calculate_sampling_strategy(self.class_counts_)

        logger.info(f"NearMiss-{self.version} target distribution: {self.sampling_strategy_}")

        self.fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Tuple:
        """Transform using NearMiss algorithm."""
        self._check_fitted()

        if y is None:
            raise ValueError("Target labels y are required")

        is_dataframe = isinstance(X, pd.DataFrame)
        X_array = X.values if is_dataframe else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # Identify minority and majority classes
        class_counts = Counter(y_array)
        minority_class = min(class_counts, key=class_counts.get)
        majority_classes = [cls for cls in class_counts if cls != minority_class]

        indices_to_keep = []

        # Always keep all minority class samples
        minority_indices = np.where(y_array == minority_class)[0]
        indices_to_keep.extend(minority_indices)

        # Get minority class samples
        X_minority = X_array[minority_indices]

        # Process each majority class
        for maj_class in majority_classes:
            maj_indices = np.where(y_array == maj_class)[0]
            X_majority = X_array[maj_indices]

            target_count = self.sampling_strategy_.get(maj_class, len(maj_indices))
            target_count = min(target_count, len(maj_indices))

            # Apply NearMiss version-specific logic
            if self.version == 1:
                selected = self._nearmiss_v1(X_majority, X_minority, target_count)
            elif self.version == 2:
                selected = self._nearmiss_v2(X_majority, X_minority, target_count)
            else:  # version == 3
                selected = self._nearmiss_v3(X_majority, X_minority, target_count)

            selected_indices = maj_indices[selected]
            indices_to_keep.extend(selected_indices)

        indices_to_keep = np.array(sorted(indices_to_keep))

        X_resampled = X_array[indices_to_keep]
        y_resampled = y_array[indices_to_keep]

        if is_dataframe:
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled = pd.Series(y_resampled, name=y.name if hasattr(y, 'name') else None)

        logger.info(f"NearMiss-{self.version} removed {len(y_array) - len(y_resampled)} samples")

        return X_resampled, y_resampled

    def fit_resample(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Tuple:
        """Fit and resample in one step."""
        return self.fit(X, y).transform(X, y)

    def _nearmiss_v1(self, X_maj: np.ndarray, X_min: np.ndarray, n_samples: int) -> np.ndarray:
        """NearMiss-1: Select majority samples closest to minority samples."""
        knn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(X_min)), n_jobs=self.n_jobs)
        knn.fit(X_min)

        # Average distance to k nearest minority samples
        distances, _ = knn.kneighbors(X_maj)
        avg_distances = distances.mean(axis=1)

        # Select samples with smallest average distance
        selected_indices = np.argsort(avg_distances)[:n_samples]
        return selected_indices

    def _nearmiss_v2(self, X_maj: np.ndarray, X_min: np.ndarray, n_samples: int) -> np.ndarray:
        """NearMiss-2: Select majority samples closest to farthest minority samples."""
        knn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(X_min)), n_jobs=self.n_jobs)
        knn.fit(X_min)

        # Average distance to k farthest minority samples
        distances, _ = knn.kneighbors(X_maj)
        # Use farthest neighbors (reverse order)
        avg_distances = distances[:, ::-1][:, :self.n_neighbors].mean(axis=1)

        # Select samples with smallest average distance to farthest
        selected_indices = np.argsort(avg_distances)[:n_samples]
        return selected_indices

    def _nearmiss_v3(self, X_maj: np.ndarray, X_min: np.ndarray, n_samples: int) -> np.ndarray:
        """NearMiss-3: For each minority sample, select closest majority samples."""
        knn = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=self.n_jobs)
        knn.fit(X_maj)

        # For each minority sample, find k nearest majority samples
        _, indices = knn.kneighbors(X_min)

        # Flatten and count occurrences
        selected_indices = indices.flatten()
        unique_indices, counts = np.unique(selected_indices, return_counts=True)

        # Prioritize samples that are neighbors to more minority samples
        sorted_indices = unique_indices[np.argsort(-counts)]

        # Take top n_samples
        return sorted_indices[:n_samples]


class TomekLinksRemover(PreprocessingStep):
    """
    Remove Tomek links from the dataset.

    A Tomek link exists between two samples from different classes
    that are each other's nearest neighbors. Removing Tomek links
    cleans the decision boundary.

    Parameters:
        sampling_strategy: Which class(es) to remove from Tomek links
            - 'auto': Remove only majority class samples from links
            - 'all': Remove both samples from each link
            - 'majority': Same as 'auto'
        n_jobs: Number of parallel jobs

    Example:
        >>> remover = TomekLinksRemover(sampling_strategy='auto')
        >>> X_resampled, y_resampled = remover.fit_resample(X, y)
    """

    def __init__(
        self,
        sampling_strategy: str = 'auto',
        n_jobs: int = 1,
        **params
    ):
        super().__init__(name="TomekLinksRemover", **params)
        self.sampling_strategy = sampling_strategy
        self.n_jobs = n_jobs

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "TomekLinksRemover":
        """Fit is not needed for Tomek links, but required by interface."""
        self.fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Tuple:
        """Remove Tomek links from the dataset."""
        self._check_fitted()

        if y is None:
            raise ValueError("Target labels y are required")

        is_dataframe = isinstance(X, pd.DataFrame)
        X_array = X.values if is_dataframe else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # Find Tomek links
        tomek_links = self._find_tomek_links(X_array, y_array)

        if len(tomek_links) == 0:
            logger.info("No Tomek links found")
            return X, y

        # Determine which samples to remove
        if self.sampling_strategy in ['auto', 'majority']:
            # Remove only majority class samples from links
            class_counts = Counter(y_array)
            majority_class = max(class_counts, key=class_counts.get)

            indices_to_remove = []
            for idx1, idx2 in tomek_links:
                if y_array[idx1] == majority_class:
                    indices_to_remove.append(idx1)
                if y_array[idx2] == majority_class:
                    indices_to_remove.append(idx2)

        elif self.sampling_strategy == 'all':
            # Remove both samples from each link
            indices_to_remove = []
            for idx1, idx2 in tomek_links:
                indices_to_remove.extend([idx1, idx2])

        else:
            raise ValueError(f"Invalid sampling_strategy: {self.sampling_strategy}")

        # Remove duplicates and create mask
        indices_to_remove = np.unique(indices_to_remove)
        mask = np.ones(len(X_array), dtype=bool)
        mask[indices_to_remove] = False

        X_resampled = X_array[mask]
        y_resampled = y_array[mask]

        if is_dataframe:
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled = pd.Series(y_resampled, name=y.name if hasattr(y, 'name') else None)

        logger.info(f"Removed {len(indices_to_remove)} samples from {len(tomek_links)} Tomek links")

        return X_resampled, y_resampled

    def fit_resample(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Tuple:
        """Fit and resample in one step."""
        return self.fit(X, y).transform(X, y)

    def _find_tomek_links(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[int, int]]:
        """Find all Tomek links in the dataset."""
        # Find nearest neighbor for each sample
        knn = NearestNeighbors(n_neighbors=2, n_jobs=self.n_jobs)  # 2 because first is self
        knn.fit(X)

        _, indices = knn.kneighbors(X)
        nearest_neighbors = indices[:, 1]  # Skip self (first neighbor)

        # Find Tomek links
        tomek_links = []

        for i in range(len(X)):
            nn_i = nearest_neighbors[i]

            # Check if they are each other's nearest neighbors and from different classes
            if nearest_neighbors[nn_i] == i and y[i] != y[nn_i]:
                # Avoid duplicates (i, nn_i) and (nn_i, i)
                if i < nn_i:
                    tomek_links.append((i, nn_i))

        return tomek_links
