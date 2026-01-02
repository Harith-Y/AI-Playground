"""
Data splitting utilities for train/validation/test splits.

This module provides flexible functions for splitting datasets into
train, validation, and test sets with various options for stratification,
shuffling, and custom split ratios.
"""

from typing import Union, Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sklearn_split
import logging

logger = logging.getLogger(__name__)


class DataSplitResult:
    """
    Container for data split results.
    
    Stores the split datasets and metadata about the split operation.
    """
    
    def __init__(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray, None],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None,
        split_ratios: Optional[Tuple[float, ...]] = None,
        stratified: bool = False,
        shuffled: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize data split result.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            X_test: Test features (optional)
            y_test: Test target (optional)
            split_ratios: Tuple of split ratios (train, val, test)
            stratified: Whether stratification was used
            shuffled: Whether data was shuffled
            random_state: Random seed used
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.split_ratios = split_ratios
        self.stratified = stratified
        self.shuffled = shuffled
        self.random_state = random_state
    
    def get_train_data(self) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray, None]]:
        """Get training data."""
        return self.X_train, self.y_train
    
    def get_val_data(self) -> Optional[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]]:
        """Get validation data if available."""
        if self.X_val is not None and self.y_val is not None:
            return self.X_val, self.y_val
        return None
    
    def get_test_data(self) -> Optional[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]]:
        """Get test data if available."""
        if self.X_test is not None and self.y_test is not None:
            return self.X_test, self.y_test
        return None
    
    def get_split_sizes(self) -> dict:
        """Get sizes of each split."""
        sizes = {
            'train': len(self.X_train),
            'val': len(self.X_val) if self.X_val is not None else 0,
            'test': len(self.X_test) if self.X_test is not None else 0,
            'total': len(self.X_train)
        }
        
        if self.X_val is not None:
            sizes['total'] += len(self.X_val)
        if self.X_test is not None:
            sizes['total'] += len(self.X_test)
        
        return sizes
    
    def __repr__(self) -> str:
        """String representation."""
        sizes = self.get_split_sizes()
        parts = [f"train={sizes['train']}"]
        
        if sizes['val'] > 0:
            parts.append(f"val={sizes['val']}")
        if sizes['test'] > 0:
            parts.append(f"test={sizes['test']}")
        
        return f"DataSplitResult({', '.join(parts)}, total={sizes['total']})"


def train_test_split(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: bool = False
) -> DataSplitResult:
    """
    Split data into train and test sets.
    
    Args:
        X: Features
        y: Target (optional for unsupervised)
        test_size: Proportion of data for test set (0.0 to 1.0)
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle data before splitting
        stratify: Whether to use stratified splitting (requires y)
    
    Returns:
        DataSplitResult containing train and test sets
    
    Raises:
        ValueError: If inputs are invalid
    
    Example:
        >>> result = train_test_split(X, y, test_size=0.2, random_state=42)
        >>> X_train, y_train = result.get_train_data()
        >>> X_test, y_test = result.get_test_data()
    """
    # Validate inputs
    _validate_split_inputs(X, y, test_size, stratify)
    
    logger.info(f"Splitting data: test_size={test_size}, shuffle={shuffle}, stratify={stratify}")
    
    # Determine stratification
    stratify_array = y if (stratify and y is not None) else None
    
    # Perform split
    if y is not None:
        X_train, X_test, y_train, y_test = sklearn_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_array
        )
    else:
        # Unsupervised (no y)
        X_train, X_test = sklearn_split(
            X,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle
        )
        y_train = None
        y_test = None
    
    # Calculate actual ratios
    train_ratio = len(X_train) / (len(X_train) + len(X_test))
    test_ratio = len(X_test) / (len(X_train) + len(X_test))
    
    logger.info(f"Split complete: train={len(X_train)}, test={len(X_test)}")
    logger.info(f"Actual ratios: train={train_ratio:.3f}, test={test_ratio:.3f}")
    
    return DataSplitResult(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        split_ratios=(train_ratio, 0.0, test_ratio),
        stratified=stratify,
        shuffled=shuffle,
        random_state=random_state
    )


def train_val_test_split(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: bool = False
) -> DataSplitResult:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Features
        y: Target (optional for unsupervised)
        train_size: Proportion of data for training (0.0 to 1.0)
        val_size: Proportion of data for validation (0.0 to 1.0)
        test_size: Proportion of data for test (0.0 to 1.0)
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle data before splitting
        stratify: Whether to use stratified splitting (requires y)
    
    Returns:
        DataSplitResult containing train, validation, and test sets
    
    Raises:
        ValueError: If inputs are invalid or sizes don't sum to 1.0
    
    Example:
        >>> result = train_val_test_split(
        ...     X, y,
        ...     train_size=0.7,
        ...     val_size=0.15,
        ...     test_size=0.15,
        ...     random_state=42
        ... )
        >>> X_train, y_train = result.get_train_data()
        >>> X_val, y_val = result.get_val_data()
        >>> X_test, y_test = result.get_test_data()
    """
    # Validate inputs
    _validate_split_inputs(X, y, test_size, stratify)
    
    # Validate sizes sum to 1.0
    total_size = train_size + val_size + test_size
    if not np.isclose(total_size, 1.0, atol=1e-6):
        raise ValueError(
            f"train_size + val_size + test_size must equal 1.0. "
            f"Got: {train_size} + {val_size} + {test_size} = {total_size}"
        )
    
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError("All split sizes must be greater than 0")
    
    logger.info(
        f"Splitting data: train={train_size}, val={val_size}, test={test_size}, "
        f"shuffle={shuffle}, stratify={stratify}"
    )
    
    # Determine stratification
    stratify_array = y if (stratify and y is not None) else None
    
    # First split: separate test set
    test_ratio = test_size / (train_size + val_size + test_size)
    
    if y is not None:
        X_temp, X_test, y_temp, y_test = sklearn_split(
            X, y,
            test_size=test_ratio,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_array
        )
    else:
        X_temp, X_test = sklearn_split(
            X,
            test_size=test_ratio,
            random_state=random_state,
            shuffle=shuffle
        )
        y_temp = None
        y_test = None
    
    # Second split: separate train and validation
    val_ratio = val_size / (train_size + val_size)
    stratify_temp = y_temp if (stratify and y_temp is not None) else None
    
    if y_temp is not None:
        X_train, X_val, y_train, y_val = sklearn_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_temp
        )
    else:
        X_train, X_val = sklearn_split(
            X_temp,
            test_size=val_ratio,
            random_state=random_state,
            shuffle=shuffle
        )
        y_train = None
        y_val = None
    
    # Calculate actual ratios
    total_samples = len(X_train) + len(X_val) + len(X_test)
    actual_train_ratio = len(X_train) / total_samples
    actual_val_ratio = len(X_val) / total_samples
    actual_test_ratio = len(X_test) / total_samples
    
    logger.info(
        f"Split complete: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )
    logger.info(
        f"Actual ratios: train={actual_train_ratio:.3f}, "
        f"val={actual_val_ratio:.3f}, test={actual_test_ratio:.3f}"
    )
    
    return DataSplitResult(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        split_ratios=(actual_train_ratio, actual_val_ratio, actual_test_ratio),
        stratified=stratify,
        shuffled=shuffle,
        random_state=random_state
    )


def split_by_ratio(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    ratios: Tuple[float, ...] = (0.7, 0.15, 0.15),
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: bool = False
) -> DataSplitResult:
    """
    Split data by custom ratios.
    
    This is a convenience function that calls train_val_test_split with
    custom ratios.
    
    Args:
        X: Features
        y: Target (optional for unsupervised)
        ratios: Tuple of split ratios (train, val, test) or (train, test)
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle data before splitting
        stratify: Whether to use stratified splitting (requires y)
    
    Returns:
        DataSplitResult containing split datasets
    
    Raises:
        ValueError: If ratios are invalid
    
    Example:
        >>> # 70/15/15 split
        >>> result = split_by_ratio(X, y, ratios=(0.7, 0.15, 0.15))
        >>> 
        >>> # 80/20 split (no validation)
        >>> result = split_by_ratio(X, y, ratios=(0.8, 0.2))
    """
    if len(ratios) == 2:
        # Train/test split only
        train_ratio, test_ratio = ratios
        return train_test_split(
            X, y,
            test_size=test_ratio,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify
        )
    elif len(ratios) == 3:
        # Train/val/test split
        train_ratio, val_ratio, test_ratio = ratios
        return train_val_test_split(
            X, y,
            train_size=train_ratio,
            val_size=val_ratio,
            test_size=test_ratio,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify
        )
    else:
        raise ValueError(
            f"ratios must be a tuple of 2 or 3 values. Got {len(ratios)} values."
        )


def temporal_split(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15
) -> DataSplitResult:
    """
    Split time-series data without shuffling.
    
    This function splits data sequentially, preserving temporal order.
    Useful for time-series forecasting where future data should not
    leak into training.
    
    Args:
        X: Features (should be ordered by time)
        y: Target (optional for unsupervised)
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        test_size: Proportion of data for test
    
    Returns:
        DataSplitResult containing sequential splits
    
    Raises:
        ValueError: If sizes don't sum to 1.0
    
    Example:
        >>> # For time-series data
        >>> result = temporal_split(
        ...     X_timeseries, y_timeseries,
        ...     train_size=0.7,
        ...     val_size=0.15,
        ...     test_size=0.15
        ... )
    """
    logger.info("Performing temporal split (no shuffling)")
    
    return train_val_test_split(
        X, y,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=None,
        shuffle=False,  # Critical: no shuffling for temporal data
        stratify=False  # No stratification for temporal data
    )


def _validate_split_inputs(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]],
    test_size: float,
    stratify: bool
) -> None:
    """
    Validate split inputs.
    
    Args:
        X: Features
        y: Target
        test_size: Test size ratio
        stratify: Whether stratification is requested
    
    Raises:
        ValueError: If inputs are invalid
    """
    # Check X is not empty
    if X is None or len(X) == 0:
        raise ValueError("X cannot be empty")
    
    # Check test_size is valid
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0.0 and 1.0. Got: {test_size}")
    
    # Check X and y have same length
    if y is not None and len(X) != len(y):
        raise ValueError(
            f"X and y must have same length. Got X: {len(X)}, y: {len(y)}"
        )
    
    # Check stratification requirements
    if stratify and y is None:
        raise ValueError("stratify=True requires y to be provided")
    
    # Check minimum samples
    min_samples = 2
    if len(X) < min_samples:
        raise ValueError(
            f"Need at least {min_samples} samples for splitting. Got: {len(X)}"
        )
    
    # Check if split will result in valid sets
    n_test = int(len(X) * test_size)
    n_train = len(X) - n_test
    
    if n_train < 1:
        raise ValueError(
            f"Training set would have {n_train} samples with test_size={test_size}. "
            f"Reduce test_size or increase dataset size."
        )
    
    if n_test < 1:
        raise ValueError(
            f"Test set would have {n_test} samples with test_size={test_size}. "
            f"Increase test_size or dataset size."
        )
    
    # Warn about very small splits
    if n_train < 5 or n_test < 5:
        logger.warning(
            f"Very small split sizes detected: train={n_train}, test={n_test}. "
            "Results may not be reliable. Consider using cross-validation instead."
        )
    
    # Additional validation for stratification
    if stratify and y is not None:
        # Check class distribution
        if hasattr(y, 'value_counts'):
            class_counts = y.value_counts()
        else:
            unique, counts = np.unique(y, return_counts=True)
            class_counts = pd.Series(counts, index=unique)
        
        min_class_count = class_counts.min()
        n_classes = len(class_counts)
        
        # Need at least 2 samples per class for stratified split
        if min_class_count < 2:
            raise ValueError(
                f"Stratified split requires at least 2 samples per class. "
                f"Found class(es) with only {min_class_count} sample(s). "
                "Either remove these classes, disable stratification, or collect more data."
            )
        
        # Check if test set can have all classes
        if n_test < n_classes:
            logger.warning(
                f"Test set ({n_test} samples) is smaller than number of classes ({n_classes}). "
                "Some classes may not appear in test set. Consider increasing test_size."
            )
    
    logger.debug("Split inputs validated successfully")


def get_split_info(result: DataSplitResult) -> dict:
    """
    Get comprehensive information about a data split.
    
    Args:
        result: DataSplitResult object
    
    Returns:
        Dictionary containing split information
    
    Example:
        >>> result = train_val_test_split(X, y)
        >>> info = get_split_info(result)
        >>> print(info['sizes'])
        >>> print(info['ratios'])
    """
    sizes = result.get_split_sizes()
    
    info = {
        'sizes': sizes,
        'ratios': result.split_ratios,
        'stratified': result.stratified,
        'shuffled': result.shuffled,
        'random_state': result.random_state,
        'has_validation': result.X_val is not None,
        'has_test': result.X_test is not None
    }
    
    # Calculate percentages
    total = sizes['total']
    info['percentages'] = {
        'train': (sizes['train'] / total * 100) if total > 0 else 0,
        'val': (sizes['val'] / total * 100) if total > 0 else 0,
        'test': (sizes['test'] / total * 100) if total > 0 else 0
    }
    
    return info
