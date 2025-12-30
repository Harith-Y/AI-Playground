"""
Precision-Recall Curve Data Generation Module

Provides comprehensive PR (Precision-Recall) curve data generation
for binary and multi-class classification models.

Features:
- Binary classification PR curves
- Multi-class PR curves (One-vs-Rest)
- Average Precision (AP) calculation
- Optimal threshold identification (F1-score maximization)
- Micro and macro averaging for multi-class
- Multiple output formats (dict, DataFrame)
- Visualization-ready data structures

Particularly useful for imbalanced datasets where ROC curves can be misleading.

Based on: ML-TO-DO.md > ML-49
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Literal, Tuple
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    precision_recall_curve as sklearn_pr_curve,
    average_precision_score,
    auc,
)
from app.utils.logger import get_logger

logger = get_logger("pr_curve")


@dataclass
class PRCurveResult:
    """
    Container for Precision-Recall curve data and statistics.
    
    Attributes:
        precision: Precision values
        recall: Recall values
        thresholds: Decision thresholds
        average_precision: Average Precision (AP) score
        optimal_threshold: Threshold that maximizes F1 score
        optimal_threshold_index: Index of optimal threshold
        optimal_f1_score: F1 score at optimal threshold
        class_name: Name of the class (for multi-class)
        n_samples: Number of samples
        n_positive: Number of positive samples
        n_negative: Number of negative samples
        baseline_precision: Baseline precision (proportion of positives)
    """
    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray
    average_precision: float
    optimal_threshold: Optional[float] = None
    optimal_threshold_index: Optional[int] = None
    optimal_f1_score: Optional[float] = None
    class_name: Optional[str] = None
    n_samples: Optional[int] = None
    n_positive: Optional[int] = None
    n_negative: Optional[int] = None
    baseline_precision: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format (JSON-serializable)."""
        result = asdict(self)
        # Convert numpy arrays to lists
        result['precision'] = self.precision.tolist()
        result['recall'] = self.recall.tolist()
        result['thresholds'] = self.thresholds.tolist()
        return result
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert PR curve data to pandas DataFrame."""
        # Note: precision and recall have one more element than thresholds
        # We'll use the first n elements to match thresholds
        n = len(self.thresholds)
        return pd.DataFrame({
            'precision': self.precision[:n],
            'recall': self.recall[:n],
            'threshold': self.thresholds
        })
    
    def get_point_at_threshold(self, threshold: float) -> Tuple[float, float]:
        """
        Get precision and recall at a specific threshold.
        
        Args:
            threshold: Decision threshold
        
        Returns:
            Tuple of (precision, recall) at the given threshold
        """
        # Find closest threshold
        idx = np.argmin(np.abs(self.thresholds - threshold))
        return float(self.precision[idx]), float(self.recall[idx])
    
    def get_threshold_at_recall(self, target_recall: float) -> float:
        """
        Get threshold that achieves target recall.
        
        Args:
            target_recall: Target recall value
        
        Returns:
            Threshold value
        """
        # Recall is in descending order, find closest
        idx = np.argmin(np.abs(self.recall[:-1] - target_recall))
        return float(self.thresholds[idx])
    
    def get_threshold_at_precision(self, target_precision: float) -> float:
        """
        Get threshold that achieves target precision.
        
        Args:
            target_precision: Target precision value
        
        Returns:
            Threshold value
        """
        # Find closest precision
        idx = np.argmin(np.abs(self.precision[:-1] - target_precision))
        return float(self.thresholds[idx])
    
    def __repr__(self) -> str:
        """String representation."""
        opt_threshold_str = f"{self.optimal_threshold:.4f}" if self.optimal_threshold is not None else "N/A"
        opt_f1_str = f"{self.optimal_f1_score:.4f}" if self.optimal_f1_score is not None else "N/A"
        return (
            f"PRCurveResult(\n"
            f"  average_precision={self.average_precision:.4f},\n"
            f"  optimal_threshold={opt_threshold_str},\n"
            f"  optimal_f1_score={opt_f1_str},\n"
            f"  n_samples={self.n_samples},\n"
            f"  class_name='{self.class_name}'\n"
            f")"
        )


@dataclass
class MultiClassPRResult:
    """
    Container for multi-class PR curve data.
    
    Attributes:
        per_class: PR curve for each class
        micro_average: Micro-averaged PR curve
        macro_average: Macro-averaged PR curve
        class_names: Names of classes
        n_classes: Number of classes
        strategy: Multi-class strategy used ('ovr')
    """
    per_class: Dict[str, PRCurveResult]
    micro_average: Optional[PRCurveResult] = None
    macro_average: Optional[PRCurveResult] = None
    class_names: Optional[List[str]] = None
    n_classes: Optional[int] = None
    strategy: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format (JSON-serializable)."""
        result = {
            'per_class': {k: v.to_dict() for k, v in self.per_class.items()},
            'class_names': self.class_names,
            'n_classes': self.n_classes,
            'strategy': self.strategy
        }
        if self.micro_average:
            result['micro_average'] = self.micro_average.to_dict()
        if self.macro_average:
            result['macro_average'] = self.macro_average.to_dict()
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MultiClassPRResult(\n"
            f"  n_classes={self.n_classes},\n"
            f"  strategy='{self.strategy}',\n"
            f"  has_micro_average={self.micro_average is not None},\n"
            f"  has_macro_average={self.macro_average is not None}\n"
            f")"
        )


class PRCurveCalculator:
    """
    Calculator for Precision-Recall curve data generation.
    
    Provides comprehensive PR curve analysis including:
    - Binary classification PR curves
    - Multi-class PR curves (One-vs-Rest)
    - Micro and macro averaging
    - Optimal threshold identification (F1 maximization)
    - Average Precision calculation
    
    PR curves are particularly useful for:
    - Imbalanced datasets
    - When positive class is more important
    - When false positives and false negatives have different costs
    
    Example:
        >>> calculator = PRCurveCalculator()
        >>> result = calculator.compute_binary(
        ...     y_true=[0, 1, 1, 0, 1],
        ...     y_score=[0.1, 0.8, 0.6, 0.2, 0.9]
        ... )
        >>> print(f"AP: {result.average_precision:.4f}")
        >>> print(f"Optimal F1: {result.optimal_f1_score:.4f}")
    """
    
    def __init__(
        self,
        pos_label: Union[int, str] = 1
    ):
        """
        Initialize PR curve calculator.
        
        Args:
            pos_label: Label of positive class for binary classification
        """
        self.pos_label = pos_label
        logger.debug(f"Initialized PRCurveCalculator with pos_label={pos_label}")
    
    def compute_binary(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_score: Union[np.ndarray, pd.Series, List],
        class_name: Optional[str] = None,
        find_optimal_threshold: bool = True
    ) -> PRCurveResult:
        """
        Compute Precision-Recall curve for binary classification.
        
        Args:
            y_true: True binary labels
            y_score: Predicted probabilities or decision scores
            class_name: Name of the positive class
            find_optimal_threshold: Whether to find optimal threshold (max F1)
        
        Returns:
            PRCurveResult with curve data and statistics
        
        Raises:
            ValueError: If inputs have incompatible shapes or invalid values
        """
        # Convert inputs to numpy arrays
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        
        # Validate inputs
        if y_true.shape[0] != y_score.shape[0]:
            raise ValueError(
                f"y_true and y_score must have same length. "
                f"Got {y_true.shape[0]} and {y_score.shape[0]}"
            )
        
        n_samples = len(y_true)
        n_positive = int(np.sum(y_true == self.pos_label))
        n_negative = n_samples - n_positive
        
        # Baseline precision (random classifier)
        baseline_precision = n_positive / n_samples if n_samples > 0 else 0.0
        
        logger.info(
            f"Computing binary PR curve: {n_samples} samples, "
            f"{n_positive} positive, {n_negative} negative, "
            f"baseline={baseline_precision:.4f}"
        )
        
        # Compute PR curve
        precision, recall, thresholds = sklearn_pr_curve(
            y_true, y_score,
            pos_label=self.pos_label
        )
        
        # Calculate Average Precision
        ap_score = average_precision_score(y_true, y_score, pos_label=self.pos_label)
        
        # Find optimal threshold (maximize F1 score)
        optimal_threshold = None
        optimal_idx = None
        optimal_f1 = None
        
        if find_optimal_threshold and len(thresholds) > 0:
            # Calculate F1 scores for each threshold
            # F1 = 2 * (precision * recall) / (precision + recall)
            # Use precision and recall arrays (excluding last element which has no threshold)
            with np.errstate(divide='ignore', invalid='ignore'):
                f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
                f1_scores = np.nan_to_num(f1_scores)  # Replace NaN with 0
            
            optimal_idx = int(np.argmax(f1_scores))
            optimal_threshold = float(thresholds[optimal_idx])
            optimal_f1 = float(f1_scores[optimal_idx])
            
            logger.debug(
                f"Optimal threshold: {optimal_threshold:.4f} "
                f"(Precision={precision[optimal_idx]:.4f}, "
                f"Recall={recall[optimal_idx]:.4f}, F1={optimal_f1:.4f})"
            )
        
        result = PRCurveResult(
            precision=precision,
            recall=recall,
            thresholds=thresholds,
            average_precision=float(ap_score),
            optimal_threshold=optimal_threshold,
            optimal_threshold_index=optimal_idx,
            optimal_f1_score=optimal_f1,
            class_name=class_name or str(self.pos_label),
            n_samples=n_samples,
            n_positive=n_positive,
            n_negative=n_negative,
            baseline_precision=float(baseline_precision)
        )
        
        logger.info(f"Binary PR curve computed: AP={ap_score:.4f}")
        return result
    
    def compute_multiclass(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_score: Union[np.ndarray, pd.DataFrame],
        class_names: Optional[List[str]] = None,
        average: Optional[Literal['micro', 'macro', 'both']] = 'both',
        find_optimal_thresholds: bool = True
    ) -> MultiClassPRResult:
        """
        Compute PR curves for multi-class classification (One-vs-Rest).
        
        Args:
            y_true: True class labels
            y_score: Predicted probabilities for each class (n_samples, n_classes)
            class_names: Names for each class
            average: Averaging strategy ('micro', 'macro', 'both', or None)
            find_optimal_thresholds: Whether to find optimal thresholds
        
        Returns:
            MultiClassPRResult with per-class and averaged curves
        
        Raises:
            ValueError: If inputs have incompatible shapes
        """
        # Convert inputs to numpy arrays
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        
        # Validate inputs
        if y_score.ndim != 2:
            raise ValueError(
                f"y_score must be 2D array with shape (n_samples, n_classes). "
                f"Got shape {y_score.shape}"
            )
        
        if y_true.shape[0] != y_score.shape[0]:
            raise ValueError(
                f"y_true and y_score must have same number of samples. "
                f"Got {y_true.shape[0]} and {y_score.shape[0]}"
            )
        
        n_samples = len(y_true)
        n_classes = y_score.shape[1]
        
        # Get unique classes
        unique_classes = np.unique(y_true)
        if len(unique_classes) != n_classes:
            logger.warning(
                f"Number of unique classes ({len(unique_classes)}) doesn't match "
                f"y_score columns ({n_classes})"
            )
        
        # Generate class names if not provided
        if class_names is None:
            class_names = [f"class_{i}" for i in range(n_classes)]
        elif len(class_names) != n_classes:
            logger.warning(
                f"class_names length ({len(class_names)}) doesn't match "
                f"number of classes ({n_classes}). Using default names."
            )
            class_names = [f"class_{i}" for i in range(n_classes)]
        
        logger.info(
            f"Computing multi-class PR curves: {n_classes} classes, "
            f"{n_samples} samples, average={average}"
        )
        
        # Compute per-class PR curves (One-vs-Rest)
        per_class_results = {}
        
        for i, class_name in enumerate(class_names):
            # Binarize: current class vs all others
            y_true_binary = (y_true == i).astype(int)
            y_score_binary = y_score[:, i]
            
            result = self.compute_binary(
                y_true=y_true_binary,
                y_score=y_score_binary,
                class_name=class_name,
                find_optimal_threshold=find_optimal_thresholds
            )
            
            per_class_results[class_name] = result
        
        # Compute micro-average (aggregate all classes)
        micro_result = None
        if average in ['micro', 'both']:
            micro_result = self._compute_micro_average(y_true, y_score, n_classes)
        
        # Compute macro-average (average of per-class curves)
        macro_result = None
        if average in ['macro', 'both']:
            macro_result = self._compute_macro_average(per_class_results)
        
        result = MultiClassPRResult(
            per_class=per_class_results,
            micro_average=micro_result,
            macro_average=macro_result,
            class_names=class_names,
            n_classes=n_classes,
            strategy='ovr'  # One-vs-Rest
        )
        
        logger.info(f"Multi-class PR curves computed for {n_classes} classes")
        return result
    
    def _compute_micro_average(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        n_classes: int
    ) -> PRCurveResult:
        """
        Compute micro-averaged PR curve.
        
        Micro-averaging aggregates contributions from all classes.
        
        Args:
            y_true: True class labels
            y_score: Predicted probabilities
            n_classes: Number of classes
        
        Returns:
            PRCurveResult for micro-average
        """
        # Binarize labels (one-hot encoding)
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Flatten to treat as binary problem
        y_true_flat = y_true_bin.ravel()
        y_score_flat = y_score.ravel()
        
        # Compute PR curve
        precision, recall, thresholds = sklearn_pr_curve(y_true_flat, y_score_flat)
        ap_score = average_precision_score(y_true_flat, y_score_flat)
        
        # Find optimal threshold
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
            f1_scores = np.nan_to_num(f1_scores)
        
        optimal_idx = int(np.argmax(f1_scores))
        optimal_threshold = float(thresholds[optimal_idx]) if len(thresholds) > 0 else None
        optimal_f1 = float(f1_scores[optimal_idx]) if len(f1_scores) > 0 else None
        
        n_positive = int(np.sum(y_true_flat))
        baseline_precision = n_positive / len(y_true_flat) if len(y_true_flat) > 0 else 0.0
        
        return PRCurveResult(
            precision=precision,
            recall=recall,
            thresholds=thresholds,
            average_precision=float(ap_score),
            optimal_threshold=optimal_threshold,
            optimal_threshold_index=optimal_idx,
            optimal_f1_score=optimal_f1,
            class_name='micro-average',
            n_samples=len(y_true_flat),
            n_positive=n_positive,
            n_negative=len(y_true_flat) - n_positive,
            baseline_precision=float(baseline_precision)
        )
    
    def _compute_macro_average(
        self,
        per_class_results: Dict[str, PRCurveResult]
    ) -> PRCurveResult:
        """
        Compute macro-averaged PR curve.
        
        Macro-averaging computes the average of per-class PR curves.
        
        Args:
            per_class_results: Dictionary of per-class PR results
        
        Returns:
            PRCurveResult for macro-average
        """
        # Interpolate all curves to common recall points
        mean_recall = np.linspace(0, 1, 100)
        precisions = []
        aps = []
        
        for result in per_class_results.values():
            # Interpolate precision at mean recall points
            # Recall is in descending order, so we need to reverse for interpolation
            recall_rev = result.recall[::-1]
            precision_rev = result.precision[::-1]
            
            precision_interp = np.interp(mean_recall, recall_rev, precision_rev)
            precisions.append(precision_interp)
            aps.append(result.average_precision)
        
        # Compute mean precision
        mean_precision = np.mean(precisions, axis=0)
        
        # Compute mean AP
        mean_ap = float(np.mean(aps))
        
        # Generate thresholds (not meaningful for macro-average)
        thresholds = np.linspace(1, 0, len(mean_recall) - 1)
        
        # Find optimal point (max F1)
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (mean_precision[:-1] * mean_recall[:-1]) / (mean_precision[:-1] + mean_recall[:-1])
            f1_scores = np.nan_to_num(f1_scores)
        
        optimal_idx = int(np.argmax(f1_scores))
        optimal_threshold = float(thresholds[optimal_idx]) if len(thresholds) > 0 else None
        optimal_f1 = float(f1_scores[optimal_idx]) if len(f1_scores) > 0 else None
        
        return PRCurveResult(
            precision=mean_precision,
            recall=mean_recall,
            thresholds=thresholds,
            average_precision=mean_ap,
            optimal_threshold=optimal_threshold,
            optimal_threshold_index=optimal_idx,
            optimal_f1_score=optimal_f1,
            class_name='macro-average',
            n_samples=None,
            n_positive=None,
            n_negative=None,
            baseline_precision=None
        )
    
    def compare_models(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_scores: Dict[str, Union[np.ndarray, pd.Series, List]],
        model_names: Optional[List[str]] = None
    ) -> Dict[str, PRCurveResult]:
        """
        Compare PR curves for multiple models.
        
        Args:
            y_true: True binary labels
            y_scores: Dictionary of model_name -> predicted scores
            model_names: Optional list of model names (uses dict keys if None)
        
        Returns:
            Dictionary of model_name -> PRCurveResult
        
        Example:
            >>> results = calculator.compare_models(
            ...     y_true=[0, 1, 1, 0],
            ...     y_scores={
            ...         'model_a': [0.1, 0.8, 0.7, 0.2],
            ...         'model_b': [0.2, 0.9, 0.6, 0.3]
            ...     }
            ... )
        """
        if model_names is None:
            model_names = list(y_scores.keys())
        
        results = {}
        for model_name in model_names:
            if model_name not in y_scores:
                logger.warning(f"Model '{model_name}' not found in y_scores")
                continue
            
            result = self.compute_binary(
                y_true=y_true,
                y_score=y_scores[model_name],
                class_name=model_name
            )
            results[model_name] = result
        
        logger.info(f"Compared {len(results)} models")
        return results


def compute_pr_curve(
    y_true: Union[np.ndarray, pd.Series, List],
    y_score: Union[np.ndarray, pd.Series, pd.DataFrame, List],
    pos_label: Union[int, str] = 1,
    class_names: Optional[List[str]] = None,
    multiclass: bool = False,
    average: Optional[Literal['micro', 'macro', 'both']] = 'both',
    find_optimal_threshold: bool = True
) -> Union[PRCurveResult, MultiClassPRResult]:
    """
    Convenience function to compute Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_score: Predicted probabilities or scores
        pos_label: Positive class label for binary classification
        class_names: Names for each class (multi-class only)
        multiclass: Whether this is multi-class classification
        average: Averaging strategy for multi-class ('micro', 'macro', 'both')
        find_optimal_threshold: Whether to find optimal threshold
    
    Returns:
        PRCurveResult for binary or MultiClassPRResult for multi-class
    
    Example:
        >>> # Binary classification
        >>> result = compute_pr_curve(
        ...     y_true=[0, 1, 1, 0, 1],
        ...     y_score=[0.1, 0.8, 0.6, 0.2, 0.9]
        ... )
        >>> print(f"AP: {result.average_precision:.4f}")
        
        >>> # Multi-class classification
        >>> result = compute_pr_curve(
        ...     y_true=[0, 1, 2, 0, 1, 2],
        ...     y_score=[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], ...],
        ...     multiclass=True,
        ...     class_names=['A', 'B', 'C']
        ... )
    """
    calculator = PRCurveCalculator(pos_label=pos_label)
    
    if multiclass:
        return calculator.compute_multiclass(
            y_true=y_true,
            y_score=y_score,
            class_names=class_names,
            average=average,
            find_optimal_thresholds=find_optimal_threshold
        )
    else:
        return calculator.compute_binary(
            y_true=y_true,
            y_score=y_score,
            find_optimal_threshold=find_optimal_threshold
        )
