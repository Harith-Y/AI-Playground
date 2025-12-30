"""
ROC Curve Data Generation Module

Provides comprehensive ROC (Receiver Operating Characteristic) curve data generation
for binary and multi-class classification models.

Features:
- Binary classification ROC curves
- Multi-class ROC curves (One-vs-Rest and One-vs-One)
- Micro and macro averaging for multi-class
- AUC (Area Under Curve) calculation
- Optimal threshold identification
- Multiple output formats (dict, DataFrame)
- Visualization-ready data structures

Based on: ML-TO-DO.md > ML-48
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Literal, Tuple
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    roc_curve as sklearn_roc_curve,
    auc,
    roc_auc_score,
)
from app.utils.logger import get_logger

logger = get_logger("roc_curve")


@dataclass
class ROCCurveResult:
    """
    Container for ROC curve data and statistics.
    
    Attributes:
        fpr: False Positive Rate values
        tpr: True Positive Rate values
        thresholds: Decision thresholds
        auc_score: Area Under Curve
        optimal_threshold: Threshold that maximizes Youden's J statistic
        optimal_threshold_index: Index of optimal threshold
        class_name: Name of the class (for multi-class)
        n_samples: Number of samples
        n_positive: Number of positive samples
        n_negative: Number of negative samples
    """
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc_score: float
    optimal_threshold: Optional[float] = None
    optimal_threshold_index: Optional[int] = None
    class_name: Optional[str] = None
    n_samples: Optional[int] = None
    n_positive: Optional[int] = None
    n_negative: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format (JSON-serializable)."""
        result = asdict(self)
        # Convert numpy arrays to lists
        result['fpr'] = self.fpr.tolist()
        result['tpr'] = self.tpr.tolist()
        result['thresholds'] = self.thresholds.tolist()
        return result
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert ROC curve data to pandas DataFrame."""
        return pd.DataFrame({
            'fpr': self.fpr,
            'tpr': self.tpr,
            'threshold': self.thresholds
        })
    
    def get_point_at_threshold(self, threshold: float) -> Tuple[float, float]:
        """
        Get FPR and TPR at a specific threshold.
        
        Args:
            threshold: Decision threshold
        
        Returns:
            Tuple of (fpr, tpr) at the given threshold
        """
        # Find closest threshold
        idx = np.argmin(np.abs(self.thresholds - threshold))
        return float(self.fpr[idx]), float(self.tpr[idx])
    
    def get_threshold_at_fpr(self, target_fpr: float) -> float:
        """
        Get threshold that achieves target FPR.
        
        Args:
            target_fpr: Target false positive rate
        
        Returns:
            Threshold value
        """
        idx = np.argmin(np.abs(self.fpr - target_fpr))
        return float(self.thresholds[idx])
    
    def get_threshold_at_tpr(self, target_tpr: float) -> float:
        """
        Get threshold that achieves target TPR.
        
        Args:
            target_tpr: Target true positive rate
        
        Returns:
            Threshold value
        """
        idx = np.argmin(np.abs(self.tpr - target_tpr))
        return float(self.thresholds[idx])
    
    def __repr__(self) -> str:
        """String representation."""
        opt_threshold_str = f"{self.optimal_threshold:.4f}" if self.optimal_threshold is not None else "N/A"
        return (
            f"ROCCurveResult(\n"
            f"  auc_score={self.auc_score:.4f},\n"
            f"  optimal_threshold={opt_threshold_str},\n"
            f"  n_samples={self.n_samples},\n"
            f"  class_name='{self.class_name}'\n"
            f")"
        )


@dataclass
class MultiClassROCResult:
    """
    Container for multi-class ROC curve data.
    
    Attributes:
        per_class: ROC curve for each class
        micro_average: Micro-averaged ROC curve
        macro_average: Macro-averaged ROC curve
        class_names: Names of classes
        n_classes: Number of classes
        strategy: Multi-class strategy used ('ovr' or 'ovo')
    """
    per_class: Dict[str, ROCCurveResult]
    micro_average: Optional[ROCCurveResult] = None
    macro_average: Optional[ROCCurveResult] = None
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
            f"MultiClassROCResult(\n"
            f"  n_classes={self.n_classes},\n"
            f"  strategy='{self.strategy}',\n"
            f"  has_micro_average={self.micro_average is not None},\n"
            f"  has_macro_average={self.macro_average is not None}\n"
            f")"
        )


class ROCCurveCalculator:
    """
    Calculator for ROC curve data generation.
    
    Provides comprehensive ROC curve analysis including:
    - Binary classification ROC curves
    - Multi-class ROC curves (One-vs-Rest)
    - Micro and macro averaging
    - Optimal threshold identification
    - AUC calculation
    
    Example:
        >>> calculator = ROCCurveCalculator()
        >>> result = calculator.compute_binary(
        ...     y_true=[0, 1, 1, 0, 1],
        ...     y_score=[0.1, 0.8, 0.6, 0.2, 0.9]
        ... )
        >>> print(f"AUC: {result.auc_score:.4f}")
        >>> print(f"Optimal threshold: {result.optimal_threshold:.4f}")
    """
    
    def __init__(
        self,
        pos_label: Union[int, str] = 1,
        drop_intermediate: bool = True
    ):
        """
        Initialize ROC curve calculator.
        
        Args:
            pos_label: Label of positive class for binary classification
            drop_intermediate: Whether to drop suboptimal thresholds
        """
        self.pos_label = pos_label
        self.drop_intermediate = drop_intermediate
        logger.debug(f"Initialized ROCCurveCalculator with pos_label={pos_label}")
    
    def compute_binary(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_score: Union[np.ndarray, pd.Series, List],
        class_name: Optional[str] = None,
        find_optimal_threshold: bool = True
    ) -> ROCCurveResult:
        """
        Compute ROC curve for binary classification.
        
        Args:
            y_true: True binary labels
            y_score: Predicted probabilities or decision scores
            class_name: Name of the positive class
            find_optimal_threshold: Whether to find optimal threshold
        
        Returns:
            ROCCurveResult with curve data and statistics
        
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
        
        logger.info(
            f"Computing binary ROC curve: {n_samples} samples, "
            f"{n_positive} positive, {n_negative} negative"
        )
        
        # Compute ROC curve
        fpr, tpr, thresholds = sklearn_roc_curve(
            y_true, y_score,
            pos_label=self.pos_label,
            drop_intermediate=self.drop_intermediate
        )
        
        # Calculate AUC
        auc_score = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's J statistic)
        optimal_threshold = None
        optimal_idx = None
        
        if find_optimal_threshold:
            # Youden's J = TPR - FPR (maximize this)
            j_scores = tpr - fpr
            optimal_idx = int(np.argmax(j_scores))
            optimal_threshold = float(thresholds[optimal_idx])
            
            logger.debug(
                f"Optimal threshold: {optimal_threshold:.4f} "
                f"(TPR={tpr[optimal_idx]:.4f}, FPR={fpr[optimal_idx]:.4f})"
            )
        
        result = ROCCurveResult(
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            auc_score=float(auc_score),
            optimal_threshold=optimal_threshold,
            optimal_threshold_index=optimal_idx,
            class_name=class_name or str(self.pos_label),
            n_samples=n_samples,
            n_positive=n_positive,
            n_negative=n_negative
        )
        
        logger.info(f"Binary ROC curve computed: AUC={auc_score:.4f}")
        return result
    
    def compute_multiclass(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_score: Union[np.ndarray, pd.DataFrame],
        class_names: Optional[List[str]] = None,
        average: Optional[Literal['micro', 'macro', 'both']] = 'both',
        find_optimal_thresholds: bool = True
    ) -> MultiClassROCResult:
        """
        Compute ROC curves for multi-class classification (One-vs-Rest).
        
        Args:
            y_true: True class labels
            y_score: Predicted probabilities for each class (n_samples, n_classes)
            class_names: Names for each class
            average: Averaging strategy ('micro', 'macro', 'both', or None)
            find_optimal_thresholds: Whether to find optimal thresholds
        
        Returns:
            MultiClassROCResult with per-class and averaged curves
        
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
            f"Computing multi-class ROC curves: {n_classes} classes, "
            f"{n_samples} samples, average={average}"
        )
        
        # Compute per-class ROC curves (One-vs-Rest)
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
        
        result = MultiClassROCResult(
            per_class=per_class_results,
            micro_average=micro_result,
            macro_average=macro_result,
            class_names=class_names,
            n_classes=n_classes,
            strategy='ovr'  # One-vs-Rest
        )
        
        logger.info(f"Multi-class ROC curves computed for {n_classes} classes")
        return result
    
    def _compute_micro_average(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        n_classes: int
    ) -> ROCCurveResult:
        """
        Compute micro-averaged ROC curve.
        
        Micro-averaging aggregates contributions from all classes.
        
        Args:
            y_true: True class labels
            y_score: Predicted probabilities
            n_classes: Number of classes
        
        Returns:
            ROCCurveResult for micro-average
        """
        # Binarize labels (one-hot encoding)
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Flatten to treat as binary problem
        y_true_flat = y_true_bin.ravel()
        y_score_flat = y_score.ravel()
        
        # Compute ROC curve
        fpr, tpr, thresholds = sklearn_roc_curve(y_true_flat, y_score_flat)
        auc_score = auc(fpr, tpr)
        
        # Find optimal threshold
        j_scores = tpr - fpr
        optimal_idx = int(np.argmax(j_scores))
        optimal_threshold = float(thresholds[optimal_idx])
        
        return ROCCurveResult(
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            auc_score=float(auc_score),
            optimal_threshold=optimal_threshold,
            optimal_threshold_index=optimal_idx,
            class_name='micro-average',
            n_samples=len(y_true_flat),
            n_positive=int(np.sum(y_true_flat)),
            n_negative=int(len(y_true_flat) - np.sum(y_true_flat))
        )
    
    def _compute_macro_average(
        self,
        per_class_results: Dict[str, ROCCurveResult]
    ) -> ROCCurveResult:
        """
        Compute macro-averaged ROC curve.
        
        Macro-averaging computes the average of per-class ROC curves.
        
        Args:
            per_class_results: Dictionary of per-class ROC results
        
        Returns:
            ROCCurveResult for macro-average
        """
        # Interpolate all curves to common FPR points
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        
        for result in per_class_results.values():
            # Interpolate TPR at mean FPR points
            tpr_interp = np.interp(mean_fpr, result.fpr, result.tpr)
            tpr_interp[0] = 0.0  # Ensure starts at 0
            tprs.append(tpr_interp)
            aucs.append(result.auc_score)
        
        # Compute mean TPR
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0  # Ensure ends at 1
        
        # Compute mean AUC
        mean_auc = float(np.mean(aucs))
        
        # Generate thresholds (not meaningful for macro-average)
        thresholds = np.linspace(1, 0, len(mean_fpr))
        
        # Find optimal point
        j_scores = mean_tpr - mean_fpr
        optimal_idx = int(np.argmax(j_scores))
        optimal_threshold = float(thresholds[optimal_idx])
        
        return ROCCurveResult(
            fpr=mean_fpr,
            tpr=mean_tpr,
            thresholds=thresholds,
            auc_score=mean_auc,
            optimal_threshold=optimal_threshold,
            optimal_threshold_index=optimal_idx,
            class_name='macro-average',
            n_samples=None,
            n_positive=None,
            n_negative=None
        )
    
    def compare_models(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_scores: Dict[str, Union[np.ndarray, pd.Series, List]],
        model_names: Optional[List[str]] = None
    ) -> Dict[str, ROCCurveResult]:
        """
        Compare ROC curves for multiple models.
        
        Args:
            y_true: True binary labels
            y_scores: Dictionary of model_name -> predicted scores
            model_names: Optional list of model names (uses dict keys if None)
        
        Returns:
            Dictionary of model_name -> ROCCurveResult
        
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


def compute_roc_curve(
    y_true: Union[np.ndarray, pd.Series, List],
    y_score: Union[np.ndarray, pd.Series, pd.DataFrame, List],
    pos_label: Union[int, str] = 1,
    class_names: Optional[List[str]] = None,
    multiclass: bool = False,
    average: Optional[Literal['micro', 'macro', 'both']] = 'both',
    find_optimal_threshold: bool = True
) -> Union[ROCCurveResult, MultiClassROCResult]:
    """
    Convenience function to compute ROC curve.
    
    Args:
        y_true: True labels
        y_score: Predicted probabilities or scores
        pos_label: Positive class label for binary classification
        class_names: Names for each class (multi-class only)
        multiclass: Whether this is multi-class classification
        average: Averaging strategy for multi-class ('micro', 'macro', 'both')
        find_optimal_threshold: Whether to find optimal threshold
    
    Returns:
        ROCCurveResult for binary or MultiClassROCResult for multi-class
    
    Example:
        >>> # Binary classification
        >>> result = compute_roc_curve(
        ...     y_true=[0, 1, 1, 0, 1],
        ...     y_score=[0.1, 0.8, 0.6, 0.2, 0.9]
        ... )
        >>> print(f"AUC: {result.auc_score:.4f}")
        
        >>> # Multi-class classification
        >>> result = compute_roc_curve(
        ...     y_true=[0, 1, 2, 0, 1, 2],
        ...     y_score=[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], ...],
        ...     multiclass=True,
        ...     class_names=['A', 'B', 'C']
        ... )
    """
    calculator = ROCCurveCalculator(pos_label=pos_label)
    
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
