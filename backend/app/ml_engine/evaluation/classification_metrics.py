"""
Classification Metrics Module

Provides comprehensive evaluation metrics for classification models including:
- Basic metrics: accuracy, precision, recall, F1-score
- Advanced metrics: AUC-ROC, AUC-PR, Matthews correlation coefficient
- Multi-class support: macro, micro, weighted averaging
- Confusion matrix utilities
- Per-class metrics breakdown

Based on: ML-TO-DO.md > ML-46
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Literal, Tuple
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss,
    balanced_accuracy_score,
)
from app.utils.logger import get_logger

logger = get_logger("classification_metrics")


@dataclass
class ClassificationMetrics:
    """
    Container for classification evaluation metrics.
    
    Attributes:
        accuracy: Overall accuracy (correct predictions / total predictions)
        precision: Precision score (TP / (TP + FP))
        recall: Recall/Sensitivity score (TP / (TP + FN))
        f1_score: F1 score (harmonic mean of precision and recall)
        auc_roc: Area Under ROC Curve (binary/multiclass)
        auc_pr: Area Under Precision-Recall Curve
        roc_curve: calculated ROC curve points (FPR, TPR, thresholds) - downsampled
        pr_curve: calculated Precision-Recall curve points - downsampled
        balanced_accuracy: Balanced accuracy (average of recall per class)
        matthews_corrcoef: Matthews Correlation Coefficient
        cohen_kappa: Cohen's Kappa score
        log_loss: Logarithmic loss (if probabilities available)
        confusion_matrix: Confusion matrix as 2D array
        per_class_metrics: Metrics breakdown per class
        support: Number of samples per class
        n_samples: Total number of samples
        n_classes: Number of classes
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    roc_curve: Optional[Dict] = None
    pr_curve: Optional[Dict] = None
    balanced_accuracy: Optional[float] = None
    matthews_corrcoef: Optional[float] = None
    cohen_kappa: Optional[float] = None
    log_loss: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    support: Optional[Dict[str, int]] = None
    n_samples: Optional[int] = None
    n_classes: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary format."""
        result = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        if self.confusion_matrix is not None:
            result['confusion_matrix'] = self.confusion_matrix.tolist()
        return result
    
    def __repr__(self) -> str:
        """String representation of metrics."""
        auc_roc_str = f"{self.auc_roc:.4f}" if self.auc_roc is not None else "N/A"
        return (
            f"ClassificationMetrics(\n"
            f"  accuracy={self.accuracy:.4f},\n"
            f"  precision={self.precision:.4f},\n"
            f"  recall={self.recall:.4f},\n"
            f"  f1_score={self.f1_score:.4f},\n"
            f"  auc_roc={auc_roc_str},\n"
            f"  n_samples={self.n_samples},\n"
            f"  n_classes={self.n_classes}\n"
            f")"
        )


class ClassificationMetricsCalculator:
    """
    Calculator for classification model evaluation metrics.
    
    Supports:
    - Binary classification
    - Multi-class classification (one-vs-rest)
    - Multiple averaging strategies (macro, micro, weighted)
    - Probability-based metrics (AUC-ROC, AUC-PR, log loss)
    
    Example:
        >>> calculator = ClassificationMetricsCalculator()
        >>> metrics = calculator.calculate_metrics(
        ...     y_true=[0, 1, 1, 0, 1],
        ...     y_pred=[0, 1, 0, 0, 1],
        ...     y_proba=[[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.8, 0.2], [0.1, 0.9]]
        ... )
        >>> print(f"Accuracy: {metrics.accuracy:.4f}")
        >>> print(f"F1 Score: {metrics.f1_score:.4f}")
    """
    
    def __init__(
        self,
        average: Literal['binary', 'micro', 'macro', 'weighted'] = 'weighted',
        pos_label: Union[int, str] = 1,
        labels: Optional[List] = None,
        zero_division: Union[Literal['warn'], int] = 'warn'
    ):
        """
        Initialize metrics calculator.
        
        Args:
            average: Averaging strategy for multi-class metrics
                - 'binary': For binary classification only
                - 'micro': Calculate globally (all classes together)
                - 'macro': Calculate per class, then average (unweighted)
                - 'weighted': Calculate per class, then weighted average by support
            pos_label: Positive class label for binary classification
            labels: List of class labels (auto-detected if None)
            zero_division: Value to return when division by zero occurs
        """
        self.average = average
        self.pos_label = pos_label
        self.labels = labels
        self.zero_division = zero_division
        logger.debug(f"Initialized ClassificationMetricsCalculator with average={average}")
    
    def calculate_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        y_proba: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        class_names: Optional[List[str]] = None,
        include_per_class: bool = True,
        include_advanced: bool = True
    ) -> ClassificationMetrics:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for AUC metrics)
                - Binary: shape (n_samples, 2) or (n_samples,)
                - Multi-class: shape (n_samples, n_classes)
            class_names: Names for each class (for reporting)
            include_per_class: Whether to calculate per-class metrics
            include_advanced: Whether to calculate advanced metrics (MCC, Kappa, etc.)
        
        Returns:
            ClassificationMetrics object with all calculated metrics
        
        Raises:
            ValueError: If inputs have incompatible shapes or invalid values
        """
        # Convert inputs to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Validate inputs
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"y_true and y_pred must have same length. "
                f"Got {y_true.shape[0]} and {y_pred.shape[0]}"
            )
        
        if y_proba is not None:
            y_proba = np.asarray(y_proba)
            if y_proba.shape[0] != y_true.shape[0]:
                raise ValueError(
                    f"y_proba must have same number of samples as y_true. "
                    f"Got {y_proba.shape[0]} and {y_true.shape[0]}"
                )
        
        # Detect number of classes
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(unique_classes)
        n_samples = len(y_true)
        
        # Determine if binary or multi-class
        is_binary = n_classes == 2
        
        # Auto-detect average strategy if not set
        if self.average == 'binary' and not is_binary:
            logger.warning(
                f"average='binary' specified but {n_classes} classes detected. "
                f"Switching to 'weighted'"
            )
            average = 'weighted'
        else:
            average = self.average if not is_binary else 'binary'
        
        logger.info(
            f"Calculating metrics for {n_classes} classes, "
            f"{n_samples} samples, average={average}"
        )
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true, y_pred,
            average=average,
            zero_division=self.zero_division,
            labels=self.labels
        )
        recall = recall_score(
            y_true, y_pred,
            average=average,
            zero_division=self.zero_division,
            labels=self.labels
        )
        f1 = f1_score(
            y_true, y_pred,
            average=average,
            zero_division=self.zero_division,
            labels=self.labels
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=self.labels)
        
        # Calculate probability-based metrics
        auc_roc = None
        auc_pr = None
        logloss = None
        roc_curve_data = None
        pr_curve_data = None
        
        if y_proba is not None:
            try:
                if is_binary:
                    # Binary classification
                    # Handle both (n, 2) and (n,) shapes
                    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                        y_proba_pos = y_proba[:, 1]
                    else:
                        y_proba_pos = y_proba.ravel()
                    
                    auc_roc = roc_auc_score(y_true, y_proba_pos)
                    auc_pr = average_precision_score(y_true, y_proba_pos)
                    
                    # Generate ROC Curve data
                    fpr, tpr, thresholds = roc_curve(y_true, y_proba_pos, pos_label=self.pos_label)
                    roc_curve_data = self._downsample_curve(fpr, tpr, thresholds)
                    
                    # Generate PR Curve data
                    precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_true, y_proba_pos, pos_label=self.pos_label)
                    pr_curve_data = self._downsample_curve(recall_curve, precision_curve, thresholds_pr)
                    
                else:
                    # Multi-class classification (one-vs-rest)
                    auc_roc = roc_auc_score(
                        y_true, y_proba,
                        multi_class='ovr',
                        average=average if average != 'binary' else 'weighted'
                    )
                    auc_pr = average_precision_score(
                        y_true, y_proba,
                        average=average if average != 'binary' else 'weighted'
                    )
                
                # Log loss
                logloss = log_loss(y_true, y_proba, labels=self.labels)
                
            except Exception as e:
                logger.warning(f"Could not calculate probability-based metrics: {e}")
        
        # Calculate advanced metrics
        balanced_acc = None
        mcc = None
        kappa = None
        
        if include_advanced:
            try:
                balanced_acc = balanced_accuracy_score(y_true, y_pred)
                mcc = matthews_corrcoef(y_true, y_pred)
                kappa = cohen_kappa_score(y_true, y_pred)
            except Exception as e:
                logger.warning(f"Could not calculate advanced metrics: {e}")
        
        # Calculate per-class metrics
        per_class_metrics = None
        support_dict = None
        
        if include_per_class:
            per_class_metrics, support_dict = self._calculate_per_class_metrics(
                y_true, y_pred, y_proba, class_names, unique_classes
            )
        
        # Create metrics object
        metrics = ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            roc_curve=roc_curve_data,
            pr_curve=pr_curve_data,
            balanced_accuracy=balanced_acc,
            matthews_corrcoef=mcc,
            cohen_kappa=kappa,
            log_loss=logloss,
            confusion_matrix=cm,
            per_class_metrics=per_class_metrics,
            support=support_dict,
            n_samples=n_samples,
            n_classes=n_classes
        )
        
        logger.info(f"Metrics calculated: accuracy={accuracy:.4f}, f1={f1:.4f}")
        return metrics

    def _downsample_curve(self, x, y, thresholds, max_points=1000):
        """
        Downsample curve points to reduce payload size.
        """
        if len(x) <= max_points:
            # Handle thresholds if it's shorter (sometimes it diffs by 1)
            t_list = thresholds.tolist() if hasattr(thresholds, 'tolist') else list(thresholds)
            
            return {
                "x": x.tolist(), 
                "y": y.tolist(), 
                "thresholds": t_list
            }
        
        # Uniform sampling
        indices = np.linspace(0, len(x) - 1, max_points).astype(int)
        
        # Ensure we keep the last point
        if indices[-1] != len(x) - 1:
            indices[-1] = len(x) - 1
            
        x_down = x[indices]
        y_down = y[indices]
        
        # Thresholds might have different length (e.g. for PR curve)
        t_down = []
        if len(thresholds) > 0:
            # Map indices to thresholds length
            t_indices = np.linspace(0, len(thresholds) - 1, max_points).astype(int)
            t_down = thresholds[t_indices].tolist()

        return {
            "x": x_down.tolist(),
            "y": y_down.tolist(),
            "thresholds": t_down
        }
    
    def _calculate_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        class_names: Optional[List[str]],
        unique_classes: np.ndarray
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
        """
        Calculate metrics for each class individually.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            class_names: Names for each class
            unique_classes: Array of unique class labels
        
        Returns:
            Tuple of (per_class_metrics_dict, support_dict)
        """
        per_class_metrics = {}
        support_dict = {}
        
        # Get class names
        if class_names is None:
            class_names = [f"class_{cls}" for cls in unique_classes]
        elif len(class_names) != len(unique_classes):
            logger.warning(
                f"class_names length ({len(class_names)}) doesn't match "
                f"number of classes ({len(unique_classes)}). Using default names."
            )
            class_names = [f"class_{cls}" for cls in unique_classes]
        
        # Calculate metrics for each class
        for idx, (cls, cls_name) in enumerate(zip(unique_classes, class_names)):
            # Binary mask for this class
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)
            
            # Calculate metrics
            precision = precision_score(
                y_true_binary, y_pred_binary,
                zero_division=self.zero_division
            )
            recall = recall_score(
                y_true_binary, y_pred_binary,
                zero_division=self.zero_division
            )
            f1 = f1_score(
                y_true_binary, y_pred_binary,
                zero_division=self.zero_division
            )
            support = int(np.sum(y_true == cls))
            
            per_class_metrics[cls_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'support': support
            }
            
            # Add AUC if probabilities available
            if y_proba is not None:
                try:
                    if y_proba.ndim == 2 and y_proba.shape[1] > idx:
                        y_proba_cls = y_proba[:, idx]
                        auc = roc_auc_score(y_true_binary, y_proba_cls)
                        per_class_metrics[cls_name]['auc_roc'] = float(auc)
                except Exception as e:
                    logger.debug(f"Could not calculate AUC for class {cls_name}: {e}")
            
            support_dict[cls_name] = support
        
        return per_class_metrics, support_dict
    
    def calculate_confusion_matrix(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        normalize: Optional[Literal['true', 'pred', 'all']] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Union[np.ndarray, List[str]]]:
        """
        Calculate confusion matrix with optional normalization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization mode
                - None: Raw counts
                - 'true': Normalize over true labels (rows)
                - 'pred': Normalize over predicted labels (columns)
                - 'all': Normalize over all samples
            class_names: Names for each class
        
        Returns:
            Dictionary with confusion matrix and class names
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        cm = confusion_matrix(y_true, y_pred, labels=self.labels, normalize=normalize)
        
        # Get class names
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        if class_names is None:
            class_names = [str(cls) for cls in unique_classes]
        
        return {
            'matrix': cm,
            'class_names': class_names,
            'normalized': normalize is not None
        }
    
    def get_classification_report(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        class_names: Optional[List[str]] = None,
        output_dict: bool = True
    ) -> Union[str, Dict]:
        """
        Generate sklearn classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names for each class
            output_dict: If True, return dict; if False, return formatted string
        
        Returns:
            Classification report as dict or string
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        return classification_report(
            y_true, y_pred,
            target_names=class_names,
            labels=self.labels,
            zero_division=self.zero_division,
            output_dict=output_dict
        )


def calculate_classification_metrics(
    y_true: Union[np.ndarray, pd.Series, List],
    y_pred: Union[np.ndarray, pd.Series, List],
    y_proba: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    average: Literal['binary', 'micro', 'macro', 'weighted'] = 'weighted',
    class_names: Optional[List[str]] = None,
    include_per_class: bool = True,
    include_advanced: bool = True
) -> ClassificationMetrics:
    """
    Convenience function to calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        average: Averaging strategy for multi-class
        class_names: Names for each class
        include_per_class: Whether to calculate per-class metrics
        include_advanced: Whether to calculate advanced metrics
    
    Returns:
        ClassificationMetrics object
    
    Example:
        >>> metrics = calculate_classification_metrics(
        ...     y_true=[0, 1, 1, 0, 1],
        ...     y_pred=[0, 1, 0, 0, 1],
        ...     average='binary'
        ... )
        >>> print(f"Accuracy: {metrics.accuracy:.4f}")
    """
    calculator = ClassificationMetricsCalculator(average=average)
    return calculator.calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=class_names,
        include_per_class=include_per_class,
        include_advanced=include_advanced
    )
