"""
Confusion Matrix Module

Provides comprehensive confusion matrix computation and analysis for classification models.

Features:
- Raw and normalized confusion matrices
- Per-class statistics (TP, FP, TN, FN)
- Error analysis and misclassification patterns
- Cost-sensitive analysis
- Multiple output formats (array, DataFrame, dict)
- Visualization-ready data structures

Based on: ML-TO-DO.md > ML-47
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Literal, Tuple
from dataclasses import dataclass, asdict
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from app.utils.logger import get_logger

logger = get_logger("confusion_matrix")


@dataclass
class ConfusionMatrixResult:
    """
    Container for confusion matrix and related statistics.
    
    Attributes:
        matrix: Raw confusion matrix (counts)
        normalized_matrix: Normalized confusion matrix (optional)
        class_names: Names of classes
        n_classes: Number of classes
        n_samples: Total number of samples
        per_class_stats: Statistics for each class (TP, FP, TN, FN, etc.)
        overall_stats: Overall statistics
        misclassification_matrix: Matrix showing misclassification patterns
        normalization_mode: How the matrix was normalized (if applicable)
    """
    matrix: np.ndarray
    normalized_matrix: Optional[np.ndarray] = None
    class_names: Optional[List[str]] = None
    n_classes: Optional[int] = None
    n_samples: Optional[int] = None
    per_class_stats: Optional[Dict[str, Dict[str, Union[int, float]]]] = None
    overall_stats: Optional[Dict[str, Union[int, float]]] = None
    misclassification_matrix: Optional[np.ndarray] = None
    normalization_mode: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format (JSON-serializable)."""
        result = asdict(self)
        # Convert numpy arrays to lists
        if self.matrix is not None:
            result['matrix'] = self.matrix.tolist()
        if self.normalized_matrix is not None:
            result['normalized_matrix'] = self.normalized_matrix.tolist()
        if self.misclassification_matrix is not None:
            result['misclassification_matrix'] = self.misclassification_matrix.tolist()
        return result
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert confusion matrix to pandas DataFrame."""
        if self.class_names is None:
            index = [f"True_{i}" for i in range(self.n_classes)]
            columns = [f"Pred_{i}" for i in range(self.n_classes)]
        else:
            index = [f"True_{name}" for name in self.class_names]
            columns = [f"Pred_{name}" for name in self.class_names]
        
        return pd.DataFrame(
            self.matrix,
            index=index,
            columns=columns
        )
    
    def to_normalized_dataframe(self) -> Optional[pd.DataFrame]:
        """Convert normalized confusion matrix to pandas DataFrame."""
        if self.normalized_matrix is None:
            return None
        
        if self.class_names is None:
            index = [f"True_{i}" for i in range(self.n_classes)]
            columns = [f"Pred_{i}" for i in range(self.n_classes)]
        else:
            index = [f"True_{name}" for name in self.class_names]
            columns = [f"Pred_{name}" for name in self.class_names]
        
        return pd.DataFrame(
            self.normalized_matrix,
            index=index,
            columns=columns
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ConfusionMatrixResult(\n"
            f"  n_classes={self.n_classes},\n"
            f"  n_samples={self.n_samples},\n"
            f"  normalized={self.normalized_matrix is not None},\n"
            f"  shape={self.matrix.shape}\n"
            f")"
        )


class ConfusionMatrixCalculator:
    """
    Calculator for confusion matrix and related statistics.
    
    Provides comprehensive confusion matrix analysis including:
    - Raw and normalized matrices
    - Per-class statistics (TP, FP, TN, FN)
    - Error analysis
    - Misclassification patterns
    - Cost-sensitive analysis
    
    Example:
        >>> calculator = ConfusionMatrixCalculator()
        >>> result = calculator.compute(
        ...     y_true=[0, 1, 2, 0, 1, 2],
        ...     y_pred=[0, 1, 2, 1, 1, 2],
        ...     class_names=['cat', 'dog', 'bird']
        ... )
        >>> print(result.matrix)
        >>> print(result.per_class_stats)
    """
    
    def __init__(
        self,
        labels: Optional[List] = None,
        sample_weight: Optional[np.ndarray] = None
    ):
        """
        Initialize confusion matrix calculator.
        
        Args:
            labels: List of class labels (auto-detected if None)
            sample_weight: Sample weights for weighted confusion matrix
        """
        self.labels = labels
        self.sample_weight = sample_weight
        logger.debug(f"Initialized ConfusionMatrixCalculator with labels={labels}")
    
    def compute(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        class_names: Optional[List[str]] = None,
        normalize: Optional[Literal['true', 'pred', 'all']] = None,
        include_stats: bool = True,
        include_misclassification: bool = True
    ) -> ConfusionMatrixResult:
        """
        Compute confusion matrix with comprehensive statistics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names for each class (for display)
            normalize: Normalization mode
                - None: Raw counts
                - 'true': Normalize over true labels (rows sum to 1)
                - 'pred': Normalize over predicted labels (columns sum to 1)
                - 'all': Normalize over all samples (all values sum to 1)
            include_stats: Whether to calculate per-class statistics
            include_misclassification: Whether to analyze misclassification patterns
        
        Returns:
            ConfusionMatrixResult with matrix and statistics
        
        Raises:
            ValueError: If inputs have incompatible shapes
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
        
        n_samples = len(y_true)
        
        # Compute raw confusion matrix
        cm = sklearn_confusion_matrix(
            y_true, y_pred,
            labels=self.labels,
            sample_weight=self.sample_weight
        )
        
        # Get class information
        if self.labels is not None:
            unique_classes = np.array(self.labels)
        else:
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        
        n_classes = len(unique_classes)
        
        # Generate class names if not provided
        if class_names is None:
            class_names = [str(cls) for cls in unique_classes]
        elif len(class_names) != n_classes:
            logger.warning(
                f"class_names length ({len(class_names)}) doesn't match "
                f"number of classes ({n_classes}). Using default names."
            )
            class_names = [str(cls) for cls in unique_classes]
        
        logger.info(
            f"Computing confusion matrix: {n_classes} classes, "
            f"{n_samples} samples, normalize={normalize}"
        )
        
        # Compute normalized matrix if requested
        normalized_cm = None
        if normalize is not None:
            normalized_cm = self._normalize_matrix(cm, normalize)
        
        # Compute per-class statistics
        per_class_stats = None
        if include_stats:
            per_class_stats = self._compute_per_class_stats(
                cm, class_names, unique_classes
            )
        
        # Compute overall statistics
        overall_stats = self._compute_overall_stats(cm, n_samples)
        
        # Analyze misclassification patterns
        misclassification_matrix = None
        if include_misclassification:
            misclassification_matrix = self._compute_misclassification_matrix(cm)
        
        result = ConfusionMatrixResult(
            matrix=cm,
            normalized_matrix=normalized_cm,
            class_names=class_names,
            n_classes=n_classes,
            n_samples=n_samples,
            per_class_stats=per_class_stats,
            overall_stats=overall_stats,
            misclassification_matrix=misclassification_matrix,
            normalization_mode=normalize
        )
        
        logger.info(f"Confusion matrix computed successfully")
        return result
    
    def _normalize_matrix(
        self,
        cm: np.ndarray,
        mode: Literal['true', 'pred', 'all']
    ) -> np.ndarray:
        """
        Normalize confusion matrix.
        
        Args:
            cm: Raw confusion matrix
            mode: Normalization mode
        
        Returns:
            Normalized confusion matrix
        """
        cm = cm.astype(float)
        
        if mode == 'true':
            # Normalize over true labels (rows)
            row_sums = cm.sum(axis=1, keepdims=True)
            # Avoid division by zero
            row_sums[row_sums == 0] = 1
            return cm / row_sums
        
        elif mode == 'pred':
            # Normalize over predicted labels (columns)
            col_sums = cm.sum(axis=0, keepdims=True)
            # Avoid division by zero
            col_sums[col_sums == 0] = 1
            return cm / col_sums
        
        elif mode == 'all':
            # Normalize over all samples
            total = cm.sum()
            if total == 0:
                return cm
            return cm / total
        
        else:
            raise ValueError(f"Invalid normalization mode: {mode}")
    
    def _compute_per_class_stats(
        self,
        cm: np.ndarray,
        class_names: List[str],
        unique_classes: np.ndarray
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        """
        Compute per-class statistics from confusion matrix.
        
        For each class, computes:
        - TP (True Positives)
        - FP (False Positives)
        - TN (True Negatives)
        - FN (False Negatives)
        - Sensitivity (Recall)
        - Specificity
        - Precision
        - F1 Score
        - Support (number of true instances)
        
        Args:
            cm: Confusion matrix
            class_names: Names of classes
            unique_classes: Array of unique class labels
        
        Returns:
            Dictionary with statistics for each class
        """
        n_classes = len(class_names)
        per_class_stats = {}
        
        for i, (cls_name, cls_label) in enumerate(zip(class_names, unique_classes)):
            # True Positives: diagonal element
            tp = int(cm[i, i])
            
            # False Positives: sum of column i excluding diagonal
            fp = int(cm[:, i].sum() - cm[i, i])
            
            # False Negatives: sum of row i excluding diagonal
            fn = int(cm[i, :].sum() - cm[i, i])
            
            # True Negatives: sum of all elements except row i and column i
            tn = int(cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i])
            
            # Support: total true instances of this class
            support = int(cm[i, :].sum())
            
            # Calculate metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # F1 Score
            if precision + sensitivity > 0:
                f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
            else:
                f1 = 0.0
            
            # Accuracy for this class
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
            per_class_stats[cls_name] = {
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn,
                'support': support,
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'precision': float(precision),
                'recall': float(sensitivity),  # Same as sensitivity
                'f1_score': float(f1),
                'accuracy': float(accuracy)
            }
        
        return per_class_stats
    
    def _compute_overall_stats(
        self,
        cm: np.ndarray,
        n_samples: int
    ) -> Dict[str, Union[int, float]]:
        """
        Compute overall statistics from confusion matrix.
        
        Args:
            cm: Confusion matrix
            n_samples: Total number of samples
        
        Returns:
            Dictionary with overall statistics
        """
        # Overall accuracy
        correct = np.trace(cm)  # Sum of diagonal
        accuracy = float(correct / n_samples) if n_samples > 0 else 0.0
        
        # Total errors
        total_errors = int(n_samples - correct)
        error_rate = float(total_errors / n_samples) if n_samples > 0 else 0.0
        
        return {
            'total_samples': n_samples,
            'correct_predictions': int(correct),
            'incorrect_predictions': total_errors,
            'accuracy': accuracy,
            'error_rate': error_rate
        }
    
    def _compute_misclassification_matrix(
        self,
        cm: np.ndarray
    ) -> np.ndarray:
        """
        Compute misclassification matrix (off-diagonal elements only).
        
        This matrix shows only the errors, with diagonal set to 0.
        Useful for analyzing which classes are confused with each other.
        
        Args:
            cm: Confusion matrix
        
        Returns:
            Misclassification matrix
        """
        misclass = cm.copy()
        np.fill_diagonal(misclass, 0)
        return misclass
    
    def compute_cost_sensitive(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        cost_matrix: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute cost-sensitive confusion matrix analysis.
        
        Useful when different types of errors have different costs.
        For example, false negatives might be more costly than false positives
        in medical diagnosis.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            cost_matrix: Cost matrix where cost_matrix[i, j] is the cost of
                        predicting class j when true class is i
            class_names: Names for each class
        
        Returns:
            Dictionary with cost analysis
        
        Example:
            >>> # False negatives cost 10, false positives cost 1
            >>> cost_matrix = np.array([[0, 1], [10, 0]])
            >>> result = calculator.compute_cost_sensitive(
            ...     y_true, y_pred, cost_matrix
            ... )
            >>> print(f"Total cost: {result['total_cost']}")
        """
        # Compute confusion matrix
        result = self.compute(y_true, y_pred, class_names, include_stats=False)
        cm = result.matrix
        
        # Validate cost matrix shape
        if cost_matrix.shape != cm.shape:
            raise ValueError(
                f"Cost matrix shape {cost_matrix.shape} doesn't match "
                f"confusion matrix shape {cm.shape}"
            )
        
        # Compute element-wise cost
        cost_per_cell = cm * cost_matrix
        
        # Total cost
        total_cost = float(cost_per_cell.sum())
        
        # Average cost per sample
        avg_cost = total_cost / result.n_samples if result.n_samples > 0 else 0.0
        
        # Cost breakdown by class
        cost_by_true_class = cost_per_cell.sum(axis=1)  # Sum over predictions
        cost_by_pred_class = cost_per_cell.sum(axis=0)  # Sum over true labels
        
        return {
            'total_cost': total_cost,
            'average_cost_per_sample': avg_cost,
            'cost_matrix': cost_per_cell,
            'cost_by_true_class': cost_by_true_class,
            'cost_by_pred_class': cost_by_pred_class,
            'confusion_matrix': cm
        }
    
    def analyze_errors(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        class_names: Optional[List[str]] = None,
        top_n: int = 5
    ) -> Dict[str, Union[List, Dict]]:
        """
        Analyze misclassification patterns.
        
        Identifies:
        - Most common misclassifications
        - Classes with highest error rates
        - Confusion pairs (which classes are most confused)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names for each class
            top_n: Number of top errors to return
        
        Returns:
            Dictionary with error analysis
        """
        result = self.compute(
            y_true, y_pred, class_names,
            include_misclassification=True
        )
        
        cm = result.matrix
        misclass = result.misclassification_matrix
        class_names = result.class_names
        
        # Find most common misclassifications
        misclass_flat = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and misclass[i, j] > 0:
                    misclass_flat.append({
                        'true_class': class_names[i],
                        'predicted_class': class_names[j],
                        'count': int(misclass[i, j]),
                        'percentage': float(misclass[i, j] / cm[i, :].sum() * 100)
                    })
        
        # Sort by count
        misclass_flat.sort(key=lambda x: x['count'], reverse=True)
        top_misclassifications = misclass_flat[:top_n]
        
        # Classes with highest error rates
        error_rates = []
        for i, cls_name in enumerate(class_names):
            total = cm[i, :].sum()
            errors = total - cm[i, i]
            error_rate = (errors / total * 100) if total > 0 else 0.0
            error_rates.append({
                'class': cls_name,
                'error_count': int(errors),
                'total_samples': int(total),
                'error_rate': float(error_rate)
            })
        
        error_rates.sort(key=lambda x: x['error_rate'], reverse=True)
        
        return {
            'top_misclassifications': top_misclassifications,
            'error_rates_by_class': error_rates,
            'total_errors': int(misclass.sum()),
            'total_samples': result.n_samples
        }


def compute_confusion_matrix(
    y_true: Union[np.ndarray, pd.Series, List],
    y_pred: Union[np.ndarray, pd.Series, List],
    class_names: Optional[List[str]] = None,
    normalize: Optional[Literal['true', 'pred', 'all']] = None,
    include_stats: bool = True,
    include_misclassification: bool = True,
    labels: Optional[List] = None
) -> ConfusionMatrixResult:
    """
    Convenience function to compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for each class
        normalize: Normalization mode ('true', 'pred', 'all', or None)
        include_stats: Whether to calculate per-class statistics
        include_misclassification: Whether to analyze misclassification patterns
        labels: List of class labels (auto-detected if None)
    
    Returns:
        ConfusionMatrixResult with matrix and statistics
    
    Example:
        >>> result = compute_confusion_matrix(
        ...     y_true=[0, 1, 2, 0, 1, 2],
        ...     y_pred=[0, 1, 2, 1, 1, 2],
        ...     class_names=['cat', 'dog', 'bird'],
        ...     normalize='true'
        ... )
        >>> print(result.matrix)
        >>> print(result.per_class_stats)
    """
    calculator = ConfusionMatrixCalculator(labels=labels)
    return calculator.compute(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        normalize=normalize,
        include_stats=include_stats,
        include_misclassification=include_misclassification
    )
