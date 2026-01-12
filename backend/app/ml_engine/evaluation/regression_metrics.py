"""
Regression Metrics Module

Provides comprehensive evaluation metrics for regression models.

Features:
- Basic metrics: MAE, MSE, RMSE, R²
- Percentage-based metrics: MAPE, SMAPE, RMSPE
- Advanced metrics: Adjusted R², Explained Variance, Max Error
- Residual analysis utilities
- Multiple output formats (dict, DataFrame)
- Visualization-ready data structures

Based on: ML-TO-DO.md > ML-50
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    max_error,
    mean_absolute_percentage_error,
    median_absolute_error,
)
from app.utils.logger import get_logger

logger = get_logger("regression_metrics")


@dataclass
class RegressionMetrics:
    """
    Container for regression evaluation metrics.
    
    Attributes:
        mae: Mean Absolute Error
        mse: Mean Squared Error
        rmse: Root Mean Squared Error
        r2: R² (Coefficient of Determination)
        adjusted_r2: Adjusted R² (accounts for number of features)
        mape: Mean Absolute Percentage Error
        smape: Symmetric Mean Absolute Percentage Error
        rmspe: Root Mean Squared Percentage Error
        explained_variance: Explained Variance Score
        max_error: Maximum Residual Error
        median_absolute_error: Median Absolute Error
        mean_residual: Mean of residuals (should be close to 0)
        std_residual: Standard deviation of residuals
        n_samples: Number of samples
        n_features: Number of features (for adjusted R²)
    """
    mae: float
    mse: float
    rmse: float
    r2: float
    adjusted_r2: Optional[float] = None
    mape: Optional[float] = None
    smape: Optional[float] = None
    rmspe: Optional[float] = None
    explained_variance: Optional[float] = None
    max_error: Optional[float] = None
    median_absolute_error: Optional[float] = None
    mean_residual: Optional[float] = None
    std_residual: Optional[float] = None
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    residual_plot: Optional[Dict] = None
    prediction_error_plot: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary format (JSON-serializable)."""
        return asdict(self)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RegressionMetrics(\n"
            f"  MAE={self.mae:.4f},\n"
            f"  RMSE={self.rmse:.4f},\n"
            f"  R²={self.r2:.4f},\n"
            f"  n_samples={self.n_samples}\n"
            f")"
        )


@dataclass
class ResidualAnalysis:
    """
    Container for residual analysis results.
    
    Attributes:
        residuals: Array of residuals (y_true - y_pred)
        standardized_residuals: Standardized residuals
        absolute_residuals: Absolute values of residuals
        percentage_errors: Percentage errors
        mean_residual: Mean of residuals
        std_residual: Standard deviation of residuals
        min_residual: Minimum residual
        max_residual: Maximum residual
        q1_residual: 25th percentile of residuals
        median_residual: Median of residuals
        q3_residual: 75th percentile of residuals
        outlier_indices: Indices of outlier residuals (|z-score| > 3)
        n_outliers: Number of outlier residuals
    """
    residuals: np.ndarray
    standardized_residuals: np.ndarray
    absolute_residuals: np.ndarray
    percentage_errors: Optional[np.ndarray] = None
    mean_residual: Optional[float] = None
    std_residual: Optional[float] = None
    min_residual: Optional[float] = None
    max_residual: Optional[float] = None
    q1_residual: Optional[float] = None
    median_residual: Optional[float] = None
    q3_residual: Optional[float] = None
    outlier_indices: Optional[np.ndarray] = None
    n_outliers: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format (JSON-serializable)."""
        result = asdict(self)
        # Convert numpy arrays to lists
        result['residuals'] = self.residuals.tolist()
        result['standardized_residuals'] = self.standardized_residuals.tolist()
        result['absolute_residuals'] = self.absolute_residuals.tolist()
        if self.percentage_errors is not None:
            result['percentage_errors'] = self.percentage_errors.tolist()
        if self.outlier_indices is not None:
            result['outlier_indices'] = self.outlier_indices.tolist()
        return result
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert residual analysis to pandas DataFrame."""
        df = pd.DataFrame({
            'residual': self.residuals,
            'standardized_residual': self.standardized_residuals,
            'absolute_residual': self.absolute_residuals
        })
        if self.percentage_errors is not None:
            df['percentage_error'] = self.percentage_errors
        return df
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ResidualAnalysis(\n"
            f"  mean={self.mean_residual:.4f},\n"
            f"  std={self.std_residual:.4f},\n"
            f"  n_outliers={self.n_outliers}\n"
            f")"
        )


class RegressionMetricsCalculator:
    """
    Calculator for regression model evaluation metrics.
    
    Provides comprehensive regression metrics including:
    - Basic error metrics (MAE, MSE, RMSE)
    - Goodness of fit (R², Adjusted R²)
    - Percentage-based metrics (MAPE, SMAPE, RMSPE)
    - Residual analysis
    - Advanced metrics
    
    Example:
        >>> calculator = RegressionMetricsCalculator()
        >>> metrics = calculator.calculate_metrics(
        ...     y_true=[1.0, 2.0, 3.0, 4.0, 5.0],
        ...     y_pred=[1.1, 2.2, 2.9, 4.1, 4.8]
        ... )
        >>> print(f"RMSE: {metrics.rmse:.4f}")
        >>> print(f"R²: {metrics.r2:.4f}")
    """
    
    def __init__(self):
        """Initialize regression metrics calculator."""
        logger.debug("Initialized RegressionMetricsCalculator")
    
    def calculate_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        n_features: Optional[int] = None,
        include_percentage_metrics: bool = True,
        include_advanced: bool = True
    ) -> RegressionMetrics:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            n_features: Number of features (for adjusted R²)
            include_percentage_metrics: Whether to calculate MAPE, SMAPE, RMSPE
            include_advanced: Whether to calculate advanced metrics
        
        Returns:
            RegressionMetrics with all calculated metrics
        
        Raises:
            ValueError: If inputs have incompatible shapes or invalid values
        """
        # Convert inputs to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Validate inputs
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"y_true and y_pred must have same shape. "
                f"Got {y_true.shape} and {y_pred.shape}"
            )
        
        if len(y_true) == 0:
            raise ValueError("y_true and y_pred cannot be empty")
        
        n_samples = len(y_true)
        
        logger.info(f"Computing regression metrics for {n_samples} samples")
        
        # Calculate basic metrics
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))
        
        # Calculate adjusted R²
        adjusted_r2 = None
        if n_features is not None and n_features > 0:
            adjusted_r2 = self._calculate_adjusted_r2(r2, n_samples, n_features)
        
        # Calculate percentage-based metrics
        mape_val = None
        smape_val = None
        rmspe_val = None
        
        if include_percentage_metrics:
            try:
                mape_val = self._calculate_mape(y_true, y_pred)
                smape_val = self._calculate_smape(y_true, y_pred)
                rmspe_val = self._calculate_rmspe(y_true, y_pred)
            except Exception as e:
                logger.warning(f"Could not calculate percentage metrics: {e}")
        
        # Calculate advanced metrics
        explained_var = None
        max_err = None
        median_ae = None
        mean_res = None
        std_res = None
        
        if include_advanced:
            try:
                explained_var = float(explained_variance_score(y_true, y_pred))
                max_err = float(max_error(y_true, y_pred))
                median_ae = float(median_absolute_error(y_true, y_pred))
                
                # Residual statistics
                residuals = y_true - y_pred
                mean_res = float(np.mean(residuals))
                std_res = float(np.std(residuals))
            except Exception as e:
                logger.warning(f"Could not calculate advanced metrics: {e}")
        
        # Calculate residuals if not already done
        if mean_res is None:
             residuals = y_true - y_pred
             mean_res = float(np.mean(residuals))
             std_res = float(np.std(residuals))
        else:
             residuals = y_true - y_pred

        # Generate plot data (downsampled)
        residual_plot_data = self._downsample_points(y_pred, residuals)
        prediction_error_data = self._downsample_points(y_true, y_pred)
        
        metrics = RegressionMetrics(
            mae=mae,
            mse=mse,
            rmse=rmse,
            r2=r2,
            adjusted_r2=adjusted_r2,
            mape=mape_val,
            smape=smape_val,
            rmspe=rmspe_val,
            explained_variance=explained_var,
            max_error=max_err,
            median_absolute_error=median_ae,
            mean_residual=mean_res,
            std_residual=std_res,
            n_samples=n_samples,
            n_features=n_features,
            residual_plot=residual_plot_data,
            prediction_error_plot=prediction_error_data
        )
        
        logger.info(f"Regression metrics computed: RMSE={rmse:.4f}, R²={r2:.4f}")
        return metrics

    def _downsample_points(self, x: np.ndarray, y: np.ndarray, max_points=1000) -> Dict:
        """
        Downsample scatter plot points to reduce payload size.
        """
        n_points = len(x)
        if n_points <= max_points:
            return {
                "x": x.tolist(), 
                "y": y.tolist()
            }
        
        # Random sampling for scatter plots is often better than uniform steps to avoid aliasing artifacts
        # But for reproducibility we'll use uniform steps or fixed seed
        indices = np.linspace(0, n_points - 1, max_points).astype(int)
        
        return {
            "x": x[indices].tolist(),
            "y": y[indices].tolist()
        }
    
    def _calculate_adjusted_r2(
        self,
        r2: float,
        n_samples: int,
        n_features: int
    ) -> float:
        """
        Calculate Adjusted R².
        
        Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)
        where n = number of samples, p = number of features
        
        Args:
            r2: R² score
            n_samples: Number of samples
            n_features: Number of features
        
        Returns:
            Adjusted R² score
        """
        if n_samples <= n_features + 1:
            logger.warning(
                f"Cannot calculate adjusted R²: n_samples ({n_samples}) <= "
                f"n_features + 1 ({n_features + 1})"
            )
            return r2
        
        adjusted = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        return float(adjusted)
    
    def _calculate_mape(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        MAPE = (1/n) * Σ|((y_true - y_pred) / y_true)| * 100
        
        Note: Undefined when y_true contains zeros.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            MAPE score (percentage)
        """
        # Check for zeros in y_true
        if np.any(y_true == 0):
            logger.warning("y_true contains zeros, MAPE may be undefined")
            # Use sklearn's implementation which handles this
            return float(mean_absolute_percentage_error(y_true, y_pred) * 100)
        
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return float(mape)
    
    def _calculate_smape(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error.
        
        SMAPE = (100/n) * Σ(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2))
        
        More robust than MAPE when y_true contains zeros.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            SMAPE score (percentage)
        """
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        # Avoid division by zero
        mask = denominator != 0
        smape = np.mean(numerator[mask] / denominator[mask]) * 100
        
        return float(smape)
    
    def _calculate_rmspe(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate Root Mean Squared Percentage Error.
        
        RMSPE = sqrt((1/n) * Σ((y_true - y_pred) / y_true)²) * 100
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            RMSPE score (percentage)
        """
        # Check for zeros in y_true
        if np.any(y_true == 0):
            logger.warning("y_true contains zeros, RMSPE may be undefined")
            # Filter out zeros
            mask = y_true != 0
            if not np.any(mask):
                return float('inf')
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        
        rmspe = np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2)) * 100
        return float(rmspe)
    
    def analyze_residuals(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        outlier_threshold: float = 3.0
    ) -> ResidualAnalysis:
        """
        Perform comprehensive residual analysis.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            outlier_threshold: Z-score threshold for outlier detection
        
        Returns:
            ResidualAnalysis with detailed residual statistics
        """
        # Convert inputs to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Calculate standardized residuals
        std_res = np.std(residuals)
        if std_res > 0:
            standardized_residuals = residuals / std_res
        else:
            standardized_residuals = np.zeros_like(residuals)
        
        # Calculate absolute residuals
        absolute_residuals = np.abs(residuals)
        
        # Calculate percentage errors (if possible)
        percentage_errors = None
        if not np.any(y_true == 0):
            percentage_errors = (residuals / y_true) * 100
        
        # Calculate statistics
        mean_res = float(np.mean(residuals))
        std_res_val = float(np.std(residuals))
        min_res = float(np.min(residuals))
        max_res = float(np.max(residuals))
        q1_res = float(np.percentile(residuals, 25))
        median_res = float(np.median(residuals))
        q3_res = float(np.percentile(residuals, 75))
        
        # Detect outliers (|z-score| > threshold)
        outlier_mask = np.abs(standardized_residuals) > outlier_threshold
        outlier_indices = np.where(outlier_mask)[0]
        n_outliers = int(np.sum(outlier_mask))
        
        logger.info(
            f"Residual analysis: mean={mean_res:.4f}, std={std_res_val:.4f}, "
            f"outliers={n_outliers}"
        )
        
        return ResidualAnalysis(
            residuals=residuals,
            standardized_residuals=standardized_residuals,
            absolute_residuals=absolute_residuals,
            percentage_errors=percentage_errors,
            mean_residual=mean_res,
            std_residual=std_res_val,
            min_residual=min_res,
            max_residual=max_res,
            q1_residual=q1_res,
            median_residual=median_res,
            q3_residual=q3_res,
            outlier_indices=outlier_indices,
            n_outliers=n_outliers
        )
    
    def compare_models(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_preds: Dict[str, Union[np.ndarray, pd.Series, List]],
        model_names: Optional[List[str]] = None
    ) -> Dict[str, RegressionMetrics]:
        """
        Compare regression metrics for multiple models.
        
        Args:
            y_true: True target values
            y_preds: Dictionary of model_name -> predicted values
            model_names: Optional list of model names (uses dict keys if None)
        
        Returns:
            Dictionary of model_name -> RegressionMetrics
        
        Example:
            >>> results = calculator.compare_models(
            ...     y_true=[1.0, 2.0, 3.0, 4.0],
            ...     y_preds={
            ...         'model_a': [1.1, 2.1, 2.9, 4.1],
            ...         'model_b': [1.2, 2.2, 3.1, 3.9]
            ...     }
            ... )
        """
        if model_names is None:
            model_names = list(y_preds.keys())
        
        results = {}
        for model_name in model_names:
            if model_name not in y_preds:
                logger.warning(f"Model '{model_name}' not found in y_preds")
                continue
            
            metrics = self.calculate_metrics(
                y_true=y_true,
                y_pred=y_preds[model_name]
            )
            results[model_name] = metrics
        
        logger.info(f"Compared {len(results)} models")
        return results


def calculate_regression_metrics(
    y_true: Union[np.ndarray, pd.Series, List],
    y_pred: Union[np.ndarray, pd.Series, List],
    n_features: Optional[int] = None,
    include_percentage_metrics: bool = True,
    include_advanced: bool = True
) -> RegressionMetrics:
    """
    Convenience function to calculate regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        n_features: Number of features (for adjusted R²)
        include_percentage_metrics: Whether to calculate MAPE, SMAPE, RMSPE
        include_advanced: Whether to calculate advanced metrics
    
    Returns:
        RegressionMetrics with all calculated metrics
    
    Example:
        >>> metrics = calculate_regression_metrics(
        ...     y_true=[1.0, 2.0, 3.0, 4.0, 5.0],
        ...     y_pred=[1.1, 2.2, 2.9, 4.1, 4.8]
        ... )
        >>> print(f"RMSE: {metrics.rmse:.4f}")
        >>> print(f"R²: {metrics.r2:.4f}")
        >>> print(f"MAPE: {metrics.mape:.2f}%")
    """
    calculator = RegressionMetricsCalculator()
    return calculator.calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        n_features=n_features,
        include_percentage_metrics=include_percentage_metrics,
        include_advanced=include_advanced
    )
