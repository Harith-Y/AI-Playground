"""
Incremental (online) training support for large datasets.

Provides partial_fit capabilities for models that support incremental learning,
allowing training on datasets too large to fit in memory.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import gc

from app.ml_engine.utils.dataset_optimizer import (
    ChunkedDataLoader,
    DatasetMetrics,
    get_optimal_chunk_size
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class IncrementalTrainer:
    """
    Trainer for incremental learning on large datasets.

    Supports models with partial_fit() method for online learning.
    """

    # Models that support incremental learning
    SUPPORTED_MODELS = {
        'sgd_classifier': 'sklearn.linear_model.SGDClassifier',
        'sgd_regressor': 'sklearn.linear_model.SGDRegressor',
        'passive_aggressive_classifier': 'sklearn.linear_model.PassiveAggressiveClassifier',
        'passive_aggressive_regressor': 'sklearn.linear_model.PassiveAggressiveRegressor',
        'perceptron': 'sklearn.linear_model.Perceptron',
        'mini_batch_kmeans': 'sklearn.cluster.MiniBatchKMeans',
        'mlp_classifier': 'sklearn.neural_network.MLPClassifier',
        'mlp_regressor': 'sklearn.neural_network.MLPRegressor',
    }

    def __init__(
        self,
        model: BaseEstimator,
        model_type: str,
        chunk_size: Optional[int] = None
    ):
        """
        Initialize incremental trainer.

        Args:
            model: Scikit-learn model with partial_fit support
            model_type: Model type identifier
            chunk_size: Chunk size for training (auto-calculated if None)
        """
        self.model = model
        self.model_type = model_type
        self.chunk_size = chunk_size

        # Verify model supports partial_fit
        if not hasattr(model, 'partial_fit'):
            raise ValueError(
                f"Model {model_type} does not support partial_fit. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.scaler = None
        self.classes_ = None
        self.n_features_in_ = None

    def fit_incremental(
        self,
        file_path: str,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        validation_split: float = 0.2,
        classes: Optional[np.ndarray] = None,
        scale_features: bool = True,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train model incrementally on large dataset.

        Args:
            file_path: Path to CSV file
            target_column: Target column name
            feature_columns: Feature column names (None = all except target)
            validation_split: Fraction of data for validation
            classes: Class labels for classification (required for classifiers)
            scale_features: Whether to scale features
            random_state: Random seed

        Returns:
            Training statistics
        """
        logger.info(f"Starting incremental training on {file_path}")

        # Get dataset metrics
        dataset_metrics = DatasetMetrics.from_file(file_path)

        # Calculate optimal chunk size if not provided
        if self.chunk_size is None:
            self.chunk_size = get_optimal_chunk_size(dataset_metrics)

        # Initialize data loader
        loader = ChunkedDataLoader(
            file_path,
            chunk_size=self.chunk_size
        )

        # First pass: Fit scaler if needed
        if scale_features:
            logger.info("Pass 1: Fitting scaler")
            self.scaler = self._fit_scaler_incremental(
                loader,
                target_column,
                feature_columns
            )

        # Second pass: Train model incrementally
        logger.info("Pass 2: Training model incrementally")
        stats = self._train_incremental(
            loader,
            target_column,
            feature_columns,
            validation_split,
            classes,
            random_state
        )

        logger.info("Incremental training completed")
        return stats

    def _fit_scaler_incremental(
        self,
        loader: ChunkedDataLoader,
        target_column: str,
        feature_columns: Optional[List[str]]
    ) -> StandardScaler:
        """Fit scaler incrementally on data chunks."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        n_samples_seen = 0

        for chunk in loader.iter_chunks():
            # Extract features
            if feature_columns:
                X = chunk[feature_columns]
            else:
                X = chunk.drop(columns=[target_column])

            # Partial fit scaler
            scaler.partial_fit(X)
            n_samples_seen += len(X)

            del chunk, X
            gc.collect()

        logger.info(f"Fitted scaler on {n_samples_seen:,} samples")
        return scaler

    def _train_incremental(
        self,
        loader: ChunkedDataLoader,
        target_column: str,
        feature_columns: Optional[List[str]],
        validation_split: float,
        classes: Optional[np.ndarray],
        random_state: int
    ) -> Dict[str, Any]:
        """Train model incrementally."""
        np.random.seed(random_state)

        train_samples = 0
        val_samples = 0
        train_score_sum = 0.0
        val_score_sum = 0.0
        n_batches = 0

        for chunk in loader.iter_chunks():
            # Extract features and target
            if feature_columns:
                X = chunk[feature_columns]
            else:
                X = chunk.drop(columns=[target_column])

            y = chunk[target_column]

            # Store number of features
            if self.n_features_in_ is None:
                self.n_features_in_ = X.shape[1]

            # Scale features
            if self.scaler is not None:
                X = pd.DataFrame(
                    self.scaler.transform(X),
                    columns=X.columns,
                    index=X.index
                )

            # Split into train/validation
            n = len(X)
            val_size = int(n * validation_split)
            indices = np.random.permutation(n)
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]

            X_train = X.iloc[train_indices]
            y_train = y.iloc[train_indices]
            X_val = X.iloc[val_indices] if val_size > 0 else None
            y_val = y.iloc[val_indices] if val_size > 0 else None

            # Partial fit on training data
            if hasattr(self.model, 'partial_fit'):
                if classes is not None:
                    # For classifiers, pass classes on first call
                    if n_batches == 0:
                        self.model.partial_fit(X_train, y_train, classes=classes)
                        self.classes_ = classes
                    else:
                        self.model.partial_fit(X_train, y_train)
                else:
                    self.model.partial_fit(X_train, y_train)

            train_samples += len(X_train)

            # Evaluate on validation set
            if X_val is not None and len(X_val) > 0:
                val_score = self.model.score(X_val, y_val)
                val_score_sum += val_score * len(X_val)
                val_samples += len(X_val)

            n_batches += 1

            # Log progress
            if n_batches % 10 == 0:
                avg_val_score = val_score_sum / val_samples if val_samples > 0 else 0
                logger.info(
                    f"Batch {n_batches}: {train_samples:,} samples trained, "
                    f"val_score: {avg_val_score:.4f}"
                )

            del chunk, X, y, X_train, y_train, X_val, y_val
            gc.collect()

        # Calculate final statistics
        avg_val_score = val_score_sum / val_samples if val_samples > 0 else None

        stats = {
            'train_samples': train_samples,
            'val_samples': val_samples,
            'n_batches': n_batches,
            'validation_score': avg_val_score,
            'n_features': self.n_features_in_,
            'incremental_training': True
        }

        logger.info(
            f"Trained on {train_samples:,} samples in {n_batches} batches, "
            f"validation score: {avg_val_score:.4f if avg_val_score else 'N/A'}"
        )

        return stats

    def predict_incremental(
        self,
        file_path: str,
        output_path: str,
        feature_columns: Optional[List[str]] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Make predictions incrementally on large dataset.

        Args:
            file_path: Input CSV path
            output_path: Output CSV path for predictions
            feature_columns: Feature column names
            batch_size: Batch size for predictions

        Returns:
            Prediction statistics
        """
        logger.info(f"Making incremental predictions: {file_path} -> {output_path}")

        if batch_size is None:
            batch_size = self.chunk_size

        loader = ChunkedDataLoader(file_path, chunk_size=batch_size)

        total_predictions = 0
        first_batch = True

        for chunk in loader.iter_chunks():
            # Extract features
            if feature_columns:
                X = chunk[feature_columns]
            else:
                X = chunk

            # Scale if scaler was fitted
            if self.scaler is not None:
                X_scaled = pd.DataFrame(
                    self.scaler.transform(X),
                    columns=X.columns,
                    index=X.index
                )
            else:
                X_scaled = X

            # Make predictions
            predictions = self.model.predict(X_scaled)

            # Create output DataFrame
            output_df = chunk.copy()
            output_df['prediction'] = predictions

            # Write to file
            mode = 'w' if first_batch else 'a'
            header = first_batch
            output_df.to_csv(output_path, mode=mode, header=header, index=False)

            total_predictions += len(predictions)
            first_batch = False

            del chunk, X, X_scaled, predictions, output_df
            gc.collect()

        logger.info(f"Made {total_predictions:,} predictions -> {output_path}")

        return {
            'total_predictions': total_predictions,
            'output_path': output_path
        }


def supports_incremental_learning(model_type: str) -> bool:
    """
    Check if model type supports incremental learning.

    Args:
        model_type: Model type identifier

    Returns:
        True if model supports partial_fit
    """
    return model_type.lower() in IncrementalTrainer.SUPPORTED_MODELS


def get_incremental_model_config(model_type: str) -> Dict[str, Any]:
    """
    Get recommended configuration for incremental learning.

    Args:
        model_type: Model type identifier

    Returns:
        Configuration dict with recommended hyperparameters
    """
    configs = {
        'sgd_classifier': {
            'loss': 'log_loss',  # For probabilistic predictions
            'penalty': 'l2',
            'alpha': 0.0001,
            'max_iter': 1000,
            'tol': 1e-3,
            'learning_rate': 'optimal',
            'warm_start': True
        },
        'sgd_regressor': {
            'loss': 'squared_error',
            'penalty': 'l2',
            'alpha': 0.0001,
            'max_iter': 1000,
            'tol': 1e-3,
            'learning_rate': 'invscaling',
            'warm_start': True
        },
        'mini_batch_kmeans': {
            'n_clusters': 8,
            'batch_size': 1000,
            'max_iter': 100,
            'init': 'k-means++',
            'reassignment_ratio': 0.01
        },
        'mlp_classifier': {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'solver': 'sgd',  # Required for partial_fit
            'alpha': 0.0001,
            'batch_size': 200,
            'learning_rate': 'adaptive',
            'max_iter': 200,
            'warm_start': True
        },
        'mlp_regressor': {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'solver': 'sgd',  # Required for partial_fit
            'alpha': 0.0001,
            'batch_size': 200,
            'learning_rate': 'adaptive',
            'max_iter': 200,
            'warm_start': True
        }
    }

    return configs.get(model_type.lower(), {})
