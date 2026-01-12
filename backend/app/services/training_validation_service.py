"""
Training Configuration Validation Service

This module provides comprehensive validation for model training configurations
before training is initiated. It ensures all prerequisites are met to prevent
training failures and provides clear error messages.

Validation includes:
- Dataset existence and ownership
- Experiment existence and ownership
- Model type availability in registry
- Target column requirements for supervised learning
- Feature column existence in dataset
- Data type compatibility
- Minimum sample requirements
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.experiment import Experiment
from app.ml_engine.model_registry import ModelRegistry
from app.utils.logger import get_logger


logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(self.message)


class TrainingConfigValidator:
    """
    Validator for training configuration.
    
    Performs comprehensive validation of training requests including:
    - Resource existence (dataset, experiment)
    - Ownership verification
    - Model type validation
    - Column validation (target, features)
    - Data compatibility checks
    
    Example:
        >>> validator = TrainingConfigValidator(db)
        >>> validator.validate_training_config(
        ...     experiment_id=exp_id,
        ...     dataset_id=ds_id,
        ...     model_type="random_forest_classifier",
        ...     target_column="species",
        ...     feature_columns=["sepal_length", "sepal_width"],
        ...     user_id=user_id
        ... )
    """
    
    def __init__(self, db: Session):
        """
        Initialize the validator.
        
        Args:
            db: Database session
        """
        self.db = db
        self.registry = ModelRegistry()
    
    def validate_training_config(
        self,
        experiment_id: UUID,
        dataset_id: UUID,
        model_type: str,
        user_id: str,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Tuple[Experiment, Dataset, Dict[str, Any]]:
        """
        Validate complete training configuration.
        
        This method performs all validation checks in sequence and returns
        validated resources if all checks pass.
        
        Args:
            experiment_id: UUID of the experiment
            dataset_id: UUID of the dataset
            model_type: Type of model to train
            user_id: UUID of the user
            target_column: Name of target column (optional)
            feature_columns: List of feature column names (optional)
            test_size: Train/test split ratio
            hyperparameters: Model hyperparameters (optional)
        
        Returns:
            Tuple of (experiment, dataset, model_info)
        
        Raises:
            ValidationError: If any validation check fails
        
        Example:
            >>> validator = TrainingConfigValidator(db)
            >>> exp, ds, model_info = validator.validate_training_config(
            ...     experiment_id=exp_id,
            ...     dataset_id=ds_id,
            ...     model_type="random_forest_classifier",
            ...     user_id=user_id,
            ...     target_column="species"
            ... )
        """
        logger.info(
            f"Validating training configuration",
            extra={
                'event': 'validation_start',
                'experiment_id': str(experiment_id),
                'dataset_id': str(dataset_id),
                'model_type': model_type,
                'user_id': user_id
            }
        )
        
        # 1. Validate experiment
        experiment = self._validate_experiment(experiment_id, user_id)
        
        # 2. Validate dataset
        dataset = self._validate_dataset(dataset_id, user_id)
        
        # 3. Validate model type
        model_info = self._validate_model_type(model_type)
        
        # 4. Validate target column for supervised learning
        self._validate_target_column(
            model_info=model_info,
            target_column=target_column,
            dataset=dataset
        )
        
        # 5. Validate feature columns if specified
        if feature_columns:
            self._validate_feature_columns(
                feature_columns=feature_columns,
                dataset=dataset,
                target_column=target_column
            )
        
        # 6. Validate test size
        self._validate_test_size(test_size)
        
        # 7. Validate hyperparameters if provided
        if hyperparameters:
            self._validate_hyperparameters(
                hyperparameters=hyperparameters,
                model_info=model_info
            )
        
        # 8. Validate dataset has sufficient samples
        self._validate_dataset_size(dataset, test_size)
        
        logger.info(
            f"Training configuration validated successfully",
            extra={
                'event': 'validation_success',
                'experiment_id': str(experiment_id),
                'dataset_id': str(dataset_id),
                'model_type': model_type
            }
        )
        
        return experiment, dataset, model_info
    
    def _validate_experiment(self, experiment_id: UUID, user_id: str) -> Experiment:
        """
        Validate that experiment exists and belongs to user.
        
        Args:
            experiment_id: UUID of the experiment
            user_id: UUID of the user
        
        Returns:
            Experiment object
        
        Raises:
            ValidationError: If experiment not found or doesn't belong to user
        """
        experiment = self.db.query(Experiment).filter(
            Experiment.id == experiment_id
        ).first()
        
        if not experiment:
            raise ValidationError(
                message=f"Experiment with id {experiment_id} not found",
                field="experiment_id"
            )
        
        if str(experiment.user_id) != user_id:
            raise ValidationError(
                message=f"Experiment {experiment_id} does not belong to user {user_id}",
                field="experiment_id"
            )
        
        logger.debug(f"Experiment validation passed: {experiment_id}")
        return experiment
    
    def _validate_dataset(self, dataset_id: UUID, user_id: str) -> Dataset:
        """
        Validate that dataset exists and belongs to user.
        
        Args:
            dataset_id: UUID of the dataset
            user_id: UUID of the user
        
        Returns:
            Dataset object
        
        Raises:
            ValidationError: If dataset not found or doesn't belong to user
        """
        dataset = self.db.query(Dataset).filter(
            Dataset.id == dataset_id
        ).first()
        
        if not dataset:
            raise ValidationError(
                message=f"Dataset with id {dataset_id} not found",
                field="dataset_id"
            )
        
        if str(dataset.user_id) != user_id:
            raise ValidationError(
                message=f"Dataset {dataset_id} does not belong to user {user_id}",
                field="dataset_id"
            )
        
        # Validate dataset file exists
        if not Path(dataset.file_path).exists():
            raise ValidationError(
                message=f"Dataset file not found at {dataset.file_path}",
                field="dataset_id"
            )
        
        logger.debug(f"Dataset validation passed: {dataset_id}")
        return dataset
    
    def _validate_model_type(self, model_type: str) -> Dict[str, Any]:
        """
        Validate that model type exists in registry.
        
        Args:
            model_type: Type of model to train
        
        Returns:
            Model info dictionary
        
        Raises:
            ValidationError: If model type not found
        """
        model_info = self.registry.get_model(model_type)
        
        if not model_info:
            # Get available models for helpful error message
            all_models = self.registry.get_all_models()
            available_types = []
            for task_models in all_models.values():
                available_types.extend([m.model_id for m in task_models])
            
            raise ValidationError(
                message=f"Model type '{model_type}' not found in registry. "
                       f"Available models: {', '.join(available_types[:10])}...",
                field="model_type"
            )
        
        logger.debug(f"Model type validation passed: {model_type}")
        return model_info
    
    def _validate_target_column(
        self,
        model_info: Dict[str, Any],
        target_column: Optional[str],
        dataset: Dataset
    ) -> None:
        """
        Validate target column for supervised learning.
        
        Args:
            model_info: Model information from registry
            target_column: Name of target column
            dataset: Dataset object
        
        Raises:
            ValidationError: If target column is missing or invalid
        """
        task_type = model_info.task_type.value
        
        # Check if target column is required
        if task_type in ['classification', 'regression']:
            if not target_column:
                raise ValidationError(
                    message=f"target_column is required for {task_type} tasks",
                    field="target_column"
                )
            
            # Load dataset to check if column exists
            try:
                df = pd.read_csv(dataset.file_path, nrows=1)
                
                if target_column not in df.columns:
                    raise ValidationError(
                        message=f"Target column '{target_column}' not found in dataset. "
                               f"Available columns: {', '.join(df.columns)}",
                        field="target_column"
                    )
                
                logger.debug(f"Target column validation passed: {target_column}")
                
            except pd.errors.EmptyDataError:
                raise ValidationError(
                    message="Dataset is empty",
                    field="dataset_id"
                )
            except Exception as e:
                raise ValidationError(
                    message=f"Failed to read dataset: {str(e)}",
                    field="dataset_id"
                )
        
        elif task_type == 'clustering':
            # Clustering doesn't need target column
            if target_column:
                logger.warning(
                    f"target_column specified for clustering task but will be ignored"
                )
    
    def _validate_feature_columns(
        self,
        feature_columns: List[str],
        dataset: Dataset,
        target_column: Optional[str]
    ) -> None:
        """
        Validate that feature columns exist in dataset.
        
        Args:
            feature_columns: List of feature column names
            dataset: Dataset object
            target_column: Name of target column (to check for overlap)
        
        Raises:
            ValidationError: If feature columns are invalid
        """
        if not feature_columns:
            return
        
        # Load dataset to check columns
        try:
            df = pd.read_csv(dataset.file_path, nrows=1)
            dataset_columns = set(df.columns)
            feature_set = set(feature_columns)
            
            # Check if all feature columns exist
            missing_columns = feature_set - dataset_columns
            if missing_columns:
                raise ValidationError(
                    message=f"Feature columns not found in dataset: {', '.join(missing_columns)}. "
                           f"Available columns: {', '.join(df.columns)}",
                    field="feature_columns"
                )
            
            # Check if target column is in feature columns
            if target_column and target_column in feature_set:
                raise ValidationError(
                    message=f"Target column '{target_column}' cannot be in feature_columns",
                    field="feature_columns"
                )
            
            # Check for duplicate feature columns
            if len(feature_columns) != len(feature_set):
                raise ValidationError(
                    message="Duplicate columns found in feature_columns",
                    field="feature_columns"
                )
            
            logger.debug(f"Feature columns validation passed: {len(feature_columns)} columns")
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                message=f"Failed to validate feature columns: {str(e)}",
                field="feature_columns"
            )
    
    def _validate_test_size(self, test_size: float) -> None:
        """
        Validate test size parameter.
        
        Args:
            test_size: Train/test split ratio
        
        Raises:
            ValidationError: If test_size is invalid
        """
        if not 0.0 < test_size < 1.0:
            raise ValidationError(
                message=f"test_size must be between 0.0 and 1.0, got {test_size}",
                field="test_size"
            )
        
        if test_size < 0.1:
            logger.warning(
                f"test_size is very small ({test_size}), may lead to unreliable evaluation"
            )
        
        if test_size > 0.5:
            logger.warning(
                f"test_size is large ({test_size}), may lead to insufficient training data"
            )
        
        logger.debug(f"Test size validation passed: {test_size}")
    
    def _validate_hyperparameters(
        self,
        hyperparameters: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> None:
        """
        Validate hyperparameters against model's expected parameters.
        
        Args:
            hyperparameters: Model hyperparameters
            model_info: Model information from registry
        
        Raises:
            ValidationError: If hyperparameters are invalid
        """
        if not hyperparameters:
            return
        
        # Get expected hyperparameters from model info
        expected_params = model_info.hyperparameters
        provided_params = set(hyperparameters.keys())
        expected_param_names = set(expected_params.keys())
        
        # Check for unexpected parameters
        unexpected_params = provided_params - expected_param_names
        
        # Ignored deprecated params (like normalize for LinearRegression in newer sklearn)
        ignored_params = {'normalize'}
        unexpected_params = unexpected_params - ignored_params
        
        if unexpected_params:
            logger.warning(
                f"Unexpected hyperparameters for {model_info.model_id}: {', '.join(unexpected_params)}. "
                f"Expected: {', '.join(expected_param_names)}"
            )
        
        # Validate parameter types and ranges
        for param_name, param_value in hyperparameters.items():
            if param_name in expected_params:
                expected_param = expected_params[param_name]
                
                # Type validation
                expected_type = expected_param.get('type')
                if expected_type == 'int' and not isinstance(param_value, int):
                    raise ValidationError(
                        message=f"Hyperparameter '{param_name}' must be an integer, got {type(param_value).__name__}",
                        field="hyperparameters"
                    )
                elif expected_type == 'float' and not isinstance(param_value, (int, float)):
                    raise ValidationError(
                        message=f"Hyperparameter '{param_name}' must be a number, got {type(param_value).__name__}",
                        field="hyperparameters"
                    )
                
                # Range validation
                if 'min' in expected_param and param_value < expected_param['min']:
                    raise ValidationError(
                        message=f"Hyperparameter '{param_name}' must be >= {expected_param['min']}, got {param_value}",
                        field="hyperparameters"
                    )
                
                if 'max' in expected_param and param_value > expected_param['max']:
                    raise ValidationError(
                        message=f"Hyperparameter '{param_name}' must be <= {expected_param['max']}, got {param_value}",
                        field="hyperparameters"
                    )
        
        logger.debug(f"Hyperparameters validation passed: {len(hyperparameters)} parameters")
    
    def _validate_dataset_size(self, dataset: Dataset, test_size: float) -> None:
        """
        Validate that dataset has sufficient samples for training.
        
        Args:
            dataset: Dataset object
            test_size: Train/test split ratio
        
        Raises:
            ValidationError: If dataset is too small
        """
        try:
            df = pd.read_csv(dataset.file_path)
            n_samples = len(df)
            
            # Minimum samples required
            min_samples = 10
            if n_samples < min_samples:
                raise ValidationError(
                    message=f"Dataset has only {n_samples} samples, minimum {min_samples} required",
                    field="dataset_id"
                )
            
            # Check if test set will have enough samples
            n_test = int(n_samples * test_size)
            if n_test < 2:
                raise ValidationError(
                    message=f"Test set would have only {n_test} samples with test_size={test_size}. "
                           f"Increase dataset size or reduce test_size",
                    field="test_size"
                )
            
            # Check if train set will have enough samples
            n_train = n_samples - n_test
            if n_train < 5:
                raise ValidationError(
                    message=f"Training set would have only {n_train} samples with test_size={test_size}. "
                           f"Increase dataset size or increase test_size",
                    field="test_size"
                )
            
            logger.debug(
                f"Dataset size validation passed: {n_samples} samples "
                f"({n_train} train, {n_test} test)"
            )
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                message=f"Failed to validate dataset size: {str(e)}",
                field="dataset_id"
            )


def get_training_validator(db: Session) -> TrainingConfigValidator:
    """
    Factory function to create a training config validator.
    
    Args:
        db: Database session
    
    Returns:
        TrainingConfigValidator instance
    
    Example:
        >>> validator = get_training_validator(db)
        >>> validator.validate_training_config(...)
    """
    return TrainingConfigValidator(db)
