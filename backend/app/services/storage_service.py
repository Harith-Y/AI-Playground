"""
Model Serialization and Storage Service

This module provides comprehensive model serialization, storage, and retrieval
functionality for trained ML models. It handles:
- Model artifact serialization (joblib format)
- Metadata persistence (training info, hyperparameters, metrics)
- File system organization (user/experiment/model hierarchy)
- Model loading and deserialization
- Storage cleanup and management

The service ensures models are stored with complete context for reproducibility
and can be loaded for inference or further training.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import joblib
import uuid

from app.core.config import settings
from app.utils.logger import get_logger
from app.ml_engine.models.base import BaseModelWrapper


logger = get_logger(__name__)


class ModelSerializationService:
    """
    Service for serializing and storing trained ML models.
    
    This service provides a centralized interface for:
    - Saving trained models with metadata
    - Loading models for inference
    - Managing model storage hierarchy
    - Cleaning up old models
    - Exporting models with dependencies
    
    Storage Structure:
        {UPLOAD_DIR}/
        └── models/
            └── {experiment_id}/
                ├── {model_run_id}.joblib          # Model artifact
                ├── {model_run_id}_metadata.json   # Training metadata
                └── {model_run_id}_config.json     # Model configuration
    
    Example:
        >>> service = ModelSerializationService()
        >>> model_path = service.save_model(
        ...     model=trained_model,
        ...     model_run_id="abc-123",
        ...     experiment_id="exp-456",
        ...     metadata={"accuracy": 0.95}
        ... )
        >>> loaded_model = service.load_model(model_path)
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the model serialization service.
        
        Args:
            base_dir: Base directory for model storage. 
                     Defaults to settings.UPLOAD_DIR/models
        """
        self.base_dir = Path(base_dir) if base_dir else Path(settings.UPLOAD_DIR) / "models"
        
        # Try to create directory, fallback to /tmp if permission denied (e.g., on Render)
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            # Test write permission
            test_file = self.base_dir / ".test_write"
            test_file.touch()
            test_file.unlink()
            logger.info(f"ModelSerializationService initialized with base_dir: {self.base_dir}")
        except (PermissionError, OSError) as e:
            logger.warning(f"Failed to initialize storage at {self.base_dir}: {e}")
            logger.warning("Falling back to /tmp/models")
            self.base_dir = Path("/tmp/models")
            self.base_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"ModelSerializationService initialized with base_dir: {self.base_dir}")
    
    def save_model(
        self,
        model: BaseModelWrapper,
        model_run_id: str,
        experiment_id: str,
        additional_metadata: Optional[Dict[str, Any]] = None,
        save_config: bool = True,
        save_metadata: bool = True
    ) -> str:
        """
        Save a trained model with its metadata and configuration.
        
        This method:
        1. Creates experiment directory if it doesn't exist
        2. Serializes the model using joblib
        3. Saves model configuration as JSON
        4. Saves training metadata as JSON
        5. Returns the path to the saved model
        
        Args:
            model: Trained model wrapper instance
            model_run_id: Unique identifier for this model run
            experiment_id: Experiment this model belongs to
            additional_metadata: Extra metadata to save (metrics, feature importance, etc.)
            save_config: Whether to save model configuration separately
            save_metadata: Whether to save training metadata separately
        
        Returns:
            str: Absolute path to the saved model file
        
        Raises:
            RuntimeError: If model hasn't been fitted
            IOError: If file writing fails
        
        Example:
            >>> service = ModelSerializationService()
            >>> model_path = service.save_model(
            ...     model=trained_rf_model,
            ...     model_run_id="run-123",
            ...     experiment_id="exp-456",
            ...     additional_metadata={
            ...         "accuracy": 0.95,
            ...         "f1_score": 0.93,
            ...         "feature_importance": {...}
            ...     }
            ... )
            >>> print(f"Model saved to: {model_path}")
        """
        if not model.is_fitted:
            raise RuntimeError("Cannot save model that hasn't been fitted")
        
        logger.info(
            f"Saving model",
            extra={
                'event': 'model_save_start',
                'model_run_id': model_run_id,
                'experiment_id': experiment_id,
                'model_type': model.config.model_type
            }
        )
        
        try:
            # Create experiment directory
            experiment_dir = self.base_dir / str(experiment_id)
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # Define file paths
            model_path = experiment_dir / f"{model_run_id}.joblib"
            config_path = experiment_dir / f"{model_run_id}_config.json"
            metadata_path = experiment_dir / f"{model_run_id}_metadata.json"
            
            # 1. Save the model artifact using the model's built-in save method
            model.save(str(model_path))
            logger.info(f"Model artifact saved to: {model_path}")
            
            # 2. Save model configuration separately (for quick inspection)
            if save_config:
                config_data = {
                    "model_run_id": model_run_id,
                    "experiment_id": experiment_id,
                    "model_type": model.config.model_type,
                    "task_type": model.get_task_type(),
                    "hyperparameters": model.config.hyperparameters,
                    "random_state": model.config.random_state,
                    "saved_at": datetime.utcnow().isoformat(),
                    "model_artifact_path": str(model_path),
                    "sklearn_version": self._get_sklearn_version()
                }
                
                with open(config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                logger.info(f"Model configuration saved to: {config_path}")
            
            # 3. Save training metadata separately
            if save_metadata:
                metadata = {
                    "model_run_id": model_run_id,
                    "experiment_id": experiment_id,
                    "training_metadata": model.metadata.to_dict(),
                    "feature_names": model._feature_names,
                    "target_name": model._target_name,
                    "n_features": len(model._feature_names),
                    "saved_at": datetime.utcnow().isoformat()
                }
                
                # Add additional metadata if provided
                if additional_metadata:
                    metadata["additional"] = additional_metadata
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Model metadata saved to: {metadata_path}")
            
            # Calculate file sizes
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            
            logger.info(
                f"Model saved successfully",
                extra={
                    'event': 'model_save_complete',
                    'model_run_id': model_run_id,
                    'experiment_id': experiment_id,
                    'model_path': str(model_path),
                    'model_size_mb': round(model_size_mb, 2),
                    'has_config': save_config,
                    'has_metadata': save_metadata
                }
            )
            
            return str(model_path)
        
        except Exception as e:
            logger.error(
                f"Failed to save model: {e}",
                extra={
                    'event': 'model_save_error',
                    'model_run_id': model_run_id,
                    'experiment_id': experiment_id,
                    'error': str(e)
                },
                exc_info=True
            )
            raise IOError(f"Failed to save model: {e}") from e
    
    def load_model(
        self,
        model_path: str,
        load_metadata: bool = False
    ) -> Tuple[BaseModelWrapper, Optional[Dict[str, Any]]]:
        """
        Load a saved model from disk.
        
        Args:
            model_path: Path to the saved model file (.joblib)
            load_metadata: Whether to also load metadata JSON
        
        Returns:
            Tuple of (model_wrapper, metadata_dict)
            metadata_dict is None if load_metadata=False or file doesn't exist
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            IOError: If loading fails
        
        Example:
            >>> service = ModelSerializationService()
            >>> model, metadata = service.load_model(
            ...     model_path="/path/to/model.joblib",
            ...     load_metadata=True
            ... )
            >>> predictions = model.predict(X_test)
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(
            f"Loading model from: {model_path}",
            extra={
                'event': 'model_load_start',
                'model_path': str(model_path)
            }
        )
        
        try:
            # Load the model using joblib
            save_dict = joblib.load(model_path)
            
            # Reconstruct the model wrapper
            from app.ml_engine.models.registry import ModelFactory
            
            config_dict = save_dict["config"]
            model_type = config_dict["model_type"]
            
            # Create model wrapper using factory
            from app.ml_engine.models.base import ModelConfig
            config = ModelConfig.from_dict(config_dict)
            model = ModelFactory.create_model(model_type, config=config)
            
            # Restore model state
            model.model = save_dict["model"]
            model._feature_names = save_dict["feature_names"]
            model._target_name = save_dict["target_name"]
            model.is_fitted = save_dict["is_fitted"]
            
            # Restore metadata
            from app.ml_engine.models.base import TrainingMetadata
            metadata_dict = save_dict["metadata"]
            model.metadata = TrainingMetadata(
                train_start_time=datetime.fromisoformat(metadata_dict["train_start_time"]) if metadata_dict.get("train_start_time") else None,
                train_end_time=datetime.fromisoformat(metadata_dict["train_end_time"]) if metadata_dict.get("train_end_time") else None,
                training_duration_seconds=metadata_dict.get("training_duration_seconds"),
                n_train_samples=metadata_dict.get("n_train_samples"),
                n_features=metadata_dict.get("n_features"),
                feature_names=metadata_dict.get("feature_names"),
                target_name=metadata_dict.get("target_name"),
                sklearn_version=metadata_dict.get("sklearn_version")
            )
            
            logger.info(
                f"Model loaded successfully",
                extra={
                    'event': 'model_load_complete',
                    'model_path': str(model_path),
                    'model_type': model_type,
                    'is_fitted': model.is_fitted
                }
            )
            
            # Load additional metadata if requested
            additional_metadata = None
            if load_metadata:
                metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        additional_metadata = json.load(f)
                    logger.info(f"Additional metadata loaded from: {metadata_path}")
            
            return model, additional_metadata
        
        except Exception as e:
            logger.error(
                f"Failed to load model: {e}",
                extra={
                    'event': 'model_load_error',
                    'model_path': str(model_path),
                    'error': str(e)
                },
                exc_info=True
            )
            raise IOError(f"Failed to load model: {e}") from e
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        Get information about a saved model without loading it.
        
        This is useful for quickly inspecting model metadata without
        loading the entire model into memory.
        
        Args:
            model_path: Path to the saved model file
        
        Returns:
            Dictionary with model information (config + metadata)
        
        Raises:
            FileNotFoundError: If model or metadata files don't exist
        
        Example:
            >>> service = ModelSerializationService()
            >>> info = service.get_model_info("/path/to/model.joblib")
            >>> print(f"Model type: {info['model_type']}")
            >>> print(f"Accuracy: {info['metrics']['accuracy']}")
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        info = {}
        
        # Load config if exists
        config_path = model_path.parent / f"{model_path.stem}_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                info["config"] = json.load(f)
        
        # Load metadata if exists
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                info["metadata"] = json.load(f)
        
        # Add file info
        info["file_info"] = {
            "path": str(model_path),
            "size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
            "created_at": datetime.fromtimestamp(model_path.stat().st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
        }
        
        return info
    
    def delete_model(self, model_path: str, delete_metadata: bool = True) -> bool:
        """
        Delete a saved model and optionally its metadata files.
        
        Args:
            model_path: Path to the model file to delete
            delete_metadata: Whether to also delete config and metadata JSON files
        
        Returns:
            bool: True if deletion succeeded, False otherwise
        
        Example:
            >>> service = ModelSerializationService()
            >>> success = service.delete_model("/path/to/model.joblib")
            >>> print(f"Deletion {'succeeded' if success else 'failed'}")
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return False
        
        try:
            # Delete model file
            model_path.unlink()
            logger.info(f"Deleted model file: {model_path}")
            
            # Delete metadata files if requested
            if delete_metadata:
                config_path = model_path.parent / f"{model_path.stem}_config.json"
                metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
                
                if config_path.exists():
                    config_path.unlink()
                    logger.info(f"Deleted config file: {config_path}")
                
                if metadata_path.exists():
                    metadata_path.unlink()
                    logger.info(f"Deleted metadata file: {metadata_path}")
            
            logger.info(
                f"Model deletion complete",
                extra={
                    'event': 'model_delete_complete',
                    'model_path': str(model_path),
                    'deleted_metadata': delete_metadata
                }
            )
            
            return True
        
        except Exception as e:
            logger.error(
                f"Failed to delete model: {e}",
                extra={
                    'event': 'model_delete_error',
                    'model_path': str(model_path),
                    'error': str(e)
                },
                exc_info=True
            )
            return False
    
    def list_experiment_models(self, experiment_id: str) -> List[Dict[str, Any]]:
        """
        List all models for a given experiment.
        
        Args:
            experiment_id: Experiment ID to list models for
        
        Returns:
            List of dictionaries with model information
        
        Example:
            >>> service = ModelSerializationService()
            >>> models = service.list_experiment_models("exp-123")
            >>> for model in models:
            ...     print(f"{model['model_run_id']}: {model['model_type']}")
        """
        experiment_dir = self.base_dir / str(experiment_id)
        
        if not experiment_dir.exists():
            logger.warning(f"Experiment directory not found: {experiment_dir}")
            return []
        
        models = []
        
        # Find all .joblib files
        for model_file in experiment_dir.glob("*.joblib"):
            try:
                info = self.get_model_info(str(model_file))
                models.append(info)
            except Exception as e:
                logger.warning(f"Failed to get info for {model_file}: {e}")
        
        return models
    
    def cleanup_experiment(self, experiment_id: str) -> int:
        """
        Delete all models for an experiment.
        
        Args:
            experiment_id: Experiment ID to clean up
        
        Returns:
            int: Number of models deleted
        
        Example:
            >>> service = ModelSerializationService()
            >>> deleted_count = service.cleanup_experiment("exp-123")
            >>> print(f"Deleted {deleted_count} models")
        """
        experiment_dir = self.base_dir / str(experiment_id)
        
        if not experiment_dir.exists():
            logger.warning(f"Experiment directory not found: {experiment_dir}")
            return 0
        
        try:
            # Count files before deletion
            model_files = list(experiment_dir.glob("*.joblib"))
            count = len(model_files)
            
            # Delete entire experiment directory
            shutil.rmtree(experiment_dir)
            
            logger.info(
                f"Cleaned up experiment directory",
                extra={
                    'event': 'experiment_cleanup',
                    'experiment_id': experiment_id,
                    'models_deleted': count
                }
            )
            
            return count
        
        except Exception as e:
            logger.error(
                f"Failed to cleanup experiment: {e}",
                extra={
                    'event': 'experiment_cleanup_error',
                    'experiment_id': experiment_id,
                    'error': str(e)
                },
                exc_info=True
            )
            return 0
    
    def export_model_package(
        self,
        model_path: str,
        output_dir: str,
        include_requirements: bool = True
    ) -> str:
        """
        Export a model with all its files as a package.
        
        Creates a directory with:
        - Model artifact (.joblib)
        - Configuration JSON
        - Metadata JSON
        - requirements.txt (optional)
        - README.md with usage instructions
        
        Args:
            model_path: Path to the model to export
            output_dir: Directory to create the package in
            include_requirements: Whether to generate requirements.txt
        
        Returns:
            str: Path to the created package directory
        
        Example:
            >>> service = ModelSerializationService()
            >>> package_path = service.export_model_package(
            ...     model_path="/path/to/model.joblib",
            ...     output_dir="/exports",
            ...     include_requirements=True
            ... )
            >>> print(f"Package created at: {package_path}")
        """
        model_path = Path(model_path)
        output_dir = Path(output_dir)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create package directory
        package_name = f"model_package_{model_path.stem}"
        package_dir = output_dir / package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy model file
            shutil.copy(model_path, package_dir / model_path.name)
            
            # Copy config and metadata if they exist
            config_path = model_path.parent / f"{model_path.stem}_config.json"
            if config_path.exists():
                shutil.copy(config_path, package_dir / config_path.name)
            
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if metadata_path.exists():
                shutil.copy(metadata_path, package_dir / metadata_path.name)
            
            # Generate requirements.txt
            if include_requirements:
                requirements = self._generate_requirements()
                with open(package_dir / "requirements.txt", 'w') as f:
                    f.write(requirements)
            
            # Generate README
            readme = self._generate_readme(model_path.stem)
            with open(package_dir / "README.md", 'w') as f:
                f.write(readme)
            
            logger.info(
                f"Model package exported",
                extra={
                    'event': 'model_export',
                    'model_path': str(model_path),
                    'package_dir': str(package_dir)
                }
            )
            
            return str(package_dir)
        
        except Exception as e:
            logger.error(f"Failed to export model package: {e}", exc_info=True)
            raise IOError(f"Failed to export model package: {e}") from e
    
    def _get_sklearn_version(self) -> str:
        """Get scikit-learn version."""
        try:
            import sklearn
            return sklearn.__version__
        except:
            return "unknown"
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt content."""
        return """# AI-Playground Model Requirements
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
"""
    
    def _generate_readme(self, model_name: str) -> str:
        """Generate README.md content."""
        return f"""# AI-Playground Model Package: {model_name}

This package contains a trained machine learning model exported from AI-Playground.

## Contents

- `{model_name}.joblib` - Trained model artifact
- `{model_name}_config.json` - Model configuration and hyperparameters
- `{model_name}_metadata.json` - Training metadata and feature information
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Loading the Model

```python
import joblib
from pathlib import Path

# Load the model
model_data = joblib.load('{model_name}.joblib')
model = model_data['model']

# Make predictions
predictions = model.predict(X_test)
```

### Model Information

See `{model_name}_config.json` for:
- Model type and hyperparameters
- Training configuration
- Feature names

See `{model_name}_metadata.json` for:
- Training duration and samples
- Feature importance (if available)
- Additional metrics

## Support

For questions or issues, refer to the AI-Playground documentation.

Generated by AI-Playground Model Serialization Service
"""


# Singleton instance
_model_serialization_service: Optional[ModelSerializationService] = None


def get_model_serialization_service() -> ModelSerializationService:
    """
    Get the singleton instance of ModelSerializationService.
    
    Returns:
        ModelSerializationService instance
    
    Example:
        >>> service = get_model_serialization_service()
        >>> model_path = service.save_model(model, run_id, exp_id)
    """
    global _model_serialization_service
    if _model_serialization_service is None:
        _model_serialization_service = ModelSerializationService()
    return _model_serialization_service
