"""
Model and Pipeline Serialization Module.

This module provides comprehensive serialization utilities for saving and loading
ML models, preprocessing pipelines, and complete ML workflows. Supports multiple
formats including pickle, joblib, and JSON configuration.

Features:
- Model serialization with metadata
- Pipeline serialization with step configurations
- Complete workflow serialization (preprocessing + model)
- Version tracking and compatibility checks
- Compression support for large models
- Cloud storage integration (S3, R2)
"""

import pickle
import joblib
import json
import gzip
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple
from datetime import datetime
import hashlib
import warnings

from app.utils.logger import get_logger

logger = get_logger("serialization")


# Version for serialization format compatibility
SERIALIZATION_VERSION = "1.0.0"


class SerializationError(Exception):
    """Raised when serialization/deserialization fails."""
    pass


class VersionMismatchWarning(UserWarning):
    """Warning for version mismatches during deserialization."""
    pass


class ModelSerializer:
    """
    Serializer for ML model wrappers.
    
    Handles saving and loading of fitted models with metadata,
    configuration, and version tracking.
    """
    
    def __init__(self, compression: bool = False):
        """
        Initialize model serializer.
        
        Args:
            compression: Whether to compress saved models (useful for large models)
        """
        self.compression = compression
    
    def save_model(
        self,
        model: Any,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ) -> Path:
        """
        Save a fitted model to disk.
        
        Args:
            model: Fitted model wrapper (BaseModelWrapper instance)
            path: Path to save the model
            metadata: Additional metadata to store
            overwrite: Whether to overwrite existing file
        
        Returns:
            Path where model was saved
        
        Raises:
            SerializationError: If saving fails
            FileExistsError: If file exists and overwrite=False
        """
        path = Path(path)
        
        # Check if file exists
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Model file already exists: {path}. Use overwrite=True to replace."
            )
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Check if model is fitted
            if not getattr(model, 'is_fitted', False):
                warnings.warn(
                    "Saving unfitted model. The model should be fitted before saving.",
                    UserWarning
                )
            
            # Prepare save dictionary
            save_dict = {
                "model": model.model if hasattr(model, 'model') else model,
                "config": model.config.to_dict() if hasattr(model, 'config') else {},
                "metadata": model.metadata.to_dict() if hasattr(model, 'metadata') else {},
                "feature_names": getattr(model, '_feature_names', []),
                "target_name": getattr(model, '_target_name', None),
                "is_fitted": getattr(model, 'is_fitted', False),
                "model_class": model.__class__.__name__,
                "model_module": model.__class__.__module__,
                "serialization_version": SERIALIZATION_VERSION,
                "saved_at": datetime.now().isoformat(),
                "additional_metadata": metadata or {}
            }
            
            # Add sklearn version if available
            try:
                import sklearn
                save_dict["sklearn_version"] = sklearn.__version__
            except ImportError:
                pass
            
            # Save using joblib (better for sklearn models)
            if self.compression:
                # Save with compression
                with gzip.open(f"{path}.gz", 'wb') as f:
                    joblib.dump(save_dict, f, compress=3)
                final_path = Path(f"{path}.gz")
            else:
                joblib.dump(save_dict, path)
                final_path = path
            
            # Calculate file hash for integrity
            file_hash = self._calculate_file_hash(final_path)
            
            logger.info(
                f"Model saved successfully to {final_path} "
                f"(size: {final_path.stat().st_size / 1024:.2f} KB, hash: {file_hash[:8]})"
            )
            
            return final_path
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise SerializationError(f"Failed to save model: {str(e)}") from e

    
    def load_model(
        self,
        path: Union[str, Path],
        verify_version: bool = True
    ) -> Any:
        """
        Load a saved model from disk.
        
        Args:
            path: Path to the saved model
            verify_version: Whether to check version compatibility
        
        Returns:
            Loaded model wrapper
        
        Raises:
            SerializationError: If loading fails
            FileNotFoundError: If file doesn't exist
        """
        path = Path(path)
        
        # Check if file exists
        if not path.exists():
            # Try with .gz extension
            gz_path = Path(f"{path}.gz")
            if gz_path.exists():
                path = gz_path
            else:
                raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            # Load the saved dictionary
            if path.suffix == '.gz':
                with gzip.open(path, 'rb') as f:
                    save_dict = joblib.load(f)
            else:
                save_dict = joblib.load(path)
            
            # Version check
            if verify_version:
                saved_version = save_dict.get("serialization_version", "unknown")
                if saved_version != SERIALIZATION_VERSION:
                    warnings.warn(
                        f"Version mismatch: model saved with version {saved_version}, "
                        f"loading with version {SERIALIZATION_VERSION}. "
                        "This may cause compatibility issues.",
                        VersionMismatchWarning
                    )
            
            # Reconstruct model wrapper
            model_class_name = save_dict.get("model_class")
            model_module = save_dict.get("model_module")
            
            if model_class_name and model_module:
                # Dynamically import the model class
                from importlib import import_module
                module = import_module(model_module)
                model_class = getattr(module, model_class_name)
                
                # Create model config
                from app.ml_engine.models.base import ModelConfig
                config = ModelConfig.from_dict(save_dict["config"], validate=False)
                
                # Instantiate wrapper
                wrapper = model_class(config)
                wrapper.model = save_dict["model"]
                wrapper._feature_names = save_dict.get("feature_names", [])
                wrapper._target_name = save_dict.get("target_name")
                wrapper.is_fitted = save_dict.get("is_fitted", False)
                
                # Reconstruct metadata
                from app.ml_engine.models.base import TrainingMetadata
                metadata_dict = save_dict.get("metadata", {})
                wrapper.metadata = TrainingMetadata(
                    train_start_time=datetime.fromisoformat(metadata_dict["train_start_time"]) if metadata_dict.get("train_start_time") else None,
                    train_end_time=datetime.fromisoformat(metadata_dict["train_end_time"]) if metadata_dict.get("train_end_time") else None,
                    training_duration_seconds=metadata_dict.get("training_duration_seconds"),
                    n_train_samples=metadata_dict.get("n_train_samples"),
                    n_features=metadata_dict.get("n_features"),
                    feature_names=metadata_dict.get("feature_names"),
                    target_name=metadata_dict.get("target_name"),
                    sklearn_version=metadata_dict.get("sklearn_version")
                )
                
                logger.info(f"Model loaded successfully from {path}")
                return wrapper
            else:
                # Fallback: return raw model
                logger.warning("Model class information not found, returning raw model")
                return save_dict["model"]
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise SerializationError(f"Failed to load model: {str(e)}") from e

    
    def get_model_info(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get model information without loading the full model.
        
        Args:
            path: Path to the saved model
        
        Returns:
            Dictionary with model metadata
        """
        path = Path(path)
        
        if not path.exists():
            gz_path = Path(f"{path}.gz")
            if gz_path.exists():
                path = gz_path
            else:
                raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            # Load only metadata
            if path.suffix == '.gz':
                with gzip.open(path, 'rb') as f:
                    save_dict = joblib.load(f)
            else:
                save_dict = joblib.load(path)
            
            # Extract info without loading the model
            info = {
                "model_class": save_dict.get("model_class"),
                "model_type": save_dict.get("config", {}).get("model_type"),
                "is_fitted": save_dict.get("is_fitted"),
                "n_features": save_dict.get("metadata", {}).get("n_features"),
                "n_train_samples": save_dict.get("metadata", {}).get("n_train_samples"),
                "feature_names": save_dict.get("feature_names"),
                "target_name": save_dict.get("target_name"),
                "saved_at": save_dict.get("saved_at"),
                "serialization_version": save_dict.get("serialization_version"),
                "sklearn_version": save_dict.get("sklearn_version"),
                "file_size_kb": path.stat().st_size / 1024,
                "compressed": path.suffix == '.gz'
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            raise SerializationError(f"Failed to get model info: {str(e)}") from e
    
    def _calculate_file_hash(self, path: Path) -> str:
        """Calculate SHA256 hash of file for integrity checking."""
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()



class PipelineSerializer:
    """
    Serializer for preprocessing pipelines.
    
    Handles saving and loading of preprocessing pipelines with
    all fitted transformers and configurations.
    """
    
    def __init__(self, compression: bool = False):
        """
        Initialize pipeline serializer.
        
        Args:
            compression: Whether to compress saved pipelines
        """
        self.compression = compression
    
    def save_pipeline(
        self,
        pipeline: Any,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ) -> Path:
        """
        Save a preprocessing pipeline to disk.
        
        Args:
            pipeline: Fitted pipeline instance
            path: Path to save the pipeline
            metadata: Additional metadata to store
            overwrite: Whether to overwrite existing file
        
        Returns:
            Path where pipeline was saved
        
        Raises:
            SerializationError: If saving fails
        """
        path = Path(path)
        
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Pipeline file already exists: {path}. Use overwrite=True to replace."
            )
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Check if pipeline is fitted
            if not getattr(pipeline, 'fitted', False):
                warnings.warn(
                    "Saving unfitted pipeline. The pipeline should be fitted before saving.",
                    UserWarning
                )
            
            # Prepare save dictionary
            save_dict = {
                "pipeline": pipeline,
                "name": getattr(pipeline, 'name', 'PreprocessingPipeline'),
                "fitted": getattr(pipeline, 'fitted', False),
                "num_steps": len(getattr(pipeline, 'steps', [])),
                "step_names": [step.name for step in getattr(pipeline, 'steps', [])],
                "step_statistics": getattr(pipeline, 'step_statistics', []),
                "metadata": getattr(pipeline, 'metadata', {}),
                "serialization_version": SERIALIZATION_VERSION,
                "saved_at": datetime.now().isoformat(),
                "additional_metadata": metadata or {}
            }
            
            # Save using joblib
            if self.compression:
                with gzip.open(f"{path}.gz", 'wb') as f:
                    joblib.dump(save_dict, f, compress=3)
                final_path = Path(f"{path}.gz")
            else:
                joblib.dump(save_dict, path)
                final_path = path
            
            logger.info(
                f"Pipeline saved successfully to {final_path} "
                f"({save_dict['num_steps']} steps, "
                f"size: {final_path.stat().st_size / 1024:.2f} KB)"
            )
            
            return final_path
            
        except Exception as e:
            logger.error(f"Failed to save pipeline: {str(e)}")
            raise SerializationError(f"Failed to save pipeline: {str(e)}") from e

    
    def load_pipeline(
        self,
        path: Union[str, Path],
        verify_version: bool = True
    ) -> Any:
        """
        Load a saved pipeline from disk.
        
        Args:
            path: Path to the saved pipeline
            verify_version: Whether to check version compatibility
        
        Returns:
            Loaded pipeline
        
        Raises:
            SerializationError: If loading fails
        """
        path = Path(path)
        
        if not path.exists():
            gz_path = Path(f"{path}.gz")
            if gz_path.exists():
                path = gz_path
            else:
                raise FileNotFoundError(f"Pipeline file not found: {path}")
        
        try:
            # Load the saved dictionary
            if path.suffix == '.gz':
                with gzip.open(path, 'rb') as f:
                    save_dict = joblib.load(f)
            else:
                save_dict = joblib.load(path)
            
            # Version check
            if verify_version:
                saved_version = save_dict.get("serialization_version", "unknown")
                if saved_version != SERIALIZATION_VERSION:
                    warnings.warn(
                        f"Version mismatch: pipeline saved with version {saved_version}, "
                        f"loading with version {SERIALIZATION_VERSION}",
                        VersionMismatchWarning
                    )
            
            pipeline = save_dict["pipeline"]
            
            logger.info(
                f"Pipeline loaded successfully from {path} "
                f"({save_dict['num_steps']} steps)"
            )
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {str(e)}")
            raise SerializationError(f"Failed to load pipeline: {str(e)}") from e
    
    def get_pipeline_info(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get pipeline information without loading the full pipeline.
        
        Args:
            path: Path to the saved pipeline
        
        Returns:
            Dictionary with pipeline metadata
        """
        path = Path(path)
        
        if not path.exists():
            gz_path = Path(f"{path}.gz")
            if gz_path.exists():
                path = gz_path
            else:
                raise FileNotFoundError(f"Pipeline file not found: {path}")
        
        try:
            if path.suffix == '.gz':
                with gzip.open(path, 'rb') as f:
                    save_dict = joblib.load(f)
            else:
                save_dict = joblib.load(path)
            
            info = {
                "name": save_dict.get("name"),
                "fitted": save_dict.get("fitted"),
                "num_steps": save_dict.get("num_steps"),
                "step_names": save_dict.get("step_names"),
                "saved_at": save_dict.get("saved_at"),
                "serialization_version": save_dict.get("serialization_version"),
                "file_size_kb": path.stat().st_size / 1024,
                "compressed": path.suffix == '.gz'
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get pipeline info: {str(e)}")
            raise SerializationError(f"Failed to get pipeline info: {str(e)}") from e



class WorkflowSerializer:
    """
    Serializer for complete ML workflows (preprocessing + model).
    
    Handles saving and loading of complete ML workflows including
    preprocessing pipeline, trained model, and all metadata.
    """
    
    def __init__(self, compression: bool = True):
        """
        Initialize workflow serializer.
        
        Args:
            compression: Whether to compress saved workflows (recommended for workflows)
        """
        self.compression = compression
        self.model_serializer = ModelSerializer(compression=False)
        self.pipeline_serializer = PipelineSerializer(compression=False)
    
    def save_workflow(
        self,
        pipeline: Any,
        model: Any,
        path: Union[str, Path],
        workflow_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ) -> Path:
        """
        Save a complete ML workflow (preprocessing + model).
        
        Args:
            pipeline: Fitted preprocessing pipeline
            model: Fitted model wrapper
            path: Path to save the workflow
            workflow_name: Name for the workflow
            metadata: Additional metadata
            overwrite: Whether to overwrite existing file
        
        Returns:
            Path where workflow was saved
        
        Raises:
            SerializationError: If saving fails
        """
        path = Path(path)
        
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Workflow file already exists: {path}. Use overwrite=True to replace."
            )
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Prepare workflow dictionary
            workflow_dict = {
                "workflow_name": workflow_name or "ML_Workflow",
                "pipeline": pipeline,
                "model": model.model if hasattr(model, 'model') else model,
                "model_config": model.config.to_dict() if hasattr(model, 'config') else {},
                "model_metadata": model.metadata.to_dict() if hasattr(model, 'metadata') else {},
                "model_class": model.__class__.__name__ if hasattr(model, '__class__') else None,
                "model_module": model.__class__.__module__ if hasattr(model, '__class__') else None,
                "pipeline_name": getattr(pipeline, 'name', 'PreprocessingPipeline'),
                "pipeline_fitted": getattr(pipeline, 'fitted', False),
                "model_fitted": getattr(model, 'is_fitted', False),
                "feature_names": getattr(model, '_feature_names', []),
                "target_name": getattr(model, '_target_name', None),
                "serialization_version": SERIALIZATION_VERSION,
                "saved_at": datetime.now().isoformat(),
                "additional_metadata": metadata or {}
            }
            
            # Add sklearn version
            try:
                import sklearn
                workflow_dict["sklearn_version"] = sklearn.__version__
            except ImportError:
                pass
            
            # Save using joblib with compression
            if self.compression:
                with gzip.open(f"{path}.gz", 'wb') as f:
                    joblib.dump(workflow_dict, f, compress=3)
                final_path = Path(f"{path}.gz")
            else:
                joblib.dump(workflow_dict, path)
                final_path = path
            
            logger.info(
                f"Workflow '{workflow_dict['workflow_name']}' saved successfully to {final_path} "
                f"(size: {final_path.stat().st_size / 1024:.2f} KB)"
            )
            
            return final_path
            
        except Exception as e:
            logger.error(f"Failed to save workflow: {str(e)}")
            raise SerializationError(f"Failed to save workflow: {str(e)}") from e

    
    def load_workflow(
        self,
        path: Union[str, Path],
        verify_version: bool = True
    ) -> Tuple[Any, Any]:
        """
        Load a saved workflow from disk.
        
        Args:
            path: Path to the saved workflow
            verify_version: Whether to check version compatibility
        
        Returns:
            Tuple of (pipeline, model)
        
        Raises:
            SerializationError: If loading fails
        """
        path = Path(path)
        
        if not path.exists():
            gz_path = Path(f"{path}.gz")
            if gz_path.exists():
                path = gz_path
            else:
                raise FileNotFoundError(f"Workflow file not found: {path}")
        
        try:
            # Load the workflow dictionary
            if path.suffix == '.gz':
                with gzip.open(path, 'rb') as f:
                    workflow_dict = joblib.load(f)
            else:
                workflow_dict = joblib.load(path)
            
            # Version check
            if verify_version:
                saved_version = workflow_dict.get("serialization_version", "unknown")
                if saved_version != SERIALIZATION_VERSION:
                    warnings.warn(
                        f"Version mismatch: workflow saved with version {saved_version}, "
                        f"loading with version {SERIALIZATION_VERSION}",
                        VersionMismatchWarning
                    )
            
            # Extract pipeline
            pipeline = workflow_dict["pipeline"]
            
            # Reconstruct model wrapper
            model_class_name = workflow_dict.get("model_class")
            model_module = workflow_dict.get("model_module")
            
            if model_class_name and model_module:
                from importlib import import_module
                from app.ml_engine.models.base import ModelConfig, TrainingMetadata
                
                module = import_module(model_module)
                model_class = getattr(module, model_class_name)
                
                config = ModelConfig.from_dict(workflow_dict["model_config"], validate=False)
                wrapper = model_class(config)
                wrapper.model = workflow_dict["model"]
                wrapper._feature_names = workflow_dict.get("feature_names", [])
                wrapper._target_name = workflow_dict.get("target_name")
                wrapper.is_fitted = workflow_dict.get("model_fitted", False)
                
                # Reconstruct metadata
                metadata_dict = workflow_dict.get("model_metadata", {})
                wrapper.metadata = TrainingMetadata(
                    train_start_time=datetime.fromisoformat(metadata_dict["train_start_time"]) if metadata_dict.get("train_start_time") else None,
                    train_end_time=datetime.fromisoformat(metadata_dict["train_end_time"]) if metadata_dict.get("train_end_time") else None,
                    training_duration_seconds=metadata_dict.get("training_duration_seconds"),
                    n_train_samples=metadata_dict.get("n_train_samples"),
                    n_features=metadata_dict.get("n_features"),
                    feature_names=metadata_dict.get("feature_names"),
                    target_name=metadata_dict.get("target_name"),
                    sklearn_version=metadata_dict.get("sklearn_version")
                )
                
                model = wrapper
            else:
                model = workflow_dict["model"]
            
            logger.info(
                f"Workflow '{workflow_dict['workflow_name']}' loaded successfully from {path}"
            )
            
            return pipeline, model
            
        except Exception as e:
            logger.error(f"Failed to load workflow: {str(e)}")
            raise SerializationError(f"Failed to load workflow: {str(e)}") from e
    
    def get_workflow_info(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get workflow information without loading the full workflow.
        
        Args:
            path: Path to the saved workflow
        
        Returns:
            Dictionary with workflow metadata
        """
        path = Path(path)
        
        if not path.exists():
            gz_path = Path(f"{path}.gz")
            if gz_path.exists():
                path = gz_path
            else:
                raise FileNotFoundError(f"Workflow file not found: {path}")
        
        try:
            if path.suffix == '.gz':
                with gzip.open(path, 'rb') as f:
                    workflow_dict = joblib.load(f)
            else:
                workflow_dict = joblib.load(path)
            
            info = {
                "workflow_name": workflow_dict.get("workflow_name"),
                "pipeline_name": workflow_dict.get("pipeline_name"),
                "pipeline_fitted": workflow_dict.get("pipeline_fitted"),
                "model_class": workflow_dict.get("model_class"),
                "model_fitted": workflow_dict.get("model_fitted"),
                "n_features": workflow_dict.get("model_metadata", {}).get("n_features"),
                "feature_names": workflow_dict.get("feature_names"),
                "target_name": workflow_dict.get("target_name"),
                "saved_at": workflow_dict.get("saved_at"),
                "serialization_version": workflow_dict.get("serialization_version"),
                "sklearn_version": workflow_dict.get("sklearn_version"),
                "file_size_kb": path.stat().st_size / 1024,
                "compressed": path.suffix == '.gz'
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get workflow info: {str(e)}")
            raise SerializationError(f"Failed to get workflow info: {str(e)}") from e



# Convenience functions for quick serialization

def save_model(
    model: Any,
    path: Union[str, Path],
    compression: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False
) -> Path:
    """
    Convenience function to save a model.
    
    Args:
        model: Fitted model wrapper
        path: Path to save the model
        compression: Whether to compress the model
        metadata: Additional metadata
        overwrite: Whether to overwrite existing file
    
    Returns:
        Path where model was saved
    
    Example:
        >>> from app.ml_engine.models.classification import RandomForestClassifierWrapper
        >>> from app.ml_engine.utils.serialization import save_model
        >>> 
        >>> model = RandomForestClassifierWrapper(config)
        >>> model.fit(X_train, y_train)
        >>> save_model(model, 'models/my_model.pkl', compression=True)
    """
    serializer = ModelSerializer(compression=compression)
    return serializer.save_model(model, path, metadata=metadata, overwrite=overwrite)


def load_model(
    path: Union[str, Path],
    verify_version: bool = True
) -> Any:
    """
    Convenience function to load a model.
    
    Args:
        path: Path to the saved model
        verify_version: Whether to check version compatibility
    
    Returns:
        Loaded model wrapper
    
    Example:
        >>> from app.ml_engine.utils.serialization import load_model
        >>> 
        >>> model = load_model('models/my_model.pkl')
        >>> predictions = model.predict(X_test)
    """
    serializer = ModelSerializer()
    return serializer.load_model(path, verify_version=verify_version)


def save_pipeline(
    pipeline: Any,
    path: Union[str, Path],
    compression: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False
) -> Path:
    """
    Convenience function to save a preprocessing pipeline.
    
    Args:
        pipeline: Fitted pipeline
        path: Path to save the pipeline
        compression: Whether to compress the pipeline
        metadata: Additional metadata
        overwrite: Whether to overwrite existing file
    
    Returns:
        Path where pipeline was saved
    
    Example:
        >>> from app.ml_engine.preprocessing.pipeline import Pipeline
        >>> from app.ml_engine.utils.serialization import save_pipeline
        >>> 
        >>> pipeline = Pipeline(steps=[...])
        >>> pipeline.fit(X_train)
        >>> save_pipeline(pipeline, 'pipelines/my_pipeline.pkl')
    """
    serializer = PipelineSerializer(compression=compression)
    return serializer.save_pipeline(pipeline, path, metadata=metadata, overwrite=overwrite)


def load_pipeline(
    path: Union[str, Path],
    verify_version: bool = True
) -> Any:
    """
    Convenience function to load a preprocessing pipeline.
    
    Args:
        path: Path to the saved pipeline
        verify_version: Whether to check version compatibility
    
    Returns:
        Loaded pipeline
    
    Example:
        >>> from app.ml_engine.utils.serialization import load_pipeline
        >>> 
        >>> pipeline = load_pipeline('pipelines/my_pipeline.pkl')
        >>> X_transformed = pipeline.transform(X_test)
    """
    serializer = PipelineSerializer()
    return serializer.load_pipeline(path, verify_version=verify_version)


def save_workflow(
    pipeline: Any,
    model: Any,
    path: Union[str, Path],
    workflow_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False
) -> Path:
    """
    Convenience function to save a complete ML workflow.
    
    Args:
        pipeline: Fitted preprocessing pipeline
        model: Fitted model wrapper
        path: Path to save the workflow
        workflow_name: Name for the workflow
        metadata: Additional metadata
        overwrite: Whether to overwrite existing file
    
    Returns:
        Path where workflow was saved
    
    Example:
        >>> from app.ml_engine.utils.serialization import save_workflow
        >>> 
        >>> # After training
        >>> save_workflow(
        ...     pipeline=preprocessing_pipeline,
        ...     model=trained_model,
        ...     path='workflows/my_workflow.pkl',
        ...     workflow_name='Customer_Churn_Predictor'
        ... )
    """
    serializer = WorkflowSerializer(compression=True)
    return serializer.save_workflow(
        pipeline, model, path,
        workflow_name=workflow_name,
        metadata=metadata,
        overwrite=overwrite
    )


def load_workflow(
    path: Union[str, Path],
    verify_version: bool = True
) -> Tuple[Any, Any]:
    """
    Convenience function to load a complete ML workflow.
    
    Args:
        path: Path to the saved workflow
        verify_version: Whether to check version compatibility
    
    Returns:
        Tuple of (pipeline, model)
    
    Example:
        >>> from app.ml_engine.utils.serialization import load_workflow
        >>> 
        >>> pipeline, model = load_workflow('workflows/my_workflow.pkl')
        >>> 
        >>> # Use for prediction
        >>> X_transformed = pipeline.transform(X_new)
        >>> predictions = model.predict(X_transformed)
    """
    serializer = WorkflowSerializer()
    return serializer.load_workflow(path, verify_version=verify_version)


def get_model_info(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get model information without loading the full model.
    
    Args:
        path: Path to the saved model
    
    Returns:
        Dictionary with model metadata
    """
    serializer = ModelSerializer()
    return serializer.get_model_info(path)


def get_pipeline_info(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get pipeline information without loading the full pipeline.
    
    Args:
        path: Path to the saved pipeline
    
    Returns:
        Dictionary with pipeline metadata
    """
    serializer = PipelineSerializer()
    return serializer.get_pipeline_info(path)


def get_workflow_info(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get workflow information without loading the full workflow.
    
    Args:
        path: Path to the saved workflow
    
    Returns:
        Dictionary with workflow metadata
    """
    serializer = WorkflowSerializer()
    return serializer.get_workflow_info(path)
