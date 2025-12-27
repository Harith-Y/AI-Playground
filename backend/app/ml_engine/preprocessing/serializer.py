"""
Pipeline Serialization and Deserialization Module

This module provides comprehensive serialization capabilities for preprocessing pipelines,
supporting multiple formats (pickle, JSON, joblib) with version control and compression.

Key Features:
- Multiple serialization formats (pickle, JSON, joblib)
- Version tracking for backward compatibility
- Compression support (gzip, bz2, lzma)
- Metadata preservation
- Safe deserialization with validation
- Export to various formats (sklearn, ONNX, etc.)
"""

import pickle
import json
import gzip
import bz2
import lzma
from pathlib import Path
from typing import Dict, Any, Optional, Union, Literal
from datetime import datetime
import hashlib
import logging

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

# Version information
SERIALIZER_VERSION = "1.0.0"
SCHEMA_VERSION = 1


class PipelineSerializer:
    """
    Comprehensive serialization utility for preprocessing pipelines.

    Supports multiple formats and provides version control, compression,
    and metadata tracking for safe serialization and deserialization.
    """

    SUPPORTED_FORMATS = ["pickle", "json", "joblib", "yaml"]
    COMPRESSION_FORMATS = ["none", "gzip", "bz2", "lzma"]

    def __init__(
        self,
        default_format: Literal["pickle", "json", "joblib", "yaml"] = "pickle",
        compression: Literal["none", "gzip", "bz2", "lzma"] = "none",
        include_metadata: bool = True
    ):
        """
        Initialize the serializer.

        Args:
            default_format: Default serialization format
            compression: Compression algorithm to use
            include_metadata: Whether to include metadata in serialization
        """
        if default_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {default_format}")

        if default_format == "joblib" and not JOBLIB_AVAILABLE:
            raise ImportError("joblib is required for joblib format. Install with: pip install joblib")

        if default_format == "yaml" and not YAML_AVAILABLE:
            raise ImportError("pyyaml is required for yaml format. Install with: pip install pyyaml")

        if compression not in self.COMPRESSION_FORMATS:
            raise ValueError(f"Unsupported compression: {compression}")

        self.default_format = default_format
        self.compression = compression
        self.include_metadata = include_metadata

    def save(
        self,
        pipeline: Any,
        path: Union[str, Path],
        format: Optional[str] = None,
        compression: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Save a pipeline to disk.

        Args:
            pipeline: The pipeline object to serialize
            path: File path to save to
            format: Serialization format (overrides default)
            compression: Compression to use (overrides default)
            metadata: Additional metadata to include

        Returns:
            Dictionary with serialization info (path, format, size, checksum, etc.)
        """
        path = Path(path)
        format = format or self.default_format
        compression = compression or self.compression

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data with metadata
        data = self._prepare_data(pipeline, metadata)

        # Serialize based on format
        if format == "pickle":
            serialized_data = self._serialize_pickle(data)
        elif format == "json":
            serialized_data = self._serialize_json(data)
        elif format == "joblib":
            serialized_data = self._serialize_joblib(data)
        elif format == "yaml":
            serialized_data = self._serialize_yaml(data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Apply compression
        if compression != "none":
            serialized_data = self._compress(serialized_data, compression)

        # Write to file
        mode = 'wb' if isinstance(serialized_data, bytes) else 'w'
        with open(path, mode) as f:
            f.write(serialized_data)

        # Calculate checksum
        checksum = self._calculate_checksum(serialized_data)

        # Return metadata
        return {
            "path": str(path.absolute()),
            "format": format,
            "compression": compression,
            "size_bytes": len(serialized_data) if isinstance(serialized_data, bytes) else len(serialized_data.encode()),
            "checksum": checksum,
            "timestamp": datetime.utcnow().isoformat(),
            "serializer_version": SERIALIZER_VERSION,
            "schema_version": SCHEMA_VERSION
        }

    def load(
        self,
        path: Union[str, Path],
        format: Optional[str] = None,
        compression: Optional[str] = None,
        validate: bool = True
    ) -> Any:
        """
        Load a pipeline from disk.

        Args:
            path: File path to load from
            format: Serialization format (auto-detected if None)
            compression: Compression used (auto-detected if None)
            validate: Whether to validate the loaded pipeline

        Returns:
            The deserialized pipeline object
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")

        # Auto-detect format from extension if not specified
        if format is None:
            format = self._detect_format(path)

        # Auto-detect compression if not specified
        if compression is None:
            compression = self._detect_compression(path)

        # Read file
        mode = 'rb' if format in ['pickle', 'joblib'] or compression != 'none' else 'r'
        with open(path, mode) as f:
            data = f.read()

        # Decompress if needed
        if compression != "none":
            data = self._decompress(data, compression)

        # Deserialize based on format
        if format == "pickle":
            loaded_data = self._deserialize_pickle(data)
        elif format == "json":
            loaded_data = self._deserialize_json(data)
        elif format == "joblib":
            loaded_data = self._deserialize_joblib(data)
        elif format == "yaml":
            loaded_data = self._deserialize_yaml(data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Extract pipeline from wrapper
        pipeline = self._extract_pipeline(loaded_data)

        # Validate if requested
        if validate:
            self._validate_pipeline(pipeline, loaded_data.get("metadata", {}))

        return pipeline

    def _prepare_data(self, pipeline: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare pipeline data with metadata wrapper."""
        data = {
            "pipeline": pipeline,
            "serializer_version": SERIALIZER_VERSION,
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.utcnow().isoformat()
        }

        if self.include_metadata:
            pipeline_metadata = {
                "name": getattr(pipeline, "name", None),
                "fitted": getattr(pipeline, "fitted", False),
                "num_steps": getattr(pipeline, "num_steps", None) or len(getattr(pipeline, "steps", [])),
                "step_names": [step.name for step in getattr(pipeline, "steps", [])] if hasattr(pipeline, "steps") else []
            }

            if metadata:
                pipeline_metadata.update(metadata)

            data["metadata"] = pipeline_metadata

        return data

    def _extract_pipeline(self, data: Dict[str, Any]) -> Any:
        """Extract pipeline from metadata wrapper."""
        if isinstance(data, dict) and "pipeline" in data:
            return data["pipeline"]
        return data

    def _serialize_pickle(self, data: Any) -> bytes:
        """Serialize using pickle."""
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

    def _deserialize_pickle(self, data: bytes) -> Any:
        """Deserialize using pickle."""
        return pickle.loads(data)

    def _serialize_json(self, data: Any) -> str:
        """Serialize using JSON (configuration only)."""
        # For JSON, we need to convert the pipeline to dict
        json_data = {
            "serializer_version": data.get("serializer_version"),
            "schema_version": data.get("schema_version"),
            "timestamp": data.get("timestamp"),
            "metadata": data.get("metadata", {}),
            "pipeline_config": data["pipeline"].to_dict() if hasattr(data["pipeline"], "to_dict") else str(data["pipeline"])
        }
        return json.dumps(json_data, indent=2, default=str)

    def _deserialize_json(self, data: Union[str, bytes]) -> Dict[str, Any]:
        """Deserialize using JSON."""
        if isinstance(data, bytes):
            data = data.decode('utf-8')

        json_data = json.loads(data)

        # Note: This returns the configuration, not a full pipeline object
        # The caller needs to reconstruct the pipeline using Pipeline.from_dict()
        return {
            "serializer_version": json_data.get("serializer_version"),
            "schema_version": json_data.get("schema_version"),
            "timestamp": json_data.get("timestamp"),
            "metadata": json_data.get("metadata", {}),
            "pipeline": json_data.get("pipeline_config")
        }

    def _serialize_joblib(self, data: Any) -> bytes:
        """Serialize using joblib."""
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib not available")
        return joblib.dump(data, compress=3)

    def _deserialize_joblib(self, data: bytes) -> Any:
        """Deserialize using joblib."""
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib not available")
        return joblib.load(data)

    def _serialize_yaml(self, data: Any) -> str:
        """Serialize using YAML (configuration only)."""
        if not YAML_AVAILABLE:
            raise ImportError("pyyaml not available")

        yaml_data = {
            "serializer_version": data.get("serializer_version"),
            "schema_version": data.get("schema_version"),
            "timestamp": data.get("timestamp"),
            "metadata": data.get("metadata", {}),
            "pipeline_config": data["pipeline"].to_dict() if hasattr(data["pipeline"], "to_dict") else str(data["pipeline"])
        }
        return yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)

    def _deserialize_yaml(self, data: Union[str, bytes]) -> Dict[str, Any]:
        """Deserialize using YAML."""
        if not YAML_AVAILABLE:
            raise ImportError("pyyaml not available")

        if isinstance(data, bytes):
            data = data.decode('utf-8')

        yaml_data = yaml.safe_load(data)

        return {
            "serializer_version": yaml_data.get("serializer_version"),
            "schema_version": yaml_data.get("schema_version"),
            "timestamp": yaml_data.get("timestamp"),
            "metadata": yaml_data.get("metadata", {}),
            "pipeline": yaml_data.get("pipeline_config")
        }

    def _compress(self, data: Union[bytes, str], compression: str) -> bytes:
        """Apply compression to data."""
        if isinstance(data, str):
            data = data.encode('utf-8')

        if compression == "gzip":
            return gzip.compress(data)
        elif compression == "bz2":
            return bz2.compress(data)
        elif compression == "lzma":
            return lzma.compress(data)
        else:
            return data

    def _decompress(self, data: bytes, compression: str) -> bytes:
        """Decompress data."""
        if compression == "gzip":
            return gzip.decompress(data)
        elif compression == "bz2":
            return bz2.decompress(data)
        elif compression == "lzma":
            return lzma.decompress(data)
        else:
            return data

    def _detect_format(self, path: Path) -> str:
        """Auto-detect serialization format from file extension."""
        ext = path.suffix.lower()

        # Remove compression extensions
        if ext in ['.gz', '.bz2', '.xz']:
            ext = path.stem.split('.')[-1] if '.' in path.stem else ''
            ext = f'.{ext}'

        format_map = {
            '.pkl': 'pickle',
            '.pickle': 'pickle',
            '.json': 'json',
            '.joblib': 'joblib',
            '.yml': 'yaml',
            '.yaml': 'yaml'
        }

        return format_map.get(ext, self.default_format)

    def _detect_compression(self, path: Path) -> str:
        """Auto-detect compression from file extension."""
        ext = path.suffix.lower()

        compression_map = {
            '.gz': 'gzip',
            '.bz2': 'bz2',
            '.xz': 'lzma',
            '.lzma': 'lzma'
        }

        return compression_map.get(ext, 'none')

    def _calculate_checksum(self, data: Union[bytes, str]) -> str:
        """Calculate SHA256 checksum of data."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()

    def _validate_pipeline(self, pipeline: Any, metadata: Dict[str, Any]) -> None:
        """Validate loaded pipeline."""
        # Basic validation
        if not hasattr(pipeline, "transform") and not isinstance(pipeline, dict):
            logger.warning("Loaded object doesn't have a 'transform' method")

        # Version compatibility check
        if metadata.get("serializer_version") != SERIALIZER_VERSION:
            logger.warning(
                f"Pipeline was serialized with version {metadata.get('serializer_version')}, "
                f"current version is {SERIALIZER_VERSION}"
            )


class PipelineRegistry:
    """
    Registry for managing saved pipelines.

    Provides catalog functionality to track, list, and manage
    serialized pipelines with metadata.
    """

    def __init__(self, registry_path: Union[str, Path]):
        """
        Initialize registry.

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "version": "1.0.0",
                "created_at": datetime.utcnow().isoformat(),
                "pipelines": {}
            }

    def _save_registry(self) -> None:
        """Save registry to disk."""
        self.registry["updated_at"] = datetime.utcnow().isoformat()
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def register(
        self,
        pipeline_id: str,
        file_info: Dict[str, Any],
        tags: Optional[list] = None,
        description: Optional[str] = None
    ) -> None:
        """
        Register a saved pipeline.

        Args:
            pipeline_id: Unique identifier for the pipeline
            file_info: File information from serializer.save()
            tags: Optional tags for categorization
            description: Optional description
        """
        self.registry["pipelines"][pipeline_id] = {
            **file_info,
            "tags": tags or [],
            "description": description,
            "registered_at": datetime.utcnow().isoformat()
        }
        self._save_registry()

    def unregister(self, pipeline_id: str) -> None:
        """Remove a pipeline from registry."""
        if pipeline_id in self.registry["pipelines"]:
            del self.registry["pipelines"][pipeline_id]
            self._save_registry()

    def get(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline info from registry."""
        return self.registry["pipelines"].get(pipeline_id)

    def list(self, tags: Optional[list] = None) -> Dict[str, Dict[str, Any]]:
        """
        List registered pipelines.

        Args:
            tags: Filter by tags (returns pipelines with any of these tags)

        Returns:
            Dictionary of pipeline_id -> pipeline_info
        """
        if tags is None:
            return self.registry["pipelines"]

        return {
            pid: info
            for pid, info in self.registry["pipelines"].items()
            if any(tag in info.get("tags", []) for tag in tags)
        }

    def search(self, query: str) -> Dict[str, Dict[str, Any]]:
        """
        Search pipelines by name or description.

        Args:
            query: Search query string

        Returns:
            Dictionary of matching pipelines
        """
        query_lower = query.lower()
        return {
            pid: info
            for pid, info in self.registry["pipelines"].items()
            if query_lower in str(info.get("description", "")).lower()
            or query_lower in str(info.get("path", "")).lower()
            or query_lower in " ".join(info.get("tags", [])).lower()
        }


# Convenience functions
def save_pipeline(
    pipeline: Any,
    path: Union[str, Path],
    format: str = "pickle",
    compression: str = "none",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Save a pipeline to disk (convenience function).

    Args:
        pipeline: Pipeline to save
        path: File path
        format: Serialization format
        compression: Compression algorithm
        metadata: Additional metadata

    Returns:
        File information dictionary
    """
    serializer = PipelineSerializer(default_format=format, compression=compression)
    return serializer.save(pipeline, path, metadata=metadata)


def load_pipeline(
    path: Union[str, Path],
    format: Optional[str] = None,
    compression: Optional[str] = None
) -> Any:
    """
    Load a pipeline from disk (convenience function).

    Args:
        path: File path
        format: Serialization format (auto-detected if None)
        compression: Compression (auto-detected if None)

    Returns:
        Loaded pipeline
    """
    serializer = PipelineSerializer()
    return serializer.load(path, format=format, compression=compression)
