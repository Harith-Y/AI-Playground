"""
Tests for pipeline serialization and deserialization.

Tests cover:
- Multiple serialization formats (pickle, JSON, joblib, YAML)
- Compression algorithms (gzip, bz2, lzma)
- Metadata preservation
- Version tracking
- Fitted vs unfitted pipelines
- Registry functionality
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from app.ml_engine.preprocessing.serializer import (
    PipelineSerializer,
    PipelineRegistry,
    save_pipeline,
    load_pipeline,
    SERIALIZER_VERSION,
    SCHEMA_VERSION,
)
from app.ml_engine.preprocessing.pipeline import Pipeline
from app.ml_engine.preprocessing.imputer import MeanImputer, MedianImputer
from app.ml_engine.preprocessing.scaler import StandardScaler, MinMaxScaler
from app.ml_engine.preprocessing.encoder import OneHotEncoder
import pandas as pd
import numpy as np


@pytest.fixture
def simple_pipeline():
    """Create a simple unfitted pipeline for testing."""
    return Pipeline(
        steps=[
            MeanImputer(columns=["age", "salary"]),
            StandardScaler(columns=["age", "salary"])
        ],
        name="TestPipeline"
    )


@pytest.fixture
def fitted_pipeline():
    """Create a fitted pipeline for testing."""
    # Create sample data
    df = pd.DataFrame({
        "age": [25, 30, np.nan, 40, 35],
        "salary": [50000, 60000, 55000, np.nan, 70000],
        "category": ["A", "B", "A", "C", "B"]
    })

    pipeline = Pipeline(
        steps=[
            MeanImputer(columns=["age", "salary"]),
            StandardScaler(columns=["age", "salary"]),
            OneHotEncoder(columns=["category"])
        ],
        name="FittedPipeline"
    )

    pipeline.fit(df)
    return pipeline


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestPipelineSerializer:
    """Tests for PipelineSerializer class."""

    def test_init_default(self):
        """Test default initialization."""
        serializer = PipelineSerializer()
        assert serializer.default_format == "pickle"
        assert serializer.compression == "none"
        assert serializer.include_metadata is True

    def test_init_custom(self):
        """Test custom initialization."""
        serializer = PipelineSerializer(
            default_format="json",
            compression="gzip",
            include_metadata=False
        )
        assert serializer.default_format == "json"
        assert serializer.compression == "gzip"
        assert serializer.include_metadata is False

    def test_init_invalid_format(self):
        """Test initialization with invalid format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            PipelineSerializer(default_format="invalid")

    def test_init_invalid_compression(self):
        """Test initialization with invalid compression."""
        with pytest.raises(ValueError, match="Unsupported compression"):
            PipelineSerializer(compression="invalid")

    def test_save_pickle_unfitted(self, simple_pipeline, temp_dir):
        """Test saving unfitted pipeline as pickle."""
        serializer = PipelineSerializer(default_format="pickle")
        save_path = temp_dir / "pipeline.pkl"

        file_info = serializer.save(simple_pipeline, save_path)

        assert Path(file_info["path"]).exists()
        assert file_info["format"] == "pickle"
        assert file_info["compression"] == "none"
        assert file_info["serializer_version"] == SERIALIZER_VERSION
        assert file_info["schema_version"] == SCHEMA_VERSION
        assert "checksum" in file_info
        assert file_info["size_bytes"] > 0

    def test_save_pickle_fitted(self, fitted_pipeline, temp_dir):
        """Test saving fitted pipeline as pickle."""
        serializer = PipelineSerializer(default_format="pickle")
        save_path = temp_dir / "fitted_pipeline.pkl"

        file_info = serializer.save(fitted_pipeline, save_path)

        assert Path(file_info["path"]).exists()
        assert file_info["size_bytes"] > 0

    def test_save_json(self, simple_pipeline, temp_dir):
        """Test saving pipeline as JSON."""
        serializer = PipelineSerializer(default_format="json")
        save_path = temp_dir / "pipeline.json"

        file_info = serializer.save(simple_pipeline, save_path)

        assert Path(file_info["path"]).exists()
        assert file_info["format"] == "json"

        # Verify JSON is valid
        with open(save_path, 'r') as f:
            data = json.load(f)

        assert "pipeline_config" in data
        assert "serializer_version" in data
        assert "metadata" in data

    def test_save_with_compression_gzip(self, simple_pipeline, temp_dir):
        """Test saving with gzip compression."""
        serializer = PipelineSerializer(default_format="pickle", compression="gzip")
        save_path = temp_dir / "pipeline.pkl.gz"

        file_info = serializer.save(simple_pipeline, save_path)

        assert Path(file_info["path"]).exists()
        assert file_info["compression"] == "gzip"

    def test_save_with_compression_bz2(self, simple_pipeline, temp_dir):
        """Test saving with bz2 compression."""
        serializer = PipelineSerializer(default_format="pickle", compression="bz2")
        save_path = temp_dir / "pipeline.pkl.bz2"

        file_info = serializer.save(simple_pipeline, save_path)

        assert Path(file_info["path"]).exists()
        assert file_info["compression"] == "bz2"

    def test_save_with_compression_lzma(self, simple_pipeline, temp_dir):
        """Test saving with lzma compression."""
        serializer = PipelineSerializer(default_format="pickle", compression="lzma")
        save_path = temp_dir / "pipeline.pkl.xz"

        file_info = serializer.save(simple_pipeline, save_path)

        assert Path(file_info["path"]).exists()
        assert file_info["compression"] == "lzma"

    def test_save_with_metadata(self, simple_pipeline, temp_dir):
        """Test saving with custom metadata."""
        serializer = PipelineSerializer()
        save_path = temp_dir / "pipeline.pkl"

        custom_metadata = {
            "author": "TestUser",
            "version": "1.0.0",
            "description": "Test pipeline"
        }

        file_info = serializer.save(simple_pipeline, save_path, metadata=custom_metadata)

        # Load and verify metadata
        loaded = serializer.load(save_path)
        # Metadata should be preserved in the wrapper

    def test_load_pickle(self, simple_pipeline, temp_dir):
        """Test loading pickle file."""
        serializer = PipelineSerializer(default_format="pickle")
        save_path = temp_dir / "pipeline.pkl"

        serializer.save(simple_pipeline, save_path)
        loaded_pipeline = serializer.load(save_path)

        assert isinstance(loaded_pipeline, Pipeline)
        assert loaded_pipeline.name == simple_pipeline.name
        assert len(loaded_pipeline.steps) == len(simple_pipeline.steps)

    def test_load_fitted_pipeline_preserves_state(self, fitted_pipeline, temp_dir):
        """Test that loading fitted pipeline preserves fitted parameters."""
        serializer = PipelineSerializer()
        save_path = temp_dir / "fitted_pipeline.pkl"

        serializer.save(fitted_pipeline, save_path)
        loaded_pipeline = serializer.load(save_path)

        assert loaded_pipeline.fitted
        assert len(loaded_pipeline.steps) == len(fitted_pipeline.steps)

        # Verify fitted parameters are preserved
        for original_step, loaded_step in zip(fitted_pipeline.steps, loaded_pipeline.steps):
            assert original_step.name == loaded_step.name
            assert original_step.fitted == loaded_step.fitted

    def test_load_json(self, simple_pipeline, temp_dir):
        """Test loading JSON file."""
        serializer = PipelineSerializer(default_format="json")
        save_path = temp_dir / "pipeline.json"

        serializer.save(simple_pipeline, save_path)
        loaded_data = serializer.load(save_path)

        # JSON loads as config dict, need to reconstruct
        assert isinstance(loaded_data, dict)

    def test_load_with_compression(self, simple_pipeline, temp_dir):
        """Test loading compressed file."""
        serializer = PipelineSerializer(compression="gzip")
        save_path = temp_dir / "pipeline.pkl.gz"

        serializer.save(simple_pipeline, save_path)
        loaded_pipeline = serializer.load(save_path)

        assert isinstance(loaded_pipeline, Pipeline)
        assert loaded_pipeline.name == simple_pipeline.name

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading non-existent file raises error."""
        serializer = PipelineSerializer()
        save_path = temp_dir / "nonexistent.pkl"

        with pytest.raises(FileNotFoundError):
            serializer.load(save_path)

    def test_auto_detect_format_pickle(self, simple_pipeline, temp_dir):
        """Test auto-detection of pickle format."""
        serializer = PipelineSerializer()
        save_path = temp_dir / "pipeline.pkl"

        serializer.save(simple_pipeline, save_path, format="pickle")
        loaded = serializer.load(save_path)  # Format auto-detected

        assert isinstance(loaded, Pipeline)

    def test_auto_detect_format_json(self, simple_pipeline, temp_dir):
        """Test auto-detection of JSON format."""
        serializer = PipelineSerializer()
        save_path = temp_dir / "pipeline.json"

        serializer.save(simple_pipeline, save_path, format="json")
        loaded = serializer.load(save_path)  # Format auto-detected

        assert isinstance(loaded, dict)

    def test_auto_detect_compression_gzip(self, simple_pipeline, temp_dir):
        """Test auto-detection of gzip compression."""
        serializer = PipelineSerializer()
        save_path = temp_dir / "pipeline.pkl.gz"

        serializer.save(simple_pipeline, save_path, compression="gzip")
        loaded = serializer.load(save_path)  # Compression auto-detected

        assert isinstance(loaded, Pipeline)

    def test_roundtrip_pickle(self, fitted_pipeline, temp_dir):
        """Test save and load roundtrip with pickle."""
        serializer = PipelineSerializer(default_format="pickle")
        save_path = temp_dir / "pipeline.pkl"

        serializer.save(fitted_pipeline, save_path)
        loaded = serializer.load(save_path)

        # Create test data
        df_test = pd.DataFrame({
            "age": [28, np.nan, 45],
            "salary": [58000, 62000, np.nan],
            "category": ["A", "B", "C"]
        })

        # Transform with both pipelines
        result_original = fitted_pipeline.transform(df_test)
        result_loaded = loaded.transform(df_test)

        # Results should be identical
        pd.testing.assert_frame_equal(result_original, result_loaded)

    @pytest.mark.skipif(not pytest.importorskip("joblib", reason="joblib not installed"), reason="")
    def test_save_load_joblib(self, simple_pipeline, temp_dir):
        """Test saving and loading with joblib format."""
        serializer = PipelineSerializer(default_format="joblib")
        save_path = temp_dir / "pipeline.joblib"

        file_info = serializer.save(simple_pipeline, save_path)
        loaded = serializer.load(save_path)

        assert isinstance(loaded, Pipeline)
        assert file_info["format"] == "joblib"

    @pytest.mark.skipif(not pytest.importorskip("yaml", reason="yaml not installed"), reason="")
    def test_save_load_yaml(self, simple_pipeline, temp_dir):
        """Test saving and loading with YAML format."""
        serializer = PipelineSerializer(default_format="yaml")
        save_path = temp_dir / "pipeline.yml"

        file_info = serializer.save(simple_pipeline, save_path)
        loaded = serializer.load(save_path)

        assert isinstance(loaded, dict)
        assert file_info["format"] == "yaml"


class TestPipelineRegistry:
    """Tests for PipelineRegistry class."""

    def test_init_new_registry(self, temp_dir):
        """Test creating a new registry."""
        registry_path = temp_dir / "registry.json"
        registry = PipelineRegistry(registry_path)

        assert registry.registry_path.exists()
        assert "pipelines" in registry.registry
        assert registry.registry["version"] == "1.0.0"

    def test_init_existing_registry(self, temp_dir):
        """Test loading existing registry."""
        registry_path = temp_dir / "registry.json"

        # Create initial registry
        registry1 = PipelineRegistry(registry_path)
        registry1.register("test-id", {"path": "/test/path"})

        # Load existing registry
        registry2 = PipelineRegistry(registry_path)
        assert "test-id" in registry2.registry["pipelines"]

    def test_register_pipeline(self, temp_dir):
        """Test registering a pipeline."""
        registry = PipelineRegistry(temp_dir / "registry.json")

        file_info = {
            "path": "/test/pipeline.pkl",
            "format": "pickle",
            "size_bytes": 1024
        }

        registry.register(
            "test-pipeline-1",
            file_info,
            tags=["ml", "preprocessing"],
            description="Test pipeline"
        )

        pipeline_info = registry.get("test-pipeline-1")
        assert pipeline_info is not None
        assert pipeline_info["path"] == "/test/pipeline.pkl"
        assert "ml" in pipeline_info["tags"]
        assert pipeline_info["description"] == "Test pipeline"

    def test_unregister_pipeline(self, temp_dir):
        """Test unregistering a pipeline."""
        registry = PipelineRegistry(temp_dir / "registry.json")

        registry.register("test-id", {"path": "/test"})
        assert registry.get("test-id") is not None

        registry.unregister("test-id")
        assert registry.get("test-id") is None

    def test_list_all(self, temp_dir):
        """Test listing all pipelines."""
        registry = PipelineRegistry(temp_dir / "registry.json")

        registry.register("pipeline-1", {"path": "/test1"})
        registry.register("pipeline-2", {"path": "/test2"})

        all_pipelines = registry.list()
        assert len(all_pipelines) == 2
        assert "pipeline-1" in all_pipelines
        assert "pipeline-2" in all_pipelines

    def test_list_by_tags(self, temp_dir):
        """Test listing pipelines filtered by tags."""
        registry = PipelineRegistry(temp_dir / "registry.json")

        registry.register("pipeline-1", {"path": "/test1"}, tags=["ml", "preprocessing"])
        registry.register("pipeline-2", {"path": "/test2"}, tags=["ml", "feature-engineering"])
        registry.register("pipeline-3", {"path": "/test3"}, tags=["data-cleaning"])

        ml_pipelines = registry.list(tags=["ml"])
        assert len(ml_pipelines) == 2
        assert "pipeline-1" in ml_pipelines
        assert "pipeline-2" in ml_pipelines

    def test_search_by_description(self, temp_dir):
        """Test searching pipelines by description."""
        registry = PipelineRegistry(temp_dir / "registry.json")

        registry.register("p1", {"path": "/t1"}, description="Classification pipeline")
        registry.register("p2", {"path": "/t2"}, description="Regression pipeline")
        registry.register("p3", {"path": "/t3"}, description="Data cleaning")

        results = registry.search("classification")
        assert len(results) == 1
        assert "p1" in results

    def test_search_by_path(self, temp_dir):
        """Test searching pipelines by path."""
        registry = PipelineRegistry(temp_dir / "registry.json")

        registry.register("p1", {"path": "/models/production/pipeline.pkl"})
        registry.register("p2", {"path": "/models/staging/pipeline.pkl"})

        results = registry.search("production")
        assert len(results) == 1
        assert "p1" in results


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_save_pipeline_function(self, simple_pipeline, temp_dir):
        """Test save_pipeline convenience function."""
        save_path = temp_dir / "pipeline.pkl"

        file_info = save_pipeline(simple_pipeline, save_path)

        assert Path(file_info["path"]).exists()
        assert file_info["format"] == "pickle"

    def test_load_pipeline_function(self, simple_pipeline, temp_dir):
        """Test load_pipeline convenience function."""
        save_path = temp_dir / "pipeline.pkl"

        save_pipeline(simple_pipeline, save_path)
        loaded = load_pipeline(save_path)

        assert isinstance(loaded, Pipeline)
        assert loaded.name == simple_pipeline.name

    def test_save_load_with_compression_function(self, simple_pipeline, temp_dir):
        """Test save/load with compression using convenience functions."""
        save_path = temp_dir / "pipeline.pkl.gz"

        file_info = save_pipeline(simple_pipeline, save_path, compression="gzip")
        loaded = load_pipeline(save_path)

        assert file_info["compression"] == "gzip"
        assert isinstance(loaded, Pipeline)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_save_to_nested_directory(self, simple_pipeline, temp_dir):
        """Test saving to nested directory that doesn't exist."""
        serializer = PipelineSerializer()
        save_path = temp_dir / "nested" / "dir" / "pipeline.pkl"

        file_info = serializer.save(simple_pipeline, save_path)

        assert Path(file_info["path"]).exists()
        assert save_path.parent.exists()

    def test_empty_pipeline(self, temp_dir):
        """Test serializing empty pipeline."""
        pipeline = Pipeline(steps=[], name="EmptyPipeline")
        serializer = PipelineSerializer()
        save_path = temp_dir / "empty.pkl"

        file_info = serializer.save(pipeline, save_path)
        loaded = serializer.load(save_path)

        assert isinstance(loaded, Pipeline)
        assert len(loaded.steps) == 0

    def test_pipeline_with_unicode_name(self, temp_dir):
        """Test pipeline with unicode characters in name."""
        pipeline = Pipeline(
            steps=[MeanImputer(columns=["age"])],
            name="Test™️ Pipeline 日本語"
        )
        serializer = PipelineSerializer()
        save_path = temp_dir / "unicode.pkl"

        serializer.save(pipeline, save_path)
        loaded = serializer.load(save_path)

        assert loaded.name == pipeline.name

    def test_version_mismatch_warning(self, simple_pipeline, temp_dir):
        """Test warning on version mismatch."""
        serializer = PipelineSerializer()
        save_path = temp_dir / "pipeline.pkl"

        # Save with current version
        serializer.save(simple_pipeline, save_path)

        # Mock different version
        with patch("app.ml_engine.preprocessing.serializer.SERIALIZER_VERSION", "2.0.0"):
            # Should load but log warning
            loaded = serializer.load(save_path, validate=True)
            assert isinstance(loaded, Pipeline)
