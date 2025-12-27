"""
Comprehensive tests for Pipeline serialization and deserialization.

Tests cover JSON serialization, pickle persistence, code generation,
versioning, and database integration.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path

from app.ml_engine.preprocessing.pipeline import (
    Pipeline,
    export_pipeline_to_sklearn_code,
    export_pipeline_to_standalone_code,
)
from app.ml_engine.preprocessing.scaler import StandardScaler, MinMaxScaler
from app.ml_engine.preprocessing.imputer import MeanImputer, MedianImputer


class TestPipelineJSONSerialization:
    """Test Pipeline JSON serialization (configuration only)."""

    def test_to_dict_basic(self):
        """Test basic serialization to dictionary."""
        pipeline = Pipeline(
            steps=[StandardScaler(columns=["x"])],
            name="TestPipeline"
        )

        config = pipeline.to_dict()

        assert config["name"] == "TestPipeline"
        assert config["fitted"] is False
        assert len(config["steps"]) == 1
        assert config["steps"][0]["class"] == "StandardScaler"
        assert "_version" in config
        assert "_schema_version" in config

    def test_to_dict_without_version(self):
        """Test serialization without version info."""
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])

        config = pipeline.to_dict(include_version=False)

        assert "_version" not in config
        assert "_schema_version" not in config

    def test_to_dict_multiple_steps(self):
        """Test serialization with multiple steps."""
        pipeline = Pipeline(steps=[
            MeanImputer(columns=["x"]),
            StandardScaler(columns=["x", "y"]),
            MinMaxScaler(columns=["z"])
        ])

        config = pipeline.to_dict()

        assert len(config["steps"]) == 3
        assert config["steps"][0]["class"] == "MeanImputer"
        assert config["steps"][1]["class"] == "StandardScaler"
        assert config["steps"][2]["class"] == "MinMaxScaler"

    def test_to_dict_preserves_metadata(self):
        """Test that metadata is preserved in serialization."""
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])
        df = pd.DataFrame({'x': [1, 2, 3]})
        pipeline.fit(df)

        config = pipeline.to_dict()

        assert "metadata" in config
        assert "fitted_at" in config["metadata"]
        assert "num_samples_fitted" in config["metadata"]

    def test_to_dict_preserves_statistics(self):
        """Test that statistics are preserved in serialization."""
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])
        df = pd.DataFrame({'x': [1, 2, 3]})
        pipeline.fit(df)

        config = pipeline.to_dict()

        assert "statistics" in config
        assert len(config["statistics"]) == 1
        assert "fit_duration_seconds" in config["statistics"][0]

    def test_from_dict_basic(self):
        """Test basic deserialization from dictionary."""
        config = {
            "name": "TestPipeline",
            "fitted": False,
            "steps": [
                {
                    "class": "StandardScaler",
                    "name": "StandardScaler",
                    "fitted": False,
                    "params": {"columns": ["x"], "with_mean": True, "with_std": True}
                }
            ],
            "metadata": {},
            "statistics": []
        }

        pipeline = Pipeline.from_dict(config)

        assert pipeline.name == "TestPipeline"
        assert len(pipeline.steps) == 1
        assert isinstance(pipeline.steps[0], StandardScaler)

    def test_from_dict_multiple_steps(self):
        """Test deserialization with multiple steps."""
        config = {
            "name": "MultiStepPipeline",
            "steps": [
                {
                    "class": "MeanImputer",
                    "name": "imputer",
                    "fitted": False,
                    "params": {"columns": ["x"]}
                },
                {
                    "class": "StandardScaler",
                    "name": "scaler",
                    "fitted": False,
                    "params": {"columns": ["x"], "with_mean": True, "with_std": True}
                }
            ],
            "metadata": {},
            "statistics": []
        }

        pipeline = Pipeline.from_dict(config)

        assert len(pipeline.steps) == 2
        assert pipeline.steps[0].name == "imputer"
        assert pipeline.steps[1].name == "scaler"

    def test_roundtrip_serialization(self):
        """Test that pipeline can be serialized and deserialized without loss."""
        original_pipeline = Pipeline(
            steps=[
                MeanImputer(columns=["x"], name="imputer"),
                StandardScaler(columns=["x", "y"], name="scaler"),
            ],
            name="RoundtripPipeline"
        )

        # Serialize
        config = original_pipeline.to_dict()

        # Deserialize
        restored_pipeline = Pipeline.from_dict(config)

        assert restored_pipeline.name == original_pipeline.name
        assert len(restored_pipeline.steps) == len(original_pipeline.steps)
        assert restored_pipeline.steps[0].name == original_pipeline.steps[0].name
        assert restored_pipeline.steps[1].name == original_pipeline.steps[1].name


class TestPipelinePickleSerialization:
    """Test Pipeline pickle serialization (with fitted parameters)."""

    def test_save_and_load_unfitted(self):
        """Test saving and loading unfitted pipeline."""
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline.pkl"

            pipeline.save(path)
            assert path.exists()

            loaded = Pipeline.load(path)
            assert loaded.name == pipeline.name
            assert len(loaded.steps) == len(pipeline.steps)
            assert not loaded.fitted

    def test_save_and_load_fitted(self):
        """Test saving and loading fitted pipeline."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])
        pipeline.fit(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline.pkl"

            pipeline.save(path)
            loaded = Pipeline.load(path)

            assert loaded.fitted
            assert loaded.steps[0].fitted

            # Verify fitted parameters are preserved
            result = loaded.transform(df)
            assert isinstance(result, pd.DataFrame)
            assert abs(result["x"].mean()) < 0.01  # Standardized

    def test_save_creates_directory(self):
        """Test that save creates parent directories if needed."""
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir1" / "subdir2" / "pipeline.pkl"

            pipeline.save(path)
            assert path.exists()

    def test_load_preserves_all_state(self):
        """Test that loading preserves all pipeline state."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [10, 20, 30]})
        pipeline = Pipeline(
            steps=[
                MeanImputer(columns=["x"]),
                StandardScaler(columns=["x", "y"])
            ],
            name="StatefulPipeline"
        )
        pipeline.fit(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline.pkl"
            pipeline.save(path)
            loaded = Pipeline.load(path)

            # Check all state is preserved
            assert loaded.name == "StatefulPipeline"
            assert loaded.fitted
            assert len(loaded.step_statistics) == 2
            assert "fitted_at" in loaded.metadata


class TestPipelineJSONFileSerialization:
    """Test Pipeline JSON file serialization (configuration files)."""

    def test_save_config_basic(self):
        """Test saving configuration to JSON file."""
        pipeline = Pipeline(
            steps=[StandardScaler(columns=["x"])],
            name="ConfigPipeline"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"

            pipeline.save_config(path)
            assert path.exists()

            # Verify JSON format
            with open(path, 'r') as f:
                config = json.load(f)
                assert config["name"] == "ConfigPipeline"

    def test_load_config_basic(self):
        """Test loading configuration from JSON file."""
        pipeline = Pipeline(
            steps=[StandardScaler(columns=["x"])],
            name="ConfigPipeline"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            pipeline.save_config(path)

            loaded = Pipeline.load_config(path)
            assert loaded.name == "ConfigPipeline"
            assert len(loaded.steps) == 1
            assert not loaded.fitted  # Config doesn't save fitted state

    def test_config_roundtrip(self):
        """Test config save/load roundtrip."""
        original = Pipeline(
            steps=[
                MeanImputer(columns=["x"]),
                StandardScaler(columns=["x", "y"])
            ],
            name="RoundtripConfig"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"

            original.save_config(path)
            loaded = Pipeline.load_config(path)

            assert loaded.name == original.name
            assert len(loaded.steps) == len(original.steps)


class TestPipelineCodeExport:
    """Test Pipeline code generation/export."""

    def test_export_to_sklearn_basic(self):
        """Test basic sklearn code export."""
        pipeline = Pipeline(
            steps=[StandardScaler(columns=["x"])],
            name="SklearnPipeline"
        )

        code = export_pipeline_to_sklearn_code(pipeline)

        assert "import pandas as pd" in code
        assert "from sklearn.pipeline import Pipeline" in code
        assert "StandardScaler" in code
        assert "pipeline = Pipeline([" in code

    def test_export_to_sklearn_without_imports(self):
        """Test sklearn export without imports."""
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])

        code = export_pipeline_to_sklearn_code(
            pipeline,
            include_imports=False
        )

        assert "import" not in code
        assert "pipeline = Pipeline([" in code

    def test_export_to_sklearn_without_comments(self):
        """Test sklearn export without comments."""
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])

        code = export_pipeline_to_sklearn_code(
            pipeline,
            include_comments=False
        )

        assert "#" not in code or code.count("#") == 0

    def test_export_to_sklearn_multiple_steps(self):
        """Test sklearn export with multiple steps."""
        pipeline = Pipeline(steps=[
            MeanImputer(columns=["x"]),
            StandardScaler(columns=["x", "y"]),
        ])

        code = export_pipeline_to_sklearn_code(pipeline)

        assert "SimpleImputer" in code  # MeanImputer mapped to SimpleImputer
        assert "StandardScaler" in code
        assert code.count("('") >= 2  # At least 2 steps

    def test_export_to_standalone_basic(self):
        """Test basic standalone code export."""
        pipeline = Pipeline(
            steps=[StandardScaler(columns=["x"])],
            name="StandalonePipeline"
        )

        code = export_pipeline_to_standalone_code(pipeline)

        assert "from app.ml_engine.preprocessing.pipeline import Pipeline" in code
        assert "from app.ml_engine.preprocessing.scaler import StandardScaler" in code
        assert "step_0 = StandardScaler" in code
        assert "pipeline = Pipeline(" in code

    def test_export_to_standalone_multiple_steps(self):
        """Test standalone export with multiple steps."""
        pipeline = Pipeline(steps=[
            MeanImputer(columns=["x"]),
            StandardScaler(columns=["x", "y"]),
            MinMaxScaler(columns=["z"])
        ])

        code = export_pipeline_to_standalone_code(pipeline)

        assert "step_0 = MeanImputer" in code
        assert "step_1 = StandardScaler" in code
        assert "step_2 = MinMaxScaler" in code
        assert "Pipeline(steps=[step_0, step_1, step_2]" in code

    def test_exported_sklearn_code_is_valid_python(self):
        """Test that exported sklearn code is syntactically valid."""
        pipeline = Pipeline(steps=[
            StandardScaler(columns=["x", "y"])
        ])

        code = export_pipeline_to_sklearn_code(pipeline)

        # Try to compile the code
        try:
            compile(code, '<string>', 'exec')
            valid = True
        except SyntaxError:
            valid = False

        assert valid, "Exported code should be valid Python"

    def test_exported_standalone_code_is_valid_python(self):
        """Test that exported standalone code is syntactically valid."""
        pipeline = Pipeline(steps=[
            StandardScaler(columns=["x", "y"])
        ])

        code = export_pipeline_to_standalone_code(pipeline)

        # Try to compile the code
        try:
            compile(code, '<string>', 'exec')
            valid = True
        except SyntaxError:
            valid = False

        assert valid, "Exported code should be valid Python"


class TestPipelineVersioning:
    """Test Pipeline versioning support."""

    def test_to_dict_includes_version(self):
        """Test that to_dict includes version information."""
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])

        config = pipeline.to_dict(include_version=True)

        assert "_version" in config
        assert "_schema_version" in config
        assert isinstance(config["_version"], str)
        assert isinstance(config["_schema_version"], int)

    def test_version_format(self):
        """Test that version follows semantic versioning."""
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])

        config = pipeline.to_dict()

        version = config["_version"]
        parts = version.split(".")
        assert len(parts) == 3  # major.minor.patch
        assert all(part.isdigit() for part in parts)

    def test_from_dict_handles_old_format_gracefully(self):
        """Test that from_dict handles configs without version info."""
        # Old format without version
        config = {
            "name": "OldFormatPipeline",
            "fitted": False,
            "steps": [
                {
                    "class": "StandardScaler",
                    "name": "StandardScaler",
                    "fitted": False,
                    "params": {"columns": ["x"], "with_mean": True, "with_std": True}
                }
            ],
            "metadata": {},
            "statistics": []
        }

        # Should load without errors
        pipeline = Pipeline.from_dict(config)

        assert pipeline.name == "OldFormatPipeline"
        assert len(pipeline.steps) == 1


class TestPipelineSerializationEdgeCases:
    """Test edge cases in Pipeline serialization."""

    def test_serialize_empty_pipeline(self):
        """Test serializing empty pipeline."""
        pipeline = Pipeline()

        config = pipeline.to_dict()

        assert config["name"] == "PreprocessingPipeline"
        assert len(config["steps"]) == 0

    def test_deserialize_empty_pipeline(self):
        """Test deserializing empty pipeline."""
        config = {
            "name": "EmptyPipeline",
            "steps": [],
            "metadata": {},
            "statistics": []
        }

        pipeline = Pipeline.from_dict(config)

        assert len(pipeline.steps) == 0

    def test_serialize_with_special_characters_in_name(self):
        """Test serializing pipeline with special characters in name."""
        pipeline = Pipeline(
            steps=[StandardScaler(columns=["x"])],
            name="Pipeline (v2.0) - Test!"
        )

        config = pipeline.to_dict()
        restored = Pipeline.from_dict(config)

        assert restored.name == "Pipeline (v2.0) - Test!"

    def test_serialize_with_none_parameters(self):
        """Test serializing steps with None parameters."""
        pipeline = Pipeline(
            steps=[StandardScaler(columns=None)]  # Auto-detect columns
        )

        config = pipeline.to_dict()
        restored = Pipeline.from_dict(config)

        assert restored.steps[0].params["columns"] is None

    def test_save_load_with_unicode_name(self):
        """Test save/load with unicode characters in name."""
        pipeline = Pipeline(
            steps=[StandardScaler(columns=["x"])],
            name="パイプライン_测试"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline.pkl"
            pipeline.save(path)
            loaded = Pipeline.load(path)

            assert loaded.name == "パイプライン_测试"


class TestPipelineSerializationIntegration:
    """Integration tests for Pipeline serialization."""

    def test_full_workflow_serialize_fit_deserialize_transform(self):
        """Test complete workflow: serialize, fit, save, load, transform."""
        # Create pipeline
        pipeline = Pipeline(
            steps=[
                MeanImputer(columns=["x"]),
                StandardScaler(columns=["x", "y"])
            ],
            name="WorkflowPipeline"
        )

        # Save config
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            pipeline.save_config(config_path)

            # Load config and fit
            loaded_pipeline = Pipeline.load_config(config_path)
            df_train = pd.DataFrame({
                'x': [1, 2, np.nan, 4, 5],
                'y': [10, 20, 30, 40, 50]
            })
            loaded_pipeline.fit(df_train)

            # Save fitted pipeline
            fitted_path = Path(tmpdir) / "fitted.pkl"
            loaded_pipeline.save(fitted_path)

            # Load fitted and transform
            final_pipeline = Pipeline.load(fitted_path)
            df_test = pd.DataFrame({'x': [6, 7], 'y': [60, 70]})
            result = final_pipeline.transform(df_test)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2

    def test_multiple_save_load_cycles(self):
        """Test that pipeline can be saved and loaded multiple times."""
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                path = Path(tmpdir) / f"pipeline_{i}.pkl"
                pipeline.save(path)
                pipeline = Pipeline.load(path)

            # Should still work after multiple cycles
            assert len(pipeline.steps) == 1
