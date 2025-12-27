"""
Comprehensive tests for the Pipeline class.

Tests cover initialization, fit/transform, step management, serialization,
error handling, and statistics tracking.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path

from app.ml_engine.preprocessing.pipeline import Pipeline, create_pipeline, load_pipeline
from app.ml_engine.preprocessing.scaler import StandardScaler, MinMaxScaler
from app.ml_engine.preprocessing.imputer import MeanImputer, MedianImputer
from app.ml_engine.preprocessing.encoder import OneHotEncoder


class TestPipelineInitialization:
    """Test Pipeline initialization."""

    def test_empty_initialization(self):
        """Test initialization with no steps."""
        pipeline = Pipeline()
        assert len(pipeline.steps) == 0
        assert pipeline.name == "PreprocessingPipeline"
        assert not pipeline.fitted
        assert pipeline.metadata["num_steps"] == 0

    def test_initialization_with_steps(self):
        """Test initialization with preprocessing steps."""
        steps = [
            StandardScaler(columns=["x"]),
            MinMaxScaler(columns=["y"])
        ]
        pipeline = Pipeline(steps=steps)
        assert len(pipeline.steps) == 2
        assert not pipeline.fitted

    def test_initialization_with_name(self):
        """Test initialization with custom name."""
        pipeline = Pipeline(name="MyPipeline")
        assert pipeline.name == "MyPipeline"

    def test_metadata_created_at(self):
        """Test that metadata includes created_at timestamp."""
        pipeline = Pipeline()
        assert "created_at" in pipeline.metadata
        assert pipeline.metadata["fitted_at"] is None


class TestPipelineFit:
    """Test Pipeline fit method."""

    def test_fit_single_step(self):
        """Test fitting pipeline with single step."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        scaler = StandardScaler(columns=["x"])
        pipeline = Pipeline(steps=[scaler])

        pipeline.fit(df)

        assert pipeline.fitted
        assert scaler.fitted
        assert len(pipeline.step_statistics) == 1

    def test_fit_multiple_steps(self):
        """Test fitting pipeline with multiple steps."""
        df = pd.DataFrame({
            'x': [1, 2, np.nan, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })

        pipeline = Pipeline(steps=[
            MeanImputer(columns=["x"]),
            StandardScaler(columns=["x", "y"])
        ])

        pipeline.fit(df)

        assert pipeline.fitted
        assert all(step.fitted for step in pipeline.steps)
        assert len(pipeline.step_statistics) == 2

    def test_fit_with_labels(self):
        """Test fitting pipeline with labels (y parameter)."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        y = pd.Series([0, 1, 0, 1, 0])

        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])
        pipeline.fit(df, y)

        assert pipeline.fitted

    def test_fit_empty_pipeline_error(self):
        """Test error when fitting empty pipeline."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        pipeline = Pipeline()

        with pytest.raises(ValueError, match="Pipeline has no steps"):
            pipeline.fit(df)

    def test_fit_step_statistics(self):
        """Test that statistics are collected during fit."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])
        pipeline.fit(df)

        stats = pipeline.step_statistics[0]
        assert "step_index" in stats
        assert "step_name" in stats
        assert "step_class" in stats
        assert "fit_duration_seconds" in stats
        assert "output_shape" in stats
        assert stats["fitted"] is True

    def test_fit_metadata_updated(self):
        """Test that metadata is updated after fit."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])
        pipeline.fit(df)

        assert pipeline.metadata["fitted_at"] is not None
        assert pipeline.metadata["num_samples_fitted"] == 5


class TestPipelineTransform:
    """Test Pipeline transform method."""

    def test_transform_single_step(self):
        """Test transforming data through single step."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])
        pipeline.fit(df)

        result = pipeline.transform(df)

        assert isinstance(result, pd.DataFrame)
        assert "x" in result.columns
        # Check that data is standardized (mean ~0, std ~1)
        assert abs(result["x"].mean()) < 0.01
        assert abs(result["x"].std() - 1.0) < 0.01

    def test_transform_multiple_steps(self):
        """Test transforming through multiple steps."""
        df = pd.DataFrame({
            'x': [1, 2, np.nan, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })

        pipeline = Pipeline(steps=[
            MeanImputer(columns=["x"]),
            StandardScaler(columns=["x", "y"])
        ])

        pipeline.fit(df)
        result = pipeline.transform(df)

        assert isinstance(result, pd.DataFrame)
        # Check no missing values after imputation
        assert result["x"].isnull().sum() == 0

    def test_transform_not_fitted_error(self):
        """Test error when transforming before fitting."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])

        with pytest.raises(RuntimeError, match="must be fitted before transform"):
            pipeline.transform(df)

    def test_transform_different_data(self):
        """Test transforming different data after fitting."""
        df_train = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        df_test = pd.DataFrame({'x': [6, 7, 8]})

        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])
        pipeline.fit(df_train)
        result = pipeline.transform(df_test)

        assert len(result) == 3
        assert "x" in result.columns


class TestPipelineFitTransform:
    """Test Pipeline fit_transform method."""

    def test_fit_transform(self):
        """Test fit_transform convenience method."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])

        result = pipeline.fit_transform(df)

        assert pipeline.fitted
        assert isinstance(result, pd.DataFrame)
        assert abs(result["x"].mean()) < 0.01

    def test_fit_transform_with_labels(self):
        """Test fit_transform with labels."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        y = pd.Series([0, 1, 0, 1, 0])

        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])
        result = pipeline.fit_transform(df, y)

        assert pipeline.fitted
        assert isinstance(result, pd.DataFrame)


class TestPipelineStepManagement:
    """Test Pipeline step management methods."""

    def test_add_step_to_end(self):
        """Test adding step to end of pipeline."""
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])
        new_step = MinMaxScaler(columns=["y"])

        pipeline.add_step(new_step)

        assert len(pipeline.steps) == 2
        assert pipeline.steps[1] == new_step
        assert not pipeline.fitted  # Should reset fitted state

    def test_add_step_at_position(self):
        """Test adding step at specific position."""
        pipeline = Pipeline(steps=[
            StandardScaler(columns=["x"]),
            MinMaxScaler(columns=["y"])
        ])
        new_step = MeanImputer(columns=["x"])

        pipeline.add_step(new_step, position=0)

        assert len(pipeline.steps) == 3
        assert pipeline.steps[0] == new_step

    def test_add_step_invalid_type_error(self):
        """Test error when adding invalid step type."""
        pipeline = Pipeline()

        with pytest.raises(TypeError, match="must be a PreprocessingStep instance"):
            pipeline.add_step("not a step")

    def test_add_step_invalid_position_error(self):
        """Test error when adding step at invalid position."""
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])

        with pytest.raises(ValueError, match="Invalid position"):
            pipeline.add_step(MinMaxScaler(columns=["y"]), position=10)

    def test_remove_step_by_index(self):
        """Test removing step by index."""
        step1 = StandardScaler(columns=["x"])
        step2 = MinMaxScaler(columns=["y"])
        pipeline = Pipeline(steps=[step1, step2])

        removed = pipeline.remove_step(0)

        assert removed == step1
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0] == step2

    def test_remove_step_invalid_index_error(self):
        """Test error when removing step at invalid index."""
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])

        with pytest.raises(IndexError, match="out of range"):
            pipeline.remove_step(10)

    def test_remove_step_by_name(self):
        """Test removing step by name."""
        step1 = StandardScaler(columns=["x"], name="scaler1")
        step2 = MinMaxScaler(columns=["y"], name="scaler2")
        pipeline = Pipeline(steps=[step1, step2])

        removed = pipeline.remove_step_by_name("scaler1")

        assert removed == step1
        assert len(pipeline.steps) == 1

    def test_remove_step_by_name_not_found_error(self):
        """Test error when removing non-existent step by name."""
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])

        with pytest.raises(ValueError, match="No step with name"):
            pipeline.remove_step_by_name("nonexistent")

    def test_get_step_by_index(self):
        """Test getting step by index."""
        step = StandardScaler(columns=["x"])
        pipeline = Pipeline(steps=[step])

        retrieved = pipeline.get_step(0)

        assert retrieved == step

    def test_get_step_by_name(self):
        """Test getting step by name."""
        step = StandardScaler(columns=["x"], name="my_scaler")
        pipeline = Pipeline(steps=[step])

        retrieved = pipeline.get_step_by_name("my_scaler")

        assert retrieved == step

    def test_get_step_by_name_not_found(self):
        """Test getting non-existent step by name returns None."""
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])

        result = pipeline.get_step_by_name("nonexistent")

        assert result is None

    def test_reorder_steps(self):
        """Test reordering pipeline steps."""
        step1 = StandardScaler(columns=["x"], name="step1")
        step2 = MinMaxScaler(columns=["y"], name="step2")
        step3 = MeanImputer(columns=["z"], name="step3")

        pipeline = Pipeline(steps=[step1, step2, step3])
        pipeline.reorder_steps([2, 0, 1])  # Move step3 to front

        assert pipeline.steps[0] == step3
        assert pipeline.steps[1] == step1
        assert pipeline.steps[2] == step2
        assert not pipeline.fitted  # Should reset fitted state

    def test_reorder_steps_invalid_length_error(self):
        """Test error when reorder list has wrong length."""
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])

        with pytest.raises(ValueError, match="must have 1 indices"):
            pipeline.reorder_steps([0, 1])

    def test_reorder_steps_invalid_indices_error(self):
        """Test error when reorder indices are invalid."""
        pipeline = Pipeline(steps=[
            StandardScaler(columns=["x"]),
            MinMaxScaler(columns=["y"])
        ])

        with pytest.raises(ValueError, match="must contain each index"):
            pipeline.reorder_steps([0, 0])  # Duplicate index

    def test_clear_steps(self):
        """Test clearing all steps from pipeline."""
        pipeline = Pipeline(steps=[
            StandardScaler(columns=["x"]),
            MinMaxScaler(columns=["y"])
        ])
        df = pd.DataFrame({'x': [1, 2, 3]})
        pipeline.fit(df)

        pipeline.clear()

        assert len(pipeline.steps) == 0
        assert not pipeline.fitted
        assert len(pipeline.step_statistics) == 0


class TestPipelineInformation:
    """Test Pipeline information and statistics methods."""

    def test_get_num_steps(self):
        """Test getting number of steps."""
        pipeline = Pipeline(steps=[
            StandardScaler(columns=["x"]),
            MinMaxScaler(columns=["y"])
        ])

        assert pipeline.get_num_steps() == 2

    def test_get_step_names(self):
        """Test getting step names."""
        pipeline = Pipeline(steps=[
            StandardScaler(columns=["x"], name="scaler"),
            MinMaxScaler(columns=["y"], name="minmax")
        ])

        names = pipeline.get_step_names()

        assert names == ["scaler", "minmax"]

    def test_get_step_statistics(self):
        """Test getting step statistics."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])
        pipeline.fit(df)

        stats = pipeline.get_step_statistics()

        assert len(stats) == 1
        assert isinstance(stats, list)

    def test_get_pipeline_summary(self):
        """Test getting comprehensive pipeline summary."""
        pipeline = Pipeline(
            steps=[StandardScaler(columns=["x"])],
            name="TestPipeline"
        )
        df = pd.DataFrame({'x': [1, 2, 3]})
        pipeline.fit(df)

        summary = pipeline.get_pipeline_summary()

        assert summary["name"] == "TestPipeline"
        assert summary["num_steps"] == 1
        assert summary["fitted"] is True
        assert "steps" in summary
        assert "metadata" in summary
        assert "statistics" in summary


class TestPipelineSerialization:
    """Test Pipeline serialization methods."""

    def test_to_dict(self):
        """Test serializing pipeline to dictionary."""
        pipeline = Pipeline(
            steps=[StandardScaler(columns=["x"])],
            name="MyPipeline"
        )

        config = pipeline.to_dict()

        assert config["name"] == "MyPipeline"
        assert config["fitted"] is False
        assert len(config["steps"]) == 1
        assert config["steps"][0]["class"] == "StandardScaler"

    def test_from_dict(self):
        """Test deserializing pipeline from dictionary."""
        config = {
            "name": "MyPipeline",
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

        assert pipeline.name == "MyPipeline"
        assert len(pipeline.steps) == 1
        assert isinstance(pipeline.steps[0], StandardScaler)

    def test_from_dict_unknown_step_error(self):
        """Test error when deserializing unknown step class."""
        config = {
            "name": "MyPipeline",
            "steps": [
                {
                    "class": "UnknownStep",
                    "name": "unknown",
                    "fitted": False,
                    "params": {}
                }
            ]
        }

        with pytest.raises(ValueError, match="Unknown preprocessing step class"):
            Pipeline.from_dict(config)

    def test_save_and_load_pickle(self):
        """Test saving and loading pipeline with pickle."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        pipeline = Pipeline(
            steps=[StandardScaler(columns=["x"])],
            name="SavedPipeline"
        )
        pipeline.fit(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline.pkl"

            # Save
            pipeline.save(path)
            assert path.exists()

            # Load
            loaded_pipeline = Pipeline.load(path)
            assert loaded_pipeline.name == "SavedPipeline"
            assert loaded_pipeline.fitted is True
            assert len(loaded_pipeline.steps) == 1

            # Test that loaded pipeline can transform
            result = loaded_pipeline.transform(df)
            assert isinstance(result, pd.DataFrame)

    def test_save_config_and_load_config_json(self):
        """Test saving and loading pipeline config as JSON."""
        pipeline = Pipeline(
            steps=[
                StandardScaler(columns=["x"]),
                MinMaxScaler(columns=["y"])
            ],
            name="ConfigPipeline"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline_config.json"

            # Save config
            pipeline.save_config(path)
            assert path.exists()

            # Verify JSON format
            with open(path, 'r') as f:
                config = json.load(f)
                assert config["name"] == "ConfigPipeline"
                assert len(config["steps"]) == 2

            # Load config
            loaded_pipeline = Pipeline.load_config(path)
            assert loaded_pipeline.name == "ConfigPipeline"
            assert len(loaded_pipeline.steps) == 2
            assert not loaded_pipeline.fitted  # Config doesn't save fitted state


class TestPipelineSpecialMethods:
    """Test Pipeline special methods."""

    def test_len(self):
        """Test __len__ method."""
        pipeline = Pipeline(steps=[
            StandardScaler(columns=["x"]),
            MinMaxScaler(columns=["y"])
        ])

        assert len(pipeline) == 2

    def test_getitem(self):
        """Test __getitem__ method (bracket notation)."""
        step1 = StandardScaler(columns=["x"])
        step2 = MinMaxScaler(columns=["y"])
        pipeline = Pipeline(steps=[step1, step2])

        assert pipeline[0] == step1
        assert pipeline[1] == step2

    def test_repr(self):
        """Test __repr__ method."""
        pipeline = Pipeline(
            steps=[StandardScaler(columns=["x"], name="scaler")],
            name="TestPipeline"
        )

        repr_str = repr(pipeline)

        assert "TestPipeline" in repr_str
        assert "1 steps" in repr_str
        assert "not fitted" in repr_str

    def test_str(self):
        """Test __str__ method."""
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])

        str_repr = str(pipeline)

        assert isinstance(str_repr, str)
        assert "Pipeline" in str_repr


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_pipeline(self):
        """Test create_pipeline convenience function."""
        steps = [
            StandardScaler(columns=["x"]),
            MinMaxScaler(columns=["y"])
        ]

        pipeline = create_pipeline(steps, name="MyPipeline")

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2
        assert pipeline.name == "MyPipeline"

    def test_load_pipeline_function(self):
        """Test load_pipeline convenience function."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])
        pipeline.fit(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline.pkl"
            pipeline.save(path)

            loaded = load_pipeline(path)

            assert isinstance(loaded, Pipeline)
            assert loaded.fitted is True


class TestPipelineIntegration:
    """Integration tests for Pipeline with multiple step types."""

    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline with multiple step types."""
        # Create sample data
        df = pd.DataFrame({
            'numeric1': [1, 2, np.nan, 4, 5],
            'numeric2': [10, 20, 30, 40, 50],
            'numeric3': [100, 200, 300, 400, 500]
        })

        # Create pipeline
        pipeline = Pipeline(steps=[
            MeanImputer(columns=["numeric1"]),
            StandardScaler(columns=["numeric1", "numeric2"]),
            MinMaxScaler(columns=["numeric3"])
        ], name="FullPipeline")

        # Fit and transform
        result = pipeline.fit_transform(df)

        # Verify transformations
        assert result["numeric1"].isnull().sum() == 0  # No missing after imputation
        assert abs(result["numeric1"].mean()) < 0.01  # Standardized
        assert abs(result["numeric2"].mean()) < 0.01  # Standardized
        assert result["numeric3"].min() >= 0  # MinMax scaled
        assert result["numeric3"].max() <= 1  # MinMax scaled

    def test_pipeline_persistence_workflow(self):
        """Test complete workflow: create, fit, save, load, transform."""
        # Training data
        df_train = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })

        # Test data
        df_test = pd.DataFrame({
            'x': [6, 7, 8],
            'y': [60, 70, 80]
        })

        # Create and fit pipeline
        pipeline = Pipeline(steps=[
            StandardScaler(columns=["x", "y"])
        ], name="PersistencePipeline")
        pipeline.fit(df_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline.pkl"

            # Save fitted pipeline
            pipeline.save(path)

            # Load in new instance
            loaded_pipeline = Pipeline.load(path)

            # Transform test data with loaded pipeline
            result = loaded_pipeline.transform(df_test)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3


class TestPipelineErrorHandling:
    """Test Pipeline error handling."""

    def test_fit_step_failure_propagation(self):
        """Test that step errors during fit are properly propagated."""
        df = pd.DataFrame({'x': [1, 2, 3]})

        # Create a pipeline with a step that will fail
        # (StandardScaler expects DataFrame, we'll force a type error)
        pipeline = Pipeline(steps=[StandardScaler(columns=["nonexistent_column"])])

        # The fit should raise RuntimeError wrapping the underlying error
        with pytest.raises(RuntimeError, match="Pipeline fitting failed"):
            pipeline.fit(df)

    def test_transform_step_failure_propagation(self):
        """Test that step errors during transform are properly propagated."""
        df_train = pd.DataFrame({'x': [1, 2, 3]})
        df_test = pd.DataFrame({'y': [4, 5, 6]})  # Different column

        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])
        pipeline.fit(df_train)

        # Transform with different columns should fail
        with pytest.raises(RuntimeError, match="Pipeline transformation failed"):
            pipeline.transform(df_test)

    def test_refit_after_step_modification(self):
        """Test that pipeline must be refitted after step modifications."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        pipeline = Pipeline(steps=[StandardScaler(columns=["x"])])
        pipeline.fit(df)

        # Add new step
        pipeline.add_step(MinMaxScaler(columns=["x"]))

        # Pipeline should no longer be fitted
        assert not pipeline.fitted

        # Transform should fail
        with pytest.raises(RuntimeError, match="must be fitted"):
            pipeline.transform(df)
