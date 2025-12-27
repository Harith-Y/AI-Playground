"""
Tests for configuration builder and recipes.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from app.ml_engine.preprocessing.config_builder import (
    ConfigBuilder,
    ConfigRecipes,
    create_config,
)
from app.ml_engine.preprocessing.config import (
    PreprocessingPreset,
    ConfigManager,
)


class TestConfigBuilder:
    """Tests for ConfigBuilder class."""

    def test_initialization(self):
        """Test ConfigBuilder initialization."""
        builder = ConfigBuilder("Test Pipeline")

        config = builder.config

        assert config["name"] == "Test Pipeline"
        assert config["version"] == "1.0.0"
        assert config["description"] == "Preprocessing configuration: Test Pipeline"
        assert config["author"] == "Unknown"
        assert config["preset"] == PreprocessingPreset.CUSTOM.value
        assert config["steps"] == []
        assert "created_at" in config
        assert "updated_at" in config

    def test_with_description(self):
        """Test setting description."""
        builder = ConfigBuilder("Test")
        builder.with_description("Custom description")

        assert builder.config["description"] == "Custom description"

    def test_with_author(self):
        """Test setting author."""
        builder = ConfigBuilder("Test")
        builder.with_author("John Doe")

        assert builder.config["author"] == "John Doe"

    def test_with_version(self):
        """Test setting version."""
        builder = ConfigBuilder("Test")
        builder.with_version("2.0.0")

        assert builder.config["version"] == "2.0.0"

    def test_with_metadata(self):
        """Test setting metadata."""
        builder = ConfigBuilder("Test")
        builder.with_metadata(project="ML", team="Data Science")

        assert builder.config["metadata"]["project"] == "ML"
        assert builder.config["metadata"]["team"] == "Data Science"

    def test_method_chaining(self):
        """Test that methods support chaining."""
        builder = (ConfigBuilder("Test")
                   .with_description("Description")
                   .with_author("Author")
                   .with_version("1.0.0")
                   .with_metadata(key="value"))

        assert builder.config["description"] == "Description"
        assert builder.config["author"] == "Author"
        assert builder.config["version"] == "1.0.0"
        assert builder.config["metadata"]["key"] == "value"

    def test_add_step(self):
        """Test adding a preprocessing step."""
        builder = ConfigBuilder("Test")
        builder.add_step("StandardScaler", "scaler", columns=["a", "b"])

        assert len(builder.config["steps"]) == 1
        step = builder.config["steps"][0]
        assert step["class"] == "StandardScaler"
        assert step["name"] == "scaler"
        assert step["params"]["columns"] == ["a", "b"]

    def test_add_imputation(self):
        """Test adding imputation step."""
        builder = ConfigBuilder("Test")

        # Mean imputation
        builder.add_imputation("mean", columns=["a", "b"])
        assert len(builder.config["steps"]) == 1
        assert builder.config["steps"][0]["class"] == "MeanImputer"
        assert builder.config["steps"][0]["name"] == "impute_mean"

        # Median imputation
        builder.add_imputation("median", columns=["c"])
        assert len(builder.config["steps"]) == 2
        assert builder.config["steps"][1]["class"] == "MedianImputer"

        # Mode imputation
        builder.add_imputation("mode", name="custom_imputer")
        assert builder.config["steps"][2]["name"] == "custom_imputer"

    def test_add_scaling(self):
        """Test adding scaling step."""
        builder = ConfigBuilder("Test")

        # Standard scaling
        builder.add_scaling("standard", columns=["a", "b"])
        assert builder.config["steps"][0]["class"] == "StandardScaler"

        # MinMax scaling
        builder.add_scaling("minmax", columns=["c"])
        assert builder.config["steps"][1]["class"] == "MinMaxScaler"

        # Robust scaling
        builder.add_scaling("robust", name="robust_scaler")
        assert builder.config["steps"][2]["class"] == "RobustScaler"

    def test_add_scaling_with_params(self):
        """Test adding scaling with additional parameters."""
        builder = ConfigBuilder("Test")
        builder.add_scaling("standard", columns=["a"], with_mean=True, with_std=True)

        step = builder.config["steps"][0]
        assert step["params"]["with_mean"] == True
        assert step["params"]["with_std"] == True

    def test_add_encoding(self):
        """Test adding encoding step."""
        builder = ConfigBuilder("Test")

        # OneHot encoding
        builder.add_encoding("onehot", columns=["category"])
        assert builder.config["steps"][0]["class"] == "OneHotEncoder"

        # Label encoding
        builder.add_encoding("label", columns=["label"])
        assert builder.config["steps"][1]["class"] == "LabelEncoder"

        # Ordinal encoding
        builder.add_encoding("ordinal", columns=["size"])
        assert builder.config["steps"][2]["class"] == "OrdinalEncoder"

    def test_add_outlier_detection(self):
        """Test adding outlier detection step."""
        builder = ConfigBuilder("Test")

        # IQR method
        builder.add_outlier_detection("iqr", columns=["value"])
        assert builder.config["steps"][0]["class"] == "IQROutlierDetector"

        # Z-score method
        builder.add_outlier_detection("zscore", columns=["score"])
        assert builder.config["steps"][1]["class"] == "ZScoreOutlierDetector"

    def test_add_sampling(self):
        """Test adding sampling step."""
        builder = ConfigBuilder("Test")

        # SMOTE
        builder.add_sampling("smote")
        assert builder.config["steps"][0]["class"] == "SMOTE"

        # Borderline SMOTE
        builder.add_sampling("borderline_smote")
        assert builder.config["steps"][1]["class"] == "BorderlineSMOTE"

        # ADASYN
        builder.add_sampling("adasyn")
        assert builder.config["steps"][2]["class"] == "ADASYN"

        # Random undersampling
        builder.add_sampling("random_under")
        assert builder.config["steps"][3]["class"] == "RandomUnderSampler"

    def test_from_preset(self):
        """Test loading from preset."""
        builder = ConfigBuilder("Test")
        builder.from_preset(PreprocessingPreset.MINIMAL)

        assert builder.config["preset"] == PreprocessingPreset.MINIMAL.value
        assert len(builder.config["steps"]) > 0

    def test_remove_step(self):
        """Test removing a step by index."""
        builder = ConfigBuilder("Test")
        builder.add_imputation("mean")
        builder.add_scaling("standard")
        builder.add_encoding("onehot")

        assert len(builder.config["steps"]) == 3

        # Remove middle step
        builder.remove_step(1)
        assert len(builder.config["steps"]) == 2
        assert builder.config["steps"][0]["class"] == "MeanImputer"
        assert builder.config["steps"][1]["class"] == "OneHotEncoder"

    def test_remove_step_invalid_index(self):
        """Test removing step with invalid index."""
        builder = ConfigBuilder("Test")
        builder.add_imputation("mean")

        # Should not raise, just log warning
        builder.remove_step(10)
        assert len(builder.config["steps"]) == 1

    def test_clear_steps(self):
        """Test clearing all steps."""
        builder = ConfigBuilder("Test")
        builder.add_imputation("mean")
        builder.add_scaling("standard")

        assert len(builder.config["steps"]) == 2

        builder.clear_steps()
        assert len(builder.config["steps"]) == 0

    def test_build(self):
        """Test building configuration."""
        builder = ConfigBuilder("Test")
        builder.add_imputation("mean")
        builder.add_scaling("standard")

        config = builder.build()

        assert config["name"] == "Test"
        assert len(config["steps"]) == 2
        assert "updated_at" in config

        # Should return a copy
        config["name"] = "Modified"
        assert builder.config["name"] == "Test"

    def test_save(self, tmp_path):
        """Test saving configuration."""
        builder = ConfigBuilder("Test")
        builder.add_imputation("mean")

        filepath = builder.save(filepath=str(tmp_path / "test.json"))

        assert Path(filepath).exists()
        assert Path(filepath).suffix == ".json"

    def test_save_yaml(self, tmp_path):
        """Test saving as YAML."""
        builder = ConfigBuilder("Test")
        builder.add_imputation("mean")

        filepath = builder.save(filepath=str(tmp_path / "test.yaml"), format="yaml")

        assert Path(filepath).exists()


class TestConfigRecipes:
    """Tests for ConfigRecipes class."""

    def test_for_classification_default(self):
        """Test classification recipe with defaults."""
        config = ConfigRecipes.for_classification()

        assert config["name"] == "Classification Pipeline"
        assert config["description"] == "Classification preprocessing pipeline"

        step_classes = [step["class"] for step in config["steps"]]

        # Should have imputation, outliers, scaling (not balancing by default)
        assert "MeanImputer" in step_classes
        assert "IQROutlierDetector" in step_classes
        assert "StandardScaler" in step_classes

    def test_for_classification_with_balancing(self):
        """Test classification recipe with class balancing."""
        config = ConfigRecipes.for_classification(balance_classes=True)

        step_classes = [step["class"] for step in config["steps"]]
        assert "SMOTE" in step_classes

    def test_for_classification_minimal(self):
        """Test minimal classification configuration."""
        config = ConfigRecipes.for_classification(
            handle_missing=False,
            remove_outliers=False,
            balance_classes=False,
            scale_features=True
        )

        step_classes = [step["class"] for step in config["steps"]]

        # Only scaling
        assert len(step_classes) == 1
        assert "StandardScaler" in step_classes

    def test_for_regression_default(self):
        """Test regression recipe with defaults."""
        config = ConfigRecipes.for_regression()

        assert config["name"] == "Regression Pipeline"
        assert config["description"] == "Regression preprocessing pipeline"

        step_classes = [step["class"] for step in config["steps"]]

        assert "MeanImputer" in step_classes
        assert "IQROutlierDetector" in step_classes
        assert "StandardScaler" in step_classes

    def test_for_regression_custom_scaling(self):
        """Test regression with custom scaling method."""
        config = ConfigRecipes.for_regression(scaling_method="minmax")

        step_classes = [step["class"] for step in config["steps"]]
        assert "MinMaxScaler" in step_classes

    def test_for_time_series(self):
        """Test time series recipe."""
        config = ConfigRecipes.for_time_series()

        assert config["name"] == "Time Series Pipeline"
        assert config["description"] == "Time series preprocessing pipeline"

        step_classes = [step["class"] for step in config["steps"]]

        # Time series uses median imputation and robust scaling
        assert "MedianImputer" in step_classes
        assert "RobustScaler" in step_classes

    def test_minimal_recipe(self):
        """Test minimal recipe."""
        config = ConfigRecipes.minimal()

        assert config["name"] == "Minimal Pipeline"
        assert config["description"] == "Minimal preprocessing - essentials only"

        step_classes = [step["class"] for step in config["steps"]]

        # Only imputation and scaling
        assert len(step_classes) == 2
        assert "MeanImputer" in step_classes
        assert "StandardScaler" in step_classes


class TestCreateConfigFunction:
    """Tests for create_config convenience function."""

    def test_create_config_function(self):
        """Test create_config function returns builder."""
        builder = create_config("Test Pipeline")

        assert isinstance(builder, ConfigBuilder)
        assert builder.config["name"] == "Test Pipeline"

    def test_create_config_chaining(self):
        """Test that create_config can be chained."""
        config = (create_config("Test")
                  .add_imputation("mean")
                  .add_scaling("standard")
                  .build())

        assert len(config["steps"]) == 2


class TestIntegration:
    """Integration tests for ConfigBuilder."""

    def test_full_builder_workflow(self):
        """Test complete builder workflow."""
        config = (ConfigBuilder("My Pipeline")
                  .with_description("Custom preprocessing pipeline")
                  .with_author("Data Scientist")
                  .with_version("1.2.0")
                  .with_metadata(project="ML Project", dataset="customers")
                  .add_imputation("mean", columns=["age", "income"])
                  .add_outlier_detection("iqr", columns=["age", "income"], threshold=1.5)
                  .add_scaling("standard", columns=["age", "income"])
                  .add_encoding("onehot", columns=["category", "region"])
                  .build())

        # Validate structure
        assert config["name"] == "My Pipeline"
        assert config["description"] == "Custom preprocessing pipeline"
        assert config["author"] == "Data Scientist"
        assert config["version"] == "1.2.0"
        assert config["metadata"]["project"] == "ML Project"

        # Validate steps
        assert len(config["steps"]) == 4

        step_names = [step["name"] for step in config["steps"]]
        assert "impute_mean" in step_names
        assert "outlier_iqr" in step_names
        assert "scale_standard" in step_names
        assert "encode_onehot" in step_names

    def test_builder_to_pipeline(self):
        """Test building pipeline from ConfigBuilder."""
        from app.ml_engine.preprocessing.pipeline import Pipeline

        config = (ConfigBuilder("Test")
                  .add_imputation("mean")
                  .add_scaling("standard")
                  .build())

        manager = ConfigManager()
        pipeline = manager.build_pipeline_from_config(config)

        assert isinstance(pipeline, Pipeline)
        assert pipeline.get_num_steps() == 2

    def test_recipe_to_pipeline_to_transform(self):
        """Test complete flow: recipe -> pipeline -> transform."""
        from app.ml_engine.preprocessing.pipeline import Pipeline

        # Create config from recipe
        config = ConfigRecipes.for_classification(
            handle_missing=True,
            remove_outliers=False,
            balance_classes=False,
            scale_features=True
        )

        # Build pipeline
        manager = ConfigManager()
        pipeline = manager.build_pipeline_from_config(config)

        # Test with data
        df = pd.DataFrame({
            'age': [25, 30, None, 40, 35],
            'income': [50000, 60000, 70000, 80000, 75000]
        })

        pipeline.fit(df)
        result = pipeline.transform(df)

        assert result is not None
        assert not result.isnull().any().any()

    def test_builder_with_preset_modification(self):
        """Test loading preset and modifying it."""
        config = (ConfigBuilder("Custom")
                  .from_preset(PreprocessingPreset.MINIMAL)
                  .add_outlier_detection("iqr")  # Add extra step
                  .with_description("Modified minimal preset")
                  .build())

        # Should have preset steps plus the additional one
        assert config["preset"] == PreprocessingPreset.MINIMAL.value
        assert len(config["steps"]) > 2  # More than minimal

        step_classes = [step["class"] for step in config["steps"]]
        assert "IQROutlierDetector" in step_classes

    def test_save_load_build_workflow(self, tmp_path):
        """Test complete workflow: build -> save -> load -> pipeline."""
        # Build config
        builder = ConfigBuilder("Test Pipeline")
        builder.add_imputation("mean")
        builder.add_scaling("standard")

        # Save
        filepath = builder.save(filepath=str(tmp_path / "config.json"))

        # Load
        manager = ConfigManager()
        loaded_config = manager.load_config(Path(filepath))

        # Build pipeline
        pipeline = manager.build_pipeline_from_config(loaded_config)

        assert pipeline.name == "Test Pipeline"
        assert pipeline.get_num_steps() == 2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_builder(self):
        """Test builder with no steps."""
        config = ConfigBuilder("Empty").build()

        assert config["name"] == "Empty"
        assert len(config["steps"]) == 0

        # Should still be valid
        manager = ConfigManager()
        assert manager.validate_config(config)

    def test_builder_with_unknown_strategy(self):
        """Test adding step with unknown strategy."""
        builder = ConfigBuilder("Test")

        # Should fall back to default
        builder.add_imputation("unknown_strategy")

        # Should use MeanImputer as default
        assert builder.config["steps"][0]["class"] == "MeanImputer"

    def test_builder_with_none_columns(self):
        """Test adding steps with None columns."""
        builder = ConfigBuilder("Test")
        builder.add_imputation("mean", columns=None)

        step = builder.config["steps"][0]
        assert step["params"]["columns"] is None

    def test_multiple_metadata_updates(self):
        """Test updating metadata multiple times."""
        builder = ConfigBuilder("Test")
        builder.with_metadata(key1="value1")
        builder.with_metadata(key2="value2")
        builder.with_metadata(key1="updated")  # Override

        assert builder.config["metadata"]["key1"] == "updated"
        assert builder.config["metadata"]["key2"] == "value2"

    def test_recipe_with_all_options_disabled(self):
        """Test recipe with all options disabled."""
        config = ConfigRecipes.for_classification(
            handle_missing=False,
            remove_outliers=False,
            balance_classes=False,
            scale_features=False
        )

        # Should have no steps
        assert len(config["steps"]) == 0

    def test_builder_updated_at_timestamp(self):
        """Test that build() updates the updated_at timestamp."""
        builder = ConfigBuilder("Test")
        builder.add_imputation("mean")

        # Get first timestamp
        config1 = builder.build()
        timestamp1 = config1["updated_at"]

        # Wait a tiny bit and build again
        import time
        time.sleep(0.01)

        config2 = builder.build()
        timestamp2 = config2["updated_at"]

        # Timestamps should be different
        assert timestamp1 != timestamp2

    def test_custom_step_name(self):
        """Test providing custom step names."""
        builder = ConfigBuilder("Test")
        builder.add_imputation("mean", name="my_custom_imputer")
        builder.add_scaling("standard", name="my_scaler")

        step_names = [step["name"] for step in builder.config["steps"]]
        assert "my_custom_imputer" in step_names
        assert "my_scaler" in step_names

    def test_builder_with_complex_params(self):
        """Test adding steps with complex parameters."""
        builder = ConfigBuilder("Test")
        builder.add_step(
            "CustomStep",
            "custom",
            param1={"nested": "value"},
            param2=[1, 2, 3],
            param3=True
        )

        step = builder.config["steps"][0]
        assert step["params"]["param1"] == {"nested": "value"}
        assert step["params"]["param2"] == [1, 2, 3]
        assert step["params"]["param3"] == True

    def test_recipe_custom_name(self):
        """Test creating recipe with custom name."""
        config = ConfigRecipes.for_classification(name="My Classification Pipeline")

        assert config["name"] == "My Classification Pipeline"

    def test_builder_clear_and_rebuild(self):
        """Test clearing steps and rebuilding."""
        builder = ConfigBuilder("Test")
        builder.add_imputation("mean")
        builder.add_scaling("standard")

        assert len(builder.config["steps"]) == 2

        builder.clear_steps()
        assert len(builder.config["steps"]) == 0

        # Rebuild with different steps
        builder.add_encoding("onehot")
        assert len(builder.config["steps"]) == 1
        assert builder.config["steps"][0]["class"] == "OneHotEncoder"
