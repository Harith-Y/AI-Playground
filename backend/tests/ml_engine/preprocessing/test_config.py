"""
Tests for preprocessing configuration management system.
"""

import pytest
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime

from app.ml_engine.preprocessing.config import (
    ConfigurationSchema,
    ConfigManager,
    PreprocessingPreset,
    load_preset,
    save_pipeline_config,
    load_pipeline_config,
)
from app.ml_engine.preprocessing.pipeline import Pipeline


class TestConfigurationSchema:
    """Tests for ConfigurationSchema validation."""

    def test_valid_configuration(self):
        """Test validation of valid configuration."""
        config = {
            "name": "Test Pipeline",
            "version": "1.0.0",
            "description": "Test description",
            "author": "Test Author",
            "preset": PreprocessingPreset.CUSTOM.value,
            "steps": [
                {
                    "class": "StandardScaler",
                    "name": "scaler",
                    "params": {}
                }
            ],
            "metadata": {}
        }

        # Should not raise
        ConfigurationSchema.validate(config)

    def test_missing_required_fields(self):
        """Test validation fails with missing required fields."""
        config = {
            "name": "Test",
            # Missing version, description, etc.
        }

        with pytest.raises(ValueError, match="Missing required fields"):
            ConfigurationSchema.validate(config)

    def test_invalid_version_format(self):
        """Test validation fails with invalid version format."""
        config = {
            "name": "Test Pipeline",
            "version": "invalid",  # Should be semver
            "description": "Test",
            "author": "Test",
            "preset": PreprocessingPreset.CUSTOM.value,
            "steps": [],
            "metadata": {}
        }

        with pytest.raises(ValueError, match="Invalid version format"):
            ConfigurationSchema.validate(config)

    def test_invalid_preset(self):
        """Test validation fails with invalid preset."""
        config = {
            "name": "Test Pipeline",
            "version": "1.0.0",
            "description": "Test",
            "author": "Test",
            "preset": "invalid_preset",
            "steps": [],
            "metadata": {}
        }

        with pytest.raises(ValueError, match="Invalid preset"):
            ConfigurationSchema.validate(config)

    def test_invalid_step_structure(self):
        """Test validation fails with invalid step structure."""
        config = {
            "name": "Test Pipeline",
            "version": "1.0.0",
            "description": "Test",
            "author": "Test",
            "preset": PreprocessingPreset.CUSTOM.value,
            "steps": [
                {
                    "class": "StandardScaler",
                    # Missing name
                    "params": {}
                }
            ],
            "metadata": {}
        }

        with pytest.raises(ValueError, match="Invalid step configuration"):
            ConfigurationSchema.validate(config)

    def test_steps_not_list(self):
        """Test validation fails when steps is not a list."""
        config = {
            "name": "Test Pipeline",
            "version": "1.0.0",
            "description": "Test",
            "author": "Test",
            "preset": PreprocessingPreset.CUSTOM.value,
            "steps": "not a list",
            "metadata": {}
        }

        with pytest.raises(ValueError, match="'steps' must be a list"):
            ConfigurationSchema.validate(config)


class TestConfigManager:
    """Tests for ConfigManager."""

    def test_initialization(self):
        """Test ConfigManager initialization."""
        manager = ConfigManager()
        assert manager.config_dir is not None
        assert manager.config_dir.exists()

    def test_create_config(self):
        """Test creating a configuration."""
        manager = ConfigManager()
        config = manager.create_config(
            name="Test Pipeline",
            description="Test description",
            author="Test Author"
        )

        assert config["name"] == "Test Pipeline"
        assert config["description"] == "Test description"
        assert config["author"] == "Test Author"
        assert config["version"] == "1.0.0"
        assert config["preset"] == PreprocessingPreset.CUSTOM.value
        assert config["steps"] == []
        assert "created_at" in config
        assert "updated_at" in config

    def test_get_preset(self):
        """Test getting preset configurations."""
        manager = ConfigManager()

        # Test all presets
        for preset in PreprocessingPreset:
            if preset != PreprocessingPreset.CUSTOM:
                config = manager.get_preset(preset)
                assert config is not None
                assert config["preset"] == preset.value
                assert "steps" in config
                assert len(config["steps"]) > 0

    def test_get_preset_custom_raises(self):
        """Test that getting CUSTOM preset raises error."""
        manager = ConfigManager()

        with pytest.raises(ValueError, match="CUSTOM preset has no predefined configuration"):
            manager.get_preset(PreprocessingPreset.CUSTOM)

    def test_save_and_load_config_json(self, tmp_path):
        """Test saving and loading configuration as JSON."""
        manager = ConfigManager(config_dir=tmp_path)

        config = manager.create_config(
            name="Test Pipeline",
            description="Test description"
        )
        config["steps"] = [
            {
                "class": "StandardScaler",
                "name": "scaler",
                "params": {}
            }
        ]

        # Save config
        filepath = manager.save_config(config, format="json")
        assert filepath.exists()
        assert filepath.suffix == ".json"

        # Load config
        loaded_config = manager.load_config(filepath)
        assert loaded_config["name"] == config["name"]
        assert loaded_config["description"] == config["description"]
        assert len(loaded_config["steps"]) == 1

    def test_save_and_load_config_yaml(self, tmp_path):
        """Test saving and loading configuration as YAML."""
        manager = ConfigManager(config_dir=tmp_path)

        config = manager.create_config(
            name="Test Pipeline",
            description="Test description"
        )

        # Save config
        filepath = manager.save_config(config, format="yaml")
        assert filepath.exists()
        assert filepath.suffix in [".yaml", ".yml"]

        # Load config
        loaded_config = manager.load_config(filepath)
        assert loaded_config["name"] == config["name"]

    def test_save_config_invalid_format(self, tmp_path):
        """Test saving with invalid format raises error."""
        manager = ConfigManager(config_dir=tmp_path)
        config = manager.create_config(name="Test")

        with pytest.raises(ValueError, match="Unsupported format"):
            manager.save_config(config, format="invalid")

    def test_validate_config(self):
        """Test configuration validation."""
        manager = ConfigManager()

        # Valid config
        config = manager.create_config(name="Test")
        assert manager.validate_config(config) == True

        # Invalid config
        invalid_config = {"name": "Test"}  # Missing required fields
        with pytest.raises(ValueError):
            manager.validate_config(invalid_config)

    def test_merge_configs(self):
        """Test merging configurations."""
        manager = ConfigManager()

        base_config = manager.create_config(
            name="Base",
            description="Base description"
        )
        base_config["steps"] = [
            {"class": "Step1", "name": "step1", "params": {}}
        ]

        override_config = {
            "description": "Override description",
            "steps": [
                {"class": "Step2", "name": "step2", "params": {}}
            ]
        }

        merged = manager.merge_configs(base_config, override_config)

        assert merged["name"] == "Base"  # Not overridden
        assert merged["description"] == "Override description"  # Overridden
        assert len(merged["steps"]) == 2  # Steps merged

    def test_create_from_pipeline(self):
        """Test creating config from Pipeline instance."""
        manager = ConfigManager()

        # Create a simple pipeline
        from app.ml_engine.preprocessing.scaler import StandardScaler
        from app.ml_engine.preprocessing.imputer import MeanImputer

        pipeline = Pipeline(
            name="Test Pipeline",
            description="Test description"
        )
        pipeline.add_step(MeanImputer(name="imputer"))
        pipeline.add_step(StandardScaler(name="scaler"))

        # Create config from pipeline
        config = manager.create_from_pipeline(pipeline, author="Test Author")

        assert config["name"] == "Test Pipeline"
        assert config["description"] == "Test description"
        assert config["author"] == "Test Author"
        assert len(config["steps"]) == 2
        assert config["steps"][0]["class"] == "MeanImputer"
        assert config["steps"][1]["class"] == "StandardScaler"

    def test_build_pipeline_from_config(self):
        """Test building Pipeline from configuration."""
        manager = ConfigManager()

        config = {
            "name": "Test Pipeline",
            "version": "1.0.0",
            "description": "Test description",
            "author": "Test",
            "preset": PreprocessingPreset.CUSTOM.value,
            "steps": [
                {
                    "class": "MeanImputer",
                    "name": "imputer",
                    "params": {}
                },
                {
                    "class": "StandardScaler",
                    "name": "scaler",
                    "params": {}
                }
            ],
            "metadata": {}
        }

        pipeline = manager.build_pipeline_from_config(config)

        assert isinstance(pipeline, Pipeline)
        assert pipeline.name == "Test Pipeline"
        assert pipeline.description == "Test description"
        assert pipeline.get_num_steps() == 2
        assert "imputer" in pipeline.get_step_names()
        assert "scaler" in pipeline.get_step_names()

    def test_build_pipeline_from_preset(self):
        """Test building Pipeline from preset."""
        manager = ConfigManager()

        pipeline = manager.build_pipeline_from_config(
            manager.get_preset(PreprocessingPreset.MINIMAL)
        )

        assert isinstance(pipeline, Pipeline)
        assert pipeline.get_num_steps() > 0

    def test_list_configs(self, tmp_path):
        """Test listing saved configurations."""
        manager = ConfigManager(config_dir=tmp_path)

        # Save multiple configs
        config1 = manager.create_config(name="Config1")
        config2 = manager.create_config(name="Config2")

        manager.save_config(config1, filepath=tmp_path / "config1.json")
        manager.save_config(config2, filepath=tmp_path / "config2.json")

        # List configs
        configs = manager.list_configs()

        assert len(configs) >= 2
        names = [c.stem for c in configs]
        assert "config1" in names
        assert "config2" in names


class TestPresets:
    """Tests for preset configurations."""

    def test_all_presets_valid(self):
        """Test that all presets produce valid configurations."""
        manager = ConfigManager()

        for preset in PreprocessingPreset:
            if preset == PreprocessingPreset.CUSTOM:
                continue

            config = manager.get_preset(preset)
            assert manager.validate_config(config)

    def test_minimal_preset(self):
        """Test minimal preset configuration."""
        manager = ConfigManager()
        config = manager.get_preset(PreprocessingPreset.MINIMAL)

        assert config["preset"] == PreprocessingPreset.MINIMAL.value
        assert len(config["steps"]) == 2  # Imputation + Scaling

        step_classes = [step["class"] for step in config["steps"]]
        assert "MeanImputer" in step_classes
        assert "StandardScaler" in step_classes

    def test_standard_preset(self):
        """Test standard preset configuration."""
        manager = ConfigManager()
        config = manager.get_preset(PreprocessingPreset.STANDARD)

        assert config["preset"] == PreprocessingPreset.STANDARD.value
        assert len(config["steps"]) >= 3  # Imputation + Outliers + Scaling

    def test_comprehensive_preset(self):
        """Test comprehensive preset configuration."""
        manager = ConfigManager()
        config = manager.get_preset(PreprocessingPreset.COMPREHENSIVE)

        assert config["preset"] == PreprocessingPreset.COMPREHENSIVE.value
        assert len(config["steps"]) >= 4  # Multiple steps

    def test_numeric_only_preset(self):
        """Test numeric-only preset configuration."""
        manager = ConfigManager()
        config = manager.get_preset(PreprocessingPreset.NUMERIC_ONLY)

        assert config["preset"] == PreprocessingPreset.NUMERIC_ONLY.value
        # Should only have numeric preprocessing steps

    def test_categorical_only_preset(self):
        """Test categorical-only preset configuration."""
        manager = ConfigManager()
        config = manager.get_preset(PreprocessingPreset.CATEGORICAL_ONLY)

        assert config["preset"] == PreprocessingPreset.CATEGORICAL_ONLY.value
        # Should only have categorical preprocessing steps

    def test_time_series_preset(self):
        """Test time series preset configuration."""
        manager = ConfigManager()
        config = manager.get_preset(PreprocessingPreset.TIME_SERIES)

        assert config["preset"] == PreprocessingPreset.TIME_SERIES.value
        # Should have time-series appropriate steps


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_load_preset_function(self):
        """Test load_preset convenience function."""
        config = load_preset(PreprocessingPreset.MINIMAL)

        assert config is not None
        assert config["preset"] == PreprocessingPreset.MINIMAL.value

    def test_load_preset_by_string(self):
        """Test load_preset with string preset name."""
        config = load_preset("minimal")

        assert config is not None
        assert config["preset"] == PreprocessingPreset.MINIMAL.value

    def test_save_and_load_pipeline_config(self, tmp_path):
        """Test save_pipeline_config and load_pipeline_config functions."""
        from app.ml_engine.preprocessing.scaler import StandardScaler

        pipeline = Pipeline(name="Test")
        pipeline.add_step(StandardScaler())

        # Save
        filepath = tmp_path / "pipeline.json"
        save_pipeline_config(pipeline, filepath, author="Test")

        assert filepath.exists()

        # Load
        loaded_config = load_pipeline_config(filepath)

        assert loaded_config["name"] == "Test"
        assert len(loaded_config["steps"]) == 1


class TestIntegration:
    """Integration tests for configuration system."""

    def test_full_workflow_json(self, tmp_path):
        """Test complete workflow: create -> save -> load -> build pipeline."""
        manager = ConfigManager(config_dir=tmp_path)

        # Create configuration
        config = manager.create_config(
            name="Integration Test",
            description="Full workflow test",
            author="Test Author"
        )

        config["steps"] = [
            {"class": "MeanImputer", "name": "imputer", "params": {}},
            {"class": "StandardScaler", "name": "scaler", "params": {}}
        ]

        # Save configuration
        filepath = manager.save_config(config, format="json")

        # Load configuration
        loaded_config = manager.load_config(filepath)

        # Build pipeline
        pipeline = manager.build_pipeline_from_config(loaded_config)

        # Test pipeline
        df = pd.DataFrame({
            'a': [1, 2, None, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })

        pipeline.fit(df)
        result = pipeline.transform(df)

        assert result is not None
        assert not result.isnull().any().any()

    def test_full_workflow_yaml(self, tmp_path):
        """Test complete workflow with YAML format."""
        manager = ConfigManager(config_dir=tmp_path)

        config = manager.create_config(name="YAML Test")
        config["steps"] = [
            {"class": "MeanImputer", "name": "imputer", "params": {}}
        ]

        # Save as YAML
        filepath = manager.save_config(config, format="yaml")

        # Load and build
        loaded_config = manager.load_config(filepath)
        pipeline = manager.build_pipeline_from_config(loaded_config)

        assert pipeline.get_num_steps() == 1

    def test_preset_to_pipeline_workflow(self):
        """Test workflow: load preset -> build pipeline -> fit/transform."""
        manager = ConfigManager()

        # Load preset
        config = manager.get_preset(PreprocessingPreset.MINIMAL)

        # Build pipeline
        pipeline = manager.build_pipeline_from_config(config)

        # Test with data
        df = pd.DataFrame({
            'num1': [1, 2, None, 4, 5],
            'num2': [10, 20, 30, 40, 50]
        })

        pipeline.fit(df)
        result = pipeline.transform(df)

        assert result is not None
        assert not result.isnull().any().any()

    def test_pipeline_to_config_to_pipeline(self, tmp_path):
        """Test roundtrip: pipeline -> config -> pipeline."""
        from app.ml_engine.preprocessing.scaler import StandardScaler
        from app.ml_engine.preprocessing.imputer import MeanImputer

        manager = ConfigManager(config_dir=tmp_path)

        # Create original pipeline
        original = Pipeline(name="Original")
        original.add_step(MeanImputer(name="imputer"))
        original.add_step(StandardScaler(name="scaler"))

        # Convert to config
        config = manager.create_from_pipeline(original, author="Test")

        # Save and load
        filepath = manager.save_config(config)
        loaded_config = manager.load_config(filepath)

        # Build new pipeline
        restored = manager.build_pipeline_from_config(loaded_config)

        # Compare
        assert restored.name == original.name
        assert restored.get_num_steps() == original.get_num_steps()
        assert restored.get_step_names() == original.get_step_names()

    def test_config_validation_in_workflow(self):
        """Test that validation catches errors during workflow."""
        manager = ConfigManager()

        # Create invalid config
        invalid_config = {
            "name": "Invalid",
            "version": "bad_version",  # Invalid semver
            "steps": []
        }

        # Should fail validation
        with pytest.raises(ValueError):
            manager.validate_config(invalid_config)

        # Should fail when building pipeline
        with pytest.raises(ValueError):
            manager.build_pipeline_from_config(invalid_config)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_steps(self):
        """Test configuration with no steps."""
        manager = ConfigManager()
        config = manager.create_config(name="Empty")

        # Valid config with no steps
        assert manager.validate_config(config)

        # Can build pipeline (though it won't do anything)
        pipeline = manager.build_pipeline_from_config(config)
        assert pipeline.get_num_steps() == 0

    def test_load_nonexistent_file(self):
        """Test loading non-existent configuration file."""
        manager = ConfigManager()

        with pytest.raises(FileNotFoundError):
            manager.load_config(Path("nonexistent.json"))

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON file."""
        manager = ConfigManager()

        # Create invalid JSON file
        filepath = tmp_path / "invalid.json"
        filepath.write_text("not valid json {")

        with pytest.raises(json.JSONDecodeError):
            manager.load_config(filepath)

    def test_save_without_permission(self):
        """Test saving to directory without write permission."""
        manager = ConfigManager()
        config = manager.create_config(name="Test")

        # Try to save to root (should fail on most systems)
        with pytest.raises((PermissionError, OSError)):
            manager.save_config(config, filepath=Path("/test.json"))

    def test_merge_with_none(self):
        """Test merging with None or empty config."""
        manager = ConfigManager()

        base = manager.create_config(name="Base")

        # Merge with empty dict
        merged = manager.merge_configs(base, {})
        assert merged["name"] == "Base"

        # Merge with partial override
        merged = manager.merge_configs(base, {"description": "New"})
        assert merged["name"] == "Base"
        assert merged["description"] == "New"

    def test_build_pipeline_with_invalid_step_class(self):
        """Test building pipeline with non-existent step class."""
        manager = ConfigManager()

        config = manager.create_config(name="Invalid")
        config["steps"] = [
            {"class": "NonExistentStep", "name": "bad", "params": {}}
        ]

        with pytest.raises((AttributeError, ImportError, ValueError)):
            manager.build_pipeline_from_config(config)

    def test_unicode_in_config(self, tmp_path):
        """Test configuration with Unicode characters."""
        manager = ConfigManager(config_dir=tmp_path)

        config = manager.create_config(
            name="Test Pipeline",
            description="Тестовое описание with émojis 🚀",
            author="作者"
        )

        # Save and load
        filepath = manager.save_config(config)
        loaded = manager.load_config(filepath)

        assert loaded["description"] == config["description"]
        assert loaded["author"] == config["author"]

    def test_very_long_config(self, tmp_path):
        """Test configuration with many steps."""
        manager = ConfigManager(config_dir=tmp_path)

        config = manager.create_config(name="Long Pipeline")

        # Add 100 steps
        for i in range(100):
            config["steps"].append({
                "class": "StandardScaler",
                "name": f"scaler_{i}",
                "params": {}
            })

        # Should still validate and save
        assert manager.validate_config(config)
        filepath = manager.save_config(config)

        # Should load successfully
        loaded = manager.load_config(filepath)
        assert len(loaded["steps"]) == 100
