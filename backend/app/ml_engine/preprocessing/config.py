"""
Configuration system for preprocessing pipelines.

Provides centralized configuration management, presets, validation,
and easy serialization/deserialization of preprocessing configurations.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from enum import Enum
import json
import yaml
from copy import deepcopy
from datetime import datetime
import pandas as pd

from app.ml_engine.preprocessing.pipeline import Pipeline
from app.utils.logger import get_logger

logger = get_logger("preprocessing_config")


class PreprocessingPreset(str, Enum):
    """Predefined preprocessing configuration presets."""

    MINIMAL = "minimal"                    # Basic preprocessing only
    STANDARD = "standard"                  # Standard preprocessing pipeline
    COMPREHENSIVE = "comprehensive"        # Full preprocessing with all steps
    NUMERIC_ONLY = "numeric_only"         # Only numeric preprocessing
    CATEGORICAL_ONLY = "categorical_only" # Only categorical preprocessing
    TIME_SERIES = "time_series"           # Time series preprocessing
    NLP = "nlp"                           # Text/NLP preprocessing
    CUSTOM = "custom"                     # User-defined configuration


class ConfigurationSchema:
    """
    Schema definition for preprocessing configurations.

    Defines the structure and validation rules for preprocessing configs.
    """

    REQUIRED_FIELDS = ["name", "version", "steps"]

    OPTIONAL_FIELDS = [
        "description",
        "author",
        "created_at",
        "updated_at",
        "metadata",
        "preset",
        "target_column",
        "feature_columns"
    ]

    STEP_FIELDS = ["class", "name", "params"]

    @classmethod
    def validate(cls, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate configuration against schema.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Validate steps structure
        if "steps" in config:
            if not isinstance(config["steps"], list):
                errors.append("'steps' must be a list")
            else:
                for idx, step in enumerate(config["steps"]):
                    if not isinstance(step, dict):
                        errors.append(f"Step {idx} must be a dictionary")
                        continue

                    for field in cls.STEP_FIELDS:
                        if field not in step:
                            errors.append(f"Step {idx} missing required field: {field}")

        # Validate version format
        if "version" in config:
            version = config["version"]
            if not isinstance(version, str):
                errors.append("'version' must be a string")
            elif not cls._is_valid_version(version):
                errors.append(f"Invalid version format: {version}")

        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def _is_valid_version(version: str) -> bool:
        """Check if version string is valid (semver format)."""
        parts = version.split(".")
        if len(parts) != 3:
            return False
        return all(part.isdigit() for part in parts)


class ConfigManager:
    """
    Centralized manager for preprocessing configurations.

    Handles loading, saving, validation, and management of preprocessing
    pipeline configurations with support for presets and templates.

    Example:
        >>> manager = ConfigManager()
        >>> config = manager.create_config("MyPipeline", preset=PreprocessingPreset.STANDARD)
        >>> manager.save_config(config, "configs/my_pipeline.json")
        >>> loaded = manager.load_config("configs/my_pipeline.json")
    """

    def __init__(self, config_dir: Optional[Path] = None, enable_validation: bool = True):
        """
        Initialize configuration manager.

        Args:
            config_dir: Optional directory for storing configurations
            enable_validation: Whether to enable automatic validation (default True)
        """
        self.config_dir = Path(config_dir) if config_dir else Path("configs/preprocessing")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.enable_validation = enable_validation

        self.presets = self._load_presets()

        # Import validator here to avoid circular imports
        if self.enable_validation:
            from app.ml_engine.preprocessing.config_validator import ConfigValidator
            self.validator = ConfigValidator()
        else:
            self.validator = None

        logger.info(f"Initialized ConfigManager with config_dir: {self.config_dir}")

    def create_config(
        self,
        name: str,
        preset: Optional[PreprocessingPreset] = None,
        description: Optional[str] = None,
        author: Optional[str] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        **metadata
    ) -> Dict[str, Any]:
        """
        Create a new preprocessing configuration.

        Args:
            name: Configuration name
            preset: Optional preset to use as base
            description: Optional description
            author: Optional author name
            steps: Optional list of preprocessing steps
            **metadata: Additional metadata fields

        Returns:
            Configuration dictionary
        """
        config = {
            "name": name,
            "version": "1.0.0",
            "description": description or f"Preprocessing configuration: {name}",
            "author": author or "Unknown",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "preset": preset.value if preset else PreprocessingPreset.CUSTOM.value,
            "steps": steps or [],
            "metadata": metadata
        }

        # Apply preset if specified
        if preset and preset != PreprocessingPreset.CUSTOM:
            preset_config = self.get_preset(preset)
            if preset_config:
                config["steps"] = deepcopy(preset_config.get("steps", []))

        logger.info(f"Created config '{name}' with preset '{config['preset']}'")

        return config

    def save_config(
        self,
        config: Dict[str, Any],
        filepath: Optional[Union[str, Path]] = None,
        format: str = "json"
    ) -> Path:
        """
        Save configuration to file.

        Args:
            config: Configuration dictionary
            filepath: Optional filepath (defaults to config_dir/name.format)
            format: File format - 'json' or 'yaml'

        Returns:
            Path where config was saved

        Raises:
            ValueError: If config is invalid or format is unsupported
        """
        # Validate configuration
        is_valid, errors = ConfigurationSchema.validate(config)
        if not is_valid:
            raise ValueError(f"Invalid configuration: {', '.join(errors)}")

        # Determine filepath
        if filepath is None:
            filename = f"{config['name'].replace(' ', '_').lower()}.{format}"
            filepath = self.config_dir / filename
        else:
            filepath = Path(filepath)

        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Update timestamp
        config["updated_at"] = datetime.now().isoformat()

        # Save based on format
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
        elif format == "yaml":
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved config to {filepath}")
        return filepath

    def load_config(
        self,
        filepath: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Load configuration from file.

        Args:
            filepath: Path to configuration file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If configuration is invalid
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        # Load based on extension
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                config = json.load(f)
        elif filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        # Validate configuration
        is_valid, errors = ConfigurationSchema.validate(config)
        if not is_valid:
            raise ValueError(f"Invalid configuration in {filepath}: {', '.join(errors)}")

        logger.info(f"Loaded config from {filepath}")
        return config

    def get_preset(self, preset: PreprocessingPreset) -> Optional[Dict[str, Any]]:
        """
        Get predefined preset configuration.

        Args:
            preset: Preset identifier

        Returns:
            Preset configuration dictionary or None
        """
        return self.presets.get(preset.value)

    def list_configs(self, pattern: str = "*") -> List[Path]:
        """
        List all configuration files in config directory.

        Args:
            pattern: Glob pattern for filtering (default: all files)

        Returns:
            List of configuration file paths
        """
        json_files = list(self.config_dir.glob(f"{pattern}.json"))
        yaml_files = list(self.config_dir.glob(f"{pattern}.yaml"))
        yml_files = list(self.config_dir.glob(f"{pattern}.yml"))

        all_files = json_files + yaml_files + yml_files
        logger.debug(f"Found {len(all_files)} config files matching '{pattern}'")

        return sorted(all_files)

    def validate_config(
        self,
        config: Dict[str, Any],
        df: Optional[pd.DataFrame] = None,
        strict: bool = False
    ) -> tuple[bool, List[str]]:
        """
        Validate a configuration.

        Args:
            config: Configuration to validate
            df: Optional DataFrame for semantic validation
            strict: If True, treat warnings as errors

        Returns:
            Tuple of (is_valid, error_messages)
        """
        # First do basic schema validation
        is_valid, errors = ConfigurationSchema.validate(config)

        # If enabled, do comprehensive validation
        if self.enable_validation and self.validator is not None:
            result = self.validator.validate(config, df=df, strict=strict)

            # Add error messages from detailed validation
            for issue in result.get_errors():
                errors.append(f"{issue.code}: {issue.message}")

            # Add warnings if in strict mode
            if strict:
                for issue in result.get_warnings():
                    errors.append(f"{issue.code}: {issue.message}")

            is_valid = is_valid and result.valid

        return is_valid, errors

    def validate_config_detailed(
        self,
        config: Dict[str, Any],
        df: Optional[pd.DataFrame] = None,
        strict: bool = False
    ):
        """
        Perform detailed validation and return ValidationResult.

        Args:
            config: Configuration to validate
            df: Optional DataFrame for semantic validation
            strict: If True, treat warnings as errors

        Returns:
            ValidationResult object with all issues

        Raises:
            RuntimeError: If validation is not enabled
        """
        if not self.enable_validation or self.validator is None:
            raise RuntimeError("Validation is not enabled for this ConfigManager")

        return self.validator.validate(config, df=df, strict=strict)

    def validate_and_raise(
        self,
        config: Dict[str, Any],
        df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Validate configuration and raise exception if errors found.

        Args:
            config: Configuration to validate
            df: Optional DataFrame for semantic validation

        Raises:
            ValueError: If validation fails
        """
        if self.enable_validation and self.validator is not None:
            self.validator.validate_and_raise(config, df)
        else:
            # Fall back to basic validation
            is_valid, errors = ConfigurationSchema.validate(config)
            if not is_valid:
                raise ValueError(f"Configuration validation failed: {', '.join(errors)}")

    def auto_fix_config(self, config: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        """
        Attempt to automatically fix common configuration issues.

        Args:
            config: Configuration to fix

        Returns:
            Tuple of (fixed_config, list_of_fixes_applied)
        """
        from app.ml_engine.preprocessing.config_validator import auto_fix_config
        return auto_fix_config(config)

    def merge_configs(
        self,
        base_config: Dict[str, Any],
        override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge two configurations (override takes precedence).

        Args:
            base_config: Base configuration
            override_config: Configuration to merge in

        Returns:
            Merged configuration
        """
        merged = deepcopy(base_config)

        # Merge top-level fields
        for key, value in override_config.items():
            if key == "steps":
                # For steps, append rather than replace
                merged["steps"] = merged.get("steps", []) + value
            elif key == "metadata":
                # Merge metadata dictionaries
                merged.setdefault("metadata", {}).update(value)
            else:
                merged[key] = value

        merged["updated_at"] = datetime.now().isoformat()

        logger.debug("Merged configurations")
        return merged

    def create_from_pipeline(
        self,
        pipeline: Pipeline,
        name: str,
        description: Optional[str] = None,
        **metadata
    ) -> Dict[str, Any]:
        """
        Create configuration from existing Pipeline instance.

        Args:
            pipeline: Pipeline instance
            name: Configuration name
            description: Optional description
            **metadata: Additional metadata

        Returns:
            Configuration dictionary
        """
        config = {
            "name": name,
            "version": "1.0.0",
            "description": description or f"Configuration from pipeline '{pipeline.name}'",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "preset": PreprocessingPreset.CUSTOM.value,
            "steps": pipeline.to_dict(include_version=False)["steps"],
            "metadata": {
                "source_pipeline": pipeline.name,
                "num_steps": len(pipeline.steps),
                "fitted": pipeline.fitted,
                **metadata
            }
        }

        logger.info(f"Created config from pipeline '{pipeline.name}'")
        return config

    def build_pipeline_from_config(
        self,
        config: Dict[str, Any],
        name: Optional[str] = None,
        validate: bool = True,
        df: Optional[pd.DataFrame] = None
    ) -> Pipeline:
        """
        Build a Pipeline instance from configuration.

        Args:
            config: Configuration dictionary
            name: Optional pipeline name (defaults to config name)
            validate: Whether to validate config before building (default True)
            df: Optional DataFrame for semantic validation

        Returns:
            Pipeline instance

        Raises:
            ValueError: If validation is enabled and config is invalid
        """
        # Validate if requested
        if validate:
            if self.enable_validation and self.validator is not None:
                # Use detailed validation
                result = self.validator.validate(config, df=df, strict=False)
                if result.has_errors():
                    error_msg = "\n".join([str(issue) for issue in result.get_errors()])
                    raise ValueError(f"Configuration validation failed:\n{error_msg}")

                # Log warnings
                if result.has_warnings():
                    for warning in result.get_warnings():
                        logger.warning(f"{warning.code}: {warning.message}")
            else:
                # Fall back to basic validation
                is_valid, errors = ConfigurationSchema.validate(config)
                if not is_valid:
                    raise ValueError(f"Configuration validation failed: {', '.join(errors)}")

        pipeline_name = name or config.get("name", "ConfiguredPipeline")

        pipeline_dict = {
            "name": pipeline_name,
            "steps": config["steps"],
            "metadata": config.get("metadata", {}),
            "statistics": []
        }

        pipeline = Pipeline.from_dict(pipeline_dict)

        logger.info(f"Built pipeline '{pipeline_name}' from config")
        return pipeline

    def _load_presets(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined configuration presets."""
        return {
            PreprocessingPreset.MINIMAL.value: {
                "name": "Minimal Preprocessing",
                "description": "Basic preprocessing with essential steps only",
                "steps": [
                    {
                        "class": "MeanImputer",
                        "name": "impute_missing",
                        "params": {"columns": None}
                    },
                    {
                        "class": "StandardScaler",
                        "name": "scale_features",
                        "params": {"columns": None, "with_mean": True, "with_std": True}
                    }
                ]
            },

            PreprocessingPreset.STANDARD.value: {
                "name": "Standard Preprocessing",
                "description": "Standard preprocessing pipeline for most datasets",
                "steps": [
                    {
                        "class": "MeanImputer",
                        "name": "impute_numeric",
                        "params": {"columns": None}
                    },
                    {
                        "class": "IQROutlierDetector",
                        "name": "remove_outliers",
                        "params": {"columns": None, "threshold": 1.5}
                    },
                    {
                        "class": "StandardScaler",
                        "name": "scale_features",
                        "params": {"columns": None, "with_mean": True, "with_std": True}
                    },
                    {
                        "class": "OneHotEncoder",
                        "name": "encode_categorical",
                        "params": {}
                    }
                ]
            },

            PreprocessingPreset.COMPREHENSIVE.value: {
                "name": "Comprehensive Preprocessing",
                "description": "Full preprocessing with all advanced steps",
                "steps": [
                    {
                        "class": "MeanImputer",
                        "name": "impute_numeric",
                        "params": {"columns": None}
                    },
                    {
                        "class": "ModeImputer",
                        "name": "impute_categorical",
                        "params": {"columns": None}
                    },
                    {
                        "class": "IQROutlierDetector",
                        "name": "detect_outliers",
                        "params": {"columns": None, "threshold": 1.5}
                    },
                    {
                        "class": "StandardScaler",
                        "name": "scale_numeric",
                        "params": {"columns": None}
                    },
                    {
                        "class": "OneHotEncoder",
                        "name": "encode_categorical",
                        "params": {}
                    },
                    {
                        "class": "SMOTE",
                        "name": "balance_classes",
                        "params": {"sampling_strategy": "auto"}
                    }
                ]
            },

            PreprocessingPreset.NUMERIC_ONLY.value: {
                "name": "Numeric Only Preprocessing",
                "description": "Preprocessing for numeric features only",
                "steps": [
                    {
                        "class": "MeanImputer",
                        "name": "impute_missing",
                        "params": {"columns": None}
                    },
                    {
                        "class": "IQROutlierDetector",
                        "name": "remove_outliers",
                        "params": {"columns": None}
                    },
                    {
                        "class": "RobustScaler",
                        "name": "robust_scale",
                        "params": {"columns": None}
                    }
                ]
            },

            PreprocessingPreset.CATEGORICAL_ONLY.value: {
                "name": "Categorical Only Preprocessing",
                "description": "Preprocessing for categorical features only",
                "steps": [
                    {
                        "class": "ModeImputer",
                        "name": "impute_missing",
                        "params": {"columns": None}
                    },
                    {
                        "class": "OneHotEncoder",
                        "name": "encode_features",
                        "params": {}
                    }
                ]
            }
        }

    def export_config_template(
        self,
        filepath: Union[str, Path],
        format: str = "json"
    ) -> None:
        """
        Export an empty configuration template.

        Args:
            filepath: Where to save template
            format: File format ('json' or 'yaml')
        """
        template = {
            "name": "MyPreprocessingPipeline",
            "version": "1.0.0",
            "description": "Description of your preprocessing pipeline",
            "author": "Your Name",
            "preset": PreprocessingPreset.CUSTOM.value,
            "steps": [
                {
                    "class": "StepClassName",
                    "name": "step_name",
                    "params": {
                        "param1": "value1",
                        "param2": "value2"
                    }
                }
            ],
            "metadata": {
                "tags": ["example", "template"],
                "notes": "Additional notes about this configuration"
            }
        }

        filepath = Path(filepath)

        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(template, f, indent=2)
        elif format == "yaml":
            with open(filepath, 'w') as f:
                yaml.dump(template, f, default_flow_style=False)

        logger.info(f"Exported template to {filepath}")


# Convenience functions

def load_preset(preset: PreprocessingPreset) -> Pipeline:
    """
    Load a preset preprocessing pipeline.

    Args:
        preset: Preset identifier

    Returns:
        Pipeline instance configured with preset

    Example:
        >>> pipeline = load_preset(PreprocessingPreset.STANDARD)
        >>> pipeline.fit_transform(df)
    """
    manager = ConfigManager()
    config = manager.get_preset(preset)

    if not config:
        raise ValueError(f"Preset not found: {preset}")

    return manager.build_pipeline_from_config(config)


def save_pipeline_config(
    pipeline: Pipeline,
    filepath: Union[str, Path],
    name: Optional[str] = None,
    description: Optional[str] = None,
    format: str = "json"
) -> Path:
    """
    Save pipeline as configuration file.

    Args:
        pipeline: Pipeline to save
        filepath: Where to save configuration
        name: Optional configuration name
        description: Optional description
        format: File format ('json' or 'yaml')

    Returns:
        Path where config was saved
    """
    manager = ConfigManager()
    config = manager.create_from_pipeline(
        pipeline,
        name=name or pipeline.name,
        description=description
    )

    return manager.save_config(config, filepath, format=format)


def load_pipeline_config(filepath: Union[str, Path]) -> Pipeline:
    """
    Load pipeline from configuration file.

    Args:
        filepath: Path to configuration file

    Returns:
        Pipeline instance
    """
    manager = ConfigManager()
    config = manager.load_config(filepath)
    return manager.build_pipeline_from_config(config)
