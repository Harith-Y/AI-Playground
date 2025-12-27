"""
Configuration builder for creating preprocessing configurations programmatically.

Provides a fluent API for building preprocessing configurations without
manually creating dictionaries.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from app.ml_engine.preprocessing.config import PreprocessingPreset, ConfigManager
from app.utils.logger import get_logger

logger = get_logger("config_builder")


class ConfigBuilder:
    """
    Fluent builder for preprocessing configurations.

    Provides a chainable API for creating preprocessing configurations
    programmatically with validation and convenience methods.

    Example:
        >>> config = (ConfigBuilder("MyPipeline")
        ...     .with_description("Custom preprocessing")
        ...     .with_author("Data Scientist")
        ...     .add_imputation("mean", columns=["age", "salary"])
        ...     .add_scaling("standard", columns=["age", "salary"])
        ...     .add_encoding("onehot", columns=["category"])
        ...     .build())
    """

    def __init__(self, name: str):
        """
        Initialize configuration builder.

        Args:
            name: Configuration name
        """
        self.config = {
            "name": name,
            "version": "1.0.0",
            "description": f"Preprocessing configuration: {name}",
            "author": "Unknown",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "preset": PreprocessingPreset.CUSTOM.value,
            "steps": [],
            "metadata": {}
        }

        logger.debug(f"Initialized ConfigBuilder for '{name}'")

    def with_description(self, description: str) -> "ConfigBuilder":
        """
        Set configuration description.

        Args:
            description: Description text

        Returns:
            Self for chaining
        """
        self.config["description"] = description
        return self

    def with_author(self, author: str) -> "ConfigBuilder":
        """
        Set configuration author.

        Args:
            author: Author name

        Returns:
            Self for chaining
        """
        self.config["author"] = author
        return self

    def with_version(self, version: str) -> "ConfigBuilder":
        """
        Set configuration version.

        Args:
            version: Version string (semver format)

        Returns:
            Self for chaining
        """
        self.config["version"] = version
        return self

    def with_metadata(self, **metadata) -> "ConfigBuilder":
        """
        Add metadata fields.

        Args:
            **metadata: Key-value pairs to add to metadata

        Returns:
            Self for chaining
        """
        self.config["metadata"].update(metadata)
        return self

    def from_preset(self, preset: PreprocessingPreset) -> "ConfigBuilder":
        """
        Load steps from a preset.

        Args:
            preset: Preset to load

        Returns:
            Self for chaining
        """
        manager = ConfigManager()
        preset_config = manager.get_preset(preset)

        if preset_config:
            self.config["steps"] = preset_config.get("steps", []).copy()
            self.config["preset"] = preset.value
            logger.debug(f"Loaded preset '{preset.value}'")

        return self

    def add_step(
        self,
        class_name: str,
        name: str,
        **params
    ) -> "ConfigBuilder":
        """
        Add a preprocessing step.

        Args:
            class_name: Step class name
            name: Step instance name
            **params: Step parameters

        Returns:
            Self for chaining
        """
        step = {
            "class": class_name,
            "name": name,
            "params": params
        }

        self.config["steps"].append(step)
        logger.debug(f"Added step: {class_name} ({name})")

        return self

    # Convenience methods for common preprocessing steps

    def add_imputation(
        self,
        strategy: str = "mean",
        columns: Optional[List[str]] = None,
        name: Optional[str] = None
    ) -> "ConfigBuilder":
        """
        Add imputation step.

        Args:
            strategy: Imputation strategy ('mean', 'median', 'mode')
            columns: Columns to impute (None = all)
            name: Optional step name

        Returns:
            Self for chaining
        """
        class_map = {
            "mean": "MeanImputer",
            "median": "MedianImputer",
            "mode": "ModeImputer"
        }

        class_name = class_map.get(strategy, "MeanImputer")
        step_name = name or f"impute_{strategy}"

        return self.add_step(class_name, step_name, columns=columns)

    def add_scaling(
        self,
        method: str = "standard",
        columns: Optional[List[str]] = None,
        name: Optional[str] = None,
        **params
    ) -> "ConfigBuilder":
        """
        Add scaling step.

        Args:
            method: Scaling method ('standard', 'minmax', 'robust')
            columns: Columns to scale (None = all)
            name: Optional step name
            **params: Additional scaler parameters

        Returns:
            Self for chaining
        """
        class_map = {
            "standard": "StandardScaler",
            "minmax": "MinMaxScaler",
            "robust": "RobustScaler"
        }

        class_name = class_map.get(method, "StandardScaler")
        step_name = name or f"scale_{method}"

        params["columns"] = columns

        return self.add_step(class_name, step_name, **params)

    def add_encoding(
        self,
        method: str = "onehot",
        columns: Optional[List[str]] = None,
        name: Optional[str] = None,
        **params
    ) -> "ConfigBuilder":
        """
        Add encoding step.

        Args:
            method: Encoding method ('onehot', 'label', 'ordinal')
            columns: Columns to encode
            name: Optional step name
            **params: Additional encoder parameters

        Returns:
            Self for chaining
        """
        class_map = {
            "onehot": "OneHotEncoder",
            "label": "LabelEncoder",
            "ordinal": "OrdinalEncoder"
        }

        class_name = class_map.get(method, "OneHotEncoder")
        step_name = name or f"encode_{method}"

        if columns:
            params["columns"] = columns

        return self.add_step(class_name, step_name, **params)

    def add_outlier_detection(
        self,
        method: str = "iqr",
        columns: Optional[List[str]] = None,
        name: Optional[str] = None,
        **params
    ) -> "ConfigBuilder":
        """
        Add outlier detection step.

        Args:
            method: Detection method ('iqr', 'zscore')
            columns: Columns to check
            name: Optional step name
            **params: Additional detector parameters

        Returns:
            Self for chaining
        """
        class_map = {
            "iqr": "IQROutlierDetector",
            "zscore": "ZScoreOutlierDetector"
        }

        class_name = class_map.get(method, "IQROutlierDetector")
        step_name = name or f"outlier_{method}"

        params["columns"] = columns

        return self.add_step(class_name, step_name, **params)

    def add_sampling(
        self,
        method: str = "smote",
        name: Optional[str] = None,
        **params
    ) -> "ConfigBuilder":
        """
        Add sampling step for class balancing.

        Args:
            method: Sampling method ('smote', 'borderline_smote', 'adasyn', 'random_under', etc.)
            name: Optional step name
            **params: Additional sampler parameters

        Returns:
            Self for chaining
        """
        class_map = {
            "smote": "SMOTE",
            "borderline_smote": "BorderlineSMOTE",
            "adasyn": "ADASYN",
            "random_under": "RandomUnderSampler",
            "nearmiss": "NearMissUnderSampler",
            "tomek": "TomekLinksRemover"
        }

        class_name = class_map.get(method, "SMOTE")
        step_name = name or f"sample_{method}"

        return self.add_step(class_name, step_name, **params)

    def remove_step(self, index: int) -> "ConfigBuilder":
        """
        Remove a step by index.

        Args:
            index: Step index to remove

        Returns:
            Self for chaining
        """
        if 0 <= index < len(self.config["steps"]):
            removed = self.config["steps"].pop(index)
            logger.debug(f"Removed step at index {index}: {removed['name']}")
        else:
            logger.warning(f"Invalid step index: {index}")

        return self

    def clear_steps(self) -> "ConfigBuilder":
        """
        Remove all steps.

        Returns:
            Self for chaining
        """
        self.config["steps"] = []
        logger.debug("Cleared all steps")
        return self

    def build(self) -> Dict[str, Any]:
        """
        Build and return the configuration.

        Returns:
            Configuration dictionary
        """
        self.config["updated_at"] = datetime.now().isoformat()
        logger.info(f"Built configuration '{self.config['name']}' with {len(self.config['steps'])} steps")
        return self.config.copy()

    def save(
        self,
        filepath: Optional[str] = None,
        format: str = "json"
    ) -> str:
        """
        Build and save configuration.

        Args:
            filepath: Optional filepath (defaults to name.format)
            format: File format ('json' or 'yaml')

        Returns:
            Path where config was saved
        """
        config = self.build()

        manager = ConfigManager()
        path = manager.save_config(config, filepath, format)

        logger.info(f"Saved configuration to {path}")
        return str(path)


# Convenience function

def create_config(name: str) -> ConfigBuilder:
    """
    Create a new configuration builder.

    Args:
        name: Configuration name

    Returns:
        ConfigBuilder instance

    Example:
        >>> config = (create_config("MyPipeline")
        ...     .add_imputation("mean")
        ...     .add_scaling("standard")
        ...     .build())
    """
    return ConfigBuilder(name)


# Pre-built configuration recipes

class ConfigRecipes:
    """
    Pre-built configuration recipes for common scenarios.
    """

    @staticmethod
    def for_classification(
        name: str = "Classification Pipeline",
        handle_missing: bool = True,
        remove_outliers: bool = True,
        balance_classes: bool = False,
        scale_features: bool = True
    ) -> Dict[str, Any]:
        """
        Create configuration for classification tasks.

        Args:
            name: Configuration name
            handle_missing: Whether to include imputation
            remove_outliers: Whether to include outlier detection
            balance_classes: Whether to include class balancing (SMOTE)
            scale_features: Whether to include scaling

        Returns:
            Configuration dictionary
        """
        builder = ConfigBuilder(name).with_description("Classification preprocessing pipeline")

        if handle_missing:
            builder.add_imputation("mean")

        if remove_outliers:
            builder.add_outlier_detection("iqr", threshold=1.5)

        if scale_features:
            builder.add_scaling("standard")

        if balance_classes:
            builder.add_sampling("smote")

        return builder.build()

    @staticmethod
    def for_regression(
        name: str = "Regression Pipeline",
        handle_missing: bool = True,
        remove_outliers: bool = True,
        scale_features: bool = True,
        scaling_method: str = "standard"
    ) -> Dict[str, Any]:
        """
        Create configuration for regression tasks.

        Args:
            name: Configuration name
            handle_missing: Whether to include imputation
            remove_outliers: Whether to include outlier detection
            scale_features: Whether to include scaling
            scaling_method: Scaling method to use

        Returns:
            Configuration dictionary
        """
        builder = ConfigBuilder(name).with_description("Regression preprocessing pipeline")

        if handle_missing:
            builder.add_imputation("mean")

        if remove_outliers:
            builder.add_outlier_detection("iqr")

        if scale_features:
            builder.add_scaling(scaling_method)

        return builder.build()

    @staticmethod
    def for_time_series(
        name: str = "Time Series Pipeline"
    ) -> Dict[str, Any]:
        """
        Create configuration for time series tasks.

        Args:
            name: Configuration name

        Returns:
            Configuration dictionary
        """
        return (ConfigBuilder(name)
            .with_description("Time series preprocessing pipeline")
            .add_imputation("median")  # Median less sensitive to outliers
            .add_scaling("robust")     # Robust scaling for time series
            .build())

    @staticmethod
    def minimal(name: str = "Minimal Pipeline") -> Dict[str, Any]:
        """
        Create minimal preprocessing configuration.

        Args:
            name: Configuration name

        Returns:
            Configuration dictionary
        """
        return (ConfigBuilder(name)
            .with_description("Minimal preprocessing - essentials only")
            .add_imputation("mean")
            .add_scaling("standard")
            .build())
