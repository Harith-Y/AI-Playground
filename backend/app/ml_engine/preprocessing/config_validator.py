"""
Comprehensive validation system for preprocessing configurations.

Provides multi-level validation:
- Schema validation (structure, types, required fields)
- Step-level validation (parameters, constraints)
- Pipeline-level validation (step order, compatibility, dependencies)
- Semantic validation (column existence, type compatibility)
- Auto-repair capabilities for common issues
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import pandas as pd

from app.utils.logger import get_logger

logger = get_logger("config_validator")


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"  # Must be fixed, prevents execution
    WARNING = "warning"  # Should be fixed, might cause issues
    INFO = "info"  # Informational, optimization suggestion


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    severity: ValidationSeverity
    code: str
    message: str
    location: str  # Where the issue is (e.g., "step[0].params.columns")
    suggestion: Optional[str] = None  # How to fix it
    auto_fixable: bool = False

    def __str__(self) -> str:
        """String representation of the issue."""
        parts = [f"[{self.severity.value.upper()}] {self.code}: {self.message}"]
        parts.append(f"  Location: {self.location}")
        if self.suggestion:
            parts.append(f"  Suggestion: {self.suggestion}")
        if self.auto_fixable:
            parts.append(f"  Auto-fixable: Yes")
        return "\n".join(parts)


@dataclass
class ValidationResult:
    """Results from validation with all issues found."""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)

    def get_errors(self) -> List[ValidationIssue]:
        """Get all error issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    def get_warnings(self) -> List[ValidationIssue]:
        """Get all warning issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def get_info(self) -> List[ValidationIssue]:
        """Get all info issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.INFO]

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the result."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.valid = False

    def __str__(self) -> str:
        """String representation of validation results."""
        if self.valid and not self.issues:
            return "Validation passed: No issues found"

        parts = [f"Validation {'passed' if self.valid else 'FAILED'}"]
        parts.append(f"  Errors: {len(self.get_errors())}")
        parts.append(f"  Warnings: {len(self.get_warnings())}")
        parts.append(f"  Info: {len(self.get_info())}")

        if self.issues:
            parts.append("\nIssues:")
            for issue in self.issues:
                parts.append(str(issue))
                parts.append("")

        return "\n".join(parts)


class StepParameterValidator:
    """Validates parameters for specific preprocessing step types."""

    # Parameter specifications for each step class
    PARAMETER_SPECS = {
        "StandardScaler": {
            "columns": {"type": (list, type(None)), "optional": True},
            "with_mean": {"type": bool, "optional": True, "default": True},
            "with_std": {"type": bool, "optional": True, "default": True},
        },
        "MinMaxScaler": {
            "columns": {"type": (list, type(None)), "optional": True},
            "feature_range": {"type": tuple, "optional": True, "default": (0, 1)},
        },
        "RobustScaler": {
            "columns": {"type": (list, type(None)), "optional": True},
            "with_centering": {"type": bool, "optional": True, "default": True},
            "with_scaling": {"type": bool, "optional": True, "default": True},
        },
        "OneHotEncoder": {
            "columns": {"type": (list, type(None)), "optional": True},
            "drop_first": {"type": bool, "optional": True, "default": False},
            "handle_unknown": {"type": str, "optional": True, "default": "error",
                              "choices": ["error", "ignore"]},
        },
        "LabelEncoder": {
            "columns": {"type": (list, type(None)), "optional": True},
        },
        "OrdinalEncoder": {
            "columns": {"type": (list, type(None)), "optional": True},
            "categories": {"type": (dict, type(None)), "optional": True},
        },
        "MeanImputer": {
            "columns": {"type": (list, type(None)), "optional": True},
        },
        "MedianImputer": {
            "columns": {"type": (list, type(None)), "optional": True},
        },
        "ModeImputer": {
            "columns": {"type": (list, type(None)), "optional": True},
        },
        "IQROutlierDetector": {
            "columns": {"type": (list, type(None)), "optional": True},
            "threshold": {"type": (int, float), "optional": True, "default": 1.5,
                         "min": 0, "max": 10},
        },
        "ZScoreOutlierDetector": {
            "columns": {"type": (list, type(None)), "optional": True},
            "threshold": {"type": (int, float), "optional": True, "default": 3.0,
                         "min": 0, "max": 10},
        },
        "SMOTE": {
            "sampling_strategy": {"type": (str, float, dict), "optional": True, "default": "auto"},
            "k_neighbors": {"type": int, "optional": True, "default": 5, "min": 1},
        },
        "BorderlineSMOTE": {
            "sampling_strategy": {"type": (str, float, dict), "optional": True, "default": "auto"},
            "k_neighbors": {"type": int, "optional": True, "default": 5, "min": 1},
        },
        "ADASYN": {
            "sampling_strategy": {"type": (str, float, dict), "optional": True, "default": "auto"},
            "n_neighbors": {"type": int, "optional": True, "default": 5, "min": 1},
        },
        "RandomUnderSampler": {
            "sampling_strategy": {"type": (str, float, dict), "optional": True, "default": "auto"},
        },
    }

    @classmethod
    def validate_step(cls, step: Dict[str, Any], step_index: int) -> List[ValidationIssue]:
        """
        Validate a single preprocessing step.

        Args:
            step: Step configuration dictionary
            step_index: Index of the step in the pipeline

        Returns:
            List of validation issues
        """
        issues = []
        location_prefix = f"steps[{step_index}]"

        step_class = step.get("class", "")
        step_name = step.get("name", "")
        params = step.get("params", {})

        # Check if step class is known
        if step_class not in cls.PARAMETER_SPECS:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="UNKNOWN_STEP_CLASS",
                message=f"Unknown step class '{step_class}'",
                location=f"{location_prefix}.class",
                suggestion="Verify the step class name is correct"
            ))
            return issues  # Can't validate params for unknown class

        # Validate parameters
        param_spec = cls.PARAMETER_SPECS[step_class]

        # Check for unknown parameters
        for param_name in params:
            if param_name not in param_spec:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="UNKNOWN_PARAMETER",
                    message=f"Unknown parameter '{param_name}' for {step_class}",
                    location=f"{location_prefix}.params.{param_name}",
                    suggestion=f"Valid parameters: {', '.join(param_spec.keys())}"
                ))

        # Validate each parameter
        for param_name, spec in param_spec.items():
            param_value = params.get(param_name)
            param_location = f"{location_prefix}.params.{param_name}"

            # Check if required parameter is missing
            if not spec.get("optional", False) and param_value is None:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_REQUIRED_PARAMETER",
                    message=f"Required parameter '{param_name}' missing for {step_class}",
                    location=param_location,
                    suggestion=f"Add '{param_name}' parameter"
                ))
                continue

            # Skip validation if parameter is not provided and is optional
            if param_value is None:
                continue

            # Validate type
            expected_types = spec["type"]
            if not isinstance(expected_types, tuple):
                expected_types = (expected_types,)

            if not isinstance(param_value, expected_types):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_PARAMETER_TYPE",
                    message=f"Parameter '{param_name}' has wrong type. Expected {expected_types}, got {type(param_value)}",
                    location=param_location,
                    suggestion=f"Change parameter type to one of {expected_types}"
                ))
                continue

            # Validate choices (if specified)
            if "choices" in spec and param_value not in spec["choices"]:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_PARAMETER_VALUE",
                    message=f"Invalid value '{param_value}' for parameter '{param_name}'",
                    location=param_location,
                    suggestion=f"Valid choices: {spec['choices']}"
                ))

            # Validate min/max (if specified)
            if "min" in spec and isinstance(param_value, (int, float)):
                if param_value < spec["min"]:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="PARAMETER_OUT_OF_RANGE",
                        message=f"Parameter '{param_name}' value {param_value} is below minimum {spec['min']}",
                        location=param_location,
                        suggestion=f"Use value >= {spec['min']}"
                    ))

            if "max" in spec and isinstance(param_value, (int, float)):
                if param_value > spec["max"]:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="PARAMETER_OUT_OF_RANGE",
                        message=f"Parameter '{param_name}' value {param_value} exceeds maximum {spec['max']}",
                        location=param_location,
                        suggestion=f"Use value <= {spec['max']}"
                    ))

        # Step-specific validations
        issues.extend(cls._validate_step_specific(step_class, params, location_prefix))

        return issues

    @classmethod
    def _validate_step_specific(cls, step_class: str, params: Dict, location: str) -> List[ValidationIssue]:
        """Step-specific validation logic."""
        issues = []

        # StandardScaler: at least one of with_mean or with_std must be True
        if step_class == "StandardScaler":
            with_mean = params.get("with_mean", True)
            with_std = params.get("with_std", True)
            if not with_mean and not with_std:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_CONFIGURATION",
                    message="StandardScaler requires at least one of with_mean or with_std to be True",
                    location=f"{location}.params",
                    suggestion="Set with_mean=True or with_std=True"
                ))

        # MinMaxScaler: feature_range must be (min, max) with min < max
        if step_class == "MinMaxScaler":
            feature_range = params.get("feature_range", (0, 1))
            if isinstance(feature_range, tuple) and len(feature_range) == 2:
                if feature_range[0] >= feature_range[1]:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="INVALID_RANGE",
                        message=f"feature_range minimum ({feature_range[0]}) must be less than maximum ({feature_range[1]})",
                        location=f"{location}.params.feature_range",
                        suggestion="Use a valid range like (0, 1)"
                    ))

        # Validate columns parameter (if present and not None)
        columns = params.get("columns")
        if columns is not None and isinstance(columns, list):
            if len(columns) == 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="EMPTY_COLUMNS_LIST",
                    message="Empty columns list - step will process no columns",
                    location=f"{location}.params.columns",
                    suggestion="Either specify columns or set to null for auto-detection",
                    auto_fixable=True
                ))

            # Check for duplicate columns
            if len(columns) != len(set(columns)):
                duplicates = [col for col in columns if columns.count(col) > 1]
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="DUPLICATE_COLUMNS",
                    message=f"Duplicate columns in list: {set(duplicates)}",
                    location=f"{location}.params.columns",
                    suggestion="Remove duplicate column names",
                    auto_fixable=True
                ))

        return issues


class PipelineValidator:
    """Validates pipeline-level configurations and step interactions."""

    # Steps that require numeric data
    NUMERIC_STEPS = {
        "StandardScaler", "MinMaxScaler", "RobustScaler",
        "MeanImputer", "MedianImputer",
        "IQROutlierDetector", "ZScoreOutlierDetector"
    }

    # Steps that require categorical data
    CATEGORICAL_STEPS = {
        "OneHotEncoder", "LabelEncoder", "OrdinalEncoder", "ModeImputer"
    }

    # Steps that modify the number of columns
    COLUMN_CHANGING_STEPS = {
        "OneHotEncoder"  # Creates new columns
    }

    # Steps that should typically come early
    EARLY_STEPS = {
        "MeanImputer", "MedianImputer", "ModeImputer"  # Imputation should be early
    }

    # Steps that should typically come late
    LATE_STEPS = {
        "SMOTE", "BorderlineSMOTE", "ADASYN",  # Sampling should be late
        "RandomUnderSampler", "NearMissUnderSampler"
    }

    @classmethod
    def validate_pipeline(cls, config: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate pipeline-level configuration.

        Args:
            config: Complete pipeline configuration

        Returns:
            List of validation issues
        """
        issues = []
        steps = config.get("steps", [])

        if not steps:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="EMPTY_PIPELINE",
                message="Pipeline has no steps",
                location="steps",
                suggestion="Add preprocessing steps to the pipeline"
            ))
            return issues

        # Check for duplicate step names
        step_names = [step.get("name", "") for step in steps]
        duplicates = [name for name in step_names if step_names.count(name) > 1]
        if duplicates:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="DUPLICATE_STEP_NAMES",
                message=f"Duplicate step names found: {set(duplicates)}",
                location="steps",
                suggestion="Each step must have a unique name",
                auto_fixable=True
            ))

        # Validate step ordering
        issues.extend(cls._validate_step_ordering(steps))

        # Validate column dependencies
        issues.extend(cls._validate_column_dependencies(steps))

        # Check for common anti-patterns
        issues.extend(cls._check_antipatterns(steps))

        return issues

    @classmethod
    def _validate_step_ordering(cls, steps: List[Dict]) -> List[ValidationIssue]:
        """Validate that steps are in a reasonable order."""
        issues = []

        # Find positions of early and late steps
        early_positions = []
        late_positions = []

        for idx, step in enumerate(steps):
            step_class = step.get("class", "")
            if step_class in cls.EARLY_STEPS:
                early_positions.append((idx, step_class))
            if step_class in cls.LATE_STEPS:
                late_positions.append((idx, step_class))

        # Check if any late step comes before early step
        for late_idx, late_class in late_positions:
            for early_idx, early_class in early_positions:
                if late_idx < early_idx:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="SUBOPTIMAL_STEP_ORDER",
                        message=f"{late_class} (step {late_idx}) should typically come after {early_class} (step {early_idx})",
                        location=f"steps[{late_idx}]",
                        suggestion=f"Consider moving {early_class} before {late_class}"
                    ))

        # Check for scaling before imputation
        for idx in range(len(steps) - 1):
            current_class = steps[idx].get("class", "")
            next_class = steps[idx + 1].get("class", "")

            if current_class in {"StandardScaler", "MinMaxScaler", "RobustScaler"}:
                if next_class in {"MeanImputer", "MedianImputer"}:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="SCALING_BEFORE_IMPUTATION",
                        message=f"Scaling (step {idx}) before imputation (step {idx+1}) may produce unexpected results",
                        location=f"steps[{idx}]",
                        suggestion="Consider imputing missing values before scaling"
                    ))

        return issues

    @classmethod
    def _validate_column_dependencies(cls, steps: List[Dict]) -> List[ValidationIssue]:
        """Validate that column dependencies are satisfied."""
        issues = []

        # Track which columns are explicitly specified
        for idx, step in enumerate(steps):
            params = step.get("params", {})
            columns = params.get("columns")
            step_class = step.get("class", "")

            # Warn if explicit columns are used after column-changing step
            if columns is not None and isinstance(columns, list):
                # Check if any previous step changes columns
                for prev_idx in range(idx):
                    prev_class = steps[prev_idx].get("class", "")
                    if prev_class in cls.COLUMN_CHANGING_STEPS:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            code="COLUMNS_AFTER_TRANSFORMATION",
                            message=f"Step {idx} specifies explicit columns after {prev_class} which modifies columns",
                            location=f"steps[{idx}].params.columns",
                            suggestion="Consider using auto-detection (columns=null) or verify column names"
                        ))
                        break

        return issues

    @classmethod
    def _check_antipatterns(cls, steps: List[Dict]) -> List[ValidationIssue]:
        """Check for common configuration anti-patterns."""
        issues = []

        # Check for multiple scalers
        scaler_indices = []
        for idx, step in enumerate(steps):
            if step.get("class", "") in {"StandardScaler", "MinMaxScaler", "RobustScaler"}:
                scaler_indices.append(idx)

        if len(scaler_indices) > 1:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="MULTIPLE_SCALERS",
                message=f"Multiple scaling steps found at positions {scaler_indices}",
                location="steps",
                suggestion="Typically only one scaling method is needed"
            ))

        # Check for multiple imputers on same columns
        imputer_steps = [s for s in steps if s.get("class", "").endswith("Imputer")]
        if len(imputer_steps) > 1:
            # Check if they have overlapping columns
            has_overlap = False
            for i in range(len(imputer_steps)):
                cols_i = imputer_steps[i].get("params", {}).get("columns")
                if cols_i is None:  # Auto-detect means potential overlap
                    has_overlap = True
                    break
                for j in range(i + 1, len(imputer_steps)):
                    cols_j = imputer_steps[j].get("params", {}).get("columns")
                    if cols_j is None or (isinstance(cols_i, list) and isinstance(cols_j, list) and set(cols_i) & set(cols_j)):
                        has_overlap = True
                        break

            if has_overlap:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    code="MULTIPLE_IMPUTERS",
                    message="Multiple imputation steps found - verify they target different columns",
                    location="steps",
                    suggestion="Consider using a single imputation strategy or ensure non-overlapping columns"
                ))

        # Check for encoding before scaling (might be intentional but worth noting)
        for idx in range(len(steps) - 1):
            current_class = steps[idx].get("class", "")
            if current_class in cls.CATEGORICAL_STEPS:
                for future_idx in range(idx + 1, len(steps)):
                    future_class = steps[future_idx].get("class", "")
                    if future_class in {"StandardScaler", "MinMaxScaler", "RobustScaler"}:
                        # This is actually OK if encoding is numeric (label/ordinal)
                        if current_class in {"LabelEncoder", "OrdinalEncoder"}:
                            continue
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            code="SCALING_ENCODED_FEATURES",
                            message=f"Scaling after {current_class} - verify this is intended",
                            location=f"steps[{future_idx}]",
                            suggestion=f"OneHot encoded features typically don't need scaling"
                        ))
                        break

        return issues


class SemanticValidator:
    """Validates configuration against actual data semantics."""

    @classmethod
    def validate_with_data(
        cls,
        config: Dict[str, Any],
        df: pd.DataFrame
    ) -> List[ValidationIssue]:
        """
        Validate configuration against actual DataFrame.

        Args:
            config: Pipeline configuration
            df: DataFrame to validate against

        Returns:
            List of validation issues
        """
        issues = []
        steps = config.get("steps", [])

        for idx, step in enumerate(steps):
            issues.extend(cls._validate_step_with_data(step, df, idx))

        return issues

    @classmethod
    def _validate_step_with_data(
        cls,
        step: Dict[str, Any],
        df: pd.DataFrame,
        step_index: int
    ) -> List[ValidationIssue]:
        """Validate a single step against DataFrame."""
        issues = []
        location_prefix = f"steps[{step_index}]"

        step_class = step.get("class", "")
        params = step.get("params", {})
        columns = params.get("columns")

        # Check if specified columns exist
        if columns is not None and isinstance(columns, list):
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_COLUMNS",
                    message=f"Columns not found in DataFrame: {missing_cols}",
                    location=f"{location_prefix}.params.columns",
                    suggestion=f"Available columns: {list(df.columns)}"
                ))
                return issues  # Can't validate further without columns

        # Get actual columns that will be processed
        if columns is None:
            # Auto-detection based on step type
            if step_class in PipelineValidator.NUMERIC_STEPS:
                actual_columns = df.select_dtypes(include=['number']).columns.tolist()
            elif step_class in PipelineValidator.CATEGORICAL_STEPS:
                actual_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            else:
                actual_columns = df.columns.tolist()
        else:
            actual_columns = columns

        # Validate column types match step requirements
        if actual_columns:
            if step_class in PipelineValidator.NUMERIC_STEPS:
                non_numeric = [col for col in actual_columns
                             if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
                if non_numeric:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="TYPE_MISMATCH",
                        message=f"{step_class} requires numeric columns, but these are non-numeric: {non_numeric}",
                        location=f"{location_prefix}.params.columns",
                        suggestion="Remove non-numeric columns or use appropriate preprocessing"
                    ))

            elif step_class in PipelineValidator.CATEGORICAL_STEPS:
                numeric = [col for col in actual_columns
                          if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
                if numeric:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="TYPE_MISMATCH",
                        message=f"{step_class} typically works with categorical columns, but these are numeric: {numeric}",
                        location=f"{location_prefix}.params.columns",
                        suggestion="Verify these numeric columns should be encoded"
                    ))

        # Check for columns with all nulls
        if actual_columns:
            all_null_cols = [col for col in actual_columns
                           if col in df.columns and df[col].isnull().all()]
            if all_null_cols:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="ALL_NULL_COLUMN",
                    message=f"Columns contain only null values: {all_null_cols}",
                    location=f"{location_prefix}.params.columns",
                    suggestion="Consider removing these columns or using a different strategy"
                ))

        # Check for columns with constant values (for scalers)
        if step_class in {"StandardScaler", "MinMaxScaler", "RobustScaler"}:
            constant_cols = [col for col in actual_columns
                           if col in df.columns and df[col].nunique() <= 1]
            if constant_cols:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="CONSTANT_COLUMN",
                    message=f"Columns have constant values (scaling will fail or produce NaN): {constant_cols}",
                    location=f"{location_prefix}.params.columns",
                    suggestion="Remove constant columns before scaling"
                ))

        # Check cardinality for encoders
        if step_class == "OneHotEncoder":
            high_cardinality = []
            for col in actual_columns:
                if col in df.columns:
                    n_unique = df[col].nunique()
                    if n_unique > 50:  # Arbitrary threshold
                        high_cardinality.append((col, n_unique))

            if high_cardinality:
                cols_str = ", ".join([f"{col} ({n})" for col, n in high_cardinality])
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="HIGH_CARDINALITY",
                    message=f"High cardinality columns will create many features: {cols_str}",
                    location=f"{location_prefix}.params.columns",
                    suggestion="Consider using LabelEncoder or feature hashing for high-cardinality columns"
                ))

        return issues


class ConfigValidator:
    """Main validator that orchestrates all validation levels."""

    def __init__(self):
        """Initialize the validator."""
        self.step_validator = StepParameterValidator()
        self.pipeline_validator = PipelineValidator()
        self.semantic_validator = SemanticValidator()

    def validate(
        self,
        config: Dict[str, Any],
        df: Optional[pd.DataFrame] = None,
        strict: bool = False
    ) -> ValidationResult:
        """
        Perform comprehensive validation of preprocessing configuration.

        Args:
            config: Configuration to validate
            df: Optional DataFrame for semantic validation
            strict: If True, treat warnings as errors

        Returns:
            ValidationResult with all issues found
        """
        result = ValidationResult(valid=True)

        logger.info("Starting configuration validation")

        # Schema validation (already done by ConfigurationSchema, but we can add more)
        steps = config.get("steps", [])

        # Step-level validation
        logger.debug(f"Validating {len(steps)} steps")
        for idx, step in enumerate(steps):
            step_issues = self.step_validator.validate_step(step, idx)
            for issue in step_issues:
                result.add_issue(issue)

        # Pipeline-level validation
        logger.debug("Validating pipeline structure")
        pipeline_issues = self.pipeline_validator.validate_pipeline(config)
        for issue in pipeline_issues:
            result.add_issue(issue)

        # Semantic validation (if data provided)
        if df is not None:
            logger.debug("Validating against DataFrame")
            semantic_issues = self.semantic_validator.validate_with_data(config, df)
            for issue in semantic_issues:
                result.add_issue(issue)

        # Apply strict mode
        if strict:
            for issue in result.issues:
                if issue.severity == ValidationSeverity.WARNING:
                    issue.severity = ValidationSeverity.ERROR
                    result.valid = False

        logger.info(f"Validation complete: {len(result.issues)} issues found")

        return result

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
        result = self.validate(config, df)

        if not result.valid:
            error_messages = [str(issue) for issue in result.get_errors()]
            raise ValueError(
                f"Configuration validation failed with {len(result.get_errors())} errors:\n" +
                "\n\n".join(error_messages)
            )


def validate_config(
    config: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
    strict: bool = False
) -> ValidationResult:
    """
    Convenience function to validate a preprocessing configuration.

    Args:
        config: Configuration to validate
        df: Optional DataFrame for semantic validation
        strict: If True, treat warnings as errors

    Returns:
        ValidationResult with all issues found

    Example:
        >>> result = validate_config(config, df)
        >>> if not result.valid:
        ...     print(result)
    """
    validator = ConfigValidator()
    return validator.validate(config, df, strict)


def auto_fix_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Attempt to automatically fix common configuration issues.

    Args:
        config: Configuration to fix

    Returns:
        Tuple of (fixed_config, list_of_fixes_applied)

    Example:
        >>> fixed_config, fixes = auto_fix_config(config)
        >>> print(f"Applied {len(fixes)} fixes")
    """
    import copy
    fixed_config = copy.deepcopy(config)
    fixes_applied = []

    steps = fixed_config.get("steps", [])

    # Fix duplicate step names
    step_names = [step.get("name", "") for step in steps]
    name_counts = {}
    for idx, step in enumerate(steps):
        name = step.get("name", f"step_{idx}")
        if name in name_counts:
            name_counts[name] += 1
            new_name = f"{name}_{name_counts[name]}"
            step["name"] = new_name
            fixes_applied.append(f"Renamed duplicate step '{name}' to '{new_name}'")
        else:
            name_counts[name] = 0
            if not step.get("name"):
                step["name"] = name
                fixes_applied.append(f"Added missing name '{name}' to step {idx}")

    # Fix empty columns lists
    for idx, step in enumerate(steps):
        params = step.get("params", {})
        if "columns" in params and isinstance(params["columns"], list) and len(params["columns"]) == 0:
            params["columns"] = None
            fixes_applied.append(f"Changed empty columns list to null for auto-detection in step {idx}")

    # Remove duplicate columns
    for idx, step in enumerate(steps):
        params = step.get("params", {})
        if "columns" in params and isinstance(params["columns"], list):
            original_len = len(params["columns"])
            params["columns"] = list(dict.fromkeys(params["columns"]))  # Preserves order
            if len(params["columns"]) < original_len:
                fixes_applied.append(f"Removed duplicate columns in step {idx}")

    return fixed_config, fixes_applied
