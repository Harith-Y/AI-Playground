"""
Tests for preprocessing configuration validation system.
"""

import pytest
import pandas as pd
import numpy as np

from app.ml_engine.preprocessing.config_validator import (
    ValidationSeverity,
    ValidationIssue,
    ValidationResult,
    StepParameterValidator,
    PipelineValidator,
    SemanticValidator,
    ConfigValidator,
    validate_config,
    auto_fix_config,
)
from app.ml_engine.preprocessing.config import PreprocessingPreset


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_create_issue(self):
        """Test creating a validation issue."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="TEST_ERROR",
            message="Test error message",
            location="test.location"
        )

        assert issue.severity == ValidationSeverity.ERROR
        assert issue.code == "TEST_ERROR"
        assert issue.message == "Test error message"
        assert issue.location == "test.location"
        assert issue.suggestion is None
        assert issue.auto_fixable is False

    def test_issue_string_representation(self):
        """Test string representation of issue."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="TEST_WARNING",
            message="Test warning",
            location="test.location",
            suggestion="Fix it this way",
            auto_fixable=True
        )

        issue_str = str(issue)
        assert "WARNING" in issue_str
        assert "TEST_WARNING" in issue_str
        assert "Test warning" in issue_str
        assert "Fix it this way" in issue_str
        assert "Auto-fixable: Yes" in issue_str


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_empty_result(self):
        """Test empty validation result."""
        result = ValidationResult(valid=True)

        assert result.valid
        assert len(result.issues) == 0
        assert not result.has_errors()
        assert not result.has_warnings()

    def test_add_issue(self):
        """Test adding issues to result."""
        result = ValidationResult(valid=True)

        # Add warning - shouldn't affect validity
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="WARN",
            message="Warning",
            location="loc"
        ))
        assert result.valid
        assert result.has_warnings()
        assert not result.has_errors()

        # Add error - should set valid to False
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="ERR",
            message="Error",
            location="loc"
        ))
        assert not result.valid
        assert result.has_errors()

    def test_get_filtered_issues(self):
        """Test getting issues by severity."""
        result = ValidationResult(valid=True)

        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="E1",
            message="Error 1",
            location="loc"
        ))
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="W1",
            message="Warning 1",
            location="loc"
        ))
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.INFO,
            code="I1",
            message="Info 1",
            location="loc"
        ))

        assert len(result.get_errors()) == 1
        assert len(result.get_warnings()) == 1
        assert len(result.get_info()) == 1


class TestStepParameterValidator:
    """Tests for StepParameterValidator."""

    def test_validate_valid_step(self):
        """Test validation of valid step configuration."""
        step = {
            "class": "StandardScaler",
            "name": "scaler",
            "params": {
                "columns": ["age", "income"],
                "with_mean": True,
                "with_std": True
            }
        }

        issues = StepParameterValidator.validate_step(step, 0)

        # Should have no errors
        errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0

    def test_unknown_step_class(self):
        """Test validation of unknown step class."""
        step = {
            "class": "UnknownStep",
            "name": "unknown",
            "params": {}
        }

        issues = StepParameterValidator.validate_step(step, 0)

        # Should have warning about unknown class
        assert any(i.code == "UNKNOWN_STEP_CLASS" for i in issues)

    def test_unknown_parameter(self):
        """Test validation of unknown parameter."""
        step = {
            "class": "StandardScaler",
            "name": "scaler",
            "params": {
                "unknown_param": "value"
            }
        }

        issues = StepParameterValidator.validate_step(step, 0)

        # Should have warning about unknown parameter
        assert any(i.code == "UNKNOWN_PARAMETER" for i in issues)

    def test_invalid_parameter_type(self):
        """Test validation of invalid parameter type."""
        step = {
            "class": "StandardScaler",
            "name": "scaler",
            "params": {
                "with_mean": "yes"  # Should be bool
            }
        }

        issues = StepParameterValidator.validate_step(step, 0)

        # Should have error about invalid type
        assert any(i.code == "INVALID_PARAMETER_TYPE" for i in issues)

    def test_invalid_parameter_value(self):
        """Test validation of invalid parameter value."""
        step = {
            "class": "OneHotEncoder",
            "name": "encoder",
            "params": {
                "handle_unknown": "invalid_choice"  # Must be 'error' or 'ignore'
            }
        }

        issues = StepParameterValidator.validate_step(step, 0)

        # Should have error about invalid value
        assert any(i.code == "INVALID_PARAMETER_VALUE" for i in issues)

    def test_parameter_out_of_range(self):
        """Test validation of out-of-range parameter."""
        step = {
            "class": "IQROutlierDetector",
            "name": "outlier",
            "params": {
                "threshold": -1.0  # Must be >= 0
            }
        }

        issues = StepParameterValidator.validate_step(step, 0)

        # Should have error about out of range
        assert any(i.code == "PARAMETER_OUT_OF_RANGE" for i in issues)

    def test_standard_scaler_specific_validation(self):
        """Test StandardScaler-specific validation."""
        step = {
            "class": "StandardScaler",
            "name": "scaler",
            "params": {
                "with_mean": False,
                "with_std": False  # Both False is invalid
            }
        }

        issues = StepParameterValidator.validate_step(step, 0)

        # Should have error about configuration
        assert any(i.code == "INVALID_CONFIGURATION" for i in issues)

    def test_minmax_scaler_feature_range(self):
        """Test MinMaxScaler feature_range validation."""
        step = {
            "class": "MinMaxScaler",
            "name": "scaler",
            "params": {
                "feature_range": (1, 0)  # Min > Max is invalid
            }
        }

        issues = StepParameterValidator.validate_step(step, 0)

        # Should have error about invalid range
        assert any(i.code == "INVALID_RANGE" for i in issues)

    def test_empty_columns_list(self):
        """Test validation of empty columns list."""
        step = {
            "class": "StandardScaler",
            "name": "scaler",
            "params": {
                "columns": []  # Empty list
            }
        }

        issues = StepParameterValidator.validate_step(step, 0)

        # Should have warning about empty columns
        assert any(i.code == "EMPTY_COLUMNS_LIST" for i in issues)

    def test_duplicate_columns(self):
        """Test validation of duplicate columns."""
        step = {
            "class": "StandardScaler",
            "name": "scaler",
            "params": {
                "columns": ["age", "income", "age"]  # Duplicate 'age'
            }
        }

        issues = StepParameterValidator.validate_step(step, 0)

        # Should have warning about duplicates
        assert any(i.code == "DUPLICATE_COLUMNS" for i in issues)


class TestPipelineValidator:
    """Tests for PipelineValidator."""

    def test_empty_pipeline(self):
        """Test validation of empty pipeline."""
        config = {"steps": []}

        issues = PipelineValidator.validate_pipeline(config)

        # Should have warning about empty pipeline
        assert any(i.code == "EMPTY_PIPELINE" for i in issues)

    def test_duplicate_step_names(self):
        """Test validation of duplicate step names."""
        config = {
            "steps": [
                {"class": "StandardScaler", "name": "scaler", "params": {}},
                {"class": "MinMaxScaler", "name": "scaler", "params": {}}  # Duplicate name
            ]
        }

        issues = PipelineValidator.validate_pipeline(config)

        # Should have error about duplicate names
        assert any(i.code == "DUPLICATE_STEP_NAMES" for i in issues)

    def test_suboptimal_step_order(self):
        """Test validation of suboptimal step ordering."""
        config = {
            "steps": [
                {"class": "SMOTE", "name": "smote", "params": {}},  # Late step
                {"class": "MeanImputer", "name": "imputer", "params": {}}  # Early step
            ]
        }

        issues = PipelineValidator.validate_pipeline(config)

        # Should have warning about step order
        assert any(i.code == "SUBOPTIMAL_STEP_ORDER" for i in issues)

    def test_scaling_before_imputation(self):
        """Test warning for scaling before imputation."""
        config = {
            "steps": [
                {"class": "StandardScaler", "name": "scaler", "params": {}},
                {"class": "MeanImputer", "name": "imputer", "params": {}}
            ]
        }

        issues = PipelineValidator.validate_pipeline(config)

        # Should have warning about scaling before imputation
        assert any(i.code == "SCALING_BEFORE_IMPUTATION" for i in issues)

    def test_columns_after_transformation(self):
        """Test warning for explicit columns after column-changing step."""
        config = {
            "steps": [
                {"class": "OneHotEncoder", "name": "encoder", "params": {"columns": ["category"]}},
                {"class": "StandardScaler", "name": "scaler", "params": {"columns": ["age"]}}
            ]
        }

        issues = PipelineValidator.validate_pipeline(config)

        # Should have warning about columns after transformation
        assert any(i.code == "COLUMNS_AFTER_TRANSFORMATION" for i in issues)

    def test_multiple_scalers(self):
        """Test warning for multiple scalers."""
        config = {
            "steps": [
                {"class": "StandardScaler", "name": "scaler1", "params": {}},
                {"class": "MinMaxScaler", "name": "scaler2", "params": {}}
            ]
        }

        issues = PipelineValidator.validate_pipeline(config)

        # Should have warning about multiple scalers
        assert any(i.code == "MULTIPLE_SCALERS" for i in issues)

    def test_multiple_imputers(self):
        """Test info about multiple imputers."""
        config = {
            "steps": [
                {"class": "MeanImputer", "name": "imputer1", "params": {}},
                {"class": "MedianImputer", "name": "imputer2", "params": {}}
            ]
        }

        issues = PipelineValidator.validate_pipeline(config)

        # Should have info about multiple imputers
        assert any(i.code == "MULTIPLE_IMPUTERS" for i in issues)


class TestSemanticValidator:
    """Tests for SemanticValidator with actual data."""

    def test_missing_columns(self):
        """Test validation of missing columns."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'income': [50000, 60000, 70000]
        })

        step = {
            "class": "StandardScaler",
            "name": "scaler",
            "params": {
                "columns": ["age", "salary"]  # 'salary' doesn't exist
            }
        }

        issues = SemanticValidator._validate_step_with_data(step, df, 0)

        # Should have error about missing columns
        assert any(i.code == "MISSING_COLUMNS" for i in issues)

    def test_type_mismatch_numeric(self):
        """Test validation of type mismatch for numeric steps."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'name': ['Alice', 'Bob', 'Charlie']
        })

        step = {
            "class": "StandardScaler",
            "name": "scaler",
            "params": {
                "columns": ["age", "name"]  # 'name' is not numeric
            }
        }

        issues = SemanticValidator._validate_step_with_data(step, df, 0)

        # Should have error about type mismatch
        assert any(i.code == "TYPE_MISMATCH" for i in issues)

    def test_type_mismatch_categorical(self):
        """Test validation of type mismatch for categorical steps."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'category': ['A', 'B', 'C']
        })

        step = {
            "class": "OneHotEncoder",
            "name": "encoder",
            "params": {
                "columns": ["age", "category"]  # 'age' is numeric
            }
        }

        issues = SemanticValidator._validate_step_with_data(step, df, 0)

        # Should have warning about numeric column
        assert any(i.code == "TYPE_MISMATCH" for i in issues)

    def test_all_null_column(self):
        """Test warning for columns with all null values."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'missing': [None, None, None]
        })

        step = {
            "class": "StandardScaler",
            "name": "scaler",
            "params": {
                "columns": ["age", "missing"]
            }
        }

        issues = SemanticValidator._validate_step_with_data(step, df, 0)

        # Should have warning about all-null column
        assert any(i.code == "ALL_NULL_COLUMN" for i in issues)

    def test_constant_column(self):
        """Test warning for constant columns with scalers."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'constant': [5, 5, 5]
        })

        step = {
            "class": "StandardScaler",
            "name": "scaler",
            "params": {
                "columns": ["age", "constant"]
            }
        }

        issues = SemanticValidator._validate_step_with_data(step, df, 0)

        # Should have warning about constant column
        assert any(i.code == "CONSTANT_COLUMN" for i in issues)

    def test_high_cardinality(self):
        """Test warning for high-cardinality columns."""
        df = pd.DataFrame({
            'id': range(100),
            'category': [f'cat_{i}' for i in range(100)]  # 100 unique values
        })

        step = {
            "class": "OneHotEncoder",
            "name": "encoder",
            "params": {
                "columns": ["category"]
            }
        }

        issues = SemanticValidator._validate_step_with_data(step, df, 0)

        # Should have warning about high cardinality
        assert any(i.code == "HIGH_CARDINALITY" for i in issues)

    def test_auto_detection_numeric(self):
        """Test auto-detection of numeric columns."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'income': [50000, 60000, 70000],
            'name': ['Alice', 'Bob', 'Charlie']
        })

        step = {
            "class": "StandardScaler",
            "name": "scaler",
            "params": {
                "columns": None  # Auto-detect
            }
        }

        issues = SemanticValidator._validate_step_with_data(step, df, 0)

        # Should not have errors (will auto-select age and income)
        errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0


class TestConfigValidator:
    """Tests for the main ConfigValidator."""

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = {
            "name": "Test Pipeline",
            "version": "1.0.0",
            "steps": [
                {"class": "MeanImputer", "name": "imputer", "params": {}},
                {"class": "StandardScaler", "name": "scaler", "params": {}}
            ]
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert result.valid
        assert not result.has_errors()

    def test_validate_with_errors(self):
        """Test validation with configuration errors."""
        config = {
            "name": "Test Pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "class": "StandardScaler",
                    "name": "scaler",
                    "params": {
                        "with_mean": False,
                        "with_std": False  # Both False is invalid
                    }
                }
            ]
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert not result.valid
        assert result.has_errors()

    def test_validate_with_data(self):
        """Test validation with DataFrame for semantic checks."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'income': [50000, 60000, 70000]
        })

        config = {
            "name": "Test Pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "class": "StandardScaler",
                    "name": "scaler",
                    "params": {
                        "columns": ["age", "missing_column"]  # Column doesn't exist
                    }
                }
            ]
        }

        validator = ConfigValidator()
        result = validator.validate(config, df=df)

        assert not result.valid
        assert result.has_errors()
        assert any("MISSING_COLUMNS" in str(i) for i in result.get_errors())

    def test_validate_strict_mode(self):
        """Test validation in strict mode (warnings become errors)."""
        config = {
            "name": "Test Pipeline",
            "version": "1.0.0",
            "steps": [
                {"class": "StandardScaler", "name": "scaler1", "params": {}},
                {"class": "MinMaxScaler", "name": "scaler2", "params": {}}
            ]
        }

        validator = ConfigValidator()

        # Normal mode - should be valid (just warnings)
        result_normal = validator.validate(config, strict=False)
        assert result_normal.valid
        assert result_normal.has_warnings()

        # Strict mode - warnings become errors
        result_strict = validator.validate(config, strict=True)
        assert not result_strict.valid

    def test_validate_and_raise(self):
        """Test validate_and_raise method."""
        config = {
            "name": "Test Pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "class": "StandardScaler",
                    "name": "scaler",
                    "params": {
                        "with_mean": False,
                        "with_std": False
                    }
                }
            ]
        }

        validator = ConfigValidator()

        with pytest.raises(ValueError, match="Configuration validation failed"):
            validator.validate_and_raise(config)


class TestValidateConfigFunction:
    """Tests for validate_config convenience function."""

    def test_validate_config_function(self):
        """Test validate_config convenience function."""
        config = {
            "name": "Test",
            "version": "1.0.0",
            "steps": [
                {"class": "StandardScaler", "name": "scaler", "params": {}}
            ]
        }

        result = validate_config(config)

        assert isinstance(result, ValidationResult)
        assert result.valid


class TestAutoFixConfig:
    """Tests for auto_fix_config function."""

    def test_fix_duplicate_step_names(self):
        """Test auto-fixing duplicate step names."""
        config = {
            "name": "Test",
            "version": "1.0.0",
            "steps": [
                {"class": "StandardScaler", "name": "scaler", "params": {}},
                {"class": "MinMaxScaler", "name": "scaler", "params": {}}
            ]
        }

        fixed_config, fixes = auto_fix_config(config)

        # Should rename second scaler
        step_names = [s["name"] for s in fixed_config["steps"]]
        assert len(set(step_names)) == 2  # All unique
        assert "scaler_1" in step_names
        assert len(fixes) > 0

    def test_fix_empty_columns_list(self):
        """Test auto-fixing empty columns lists."""
        config = {
            "name": "Test",
            "version": "1.0.0",
            "steps": [
                {
                    "class": "StandardScaler",
                    "name": "scaler",
                    "params": {"columns": []}
                }
            ]
        }

        fixed_config, fixes = auto_fix_config(config)

        # Should change empty list to None
        assert fixed_config["steps"][0]["params"]["columns"] is None
        assert len(fixes) > 0

    def test_fix_duplicate_columns(self):
        """Test auto-fixing duplicate columns in list."""
        config = {
            "name": "Test",
            "version": "1.0.0",
            "steps": [
                {
                    "class": "StandardScaler",
                    "name": "scaler",
                    "params": {"columns": ["age", "income", "age"]}
                }
            ]
        }

        fixed_config, fixes = auto_fix_config(config)

        # Should remove duplicate 'age'
        columns = fixed_config["steps"][0]["params"]["columns"]
        assert len(columns) == 2
        assert columns == ["age", "income"]  # Order preserved
        assert len(fixes) > 0

    def test_fix_missing_step_names(self):
        """Test auto-fixing missing step names."""
        config = {
            "name": "Test",
            "version": "1.0.0",
            "steps": [
                {"class": "StandardScaler", "name": "", "params": {}}
            ]
        }

        fixed_config, fixes = auto_fix_config(config)

        # Should add a name
        assert fixed_config["steps"][0]["name"] != ""
        assert len(fixes) > 0


class TestIntegration:
    """Integration tests for the validation system."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        df = pd.DataFrame({
            'age': [25, 30, 35, 40],
            'income': [50000, 60000, 70000, 80000],
            'category': ['A', 'B', 'A', 'C']
        })

        config = {
            "name": "Full Test Pipeline",
            "version": "1.0.0",
            "steps": [
                {"class": "MeanImputer", "name": "imputer", "params": {"columns": ["age", "income"]}},
                {"class": "StandardScaler", "name": "scaler", "params": {"columns": ["age", "income"]}},
                {"class": "OneHotEncoder", "name": "encoder", "params": {"columns": ["category"]}}
            ]
        }

        result = validate_config(config, df=df)

        # Should be valid
        assert result.valid
        assert not result.has_errors()

    def test_validation_with_config_manager(self):
        """Test validation integration with ConfigManager."""
        from app.ml_engine.preprocessing.config import ConfigManager

        manager = ConfigManager(enable_validation=True)

        config = manager.create_config(
            name="Test",
            description="Test config"
        )

        # Add valid steps
        config["steps"] = [
            {"class": "MeanImputer", "name": "imputer", "params": {}},
            {"class": "StandardScaler", "name": "scaler", "params": {}}
        ]

        # Should validate without errors
        is_valid, errors = manager.validate_config(config)
        assert is_valid
        assert len(errors) == 0

    def test_build_pipeline_with_validation(self):
        """Test building pipeline with validation enabled."""
        from app.ml_engine.preprocessing.config import ConfigManager

        manager = ConfigManager(enable_validation=True)

        # Valid config
        config = {
            "name": "Test",
            "version": "1.0.0",
            "steps": [
                {"class": "MeanImputer", "name": "imputer", "params": {}}
            ]
        }

        # Should build successfully
        pipeline = manager.build_pipeline_from_config(config, validate=True)
        assert pipeline is not None

        # Invalid config
        invalid_config = {
            "name": "Test",
            "version": "1.0.0",
            "steps": [
                {
                    "class": "StandardScaler",
                    "name": "scaler",
                    "params": {"with_mean": False, "with_std": False}
                }
            ]
        }

        # Should raise error
        with pytest.raises(ValueError, match="validation failed"):
            manager.build_pipeline_from_config(invalid_config, validate=True)
