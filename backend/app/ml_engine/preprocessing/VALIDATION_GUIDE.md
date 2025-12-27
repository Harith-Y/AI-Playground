# Preprocessing Configuration Validation System

A comprehensive validation framework for preprocessing pipeline configurations with multi-level validation, detailed error reporting, and auto-repair capabilities.

## Overview

The validation system provides four levels of validation:
1. **Schema Validation** - Structure and required fields
2. **Step-Level Validation** - Parameter types, values, and constraints
3. **Pipeline-Level Validation** - Step ordering, compatibility, and anti-patterns
4. **Semantic Validation** - Data-aware validation (column existence, types)

## Quick Start

```python
from app.ml_engine.preprocessing.config_validator import validate_config
import pandas as pd

# Create your configuration
config = {
    "name": "My Pipeline",
    "version": "1.0.0",
    "steps": [
        {"class": "MeanImputer", "name": "imputer", "params": {"columns": ["age"]}},
        {"class": "StandardScaler", "name": "scaler", "params": {"columns": ["age"]}}
    ]
}

# Validate without data
result = validate_config(config)
if not result.valid:
    print(result)  # Pretty-printed validation report

# Validate with DataFrame for semantic checks
df = pd.DataFrame({"age": [25, 30, 35]})
result = validate_config(config, df=df)
```

## Validation Levels

### 1. Schema Validation

Validates basic structure and required fields:
- Required fields: `name`, `version`, `steps`
- Version format (semver: `x.y.z`)
- Step structure: `class`, `name`, `params`

### 2. Step-Level Validation

Validates individual preprocessing steps:

**Parameter Type Checking:**
```python
# ERROR: with_mean should be bool, not string
{"class": "StandardScaler", "params": {"with_mean": "yes"}}
```

**Parameter Value Validation:**
```python
# ERROR: handle_unknown must be 'error' or 'ignore'
{"class": "OneHotEncoder", "params": {"handle_unknown": "invalid"}}
```

**Range Validation:**
```python
# ERROR: threshold must be between 0 and 10
{"class": "IQROutlierDetector", "params": {"threshold": -1.0}}
```

**Step-Specific Rules:**
```python
# ERROR: StandardScaler needs at least one of with_mean or with_std
{"class": "StandardScaler", "params": {"with_mean": False, "with_std": False}}

# ERROR: MinMaxScaler feature_range min must be < max
{"class": "MinMaxScaler", "params": {"feature_range": (1, 0)}}
```

### 3. Pipeline-Level Validation

Validates step interactions and ordering:

**Duplicate Step Names:**
```python
# ERROR: Each step must have unique name
steps = [
    {"class": "StandardScaler", "name": "scaler", "params": {}},
    {"class": "MinMaxScaler", "name": "scaler", "params": {}}
]
```

**Suboptimal Step Ordering:**
```python
# WARNING: Imputation should typically come before sampling
steps = [
    {"class": "SMOTE", "name": "sampler", "params": {}},
    {"class": "MeanImputer", "name": "imputer", "params": {}}
]
```

**Anti-Patterns:**
```python
# WARNING: Multiple scalers detected
steps = [
    {"class": "StandardScaler", "name": "scaler1", "params": {}},
    {"class": "MinMaxScaler", "name": "scaler2", "params": {}}
]

# WARNING: Scaling before imputation may cause issues
steps = [
    {"class": "StandardScaler", "name": "scaler", "params": {}},
    {"class": "MeanImputer", "name": "imputer", "params": {}}
]
```

### 4. Semantic Validation

Validates against actual DataFrame:

**Missing Columns:**
```python
# ERROR: Column 'salary' not in DataFrame
df = pd.DataFrame({"age": [25, 30]})
step = {"class": "StandardScaler", "params": {"columns": ["age", "salary"]}}
```

**Type Mismatches:**
```python
# ERROR: StandardScaler requires numeric columns
df = pd.DataFrame({"name": ["Alice", "Bob"]})
step = {"class": "StandardScaler", "params": {"columns": ["name"]}}
```

**Constant Columns:**
```python
# WARNING: Constant column will fail scaling
df = pd.DataFrame({"age": [25, 25, 25]})
step = {"class": "StandardScaler", "params": {"columns": ["age"]}}
```

**High Cardinality:**
```python
# WARNING: High cardinality will create many features
df = pd.DataFrame({"id": range(1000)})
step = {"class": "OneHotEncoder", "params": {"columns": ["id"]}}
```

## Severity Levels

Validation issues have three severity levels:

- **ERROR**: Must be fixed, prevents execution
- **WARNING**: Should be fixed, might cause issues
- **INFO**: Informational, optimization suggestion

## Detailed Validation

For detailed validation results:

```python
from app.ml_engine.preprocessing.config_validator import ConfigValidator

validator = ConfigValidator()
result = validator.validate(config, df=df)

# Access specific issue types
errors = result.get_errors()
warnings = result.get_warnings()
info = result.get_info()

# Pretty print all issues
print(result)

# Example output:
"""
Validation FAILED
  Errors: 2
  Warnings: 1
  Info: 0

Issues:
[ERROR] MISSING_COLUMNS: Columns not found in DataFrame: ['salary']
  Location: steps[0].params.columns
  Suggestion: Available columns: ['age', 'income']

[ERROR] INVALID_PARAMETER_TYPE: Parameter 'with_mean' has wrong type
  Location: steps[1].params.with_mean
  Suggestion: Change parameter type to bool

[WARNING] MULTIPLE_SCALERS: Multiple scaling steps found at positions [1, 2]
  Location: steps
  Suggestion: Typically only one scaling method is needed
"""
```

## Auto-Repair

Automatically fix common issues:

```python
from app.ml_engine.preprocessing.config_validator import auto_fix_config

config = {
    "name": "Test",
    "version": "1.0.0",
    "steps": [
        {"class": "StandardScaler", "name": "scaler", "params": {}},
        {"class": "MinMaxScaler", "name": "scaler", "params": {}}  # Duplicate name
    ]
}

fixed_config, fixes_applied = auto_fix_config(config)
print(fixes_applied)
# ["Renamed duplicate step 'scaler' to 'scaler_1'"]
```

Auto-fixable issues:
- Duplicate step names (renamed with suffix)
- Empty columns lists (changed to `null`)
- Duplicate columns in lists (removed)
- Missing step names (auto-generated)

## Integration with ConfigManager

ConfigManager has built-in validation:

```python
from app.ml_engine.preprocessing.config import ConfigManager

# Enable validation (default)
manager = ConfigManager(enable_validation=True)

# Simple validation
is_valid, errors = manager.validate_config(config)

# Detailed validation
result = manager.validate_config_detailed(config, df=df)

# Validate and raise exception if errors
manager.validate_and_raise(config, df=df)

# Build pipeline with validation
pipeline = manager.build_pipeline_from_config(
    config,
    validate=True,  # Validate before building
    df=df  # Optional: semantic validation
)

# Auto-fix config
fixed_config, fixes = manager.auto_fix_config(config)
```

## Validation Error Codes

### Step-Level Codes
- `UNKNOWN_STEP_CLASS` - Step class not recognized
- `UNKNOWN_PARAMETER` - Parameter not valid for step
- `MISSING_REQUIRED_PARAMETER` - Required parameter missing
- `INVALID_PARAMETER_TYPE` - Wrong parameter type
- `INVALID_PARAMETER_VALUE` - Value not in allowed choices
- `PARAMETER_OUT_OF_RANGE` - Value outside min/max range
- `INVALID_CONFIGURATION` - Invalid step configuration
- `INVALID_RANGE` - Invalid range (min >= max)
- `EMPTY_COLUMNS_LIST` - Empty columns list
- `DUPLICATE_COLUMNS` - Duplicate column names

### Pipeline-Level Codes
- `EMPTY_PIPELINE` - Pipeline has no steps
- `DUPLICATE_STEP_NAMES` - Non-unique step names
- `SUBOPTIMAL_STEP_ORDER` - Steps in suboptimal order
- `SCALING_BEFORE_IMPUTATION` - Scaler before imputer
- `COLUMNS_AFTER_TRANSFORMATION` - Explicit columns after column-changing step
- `MULTIPLE_SCALERS` - Multiple scaling steps
- `MULTIPLE_IMPUTERS` - Multiple imputation steps
- `SCALING_ENCODED_FEATURES` - Scaling after one-hot encoding

### Semantic Codes
- `MISSING_COLUMNS` - Columns not in DataFrame
- `TYPE_MISMATCH` - Column type incompatible with step
- `ALL_NULL_COLUMN` - Column contains only nulls
- `CONSTANT_COLUMN` - Column has constant values
- `HIGH_CARDINALITY` - Too many unique values

## Strict Mode

In strict mode, warnings are treated as errors:

```python
# Normal mode: warnings don't prevent building
result = validator.validate(config, strict=False)
if result.valid:  # True even with warnings
    pipeline = build_pipeline(config)

# Strict mode: warnings become errors
result = validator.validate(config, strict=True)
if result.valid:  # False if any warnings
    pipeline = build_pipeline(config)
```

## Best Practices

1. **Always validate before deployment:**
   ```python
   manager.validate_and_raise(config)  # Raises on errors
   pipeline = manager.build_pipeline_from_config(config)
   ```

2. **Use semantic validation in development:**
   ```python
   # Validate against sample data
   sample_df = df.head(100)
   result = validate_config(config, df=sample_df)
   ```

3. **Review warnings carefully:**
   ```python
   result = validate_config(config)
   for warning in result.get_warnings():
       print(f"⚠️  {warning.message}")
       if warning.suggestion:
           print(f"   💡 {warning.suggestion}")
   ```

4. **Use auto-fix for quick corrections:**
   ```python
   config, fixes = auto_fix_config(config)
   if fixes:
       print(f"Applied {len(fixes)} fixes:")
       for fix in fixes:
           print(f"  - {fix}")
   ```

5. **Log validation results:**
   ```python
   result = manager.validate_config_detailed(config, df=df)
   logger.info(f"Validation: {len(result.issues)} issues found")
   for issue in result.issues:
       if issue.severity == ValidationSeverity.ERROR:
           logger.error(str(issue))
       elif issue.severity == ValidationSeverity.WARNING:
           logger.warning(str(issue))
   ```

## Supported Steps

The validator has built-in parameter specifications for:

**Scalers:**
- StandardScaler
- MinMaxScaler
- RobustScaler

**Encoders:**
- OneHotEncoder
- LabelEncoder
- OrdinalEncoder

**Imputers:**
- MeanImputer
- MedianImputer
- ModeImputer

**Outlier Detection:**
- IQROutlierDetector
- ZScoreOutlierDetector

**Sampling:**
- SMOTE
- BorderlineSMOTE
- ADASYN
- RandomUnderSampler

## Extending the Validator

To add validation for custom steps:

```python
from app.ml_engine.preprocessing.config_validator import StepParameterValidator

# Add parameter spec for custom step
StepParameterValidator.PARAMETER_SPECS["CustomStep"] = {
    "custom_param": {
        "type": str,
        "optional": False,
        "choices": ["option1", "option2"]
    },
    "threshold": {
        "type": (int, float),
        "optional": True,
        "default": 1.0,
        "min": 0,
        "max": 10
    }
}
```

## Examples

### Example 1: Complete Validation Workflow

```python
from app.ml_engine.preprocessing.config import ConfigManager
import pandas as pd

# Load data
df = pd.read_csv("data.csv")

# Create manager with validation
manager = ConfigManager(enable_validation=True)

# Create config
config = manager.create_config("My Pipeline")
config["steps"] = [
    {"class": "MeanImputer", "name": "imputer", "params": {"columns": ["age", "income"]}},
    {"class": "StandardScaler", "name": "scaler", "params": {"columns": ["age", "income"]}},
    {"class": "OneHotEncoder", "name": "encoder", "params": {"columns": ["category"]}}
]

# Validate with data
result = manager.validate_config_detailed(config, df=df)

if result.has_errors():
    print("❌ Validation failed:")
    for error in result.get_errors():
        print(f"  - {error.message}")

    # Try auto-fix
    fixed_config, fixes = manager.auto_fix_config(config)
    if fixes:
        print("\n🔧 Applied fixes:")
        for fix in fixes:
            print(f"  - {fix}")
        config = fixed_config
else:
    print("✅ Validation passed!")

if result.has_warnings():
    print("\n⚠️  Warnings:")
    for warning in result.get_warnings():
        print(f"  - {warning.message}")

# Build pipeline
pipeline = manager.build_pipeline_from_config(config, validate=True, df=df)
```

### Example 2: CI/CD Integration

```python
def validate_pipeline_config(config_path, data_path):
    """Validate pipeline config for CI/CD."""
    from app.ml_engine.preprocessing.config import ConfigManager
    import pandas as pd

    manager = ConfigManager(enable_validation=True)
    config = manager.load_config(config_path)
    df = pd.read_csv(data_path, nrows=1000)  # Sample for validation

    result = manager.validate_config_detailed(config, df=df, strict=True)

    if not result.valid:
        print(f"::error::Validation failed with {len(result.get_errors())} errors")
        for error in result.get_errors():
            print(f"::error file={config_path}::{error.message}")
        return False

    print(f"::notice::Validation passed with {len(result.get_warnings())} warnings")
    return True

if __name__ == "__main__":
    import sys
    success = validate_pipeline_config("config.json", "sample_data.csv")
    sys.exit(0 if success else 1)
```

## Summary

The validation system provides:

✅ **Multi-level validation** - Schema, step, pipeline, and semantic
✅ **Detailed error reporting** - Clear messages with suggestions
✅ **Severity levels** - ERROR, WARNING, INFO
✅ **Auto-repair** - Automatic fixes for common issues
✅ **Data-aware** - Validates against actual DataFrames
✅ **Integrated** - Works seamlessly with ConfigManager
✅ **Extensible** - Easy to add custom step validators
✅ **Well-tested** - 50+ comprehensive tests

Use validation early and often to catch configuration errors before they cause runtime failures!
