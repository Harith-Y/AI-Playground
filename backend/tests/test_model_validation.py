"""
Unit tests for model parameter validation.

Tests the validation system, parameter specs, and model schemas.
"""

import pytest
from app.ml_engine.models.validation import (
    ParameterType,
    ParameterSpec,
    ModelParameterSchema,
    ValidationError,
    validate_model_config,
    get_model_defaults,
    get_parameter_info,
    get_available_models_with_schemas,
)
from app.ml_engine.models import ModelConfig, create_model


class TestParameterSpec:
    """Test ParameterSpec class."""
    
    def test_create_parameter_spec(self):
        """Test creating a parameter specification."""
        spec = ParameterSpec(
            name='n_estimators',
            param_type=ParameterType.INT,
            required=False,
            default=100,
            min_value=1,
            description='Number of trees'
        )
        
        assert spec.name == 'n_estimators'
        assert ParameterType.INT in spec.param_type
        assert spec.default == 100
        assert spec.min_value == 1
    
    def test_validate_valid_int(self):
        """Test validating a valid integer."""
        spec = ParameterSpec('n_estimators', ParameterType.INT, min_value=1)
        
        is_valid, error = spec.validate(100)
        
        assert is_valid
        assert error is None
    
    def test_validate_invalid_type(self):
        """Test validating invalid type."""
        spec = ParameterSpec('n_estimators', ParameterType.INT)
        
        is_valid, error = spec.validate('100')  # String instead of int
        
        assert not is_valid
        assert 'must be of type int' in error
    
    def test_validate_out_of_range(self):
        """Test validating out of range value."""
        spec = ParameterSpec('n_estimators', ParameterType.INT, min_value=1, max_value=1000)
        
        is_valid, error = spec.validate(-10)
        
        assert not is_valid
        assert 'must be >=' in error
    
    def test_validate_allowed_values(self):
        """Test validating against allowed values."""
        spec = ParameterSpec(
            'criterion',
            ParameterType.STRING,
            allowed_values=['gini', 'entropy']
        )
        
        # Valid value
        is_valid, error = spec.validate('gini')
        assert is_valid
        
        # Invalid value
        is_valid, error = spec.validate('invalid')
        assert not is_valid
        assert 'must be one of' in error
    
    def test_validate_multiple_types(self):
        """Test validating parameter with multiple allowed types."""
        spec = ParameterSpec(
            'max_depth',
            [ParameterType.INT, ParameterType.NONE],
            min_value=1
        )
        
        # Valid int
        is_valid, error = spec.validate(10)
        assert is_valid
        
        # Valid None
        is_valid, error = spec.validate(None)
        assert is_valid
    
    def test_validate_float_accepts_int(self):
        """Test that float type accepts integers."""
        spec = ParameterSpec('alpha', ParameterType.FLOAT, min_value=0.0)
        
        is_valid, error = spec.validate(1)  # Int value
        
        assert is_valid
    
    def test_validate_length(self):
        """Test validating length constraints."""
        spec = ParameterSpec(
            'feature_names',
            ParameterType.LIST,
            min_length=1,
            max_length=10
        )
        
        # Valid length
        is_valid, error = spec.validate(['a', 'b', 'c'])
        assert is_valid
        
        # Too short
        is_valid, error = spec.validate([])
        assert not is_valid
        assert 'length >=' in error


class TestModelParameterSchema:
    """Test ModelParameterSchema class."""
    
    def test_create_schema(self):
        """Test creating a model parameter schema."""
        params = [
            ParameterSpec('n_estimators', ParameterType.INT, default=100, min_value=1),
            ParameterSpec('max_depth', [ParameterType.INT, ParameterType.NONE], default=None)
        ]
        
        schema = ModelParameterSchema('test_model', params)
        
        assert schema.model_id == 'test_model'
        assert len(schema.parameters) == 2
        assert 'n_estimators' in schema.parameters
    
    def test_validate_valid_params(self):
        """Test validating valid parameters."""
        params = [
            ParameterSpec('n_estimators', ParameterType.INT, default=100, min_value=1)
        ]
        schema = ModelParameterSchema('test_model', params)
        
        is_valid, errors = schema.validate({'n_estimators': 50})
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_invalid_params(self):
        """Test validating invalid parameters."""
        params = [
            ParameterSpec('n_estimators', ParameterType.INT, min_value=1)
        ]
        schema = ModelParameterSchema('test_model', params)
        
        is_valid, errors = schema.validate({'n_estimators': -10})
        
        assert not is_valid
        assert len(errors) > 0
    
    def test_validate_missing_required_param(self):
        """Test validating with missing required parameter."""
        params = [
            ParameterSpec('n_clusters', ParameterType.INT, required=True)
        ]
        schema = ModelParameterSchema('test_model', params)
        
        is_valid, errors = schema.validate({})
        
        assert not is_valid
        assert any('missing' in error.lower() for error in errors)
    
    def test_validate_unknown_param_strict(self):
        """Test validating unknown parameter in strict mode."""
        params = [
            ParameterSpec('n_estimators', ParameterType.INT, default=100)
        ]
        schema = ModelParameterSchema('test_model', params)
        
        is_valid, errors = schema.validate(
            {'n_estimators': 100, 'unknown_param': 'value'},
            strict=True
        )
        
        assert not is_valid
        assert any('unknown' in error.lower() for error in errors)
    
    def test_validate_unknown_param_non_strict(self):
        """Test validating unknown parameter in non-strict mode."""
        params = [
            ParameterSpec('n_estimators', ParameterType.INT, default=100)
        ]
        schema = ModelParameterSchema('test_model', params)
        
        is_valid, errors = schema.validate(
            {'n_estimators': 100, 'unknown_param': 'value'},
            strict=False
        )
        
        assert is_valid  # Unknown params allowed in non-strict mode
    
    def test_get_defaults(self):
        """Test getting default values."""
        params = [
            ParameterSpec('n_estimators', ParameterType.INT, default=100),
            ParameterSpec('max_depth', ParameterType.INT, default=10),
            ParameterSpec('min_samples_split', ParameterType.INT)  # No default
        ]
        schema = ModelParameterSchema('test_model', params)
        
        defaults = schema.get_defaults()
        
        assert defaults['n_estimators'] == 100
        assert defaults['max_depth'] == 10
        assert 'min_samples_split' not in defaults
    
    def test_get_parameter_info(self):
        """Test getting parameter information."""
        params = [
            ParameterSpec(
                'n_estimators',
                ParameterType.INT,
                default=100,
                min_value=1,
                description='Number of trees'
            )
        ]
        schema = ModelParameterSchema('test_model', params)
        
        info = schema.get_parameter_info('n_estimators')
        
        assert info is not None
        assert info['name'] == 'n_estimators'
        assert info['default'] == 100
        assert info['min_value'] == 1
        assert info['description'] == 'Number of trees'


class TestValidateModelConfig:
    """Test validate_model_config function."""
    
    def test_validate_valid_config(self):
        """Test validating a valid configuration."""
        is_valid, errors = validate_model_config(
            'random_forest_classifier',
            {'n_estimators': 100, 'max_depth': 10}
        )
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_invalid_config(self):
        """Test validating an invalid configuration."""
        is_valid, errors = validate_model_config(
            'random_forest_classifier',
            {'n_estimators': -10}  # Invalid
        )
        
        assert not is_valid
        assert len(errors) > 0
    
    def test_validate_unknown_model(self):
        """Test validating config for unknown model."""
        is_valid, errors = validate_model_config(
            'unknown_model',
            {'param': 'value'}
        )
        
        # Should return valid (no schema available)
        assert is_valid
    
    def test_validate_multiple_errors(self):
        """Test validating config with multiple errors."""
        is_valid, errors = validate_model_config(
            'random_forest_classifier',
            {
                'n_estimators': -10,  # Invalid range
                'criterion': 'invalid'  # Invalid value
            }
        )
        
        assert not is_valid
        assert len(errors) >= 2


class TestGetModelDefaults:
    """Test get_model_defaults function."""
    
    def test_get_defaults_random_forest(self):
        """Test getting defaults for random forest."""
        defaults = get_model_defaults('random_forest_classifier')
        
        assert isinstance(defaults, dict)
        assert 'n_estimators' in defaults
        assert defaults['n_estimators'] == 100
        assert 'criterion' in defaults
        assert defaults['criterion'] == 'gini'
    
    def test_get_defaults_logistic_regression(self):
        """Test getting defaults for logistic regression."""
        defaults = get_model_defaults('logistic_regression')
        
        assert isinstance(defaults, dict)
        assert 'C' in defaults
        assert defaults['C'] == 1.0
        assert 'penalty' in defaults
    
    def test_get_defaults_unknown_model(self):
        """Test getting defaults for unknown model."""
        defaults = get_model_defaults('unknown_model')
        
        assert isinstance(defaults, dict)
        assert len(defaults) == 0


class TestGetParameterInfo:
    """Test get_parameter_info function."""
    
    def test_get_parameter_info_valid(self):
        """Test getting info for valid parameter."""
        info = get_parameter_info('random_forest_classifier', 'n_estimators')
        
        assert info is not None
        assert info['name'] == 'n_estimators'
        assert info['type'] == ['int']
        assert info['min_value'] == 1
        assert info['default'] == 100
        assert 'description' in info
    
    def test_get_parameter_info_invalid_model(self):
        """Test getting info for invalid model."""
        info = get_parameter_info('unknown_model', 'param')
        
        assert info is None
    
    def test_get_parameter_info_invalid_param(self):
        """Test getting info for invalid parameter."""
        info = get_parameter_info('random_forest_classifier', 'unknown_param')
        
        assert info is None


class TestGetAvailableModelsWithSchemas:
    """Test get_available_models_with_schemas function."""
    
    def test_get_available_models(self):
        """Test getting available models."""
        models = get_available_models_with_schemas()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert 'random_forest_classifier' in models
        assert 'logistic_regression' in models
        assert 'kmeans' in models


class TestModelConfigValidation:
    """Test ModelConfig validation integration."""
    
    def test_config_with_valid_params(self):
        """Test creating config with valid parameters."""
        config = ModelConfig(
            model_type='random_forest_classifier',
            hyperparameters={'n_estimators': 100, 'max_depth': 10},
            validate=True
        )
        
        assert config.hyperparameters['n_estimators'] == 100
    
    def test_config_with_invalid_params(self):
        """Test creating config with invalid parameters."""
        with pytest.raises(ValueError, match="Invalid hyperparameters"):
            ModelConfig(
                model_type='random_forest_classifier',
                hyperparameters={'n_estimators': -10},
                validate=True
            )
    
    def test_config_validation_disabled(self):
        """Test creating config with validation disabled."""
        config = ModelConfig(
            model_type='random_forest_classifier',
            hyperparameters={'n_estimators': -10},  # Invalid
            validate=False
        )
        
        # Should not raise error
        assert config.hyperparameters['n_estimators'] == -10
    
    def test_config_strict_mode(self):
        """Test config in strict mode."""
        with pytest.raises(ValueError, match="Unknown parameter"):
            ModelConfig(
                model_type='random_forest_classifier',
                hyperparameters={'n_estimators': 100, 'unknown': 'value'},
                validate=True,
                strict=True
            )
    
    def test_config_from_dict_with_validation(self):
        """Test creating config from dict with validation."""
        config_dict = {
            'model_type': 'random_forest_classifier',
            'hyperparameters': {'n_estimators': 100},
            'random_state': 42
        }
        
        config = ModelConfig.from_dict(config_dict, validate=True)
        
        assert config.hyperparameters['n_estimators'] == 100


class TestCreateModelWithValidation:
    """Test create_model with validation."""
    
    def test_create_model_valid_params(self):
        """Test creating model with valid parameters."""
        model = create_model(
            'random_forest_classifier',
            n_estimators=100,
            max_depth=10,
            validate=True
        )
        
        assert model.config.hyperparameters['n_estimators'] == 100
    
    def test_create_model_invalid_params(self):
        """Test creating model with invalid parameters."""
        with pytest.raises(ValueError, match="Invalid hyperparameters"):
            create_model(
                'random_forest_classifier',
                n_estimators=-10,
                validate=True
            )
    
    def test_create_model_validation_disabled(self):
        """Test creating model with validation disabled."""
        model = create_model(
            'random_forest_classifier',
            n_estimators=-10,  # Invalid
            validate=False
        )
        
        # Should not raise error
        assert model.config.hyperparameters['n_estimators'] == -10


class TestSpecificModelSchemas:
    """Test specific model validation schemas."""
    
    def test_random_forest_classifier_schema(self):
        """Test random forest classifier schema."""
        # Valid config
        is_valid, errors = validate_model_config(
            'random_forest_classifier',
            {
                'n_estimators': 100,
                'criterion': 'gini',
                'max_depth': 10,
                'min_samples_split': 2
            }
        )
        assert is_valid
        
        # Invalid criterion
        is_valid, errors = validate_model_config(
            'random_forest_classifier',
            {'criterion': 'invalid'}
        )
        assert not is_valid
    
    def test_logistic_regression_schema(self):
        """Test logistic regression schema."""
        # Valid config
        is_valid, errors = validate_model_config(
            'logistic_regression',
            {'C': 1.0, 'penalty': 'l2', 'max_iter': 100}
        )
        assert is_valid
        
        # Invalid C (negative)
        is_valid, errors = validate_model_config(
            'logistic_regression',
            {'C': -1.0}
        )
        assert not is_valid
    
    def test_kmeans_schema(self):
        """Test K-Means schema."""
        # Valid config
        is_valid, errors = validate_model_config(
            'kmeans',
            {'n_clusters': 3, 'max_iter': 300}
        )
        assert is_valid
        
        # Invalid n_clusters (zero)
        is_valid, errors = validate_model_config(
            'kmeans',
            {'n_clusters': 0}
        )
        assert not is_valid
    
    def test_ridge_regression_schema(self):
        """Test Ridge regression schema."""
        # Valid config
        is_valid, errors = validate_model_config(
            'ridge_regression',
            {'alpha': 1.0, 'fit_intercept': True}
        )
        assert is_valid
        
        # Invalid alpha (negative)
        is_valid, errors = validate_model_config(
            'ridge_regression',
            {'alpha': -1.0}
        )
        assert not is_valid


class TestValidationErrorMessages:
    """Test validation error messages."""
    
    def test_type_error_message(self):
        """Test type error message."""
        is_valid, errors = validate_model_config(
            'random_forest_classifier',
            {'n_estimators': '100'}  # String instead of int
        )
        
        assert not is_valid
        assert any('type' in error.lower() for error in errors)
    
    def test_range_error_message(self):
        """Test range error message."""
        is_valid, errors = validate_model_config(
            'random_forest_classifier',
            {'n_estimators': -10}
        )
        
        assert not is_valid
        assert any('>=' in error for error in errors)
    
    def test_value_error_message(self):
        """Test value error message."""
        is_valid, errors = validate_model_config(
            'random_forest_classifier',
            {'criterion': 'invalid'}
        )
        
        assert not is_valid
        assert any('must be one of' in error for error in errors)
