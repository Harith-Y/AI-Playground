"""
Unit tests for preprocessing code generator.

Tests cover:
- Basic code generation
- Different preprocessing steps
- Multiple output formats (script, function, class)
- Edge cases
"""

import pytest
from app.ml_engine.code_generation.preprocessing_generator import (
    PreprocessingCodeGenerator,
    generate_preprocessing_code
)


class TestPreprocessingCodeGenerator:
    """Test suite for PreprocessingCodeGenerator."""
    
    def test_initialization(self):
        """Test generator initialization."""
        generator = PreprocessingCodeGenerator()
        assert generator is not None
    
    def test_generate_empty_config(self):
        """Test generation with empty configuration."""
        config = {
            'preprocessing_steps': [],
            'dataset_info': {}
        }
        
        generator = PreprocessingCodeGenerator()
        code = generator.generate(config)
        
        assert isinstance(code, str)
        assert len(code) > 0
        assert 'import pandas as pd' in code
    
    def test_generate_with_imputation(self):
        """Test generation with missing value imputation."""
        config = {
            'preprocessing_steps': [
                {
                    'type': 'missing_value_imputation',
                    'name': 'Mean Imputation',
                    'parameters': {
                        'strategy': 'mean',
                        'columns': ['age', 'income']
                    }
                }
            ],
            'dataset_info': {
                'file_path': 'data.csv',
                'file_format': 'csv'
            }
        }
        
        generator = PreprocessingCodeGenerator()
        code = generator.generate(config)
        
        assert 'SimpleImputer' in code
        assert "strategy='mean'" in code
        assert 'age' in code or 'income' in code
    
    def test_generate_with_scaling(self):
        """Test generation with feature scaling."""
        config = {
            'preprocessing_steps': [
                {
                    'type': 'scaling',
                    'name': 'Standard Scaling',
                    'parameters': {
                        'scaler': 'standard',
                        'columns': ['feature1', 'feature2']
                    }
                }
            ],
            'dataset_info': {}
        }
        
        generator = PreprocessingCodeGenerator()
        code = generator.generate(config)
        
        assert 'StandardScaler' in code
        assert 'feature1' in code or 'feature2' in code
    
    def test_generate_with_encoding(self):
        """Test generation with categorical encoding."""
        config = {
            'preprocessing_steps': [
                {
                    'type': 'encoding',
                    'name': 'OneHot Encoding',
                    'parameters': {
                        'encoder': 'onehot',
                        'columns': ['category', 'type']
                    }
                }
            ],
            'dataset_info': {}
        }
        
        generator = PreprocessingCodeGenerator()
        code = generator.generate(config)
        
        assert 'OneHotEncoder' in code
        assert 'category' in code or 'type' in code
    
    def test_generate_with_outlier_detection(self):
        """Test generation with outlier detection."""
        config = {
            'preprocessing_steps': [
                {
                    'type': 'outlier_detection',
                    'name': 'IQR Outlier Detection',
                    'parameters': {
                        'method': 'iqr',
                        'threshold': 1.5,
                        'action': 'clip',
                        'columns': ['price', 'quantity']
                    }
                }
            ],
            'dataset_info': {}
        }
        
        generator = PreprocessingCodeGenerator()
        code = generator.generate(config)
        
        assert 'IQR' in code or 'iqr' in code
        assert '1.5' in code
        assert 'clip' in code
    
    def test_generate_multiple_steps(self):
        """Test generation with multiple preprocessing steps."""
        config = {
            'preprocessing_steps': [
                {
                    'type': 'missing_value_imputation',
                    'parameters': {'strategy': 'mean', 'columns': ['age']}
                },
                {
                    'type': 'scaling',
                    'parameters': {'scaler': 'standard', 'columns': ['age', 'income']}
                },
                {
                    'type': 'encoding',
                    'parameters': {'encoder': 'onehot', 'columns': ['category']}
                }
            ],
            'dataset_info': {}
        }
        
        generator = PreprocessingCodeGenerator()
        code = generator.generate(config)
        
        # Should contain all step types
        assert 'SimpleImputer' in code
        assert 'StandardScaler' in code
        assert 'OneHotEncoder' in code
        
        # Should have step comments
        assert 'Step 1:' in code
        assert 'Step 2:' in code
        assert 'Step 3:' in code
    
    def test_generate_without_imports(self):
        """Test generation without import statements."""
        config = {
            'preprocessing_steps': [],
            'dataset_info': {}
        }
        
        generator = PreprocessingCodeGenerator()
        code = generator.generate(config, include_imports=False)
        
        assert 'import pandas' not in code
    
    def test_generate_without_data_loading(self):
        """Test generation without data loading code."""
        config = {
            'preprocessing_steps': [],
            'dataset_info': {}
        }
        
        generator = PreprocessingCodeGenerator()
        code = generator.generate(config, include_data_loading=False)
        
        assert 'load_data' not in code
    
    def test_generate_preprocessing_class(self):
        """Test generation of preprocessing class."""
        config = {
            'preprocessing_steps': [
                {
                    'type': 'scaling',
                    'parameters': {'scaler': 'standard', 'columns': ['feature1']}
                }
            ],
            'dataset_info': {}
        }
        
        generator = PreprocessingCodeGenerator()
        code = generator.generate_preprocessing_class(config)
        
        assert 'class DataPreprocessor:' in code
        assert 'def fit(' in code
        assert 'def transform(' in code
        assert 'def fit_transform(' in code


class TestConvenienceFunction:
    """Test suite for convenience function."""
    
    def test_generate_preprocessing_code_script(self):
        """Test convenience function with script format."""
        config = {
            'preprocessing_steps': [],
            'dataset_info': {}
        }
        
        code = generate_preprocessing_code(config, output_format='script')
        
        assert isinstance(code, str)
        assert len(code) > 0
    
    def test_generate_preprocessing_code_function(self):
        """Test convenience function with function format."""
        config = {
            'preprocessing_steps': [],
            'dataset_info': {}
        }
        
        code = generate_preprocessing_code(config, output_format='function')
        
        assert isinstance(code, str)
        assert 'def preprocess_data' in code
    
    def test_generate_preprocessing_code_class(self):
        """Test convenience function with class format."""
        config = {
            'preprocessing_steps': [],
            'dataset_info': {}
        }
        
        code = generate_preprocessing_code(config, output_format='class')
        
        assert isinstance(code, str)
        assert 'class DataPreprocessor' in code
    
    def test_invalid_output_format(self):
        """Test error handling for invalid output format."""
        config = {'preprocessing_steps': []}
        
        with pytest.raises(ValueError, match="Invalid output_format"):
            generate_preprocessing_code(config, output_format='invalid')


class TestEdgeCases:
    """Test edge cases."""
    
    def test_unknown_step_type(self):
        """Test handling of unknown step type."""
        config = {
            'preprocessing_steps': [
                {
                    'type': 'unknown_step_type',
                    'parameters': {}
                }
            ],
            'dataset_info': {}
        }
        
        generator = PreprocessingCodeGenerator()
        code = generator.generate(config)
        
        # Should not crash, just skip unknown step
        assert isinstance(code, str)
    
    def test_missing_parameters(self):
        """Test handling of missing parameters."""
        config = {
            'preprocessing_steps': [
                {
                    'type': 'scaling',
                    # Missing 'parameters' key
                }
            ],
            'dataset_info': {}
        }
        
        generator = PreprocessingCodeGenerator()
        code = generator.generate(config)
        
        # Should use defaults
        assert isinstance(code, str)
