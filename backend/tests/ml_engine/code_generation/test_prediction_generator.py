"""
Tests for Prediction Code Generator

Tests the generation of Python code for model predictions from configurations.
"""

import pytest
from app.ml_engine.code_generation.prediction_generator import (
    PredictionCodeGenerator,
    generate_prediction_code
)


class TestPredictionCodeGenerator:
    """Test PredictionCodeGenerator class."""
    
    def test_initialization(self):
        """Test generator initialization."""
        generator = PredictionCodeGenerator()
        assert generator is not None
    
    def test_generate_classification_script(self):
        """Test generating classification prediction script."""
        config = {
            'task_type': 'classification',
            'experiment_name': 'Customer Churn Prediction',
            'model_path': 'churn_model.pkl',
            'include_probabilities': True,
            'save_predictions': True
        }
        
        generator = PredictionCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Check essential components
        assert 'import numpy as np' in code
        assert 'import pandas as pd' in code
        assert 'import pickle' in code
        assert 'def load_model(' in code
        assert 'def predict(' in code
        assert 'predict_proba' in code
        assert 'def save_predictions(' in code
    
    def test_generate_regression_script(self):
        """Test generating regression prediction script."""
        config = {
            'task_type': 'regression',
            'experiment_name': 'House Price Prediction',
            'model_path': 'price_model.pkl',
            'save_predictions': True
        }
        
        generator = PredictionCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Check regression-specific components
        assert 'def predict(' in code
        assert 'mean_prediction' in code
        assert 'std_prediction' in code
        assert 'model.predict(X)' in code
    
    def test_generate_clustering_script(self):
        """Test generating clustering prediction script."""
        config = {
            'task_type': 'clustering',
            'experiment_name': 'Customer Segmentation',
            'model_path': 'clustering_model.pkl'
        }
        
        generator = PredictionCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Check clustering-specific components
        assert 'def predict(' in code
        assert 'cluster_labels' in code
        assert 'cluster_distribution' in code
    
    def test_generate_function_format(self):
        """Test generating prediction function."""
        config = {
            'task_type': 'classification',
            'model_path': 'model.pkl'
        }
        
        generator = PredictionCodeGenerator()
        code = generator.generate(config, output_format='function')
        
        # Check function structure
        assert 'def predict(' in code
        assert 'model.predict(X)' in code
        # Should not have imports or main block
        assert 'if __name__' not in code
    
    def test_generate_api_format(self):
        """Test generating FastAPI prediction service."""
        config = {
            'task_type': 'classification',
            'experiment_name': 'ML API',
            'model_path': 'model.pkl'
        }
        
        generator = PredictionCodeGenerator()
        code = generator.generate(config, output_format='api')
        
        # Check API components
        assert 'from fastapi import FastAPI' in code
        assert 'app = FastAPI(' in code
        assert '@app.post("/predict"' in code
        assert '@app.get("/health"' in code
        assert 'class PredictionRequest' in code
        assert 'class PredictionResponse' in code
    
    def test_generate_module_format(self):
        """Test generating prediction module."""
        config = {
            'task_type': 'classification',
            'model_path': 'model.pkl',
            'save_predictions': True
        }
        
        generator = PredictionCodeGenerator()
        code = generator.generate(config, output_format='module')
        
        # Check module structure
        assert '"""' in code  # Module docstring
        assert 'MODEL_PATH' in code
        assert 'def load_model(' in code
        assert 'def predict(' in code
        assert 'def save_predictions(' in code
        assert 'if __name__ == \'__main__\':' in code
    
    def test_with_preprocessing(self):
        """Test with preprocessing option."""
        config = {
            'task_type': 'classification',
            'model_path': 'model.pkl',
            'include_preprocessing': True,
            'preprocessing_path': 'preprocessor.pkl'
        }
        
        generator = PredictionCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Check preprocessing components
        assert 'def load_preprocessor(' in code
        assert 'def predict_with_preprocessing(' in code
        assert 'preprocessor.transform' in code
        assert 'from sklearn.preprocessing import' in code
    
    def test_batch_prediction(self):
        """Test batch prediction option."""
        config = {
            'task_type': 'classification',
            'model_path': 'model.pkl',
            'batch_prediction': True
        }
        
        generator = PredictionCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Check batch prediction components
        assert 'def predict_batch(' in code
        assert 'batch_size' in code
        assert 'n_batches' in code
    
    def test_save_predictions_option(self):
        """Test save predictions option."""
        config = {
            'task_type': 'classification',
            'model_path': 'model.pkl',
            'save_predictions': True,
            'output_path': 'my_predictions.csv'
        }
        
        generator = PredictionCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Check save predictions components
        assert 'def save_predictions(' in code
        assert 'my_predictions.csv' in code
        assert 'to_csv' in code
    
    def test_without_imports(self):
        """Test generating code without imports."""
        config = {
            'task_type': 'classification',
            'model_path': 'model.pkl'
        }
        
        generator = PredictionCodeGenerator()
        code = generator.generate(config, output_format='script', include_imports=False)
        
        # Should not have imports
        assert 'import numpy' not in code
        assert 'import pandas' not in code
        # But should have prediction code
        assert 'def predict(' in code


class TestConvenienceFunction:
    """Test convenience function."""
    
    def test_generate_prediction_code_function(self):
        """Test generate_prediction_code convenience function."""
        config = {
            'task_type': 'classification',
            'model_path': 'model.pkl'
        }
        
        code = generate_prediction_code(config)
        
        # Check it works
        assert 'def predict(' in code
        assert 'model.predict(X)' in code
    
    def test_generate_prediction_code_formats(self):
        """Test different output formats."""
        config = {
            'task_type': 'regression',
            'model_path': 'model.pkl'
        }
        
        # Test script format
        script = generate_prediction_code(config, output_format='script')
        assert 'import numpy' in script
        assert 'if __name__' in script
        
        # Test function format
        function = generate_prediction_code(config, output_format='function')
        assert 'def predict(' in function
        assert 'import numpy' not in function
        
        # Test API format
        api = generate_prediction_code(config, output_format='api')
        assert 'FastAPI' in api
        assert '@app.post' in api
        
        # Test module format
        module = generate_prediction_code(config, output_format='module')
        assert '"""' in module  # Docstring
        assert 'MODEL_PATH' in module


class TestTaskTypes:
    """Test support for different task types."""
    
    @pytest.mark.parametrize('task_type,expected_content', [
        ('classification', ['predict_proba', 'probabilities', 'classes']),
        ('regression', ['mean_prediction', 'std_prediction', 'min_prediction']),
        ('clustering', ['cluster_labels', 'cluster_distribution', 'n_clusters']),
    ])
    def test_task_type_specific_code(self, task_type, expected_content):
        """Test correct code for different task types."""
        config = {
            'task_type': task_type,
            'model_path': 'model.pkl'
        }
        
        code = generate_prediction_code(config)
        
        for content in expected_content:
            assert content in code


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_config(self):
        """Test with empty configuration."""
        config = {}
        
        # Should use defaults
        code = generate_prediction_code(config)
        
        assert 'def predict(' in code
        assert 'classification' in code.lower()  # Default task type
    
    def test_invalid_output_format(self):
        """Test with invalid output format."""
        config = {
            'task_type': 'classification',
            'model_path': 'model.pkl'
        }
        
        generator = PredictionCodeGenerator()
        
        with pytest.raises(ValueError, match="Unknown output format"):
            generator.generate(config, output_format='invalid')
    
    def test_minimal_config(self):
        """Test with minimal configuration."""
        config = {
            'task_type': 'classification'
        }
        
        # Should use defaults for everything else
        code = generate_prediction_code(config)
        
        assert 'def predict(' in code
        assert 'model.pkl' in code  # Default model path
    
    def test_custom_paths(self):
        """Test with custom paths."""
        config = {
            'task_type': 'regression',
            'model_path': 'custom/path/model.pkl',
            'output_path': 'custom/output/predictions.csv'
        }
        
        code = generate_prediction_code(config)
        
        assert 'custom/path/model.pkl' in code
        assert 'custom/output/predictions.csv' in code


class TestAPIGeneration:
    """Test FastAPI generation."""
    
    def test_api_has_all_endpoints(self):
        """Test that API has all required endpoints."""
        config = {
            'task_type': 'classification',
            'model_path': 'model.pkl'
        }
        
        code = generate_prediction_code(config, output_format='api')
        
        # Check all endpoints
        assert '@app.post("/predict"' in code
        assert '@app.get("/health"' in code
        assert '@app.get("/"' in code
    
    def test_api_has_request_response_models(self):
        """Test that API has Pydantic models."""
        config = {
            'task_type': 'classification',
            'model_path': 'model.pkl'
        }
        
        code = generate_prediction_code(config, output_format='api')
        
        # Check Pydantic models
        assert 'class PredictionRequest(BaseModel)' in code
        assert 'class PredictionResponse(BaseModel)' in code
        assert 'features: List[List[float]]' in code
    
    def test_api_has_startup_event(self):
        """Test that API has startup event for model loading."""
        config = {
            'task_type': 'classification',
            'model_path': 'model.pkl'
        }
        
        code = generate_prediction_code(config, output_format='api')
        
        # Check startup event
        assert '@app.on_event("startup")' in code
        assert 'async def load_model_startup()' in code
