"""
Tests for Model Training Code Generator

Tests the generation of Python code for model training from configurations.
"""

import pytest
from app.ml_engine.code_generation.training_generator import (
    TrainingCodeGenerator,
    generate_training_code
)


class TestTrainingCodeGenerator:
    """Test TrainingCodeGenerator class."""
    
    def test_initialization(self):
        """Test generator initialization."""
        generator = TrainingCodeGenerator()
        assert generator is not None
    
    def test_generate_basic_regression_script(self):
        """Test generating basic regression training script."""
        config = {
            'model_type': 'linear_regression',
            'task_type': 'regression',
            'experiment_name': 'House Price Prediction',
            'hyperparameters': {},
            'target_column': 'price'
        }
        
        generator = TrainingCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Check essential components
        assert 'import pandas as pd' in code
        assert 'import numpy as np' in code
        assert 'from sklearn.linear_model import LinearRegression' in code
        assert 'LinearRegression(' in code
        assert 'train_test_split' in code
        assert 'model.fit(X_train, y_train)' in code
        assert 'pickle.dump' in code
    
    def test_generate_classification_script(self):
        """Test generating classification training script."""
        config = {
            'model_type': 'random_forest_classifier',
            'task_type': 'classification',
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2
            },
            'target_column': 'label'
        }
        
        generator = TrainingCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Check model-specific components
        assert 'from sklearn.ensemble import RandomForestClassifier' in code
        assert 'RandomForestClassifier(' in code
        assert 'n_estimators=100' in code
        assert 'max_depth=10' in code
        assert 'min_samples_split=2' in code
        assert 'stratify=y' in code  # Classification should stratify
    
    def test_generate_clustering_script(self):
        """Test generating clustering training script."""
        config = {
            'model_type': 'kmeans',
            'task_type': 'clustering',
            'hyperparameters': {
                'n_clusters': 3,
                'max_iter': 300
            }
        }
        
        generator = TrainingCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Check clustering-specific components
        assert 'from sklearn.cluster import KMeans' in code
        assert 'KMeans(' in code
        assert 'n_clusters=3' in code
        assert 'max_iter=300' in code
    
    def test_generate_training_function(self):
        """Test generating training function."""
        config = {
            'model_type': 'logistic_regression',
            'task_type': 'classification',
            'hyperparameters': {
                'C': 1.0,
                'max_iter': 1000
            }
        }
        
        generator = TrainingCodeGenerator()
        code = generator.generate(config, output_format='function')
        
        # Check function structure
        assert 'def train_model(' in code
        assert 'X_train, y_train' in code
        assert 'X_val=None, y_val=None' in code
        assert 'LogisticRegression(' in code
        assert 'C=1.0' in code
        assert 'max_iter=1000' in code
        assert 'return model' in code
    
    def test_generate_training_class(self):
        """Test generating training class."""
        config = {
            'model_type': 'gradient_boosting_classifier',
            'task_type': 'classification',
            'hyperparameters': {
                'n_estimators': 100,
                'learning_rate': 0.1
            }
        }
        
        generator = TrainingCodeGenerator()
        code = generator.generate(config, output_format='class')
        
        # Check class structure
        assert 'class ModelTrainer:' in code
        assert 'def __init__(self' in code
        assert 'def train(self' in code
        assert 'def predict(self' in code
        assert 'def save(self' in code
        assert 'def load(self' in code
        assert 'GradientBoostingClassifier(' in code
        assert 'n_estimators=100' in code
        assert 'learning_rate=0.1' in code
    
    def test_generate_without_imports(self):
        """Test generating code without imports."""
        config = {
            'model_type': 'ridge',
            'task_type': 'regression',
            'hyperparameters': {'alpha': 1.0}
        }
        
        generator = TrainingCodeGenerator()
        code = generator.generate(config, output_format='script', include_imports=False)
        
        # Should not have imports
        assert 'import pandas' not in code
        assert 'import numpy' not in code
        # But should have training code
        assert 'Ridge(' in code
        assert 'alpha=1.0' in code
    
    def test_hyperparameter_formatting(self):
        """Test hyperparameter formatting in generated code."""
        config = {
            'model_type': 'svc',
            'task_type': 'classification',
            'hyperparameters': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True
            }
        }
        
        generator = TrainingCodeGenerator()
        code = generator.generate(config, output_format='function')
        
        # Check all hyperparameters are included
        assert 'C=1.0' in code
        assert "kernel='rbf'" in code
        assert "gamma='scale'" in code
        assert 'probability=True' in code
    
    def test_model_saving_included(self):
        """Test that model saving code is included when requested."""
        config = {
            'model_type': 'linear_regression',
            'task_type': 'regression',
            'save_model': True,
            'model_path': 'my_model.pkl'
        }
        
        generator = TrainingCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Check model saving code
        assert 'def save_model(' in code
        assert 'pickle.dump' in code
        assert 'my_model.pkl' in code
    
    def test_random_state_configuration(self):
        """Test random state configuration."""
        config = {
            'model_type': 'random_forest_regressor',
            'task_type': 'regression',
            'random_state': 123,
            'hyperparameters': {'n_estimators': 50}
        }
        
        generator = TrainingCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Check random state is set
        assert 'RANDOM_STATE = 123' in code
        assert 'random_state=RANDOM_STATE' in code or 'random_state=123' in code
    
    def test_test_size_configuration(self):
        """Test test size configuration."""
        config = {
            'model_type': 'lasso',
            'task_type': 'regression',
            'test_size': 0.3,
            'hyperparameters': {}
        }
        
        generator = TrainingCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Check test size
        assert 'test_size: float = 0.3' in code or 'test_size=0.3' in code


class TestConvenienceFunction:
    """Test convenience function."""
    
    def test_generate_training_code_function(self):
        """Test generate_training_code convenience function."""
        config = {
            'model_type': 'decision_tree_classifier',
            'task_type': 'classification',
            'hyperparameters': {
                'max_depth': 5,
                'min_samples_leaf': 10
            }
        }
        
        code = generate_training_code(config)
        
        # Check it works
        assert 'DecisionTreeClassifier(' in code
        assert 'max_depth=5' in code
        assert 'min_samples_leaf=10' in code
    
    def test_generate_training_code_formats(self):
        """Test different output formats."""
        config = {
            'model_type': 'knn_classifier',
            'task_type': 'classification',
            'hyperparameters': {'n_neighbors': 5}
        }
        
        # Test script format
        script = generate_training_code(config, output_format='script')
        assert 'import pandas' in script
        assert 'def split_data(' in script
        
        # Test function format
        function = generate_training_code(config, output_format='function')
        assert 'def train_model(' in function
        assert 'import pandas' not in function
        
        # Test class format
        class_code = generate_training_code(config, output_format='class')
        assert 'class ModelTrainer:' in class_code


class TestModelSupport:
    """Test support for different model types."""
    
    @pytest.mark.parametrize('model_type,expected_import,expected_class', [
        ('linear_regression', 'sklearn.linear_model', 'LinearRegression'),
        ('ridge', 'sklearn.linear_model', 'Ridge'),
        ('lasso', 'sklearn.linear_model', 'Lasso'),
        ('random_forest_regressor', 'sklearn.ensemble', 'RandomForestRegressor'),
        ('logistic_regression', 'sklearn.linear_model', 'LogisticRegression'),
        ('random_forest_classifier', 'sklearn.ensemble', 'RandomForestClassifier'),
        ('svc', 'sklearn.svm', 'SVC'),
        ('kmeans', 'sklearn.cluster', 'KMeans'),
        ('dbscan', 'sklearn.cluster', 'DBSCAN'),
    ])
    def test_model_imports(self, model_type, expected_import, expected_class):
        """Test correct imports for different model types."""
        config = {
            'model_type': model_type,
            'task_type': 'regression',
            'hyperparameters': {}
        }
        
        code = generate_training_code(config)
        
        assert expected_import in code
        assert f'{expected_class}(' in code


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_hyperparameters(self):
        """Test with empty hyperparameters."""
        config = {
            'model_type': 'linear_regression',
            'task_type': 'regression',
            'hyperparameters': {}
        }
        
        code = generate_training_code(config)
        
        # Should still generate valid code
        assert 'LinearRegression(' in code
        assert 'model.fit(' in code
    
    def test_invalid_output_format(self):
        """Test with invalid output format."""
        config = {
            'model_type': 'linear_regression',
            'task_type': 'regression'
        }
        
        generator = TrainingCodeGenerator()
        
        with pytest.raises(ValueError, match="Unknown output format"):
            generator.generate(config, output_format='invalid')
    
    def test_minimal_config(self):
        """Test with minimal configuration."""
        config = {
            'model_type': 'linear_regression'
        }
        
        # Should use defaults
        code = generate_training_code(config)
        
        assert 'LinearRegression(' in code
        assert 'RANDOM_STATE = 42' in code  # Default random state
    
    def test_string_hyperparameters(self):
        """Test with string hyperparameters."""
        config = {
            'model_type': 'svc',
            'task_type': 'classification',
            'hyperparameters': {
                'kernel': 'rbf',
                'gamma': 'scale'
            }
        }
        
        code = generate_training_code(config)
        
        # String values should be quoted
        assert "kernel='rbf'" in code
        assert "gamma='scale'" in code
