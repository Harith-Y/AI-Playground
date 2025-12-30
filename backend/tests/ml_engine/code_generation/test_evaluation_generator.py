"""
Tests for Evaluation Code Generator

Tests the generation of Python code for model evaluation from configurations.
"""

import pytest
from app.ml_engine.code_generation.evaluation_generator import (
    EvaluationCodeGenerator,
    generate_evaluation_code
)


class TestEvaluationCodeGenerator:
    """Test EvaluationCodeGenerator class."""
    
    def test_initialization(self):
        """Test generator initialization."""
        generator = EvaluationCodeGenerator()
        assert generator is not None
    
    def test_generate_classification_script(self):
        """Test generating classification evaluation script."""
        config = {
            'task_type': 'classification',
            'experiment_name': 'Customer Churn Prediction',
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            'include_confusion_matrix': True,
            'include_roc_curve': True
        }
        
        generator = EvaluationCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Check essential components
        assert 'import numpy as np' in code
        assert 'import pandas as pd' in code
        assert 'from sklearn.metrics import' in code
        assert 'accuracy_score' in code
        assert 'precision_score' in code
        assert 'recall_score' in code
        assert 'f1_score' in code
        assert 'roc_auc_score' in code
        assert 'def evaluate_model(' in code
        assert 'confusion_matrix' in code
        assert 'def plot_confusion_matrix(' in code
        assert 'def plot_roc_curve(' in code
    
    def test_generate_regression_script(self):
        """Test generating regression evaluation script."""
        config = {
            'task_type': 'regression',
            'experiment_name': 'House Price Prediction',
            'metrics': ['mae', 'mse', 'rmse', 'r2', 'mape'],
            'include_residual_analysis': True
        }
        
        generator = EvaluationCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Check regression-specific components
        assert 'mean_absolute_error' in code
        assert 'mean_squared_error' in code
        assert 'r2_score' in code
        assert 'mean_absolute_percentage_error' in code
        assert 'residuals' in code
        assert 'def evaluate_model(' in code
    
    def test_generate_clustering_script(self):
        """Test generating clustering evaluation script."""
        config = {
            'task_type': 'clustering',
            'experiment_name': 'Customer Segmentation',
            'metrics': ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        }
        
        generator = EvaluationCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Check clustering-specific components
        assert 'silhouette_score' in code
        assert 'calinski_harabasz_score' in code
        assert 'davies_bouldin_score' in code
        assert 'def evaluate_model(' in code
        assert 'cluster_sizes' in code
    
    def test_generate_function_format(self):
        """Test generating evaluation function."""
        config = {
            'task_type': 'classification',
            'metrics': ['accuracy', 'f1']
        }
        
        generator = EvaluationCodeGenerator()
        code = generator.generate(config, output_format='function')
        
        # Check function structure
        assert 'def evaluate_model(' in code
        assert 'accuracy_score' in code
        assert 'f1_score' in code
        assert 'return results' in code
        # Should not have imports or main block
        assert 'if __name__' not in code
    
    def test_generate_module_format(self):
        """Test generating evaluation module."""
        config = {
            'task_type': 'classification',
            'metrics': ['accuracy', 'precision', 'recall'],
            'include_confusion_matrix': True
        }
        
        generator = EvaluationCodeGenerator()
        code = generator.generate(config, output_format='module')
        
        # Check module structure
        assert '"""' in code  # Module docstring
        assert 'RANDOM_STATE' in code
        assert 'RESULTS_PATH' in code
        assert 'def evaluate_model(' in code
        assert 'def save_results(' in code
        assert 'if __name__ == \'__main__\':' in code
        assert 'from evaluate import' in code
    
    def test_default_metrics_classification(self):
        """Test default metrics for classification."""
        config = {
            'task_type': 'classification',
            'metrics': []  # Empty, should use defaults
        }
        
        generator = EvaluationCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Should include default classification metrics
        assert 'accuracy_score' in code
        assert 'precision_score' in code
        assert 'recall_score' in code
        assert 'f1_score' in code
        assert 'roc_auc_score' in code
    
    def test_default_metrics_regression(self):
        """Test default metrics for regression."""
        config = {
            'task_type': 'regression',
            'metrics': []  # Empty, should use defaults
        }
        
        generator = EvaluationCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Should include default regression metrics
        assert 'mean_absolute_error' in code
        assert 'mean_squared_error' in code
        assert 'r2_score' in code
    
    def test_default_metrics_clustering(self):
        """Test default metrics for clustering."""
        config = {
            'task_type': 'clustering',
            'metrics': []  # Empty, should use defaults
        }
        
        generator = EvaluationCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Should include default clustering metrics
        assert 'silhouette_score' in code
        assert 'calinski_harabasz_score' in code
        assert 'davies_bouldin_score' in code
    
    def test_without_imports(self):
        """Test generating code without imports."""
        config = {
            'task_type': 'classification',
            'metrics': ['accuracy']
        }
        
        generator = EvaluationCodeGenerator()
        code = generator.generate(config, output_format='script', include_imports=False)
        
        # Should not have imports
        assert 'import numpy' not in code
        assert 'import pandas' not in code
        # But should have evaluation code
        assert 'def evaluate_model(' in code
        assert 'accuracy_score' in code
    
    def test_save_results_option(self):
        """Test save results option."""
        config = {
            'task_type': 'classification',
            'metrics': ['accuracy'],
            'save_results': True,
            'results_path': 'my_results.json'
        }
        
        generator = EvaluationCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Check save results function
        assert 'def save_results(' in code
        assert 'my_results.json' in code
        assert 'json.dump' in code
    
    def test_confusion_matrix_option(self):
        """Test confusion matrix option."""
        config = {
            'task_type': 'classification',
            'metrics': ['accuracy'],
            'include_confusion_matrix': True
        }
        
        generator = EvaluationCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Check confusion matrix code
        assert 'confusion_matrix' in code
        assert 'def plot_confusion_matrix(' in code
        assert 'matplotlib' in code or 'seaborn' in code
    
    def test_roc_curve_option(self):
        """Test ROC curve option."""
        config = {
            'task_type': 'classification',
            'metrics': ['accuracy'],
            'include_roc_curve': True
        }
        
        generator = EvaluationCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Check ROC curve code
        assert 'roc_curve' in code
        assert 'def plot_roc_curve(' in code
        assert 'matplotlib' in code
    
    def test_residual_analysis_option(self):
        """Test residual analysis option."""
        config = {
            'task_type': 'regression',
            'metrics': ['mae'],
            'include_residual_analysis': True
        }
        
        generator = EvaluationCodeGenerator()
        code = generator.generate(config, output_format='script')
        
        # Check residual analysis code
        assert 'residuals' in code
        assert 'y_test - y_pred' in code
        assert 'np.mean(residuals)' in code
        assert 'np.std(residuals)' in code


class TestConvenienceFunction:
    """Test convenience function."""
    
    def test_generate_evaluation_code_function(self):
        """Test generate_evaluation_code convenience function."""
        config = {
            'task_type': 'classification',
            'metrics': ['accuracy', 'f1']
        }
        
        code = generate_evaluation_code(config)
        
        # Check it works
        assert 'def evaluate_model(' in code
        assert 'accuracy_score' in code
        assert 'f1_score' in code
    
    def test_generate_evaluation_code_formats(self):
        """Test different output formats."""
        config = {
            'task_type': 'regression',
            'metrics': ['mae', 'rmse']
        }
        
        # Test script format
        script = generate_evaluation_code(config, output_format='script')
        assert 'import numpy' in script
        assert 'if __name__' in script
        
        # Test function format
        function = generate_evaluation_code(config, output_format='function')
        assert 'def evaluate_model(' in function
        assert 'import numpy' not in function
        
        # Test module format
        module = generate_evaluation_code(config, output_format='module')
        assert '"""' in module  # Docstring
        assert 'RANDOM_STATE' in module


class TestTaskTypes:
    """Test support for different task types."""
    
    @pytest.mark.parametrize('task_type,expected_metrics', [
        ('classification', ['accuracy_score', 'precision_score', 'recall_score']),
        ('regression', ['mean_absolute_error', 'mean_squared_error', 'r2_score']),
        ('clustering', ['silhouette_score', 'calinski_harabasz_score']),
    ])
    def test_task_type_metrics(self, task_type, expected_metrics):
        """Test correct metrics for different task types."""
        config = {
            'task_type': task_type,
            'metrics': []  # Use defaults
        }
        
        code = generate_evaluation_code(config)
        
        for metric in expected_metrics:
            assert metric in code


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_config(self):
        """Test with empty configuration."""
        config = {}
        
        # Should use defaults
        code = generate_evaluation_code(config)
        
        assert 'def evaluate_model(' in code
        assert 'classification' in code.lower()  # Default task type
    
    def test_invalid_output_format(self):
        """Test with invalid output format."""
        config = {
            'task_type': 'classification'
        }
        
        generator = EvaluationCodeGenerator()
        
        with pytest.raises(ValueError, match="Unknown output format"):
            generator.generate(config, output_format='invalid')
    
    def test_minimal_config(self):
        """Test with minimal configuration."""
        config = {
            'task_type': 'classification'
        }
        
        # Should use defaults for everything else
        code = generate_evaluation_code(config)
        
        assert 'def evaluate_model(' in code
        assert 'accuracy_score' in code
    
    def test_custom_results_path(self):
        """Test with custom results path."""
        config = {
            'task_type': 'regression',
            'results_path': 'custom/path/results.json'
        }
        
        code = generate_evaluation_code(config)
        
        assert 'custom/path/results.json' in code
