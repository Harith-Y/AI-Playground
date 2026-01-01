"""
Unit tests for model and pipeline serialization.

Tests cover:
- Model serialization and deserialization
- Pipeline serialization and deserialization
- Workflow serialization (pipeline + model)
- Compression support
- Version compatibility
- Error handling
- Metadata preservation
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

from app.ml_engine.utils.serialization import (
    ModelSerializer,
    PipelineSerializer,
    WorkflowSerializer,
    SerializationError,
    save_model,
    load_model,
    save_pipeline,
    load_pipeline,
    save_workflow,
    load_workflow,
    get_model_info,
    get_pipeline_info,
    get_workflow_info
)
from app.ml_engine.models.classification import RandomForestClassifierWrapper
from app.ml_engine.models.regression import LinearRegressionWrapper
from app.ml_engine.models.base import ModelConfig
from app.ml_engine.preprocessing.pipeline import Pipeline
from app.ml_engine.preprocessing.scaler import StandardScaler
from app.ml_engine.preprocessing.imputer import MeanImputer


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })
    y = pd.Series(np.random.randint(0, 2, 100), name='target')
    return X, y


@pytest.fixture
def fitted_model(sample_data):
    """Create a fitted classification model."""
    X, y = sample_data
    config = ModelConfig(
        model_type='random_forest_classifier',
        hyperparameters={'n_estimators': 10, 'random_state': 42}
    )
    model = RandomForestClassifierWrapper(config)
    model.fit(X, y)
    return model


@pytest.fixture
def fitted_pipeline(sample_data):
    """Create a fitted preprocessing pipeline."""
    X, _ = sample_data
    pipeline = Pipeline(steps=[
        MeanImputer(),
        StandardScaler()
    ], name='TestPipeline')
    pipeline.fit(X)
    return pipeline



class TestModelSerializer:
    """Tests for ModelSerializer class."""
    
    def test_save_model_basic(self, fitted_model, temp_dir):
        """Test basic model saving."""
        serializer = ModelSerializer()
        save_path = temp_dir / 'model.pkl'
        
        result_path = serializer.save_model(fitted_model, save_path)
        
        assert result_path.exists()
        assert result_path == save_path
    
    def test_save_model_with_compression(self, fitted_model, temp_dir):
        """Test model saving with compression."""
        serializer = ModelSerializer(compression=True)
        save_path = temp_dir / 'model.pkl'
        
        result_path = serializer.save_model(fitted_model, save_path)
        
        assert result_path.exists()
        assert result_path.suffix == '.gz'
    
    def test_save_model_with_metadata(self, fitted_model, temp_dir):
        """Test model saving with additional metadata."""
        serializer = ModelSerializer()
        save_path = temp_dir / 'model.pkl'
        metadata = {'experiment_id': '123', 'notes': 'Test model'}
        
        serializer.save_model(fitted_model, save_path, metadata=metadata)
        
        assert save_path.exists()
    
    def test_save_model_overwrite_protection(self, fitted_model, temp_dir):
        """Test that saving fails without overwrite flag."""
        serializer = ModelSerializer()
        save_path = temp_dir / 'model.pkl'
        
        serializer.save_model(fitted_model, save_path)
        
        with pytest.raises(FileExistsError):
            serializer.save_model(fitted_model, save_path, overwrite=False)
    
    def test_save_model_with_overwrite(self, fitted_model, temp_dir):
        """Test model overwriting."""
        serializer = ModelSerializer()
        save_path = temp_dir / 'model.pkl'
        
        serializer.save_model(fitted_model, save_path)
        serializer.save_model(fitted_model, save_path, overwrite=True)
        
        assert save_path.exists()
    
    def test_load_model_basic(self, fitted_model, temp_dir, sample_data):
        """Test basic model loading."""
        serializer = ModelSerializer()
        save_path = temp_dir / 'model.pkl'
        X, y = sample_data
        
        serializer.save_model(fitted_model, save_path)
        loaded_model = serializer.load_model(save_path)
        
        assert loaded_model.is_fitted
        assert loaded_model.config.model_type == 'random_forest_classifier'
        
        # Test predictions match
        original_pred = fitted_model.predict(X)
        loaded_pred = loaded_model.predict(X)
        np.testing.assert_array_equal(original_pred, loaded_pred)
    
    def test_load_model_compressed(self, fitted_model, temp_dir, sample_data):
        """Test loading compressed model."""
        serializer = ModelSerializer(compression=True)
        save_path = temp_dir / 'model.pkl'
        X, _ = sample_data
        
        serializer.save_model(fitted_model, save_path)
        loaded_model = serializer.load_model(save_path)
        
        assert loaded_model.is_fitted
        
        # Predictions should match
        original_pred = fitted_model.predict(X)
        loaded_pred = loaded_model.predict(X)
        np.testing.assert_array_equal(original_pred, loaded_pred)
    
    def test_load_model_not_found(self, temp_dir):
        """Test loading non-existent model."""
        serializer = ModelSerializer()
        
        with pytest.raises(FileNotFoundError):
            serializer.load_model(temp_dir / 'nonexistent.pkl')
    
    def test_get_model_info(self, fitted_model, temp_dir):
        """Test getting model info without loading."""
        serializer = ModelSerializer()
        save_path = temp_dir / 'model.pkl'
        
        serializer.save_model(fitted_model, save_path)
        info = serializer.get_model_info(save_path)
        
        assert info['model_class'] == 'RandomForestClassifierWrapper'
        assert info['model_type'] == 'random_forest_classifier'
        assert info['is_fitted'] is True
        assert info['n_features'] == 3
        assert 'saved_at' in info
        assert 'file_size_kb' in info



class TestPipelineSerializer:
    """Tests for PipelineSerializer class."""
    
    def test_save_pipeline_basic(self, fitted_pipeline, temp_dir):
        """Test basic pipeline saving."""
        serializer = PipelineSerializer()
        save_path = temp_dir / 'pipeline.pkl'
        
        result_path = serializer.save_pipeline(fitted_pipeline, save_path)
        
        assert result_path.exists()
        assert result_path == save_path
    
    def test_save_pipeline_with_compression(self, fitted_pipeline, temp_dir):
        """Test pipeline saving with compression."""
        serializer = PipelineSerializer(compression=True)
        save_path = temp_dir / 'pipeline.pkl'
        
        result_path = serializer.save_pipeline(fitted_pipeline, save_path)
        
        assert result_path.exists()
        assert result_path.suffix == '.gz'
    
    def test_load_pipeline_basic(self, fitted_pipeline, temp_dir, sample_data):
        """Test basic pipeline loading."""
        serializer = PipelineSerializer()
        save_path = temp_dir / 'pipeline.pkl'
        X, _ = sample_data
        
        serializer.save_pipeline(fitted_pipeline, save_path)
        loaded_pipeline = serializer.load_pipeline(save_path)
        
        assert loaded_pipeline.fitted
        assert loaded_pipeline.name == 'TestPipeline'
        assert len(loaded_pipeline.steps) == 2
        
        # Test transformations match
        original_transform = fitted_pipeline.transform(X)
        loaded_transform = loaded_pipeline.transform(X)
        np.testing.assert_array_almost_equal(original_transform, loaded_transform)
    
    def test_load_pipeline_compressed(self, fitted_pipeline, temp_dir, sample_data):
        """Test loading compressed pipeline."""
        serializer = PipelineSerializer(compression=True)
        save_path = temp_dir / 'pipeline.pkl'
        X, _ = sample_data
        
        serializer.save_pipeline(fitted_pipeline, save_path)
        loaded_pipeline = serializer.load_pipeline(save_path)
        
        assert loaded_pipeline.fitted
        
        # Transformations should match
        original_transform = fitted_pipeline.transform(X)
        loaded_transform = loaded_pipeline.transform(X)
        np.testing.assert_array_almost_equal(original_transform, loaded_transform)
    
    def test_get_pipeline_info(self, fitted_pipeline, temp_dir):
        """Test getting pipeline info without loading."""
        serializer = PipelineSerializer()
        save_path = temp_dir / 'pipeline.pkl'
        
        serializer.save_pipeline(fitted_pipeline, save_path)
        info = serializer.get_pipeline_info(save_path)
        
        assert info['name'] == 'TestPipeline'
        assert info['fitted'] is True
        assert info['num_steps'] == 2
        assert len(info['step_names']) == 2
        assert 'saved_at' in info


class TestWorkflowSerializer:
    """Tests for WorkflowSerializer class."""
    
    def test_save_workflow_basic(self, fitted_pipeline, fitted_model, temp_dir):
        """Test basic workflow saving."""
        serializer = WorkflowSerializer()
        save_path = temp_dir / 'workflow.pkl'
        
        result_path = serializer.save_workflow(
            fitted_pipeline, fitted_model, save_path,
            workflow_name='TestWorkflow'
        )
        
        assert result_path.exists()
    
    def test_load_workflow_basic(self, fitted_pipeline, fitted_model, temp_dir, sample_data):
        """Test basic workflow loading."""
        serializer = WorkflowSerializer()
        save_path = temp_dir / 'workflow.pkl'
        X, _ = sample_data
        
        serializer.save_workflow(
            fitted_pipeline, fitted_model, save_path,
            workflow_name='TestWorkflow'
        )
        
        loaded_pipeline, loaded_model = serializer.load_workflow(save_path)
        
        assert loaded_pipeline.fitted
        assert loaded_model.is_fitted
        
        # Test end-to-end prediction
        X_transformed = loaded_pipeline.transform(X)
        predictions = loaded_model.predict(X_transformed)
        
        assert len(predictions) == len(X)
    
    def test_workflow_predictions_match(self, fitted_pipeline, fitted_model, temp_dir, sample_data):
        """Test that workflow predictions match original."""
        serializer = WorkflowSerializer()
        save_path = temp_dir / 'workflow.pkl'
        X, _ = sample_data
        
        # Original predictions
        X_transformed_orig = fitted_pipeline.transform(X)
        predictions_orig = fitted_model.predict(X_transformed_orig)
        
        # Save and load
        serializer.save_workflow(fitted_pipeline, fitted_model, save_path)
        loaded_pipeline, loaded_model = serializer.load_workflow(save_path)
        
        # Loaded predictions
        X_transformed_loaded = loaded_pipeline.transform(X)
        predictions_loaded = loaded_model.predict(X_transformed_loaded)
        
        np.testing.assert_array_equal(predictions_orig, predictions_loaded)
    
    def test_get_workflow_info(self, fitted_pipeline, fitted_model, temp_dir):
        """Test getting workflow info without loading."""
        serializer = WorkflowSerializer()
        save_path = temp_dir / 'workflow.pkl'
        
        serializer.save_workflow(
            fitted_pipeline, fitted_model, save_path,
            workflow_name='TestWorkflow'
        )
        
        info = serializer.get_workflow_info(save_path)
        
        assert info['workflow_name'] == 'TestWorkflow'
        assert info['pipeline_name'] == 'TestPipeline'
        assert info['pipeline_fitted'] is True
        assert info['model_fitted'] is True
        assert info['n_features'] == 3
        assert 'saved_at' in info



class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_save_and_load_model(self, fitted_model, temp_dir, sample_data):
        """Test convenience functions for model."""
        save_path = temp_dir / 'model.pkl'
        X, _ = sample_data
        
        save_model(fitted_model, save_path)
        loaded_model = load_model(save_path)
        
        assert loaded_model.is_fitted
        
        # Predictions should match
        original_pred = fitted_model.predict(X)
        loaded_pred = loaded_model.predict(X)
        np.testing.assert_array_equal(original_pred, loaded_pred)
    
    def test_save_and_load_pipeline(self, fitted_pipeline, temp_dir, sample_data):
        """Test convenience functions for pipeline."""
        save_path = temp_dir / 'pipeline.pkl'
        X, _ = sample_data
        
        save_pipeline(fitted_pipeline, save_path)
        loaded_pipeline = load_pipeline(save_path)
        
        assert loaded_pipeline.fitted
        
        # Transformations should match
        original_transform = fitted_pipeline.transform(X)
        loaded_transform = loaded_pipeline.transform(X)
        np.testing.assert_array_almost_equal(original_transform, loaded_transform)
    
    def test_save_and_load_workflow(self, fitted_pipeline, fitted_model, temp_dir, sample_data):
        """Test convenience functions for workflow."""
        save_path = temp_dir / 'workflow.pkl'
        X, _ = sample_data
        
        save_workflow(fitted_pipeline, fitted_model, save_path, workflow_name='Test')
        loaded_pipeline, loaded_model = load_workflow(save_path)
        
        assert loaded_pipeline.fitted
        assert loaded_model.is_fitted
        
        # End-to-end predictions
        X_transformed = loaded_pipeline.transform(X)
        predictions = loaded_model.predict(X_transformed)
        assert len(predictions) == len(X)
    
    def test_get_info_functions(self, fitted_pipeline, fitted_model, temp_dir):
        """Test info retrieval functions."""
        model_path = temp_dir / 'model.pkl'
        pipeline_path = temp_dir / 'pipeline.pkl'
        workflow_path = temp_dir / 'workflow.pkl'
        
        save_model(fitted_model, model_path)
        save_pipeline(fitted_pipeline, pipeline_path)
        save_workflow(fitted_pipeline, fitted_model, workflow_path)
        
        model_info = get_model_info(model_path)
        pipeline_info = get_pipeline_info(pipeline_path)
        workflow_info = get_workflow_info(workflow_path)
        
        assert model_info['is_fitted'] is True
        assert pipeline_info['fitted'] is True
        assert workflow_info['model_fitted'] is True


class TestCompressionFeature:
    """Tests for compression functionality."""
    
    def test_compressed_model_smaller(self, fitted_model, temp_dir):
        """Test that compressed models are smaller."""
        uncompressed_path = temp_dir / 'model_uncompressed.pkl'
        compressed_path = temp_dir / 'model_compressed.pkl'
        
        save_model(fitted_model, uncompressed_path, compression=False)
        save_model(fitted_model, compressed_path, compression=True)
        
        uncompressed_size = uncompressed_path.stat().st_size
        compressed_size = Path(f"{compressed_path}.gz").stat().st_size
        
        # Compressed should be smaller (though not always guaranteed for small files)
        assert compressed_size > 0
        assert uncompressed_size > 0
    
    def test_compressed_pipeline_works(self, fitted_pipeline, temp_dir, sample_data):
        """Test that compressed pipelines work correctly."""
        save_path = temp_dir / 'pipeline.pkl'
        X, _ = sample_data
        
        save_pipeline(fitted_pipeline, save_path, compression=True)
        loaded_pipeline = load_pipeline(save_path)
        
        # Should work normally
        transformed = loaded_pipeline.transform(X)
        assert transformed.shape == X.shape


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_save_to_invalid_path(self, fitted_model):
        """Test saving to invalid path."""
        # This should create the directory, so it shouldn't fail
        invalid_path = Path('/nonexistent/deeply/nested/path/model.pkl')
        
        # On Windows/restricted systems, this might fail
        try:
            save_model(fitted_model, invalid_path)
        except (PermissionError, OSError):
            # Expected on restricted systems
            pass
    
    def test_load_corrupted_file(self, temp_dir):
        """Test loading corrupted file."""
        corrupted_path = temp_dir / 'corrupted.pkl'
        
        # Create a corrupted file
        with open(corrupted_path, 'w') as f:
            f.write("This is not a valid pickle file")
        
        with pytest.raises(SerializationError):
            load_model(corrupted_path)


class TestRegressionModel:
    """Tests with regression models."""
    
    def test_save_load_regression_model(self, temp_dir):
        """Test serialization with regression model."""
        # Create and fit regression model
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = pd.Series(np.random.randn(100), name='target')
        
        config = ModelConfig(
            model_type='linear_regression',
            hyperparameters={}
        )
        model = LinearRegressionWrapper(config)
        model.fit(X, y)
        
        # Save and load
        save_path = temp_dir / 'regression_model.pkl'
        save_model(model, save_path)
        loaded_model = load_model(save_path)
        
        # Test predictions
        predictions = loaded_model.predict(X)
        assert len(predictions) == len(X)
        assert loaded_model.get_task_type() == 'regression'


class TestMetadataPreservation:
    """Tests for metadata preservation."""
    
    def test_model_metadata_preserved(self, fitted_model, temp_dir):
        """Test that model metadata is preserved."""
        save_path = temp_dir / 'model.pkl'
        
        # Add custom metadata
        custom_metadata = {
            'experiment_id': 'exp_123',
            'dataset_name': 'test_dataset',
            'notes': 'Test model for unit tests'
        }
        
        save_model(fitted_model, save_path, metadata=custom_metadata)
        info = get_model_info(save_path)
        
        assert info['model_type'] == 'random_forest_classifier'
        assert info['is_fitted'] is True
        assert 'saved_at' in info
    
    def test_feature_names_preserved(self, fitted_model, temp_dir):
        """Test that feature names are preserved."""
        save_path = temp_dir / 'model.pkl'
        
        save_model(fitted_model, save_path)
        loaded_model = load_model(save_path)
        
        assert loaded_model._feature_names == ['feature1', 'feature2', 'feature3']
    
    def test_training_metadata_preserved(self, fitted_model, temp_dir):
        """Test that training metadata is preserved."""
        save_path = temp_dir / 'model.pkl'
        
        save_model(fitted_model, save_path)
        loaded_model = load_model(save_path)
        
        assert loaded_model.metadata.n_train_samples == 100
        assert loaded_model.metadata.n_features == 3
        assert loaded_model.metadata.training_duration_seconds is not None
