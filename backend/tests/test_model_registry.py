"""
Unit tests for model registry and factory.

Tests the ModelFactory, model creation, and registry functionality.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from app.ml_engine.models import (
    ModelFactory,
    create_model,
    ModelConfig,
    BaseModelWrapper,
    # Classification
    LogisticRegressionWrapper,
    RandomForestClassifierWrapper,
    SVMClassifierWrapper,
    # Regression
    LinearRegressionWrapper,
    RidgeRegressionWrapper,
    RandomForestRegressorWrapper,
    # Clustering
    KMeansWrapper,
    DBSCANWrapper,
)


class TestModelFactory:
    """Test ModelFactory functionality."""
    
    def test_create_model_with_hyperparameters(self):
        """Test creating model with hyperparameters."""
        model = ModelFactory.create_model(
            'random_forest_classifier',
            n_estimators=50,
            max_depth=5
        )
        
        assert isinstance(model, RandomForestClassifierWrapper)
        assert model.config.model_type == 'random_forest_classifier'
        assert model.config.hyperparameters['n_estimators'] == 50
        assert model.config.hyperparameters['max_depth'] == 5
    
    def test_create_model_with_config(self):
        """Test creating model with ModelConfig."""
        config = ModelConfig(
            model_type='logistic_regression',
            hyperparameters={'C': 0.5, 'max_iter': 200}
        )
        
        model = ModelFactory.create_model('logistic_regression', config=config)
        
        assert isinstance(model, LogisticRegressionWrapper)
        assert model.config.hyperparameters['C'] == 0.5
        assert model.config.hyperparameters['max_iter'] == 200
    
    def test_create_model_invalid_id(self):
        """Test creating model with invalid ID."""
        with pytest.raises(ValueError, match="not found in registry"):
            ModelFactory.create_model('invalid_model_id')
    
    def test_get_available_models(self):
        """Test getting available models."""
        models = ModelFactory.get_available_models()
        
        assert isinstance(models, dict)
        assert len(models) > 0
        assert 'random_forest_classifier' in models
        assert 'linear_regression' in models
        assert 'kmeans' in models
    
    def test_is_model_available(self):
        """Test checking if model is available."""
        assert ModelFactory.is_model_available('random_forest_classifier')
        assert ModelFactory.is_model_available('linear_regression')
        assert not ModelFactory.is_model_available('invalid_model')
    
    def test_create_model_convenience_function(self):
        """Test convenience create_model function."""
        model = create_model('random_forest_classifier', n_estimators=100)
        
        assert isinstance(model, RandomForestClassifierWrapper)
        assert model.config.hyperparameters['n_estimators'] == 100


class TestClassificationModels:
    """Test classification model creation."""
    
    @pytest.fixture
    def classification_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        return pd.DataFrame(X), pd.Series(y)
    
    def test_create_logistic_regression(self):
        """Test creating logistic regression model."""
        model = create_model('logistic_regression', C=1.0, max_iter=100)
        
        assert isinstance(model, LogisticRegressionWrapper)
        assert model.get_task_type() == 'classification'
        assert not model.is_fitted
    
    def test_create_random_forest_classifier(self):
        """Test creating random forest classifier."""
        model = create_model(
            'random_forest_classifier',
            n_estimators=50,
            max_depth=10
        )
        
        assert isinstance(model, RandomForestClassifierWrapper)
        assert model.get_task_type() == 'classification'
    
    def test_create_svm_classifier(self):
        """Test creating SVM classifier."""
        model = create_model('svm_classifier', C=1.0, kernel='rbf')
        
        assert isinstance(model, SVMClassifierWrapper)
        assert model.get_task_type() == 'classification'
    
    def test_fit_classification_model(self, classification_data):
        """Test fitting a classification model."""
        X, y = classification_data
        model = create_model('logistic_regression', max_iter=200)
        
        # Fit model
        model.fit(X, y)
        
        assert model.is_fitted
        assert model.metadata.n_train_samples == len(X)
        assert model.metadata.n_features == X.shape[1]
    
    def test_predict_classification_model(self, classification_data):
        """Test prediction with classification model."""
        X, y = classification_data
        model = create_model('logistic_regression', max_iter=200)
        
        # Fit and predict
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert predictions.dtype in [np.int32, np.int64]
    
    def test_predict_proba_classification_model(self, classification_data):
        """Test probability prediction."""
        X, y = classification_data
        model = create_model('logistic_regression', max_iter=200)
        
        model.fit(X, y)
        probas = model.predict_proba(X)
        
        assert probas.shape[0] == len(X)
        assert probas.shape[1] == 2  # Binary classification
        assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1


class TestRegressionModels:
    """Test regression model creation."""
    
    @pytest.fixture
    def regression_data(self):
        """Create sample regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=10,
            n_informative=5,
            random_state=42
        )
        return pd.DataFrame(X), pd.Series(y)
    
    def test_create_linear_regression(self):
        """Test creating linear regression model."""
        model = create_model('linear_regression')
        
        assert isinstance(model, LinearRegressionWrapper)
        assert model.get_task_type() == 'regression'
    
    def test_create_ridge_regression(self):
        """Test creating ridge regression model."""
        model = create_model('ridge_regression', alpha=1.0)
        
        assert isinstance(model, RidgeRegressionWrapper)
        assert model.get_task_type() == 'regression'
    
    def test_create_random_forest_regressor(self):
        """Test creating random forest regressor."""
        model = create_model(
            'random_forest_regressor',
            n_estimators=50,
            max_depth=10
        )
        
        assert isinstance(model, RandomForestRegressorWrapper)
        assert model.get_task_type() == 'regression'
    
    def test_fit_regression_model(self, regression_data):
        """Test fitting a regression model."""
        X, y = regression_data
        model = create_model('linear_regression')
        
        model.fit(X, y)
        
        assert model.is_fitted
        assert model.metadata.n_train_samples == len(X)
    
    def test_predict_regression_model(self, regression_data):
        """Test prediction with regression model."""
        X, y = regression_data
        model = create_model('linear_regression')
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert predictions.dtype in [np.float32, np.float64]
    
    def test_score_regression_model(self, regression_data):
        """Test scoring regression model."""
        X, y = regression_data
        model = create_model('linear_regression')
        
        model.fit(X, y)
        score = model.score(X, y)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0  # R² score


class TestClusteringModels:
    """Test clustering model creation."""
    
    @pytest.fixture
    def clustering_data(self):
        """Create sample clustering data."""
        X, _ = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_clusters_per_class=1,
            random_state=42
        )
        return pd.DataFrame(X)
    
    def test_create_kmeans(self):
        """Test creating K-Means model."""
        model = create_model('kmeans', n_clusters=3)
        
        assert isinstance(model, KMeansWrapper)
        assert model.get_task_type() == 'clustering'
    
    def test_create_dbscan(self):
        """Test creating DBSCAN model."""
        model = create_model('dbscan', eps=0.5, min_samples=5)
        
        assert isinstance(model, DBSCANWrapper)
        assert model.get_task_type() == 'clustering'
    
    def test_fit_clustering_model(self, clustering_data):
        """Test fitting a clustering model."""
        X = clustering_data
        model = create_model('kmeans', n_clusters=3)
        
        model.fit(X)
        
        assert model.is_fitted
        assert model.metadata.n_train_samples == len(X)
    
    def test_predict_clustering_model(self, clustering_data):
        """Test prediction with clustering model."""
        X = clustering_data
        model = create_model('kmeans', n_clusters=3)
        
        model.fit(X)
        labels = model.predict(X)
        
        assert len(labels) == len(X)
        assert labels.dtype in [np.int32, np.int64]
        assert len(np.unique(labels)) <= 3  # At most 3 clusters
    
    def test_get_cluster_labels(self, clustering_data):
        """Test getting cluster labels."""
        X = clustering_data
        model = create_model('kmeans', n_clusters=3)
        
        model.fit(X)
        labels = model.get_labels()
        
        assert labels is not None
        assert len(labels) == len(X)
    
    def test_get_cluster_centers(self, clustering_data):
        """Test getting cluster centers."""
        X = clustering_data
        model = create_model('kmeans', n_clusters=3)
        
        model.fit(X)
        centers = model.get_cluster_centers()
        
        assert centers is not None
        assert centers.shape[0] == 3  # 3 clusters
        assert centers.shape[1] == X.shape[1]  # Same number of features


class TestModelConfig:
    """Test ModelConfig functionality."""
    
    def test_create_config(self):
        """Test creating model configuration."""
        config = ModelConfig(
            model_type='random_forest_classifier',
            hyperparameters={'n_estimators': 100, 'max_depth': 10},
            random_state=42
        )
        
        assert config.model_type == 'random_forest_classifier'
        assert config.hyperparameters['n_estimators'] == 100
        assert config.random_state == 42
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = ModelConfig(
            model_type='logistic_regression',
            hyperparameters={'C': 1.0},
            random_state=42
        )
        
        config_dict = config.to_dict()
        
        assert config_dict['model_type'] == 'logistic_regression'
        assert config_dict['hyperparameters']['C'] == 1.0
        assert config_dict['random_state'] == 42
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'model_type': 'random_forest_classifier',
            'hyperparameters': {'n_estimators': 50},
            'random_state': 42
        }
        
        config = ModelConfig.from_dict(config_dict)
        
        assert config.model_type == 'random_forest_classifier'
        assert config.hyperparameters['n_estimators'] == 50
        assert config.random_state == 42
    
    def test_config_validation_enabled(self):
        """Test config with validation enabled."""
        # Valid config should work
        config = ModelConfig(
            model_type='random_forest_classifier',
            hyperparameters={'n_estimators': 100},
            validate=True
        )
        assert config.hyperparameters['n_estimators'] == 100
    
    def test_config_validation_invalid_params(self):
        """Test config validation with invalid parameters."""
        with pytest.raises(ValueError, match="Invalid hyperparameters"):
            ModelConfig(
                model_type='random_forest_classifier',
                hyperparameters={'n_estimators': -10},  # Invalid
                validate=True
            )
    
    def test_config_validation_disabled(self):
        """Test config with validation disabled."""
        # Invalid params should be allowed when validation is disabled
        config = ModelConfig(
            model_type='random_forest_classifier',
            hyperparameters={'n_estimators': -10},  # Invalid but allowed
            validate=False
        )
        assert config.hyperparameters['n_estimators'] == -10


class TestModelFeatures:
    """Test model wrapper features."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y_series = pd.Series(y, name='target')
        
        model = create_model('random_forest_classifier', n_estimators=10, random_state=42)
        model.fit(X_df, y_series)
        return model, X_df, y_series
    
    def test_get_feature_importance(self, fitted_model):
        """Test getting feature importance."""
        model, X, y = fitted_model
        
        importance = model.get_feature_importance()
        
        assert importance is not None
        assert isinstance(importance, dict)
        assert len(importance) == X.shape[1]
        assert all(isinstance(v, (int, float)) for v in importance.values())
    
    def test_get_params(self, fitted_model):
        """Test getting model parameters."""
        model, _, _ = fitted_model
        
        params = model.get_params()
        
        assert isinstance(params, dict)
        assert 'n_estimators' in params
    
    def test_set_params(self):
        """Test setting model parameters."""
        model = create_model('random_forest_classifier', n_estimators=10)
        
        model.set_params(n_estimators=20, max_depth=5)
        
        assert model.config.hyperparameters['n_estimators'] == 20
        assert model.config.hyperparameters['max_depth'] == 5
    
    def test_model_repr(self):
        """Test model string representation."""
        model = create_model('random_forest_classifier')
        
        repr_str = repr(model)
        
        assert 'RandomForestClassifierWrapper' in repr_str
        assert 'random_forest_classifier' in repr_str
        assert 'not fitted' in repr_str
    
    def test_predict_before_fit_raises_error(self):
        """Test that prediction before fitting raises error."""
        X, _ = make_classification(n_samples=10, n_features=5, random_state=42)
        model = create_model('logistic_regression')
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(X)
    
    def test_predict_proba_before_fit_raises_error(self):
        """Test that predict_proba before fitting raises error."""
        X, _ = make_classification(n_samples=10, n_features=5, random_state=42)
        model = create_model('logistic_regression')
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict_proba(X)


class TestModelMetadata:
    """Test model training metadata."""
    
    def test_metadata_after_training(self):
        """Test metadata is populated after training."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y_series = pd.Series(y, name='target')
        
        model = create_model('logistic_regression', max_iter=200)
        model.fit(X_df, y_series)
        
        metadata = model.metadata
        
        assert metadata.train_start_time is not None
        assert metadata.train_end_time is not None
        assert metadata.training_duration_seconds is not None
        assert metadata.training_duration_seconds > 0
        assert metadata.n_train_samples == 100
        assert metadata.n_features == 10
        assert len(metadata.feature_names) == 10
        assert metadata.target_name == 'target'
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        model = create_model('logistic_regression', max_iter=200)
        model.fit(X, y)
        
        metadata_dict = model.metadata.to_dict()
        
        assert isinstance(metadata_dict, dict)
        assert 'train_start_time' in metadata_dict
        assert 'training_duration_seconds' in metadata_dict
        assert 'n_train_samples' in metadata_dict


class TestModelSerialization:
    """Test model save/load functionality."""
    
    def test_save_and_load_model(self, tmp_path):
        """Test saving and loading a model."""
        # Create and fit model
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        
        model = create_model('random_forest_classifier', n_estimators=10, random_state=42)
        model.fit(X_df, y)
        
        # Save model
        model_path = tmp_path / "test_model.joblib"
        model.save(model_path)
        
        assert model_path.exists()
        
        # Load model
        loaded_model = RandomForestClassifierWrapper.load(model_path)
        
        assert loaded_model.is_fitted
        assert loaded_model.config.model_type == 'random_forest_classifier'
        assert loaded_model.metadata.n_train_samples == 100
        
        # Test predictions match
        original_pred = model.predict(X_df)
        loaded_pred = loaded_model.predict(X_df)
        
        assert np.array_equal(original_pred, loaded_pred)
    
    def test_save_unfitted_model_raises_error(self, tmp_path):
        """Test that saving unfitted model raises error."""
        model = create_model('logistic_regression')
        model_path = tmp_path / "test_model.joblib"
        
        with pytest.raises(RuntimeError, match="hasn't been fitted"):
            model.save(model_path)
    
    def test_load_nonexistent_model_raises_error(self, tmp_path):
        """Test that loading nonexistent model raises error."""
        model_path = tmp_path / "nonexistent.joblib"
        
        with pytest.raises(FileNotFoundError):
            RandomForestClassifierWrapper.load(model_path)


class TestMultipleModelTypes:
    """Test creating and using multiple model types."""
    
    def test_all_classification_models(self):
        """Test creating all classification models."""
        classification_models = [
            'logistic_regression',
            'decision_tree_classifier',
            'random_forest_classifier',
            'svm_classifier',
            'knn_classifier',
        ]
        
        for model_id in classification_models:
            model = create_model(model_id)
            assert model.get_task_type() == 'classification'
            assert not model.is_fitted
    
    def test_all_regression_models(self):
        """Test creating all regression models."""
        regression_models = [
            'linear_regression',
            'ridge_regression',
            'lasso_regression',
            'random_forest_regressor',
        ]
        
        for model_id in regression_models:
            model = create_model(model_id)
            assert model.get_task_type() == 'regression'
            assert not model.is_fitted
    
    def test_all_clustering_models(self):
        """Test creating all clustering models."""
        clustering_models = [
            'kmeans',
            'dbscan',
            'agglomerative_clustering',
            'gaussian_mixture',
        ]
        
        for model_id in clustering_models:
            model = create_model(model_id)
            assert model.get_task_type() == 'clustering'
            assert not model.is_fitted
