"""
Unit tests for training module.

Tests the training functions, data splitting, and training results.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from app.ml_engine.models import create_model
from app.ml_engine.training import (
    train_model,
    evaluate_model,
    predict_with_model,
    get_model_info,
    TrainingResult,
    # Data splitting
    train_test_split,
    train_val_test_split,
    split_by_ratio,
    temporal_split,
    get_split_info,
    DataSplitResult,
)


class TestTrainModel:
    """Test train_model function."""
    
    @pytest.fixture
    def classification_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            random_state=42
        )
        return pd.DataFrame(X), pd.Series(y)
    
    @pytest.fixture
    def regression_data(self):
        """Create sample regression data."""
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            random_state=42
        )
        return pd.DataFrame(X), pd.Series(y)
    
    def test_train_classification_model(self, classification_data):
        """Test training a classification model."""
        X, y = classification_data
        model = create_model('logistic_regression', max_iter=200)
        
        result = train_model(model, X, y)
        
        assert isinstance(result, TrainingResult)
        assert result.model is model
        assert model.is_fitted
        assert result.train_score is not None
        assert 0.0 <= result.train_score <= 1.0
    
    def test_train_with_validation_set(self, classification_data):
        """Test training with validation set."""
        X, y = classification_data
        
        # Split data
        split_result = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train, y_train = split_result.get_train_data()
        X_val, y_val = split_result.get_test_data()
        
        # Train model
        model = create_model('logistic_regression', max_iter=200)
        result = train_model(model, X_train, y_train, X_val=X_val, y_val=y_val)
        
        assert result.val_score is not None
        assert 0.0 <= result.val_score <= 1.0
    
    def test_train_with_test_set(self, classification_data):
        """Test training with test set."""
        X, y = classification_data
        
        # Split data
        split_result = train_val_test_split(X, y, random_state=42)
        X_train, y_train = split_result.get_train_data()
        X_val, y_val = split_result.get_val_data()
        X_test, y_test = split_result.get_test_data()
        
        # Train model
        model = create_model('logistic_regression', max_iter=200)
        result = train_model(
            model, X_train, y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test
        )
        
        assert result.train_score is not None
        assert result.val_score is not None
        assert result.test_score is not None
    
    def test_train_regression_model(self, regression_data):
        """Test training a regression model."""
        X, y = regression_data
        model = create_model('linear_regression')
        
        result = train_model(model, X, y)
        
        assert isinstance(result, TrainingResult)
        assert result.train_score is not None
        assert result.train_score >= 0.0  # R² can be negative but usually positive
    
    def test_train_clustering_model(self):
        """Test training a clustering model."""
        X, _ = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X)
        
        model = create_model('kmeans', n_clusters=3)
        result = train_model(model, X_df)
        
        assert isinstance(result, TrainingResult)
        assert model.is_fitted
        # Clustering doesn't have train_score in the same way
    
    def test_training_result_attributes(self, classification_data):
        """Test TrainingResult attributes."""
        X, y = classification_data
        model = create_model('logistic_regression', max_iter=200)
        
        result = train_model(model, X, y)
        
        assert hasattr(result, 'model')
        assert hasattr(result, 'train_score')
        assert hasattr(result, 'val_score')
        assert hasattr(result, 'test_score')
        assert hasattr(result, 'training_time')
        assert result.training_time > 0


class TestEvaluateModel:
    """Test evaluate_model function."""
    
    @pytest.fixture
    def fitted_classification_model(self):
        """Create a fitted classification model."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)
        
        model = create_model('logistic_regression', max_iter=200)
        model.fit(X_df, y_series)
        
        return model, X_df, y_series
    
    def test_evaluate_classification_model(self, fitted_classification_model):
        """Test evaluating a classification model."""
        model, X, y = fitted_classification_model
        
        score = evaluate_model(model, X, y)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_evaluate_unfitted_model_raises_error(self):
        """Test that evaluating unfitted model raises error."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        model = create_model('logistic_regression')
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            evaluate_model(model, X, y)


class TestPredictWithModel:
    """Test predict_with_model function."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X)
        
        model = create_model('logistic_regression', max_iter=200)
        model.fit(X_df, y)
        
        return model, X_df
    
    def test_predict_with_model(self, fitted_model):
        """Test making predictions."""
        model, X = fitted_model
        
        predictions = predict_with_model(model, X)
        
        assert len(predictions) == len(X)
        assert predictions.dtype in [np.int32, np.int64]
    
    def test_predict_with_unfitted_model_raises_error(self):
        """Test that prediction with unfitted model raises error."""
        X, _ = make_classification(n_samples=50, n_features=5, random_state=42)
        model = create_model('logistic_regression')
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            predict_with_model(model, X)


class TestGetModelInfo:
    """Test get_model_info function."""
    
    def test_get_model_info_unfitted(self):
        """Test getting info from unfitted model."""
        model = create_model('random_forest_classifier', n_estimators=100)
        
        info = get_model_info(model)
        
        assert isinstance(info, dict)
        assert info['model_type'] == 'random_forest_classifier'
        assert info['task_type'] == 'classification'
        assert info['is_fitted'] is False
        assert 'hyperparameters' in info
    
    def test_get_model_info_fitted(self):
        """Test getting info from fitted model."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        
        model = create_model('random_forest_classifier', n_estimators=10)
        model.fit(X_df, y)
        
        info = get_model_info(model)
        
        assert info['is_fitted'] is True
        assert info['n_train_samples'] == 100
        assert info['n_features'] == 10
        assert len(info['feature_names']) == 10


class TestTrainTestSplit:
    """Test train_test_split function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        return pd.DataFrame(X), pd.Series(y)
    
    def test_basic_split(self, sample_data):
        """Test basic train/test split."""
        X, y = sample_data
        
        result = train_test_split(X, y, test_size=0.2, random_state=42)
        
        assert isinstance(result, DataSplitResult)
        assert len(result.X_train) == 80
        assert len(result.X_test) == 20
        assert len(result.y_train) == 80
        assert len(result.y_test) == 20
    
    def test_split_with_different_test_size(self, sample_data):
        """Test split with different test size."""
        X, y = sample_data
        
        result = train_test_split(X, y, test_size=0.3, random_state=42)
        
        assert len(result.X_train) == 70
        assert len(result.X_test) == 30
    
    def test_split_with_stratification(self, sample_data):
        """Test split with stratification."""
        X, y = sample_data
        
        result = train_test_split(X, y, test_size=0.2, stratify=True, random_state=42)
        
        # Check class distribution is maintained
        train_dist = np.bincount(result.y_train) / len(result.y_train)
        test_dist = np.bincount(result.y_test) / len(result.y_test)
        
        assert np.allclose(train_dist, test_dist, atol=0.1)
    
    def test_split_without_shuffle(self, sample_data):
        """Test split without shuffling."""
        X, y = sample_data
        
        result = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        assert result.shuffled is False
        # First 80 samples should be train, last 20 should be test
        assert len(result.X_train) == 80
        assert len(result.X_test) == 20
    
    def test_split_unsupervised(self):
        """Test split without target (unsupervised)."""
        X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
        X_df = pd.DataFrame(X)
        
        result = train_test_split(X_df, test_size=0.2, random_state=42)
        
        assert len(result.X_train) == 80
        assert len(result.X_test) == 20
        assert result.y_train is None
        assert result.y_test is None
    
    def test_split_invalid_test_size(self, sample_data):
        """Test split with invalid test size."""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="test_size must be between"):
            train_test_split(X, y, test_size=1.5)
    
    def test_split_empty_data(self):
        """Test split with empty data."""
        X = pd.DataFrame()
        y = pd.Series()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            train_test_split(X, y, test_size=0.2)
    
    def test_get_train_data(self, sample_data):
        """Test getting train data."""
        X, y = sample_data
        result = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train, y_train = result.get_train_data()
        
        assert len(X_train) == 80
        assert len(y_train) == 80
    
    def test_get_test_data(self, sample_data):
        """Test getting test data."""
        X, y = sample_data
        result = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_test, y_test = result.get_test_data()
        
        assert X_test is not None
        assert y_test is not None
        assert len(X_test) == 20
        assert len(y_test) == 20


class TestTrainValTestSplit:
    """Test train_val_test_split function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        return pd.DataFrame(X), pd.Series(y)
    
    def test_basic_three_way_split(self, sample_data):
        """Test basic train/val/test split."""
        X, y = sample_data
        
        result = train_val_test_split(
            X, y,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15,
            random_state=42
        )
        
        assert isinstance(result, DataSplitResult)
        # Allow for rounding differences (69-70, 15-16, etc.)
        assert 69 <= len(result.X_train) <= 70
        assert 15 <= len(result.X_val) <= 16
        assert 15 <= len(result.X_test) <= 16
        assert len(result.X_train) + len(result.X_val) + len(result.X_test) == 100
    
    def test_split_with_stratification(self, sample_data):
        """Test three-way split with stratification."""
        X, y = sample_data
        
        result = train_val_test_split(
            X, y,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15,
            stratify=True,
            random_state=42
        )
        
        # Check class distribution is maintained
        train_dist = np.bincount(result.y_train) / len(result.y_train)
        val_dist = np.bincount(result.y_val) / len(result.y_val)
        test_dist = np.bincount(result.y_test) / len(result.y_test)
        
        assert np.allclose(train_dist, val_dist, atol=0.15)
        assert np.allclose(train_dist, test_dist, atol=0.15)
    
    def test_split_sizes_sum_to_one(self, sample_data):
        """Test that split sizes must sum to 1.0."""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="must equal 1.0"):
            train_val_test_split(
                X, y,
                train_size=0.6,
                val_size=0.2,
                test_size=0.1  # Sum = 0.9, not 1.0
            )
    
    def test_get_val_data(self, sample_data):
        """Test getting validation data."""
        X, y = sample_data
        result = train_val_test_split(X, y, random_state=42)
        
        X_val, y_val = result.get_val_data()
        
        assert X_val is not None
        assert y_val is not None
        # Allow for rounding (15-16)
        assert 15 <= len(X_val) <= 16
        assert 15 <= len(y_val) <= 16
    
    def test_get_split_sizes(self, sample_data):
        """Test getting split sizes."""
        X, y = sample_data
        result = train_val_test_split(X, y, random_state=42)
        
        sizes = result.get_split_sizes()
        
        # Allow for rounding
        assert 69 <= sizes['train'] <= 70
        assert 15 <= sizes['val'] <= 16
        assert 15 <= sizes['test'] <= 16
        assert sizes['total'] == 100


class TestSplitByRatio:
    """Test split_by_ratio function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        return pd.DataFrame(X), pd.Series(y)
    
    def test_two_way_split(self, sample_data):
        """Test two-way split with custom ratios."""
        X, y = sample_data
        
        result = split_by_ratio(X, y, ratios=(0.8, 0.2), random_state=42)
        
        assert len(result.X_train) == 80
        assert len(result.X_test) == 20
    
    def test_three_way_split(self, sample_data):
        """Test three-way split with custom ratios."""
        X, y = sample_data
        
        result = split_by_ratio(X, y, ratios=(0.6, 0.2, 0.2), random_state=42)
        
        assert len(result.X_train) == 60
        assert len(result.X_val) == 20
        assert len(result.X_test) == 20
    
    def test_invalid_ratios(self, sample_data):
        """Test split with invalid number of ratios."""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="must be a tuple of 2 or 3"):
            split_by_ratio(X, y, ratios=(0.5, 0.3, 0.1, 0.1))


class TestTemporalSplit:
    """Test temporal_split function."""
    
    @pytest.fixture
    def time_series_data(self):
        """Create time series data."""
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            'feature_1': np.arange(n),
            'feature_2': np.random.randn(n)
        })
        y = pd.Series(np.arange(n) + np.random.randn(n) * 0.1)
        return X, y
    
    def test_temporal_split_preserves_order(self, time_series_data):
        """Test that temporal split preserves order."""
        X, y = time_series_data
        
        result = temporal_split(
            X, y,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15
        )
        
        # Check that data is not shuffled
        assert result.shuffled is False
        
        # Check that train comes before val comes before test
        assert result.X_train.iloc[-1, 0] < result.X_val.iloc[0, 0]
        assert result.X_val.iloc[-1, 0] < result.X_test.iloc[0, 0]
    
    def test_temporal_split_no_stratification(self, time_series_data):
        """Test that temporal split doesn't use stratification."""
        X, y = time_series_data
        
        result = temporal_split(X, y)
        
        assert result.stratified is False


class TestGetSplitInfo:
    """Test get_split_info function."""
    
    def test_get_split_info(self):
        """Test getting split information."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)
        
        result = train_val_test_split(X_df, y_series, random_state=42)
        info = get_split_info(result)
        
        assert isinstance(info, dict)
        assert 'sizes' in info
        assert 'ratios' in info
        assert 'stratified' in info
        assert 'shuffled' in info
        assert 'percentages' in info
        
        # Allow for rounding
        assert 69 <= info['sizes']['train'] <= 70
        assert 15 <= info['sizes']['val'] <= 16
        assert 15 <= info['sizes']['test'] <= 16
        assert info['has_validation'] is True
        assert info['has_test'] is True


class TestDataSplitResult:
    """Test DataSplitResult class."""
    
    def test_data_split_result_repr(self):
        """Test DataSplitResult string representation."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        result = train_test_split(X, y, test_size=0.2, random_state=42)
        
        repr_str = repr(result)
        
        assert 'DataSplitResult' in repr_str
        assert 'train=80' in repr_str
        assert 'test=20' in repr_str
        assert 'total=100' in repr_str


class TestIntegrationTrainingAndSplitting:
    """Test integration of training and data splitting."""
    
    def test_complete_ml_pipeline(self):
        """Test complete ML pipeline with splitting and training."""
        # Generate data
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            random_state=42
        )
        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)
        
        # Split data
        split_result = train_val_test_split(
            X_df, y_series,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15,
            random_state=42,
            stratify=True
        )
        
        X_train, y_train = split_result.get_train_data()
        X_val, y_val = split_result.get_val_data()
        X_test, y_test = split_result.get_test_data()
        
        # Create and train model
        model = create_model(
            'random_forest_classifier',
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        
        train_result = train_model(
            model,
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test
        )
        
        # Verify results
        assert model.is_fitted
        assert train_result.train_score is not None
        assert train_result.val_score is not None
        assert train_result.test_score is not None
        assert 0.0 <= train_result.train_score <= 1.0
        assert 0.0 <= train_result.val_score <= 1.0
        assert 0.0 <= train_result.test_score <= 1.0
        
        # Make predictions
        predictions = predict_with_model(model, X_test)
        assert len(predictions) == len(X_test)
        
        # Get model info
        info = get_model_info(model)
        assert info['is_fitted'] is True
        assert info['n_train_samples'] == len(X_train)
    
    def test_pipeline_with_different_models(self):
        """Test pipeline with different model types."""
        X, y = make_classification(n_samples=150, n_features=8, random_state=42)
        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)
        
        # Split data once
        split_result = train_test_split(X_df, y_series, test_size=0.2, random_state=42)
        X_train, y_train = split_result.get_train_data()
        X_test, y_test = split_result.get_test_data()
        
        # Test multiple models
        models = [
            'logistic_regression',
            'random_forest_classifier',
            'decision_tree_classifier'
        ]
        
        results = {}
        for model_id in models:
            model = create_model(model_id, random_state=42)
            result = train_model(model, X_train, y_train, X_test=X_test, y_test=y_test)
            results[model_id] = result.test_score
        
        # All models should have valid scores
        for model_id, score in results.items():
            assert score is not None
            assert 0.0 <= score <= 1.0
