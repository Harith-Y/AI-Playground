"""
Tests for Optimized Inference Engine

Tests ML-73: Optimize inference speed
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from app.ml_engine.inference import OptimizedPredictor, ModelCache, get_optimized_predictor
from app.ml_engine.models.registry import ModelFactory
from app.ml_engine.models.base import ModelConfig


@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_model(temp_dir):
    """Create a sample trained model."""
    config = ModelConfig(
        model_type="random_forest_classifier",
        hyperparameters={"n_estimators": 10, "max_depth": 3},
        random_state=42
    )

    model = ModelFactory.create_model("random_forest_classifier", config=config)

    # Train with sample data
    X_train = np.random.randn(100, 5)
    y_train = np.random.randint(0, 2, 100)
    model.fit(X_train, y_train)

    # Save model
    model_path = Path(temp_dir) / "test_model.joblib"
    model.save(str(model_path))

    return str(model_path)


@pytest.fixture
def predictor():
    """Create predictor instance."""
    pred = OptimizedPredictor(cache_size=5)
    pred.clear_cache()  # Ensure clean state
    return pred


class TestModelCache:
    """Tests for ModelCache class."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = ModelCache(max_cache_size=10)
        assert cache.max_cache_size == 10
        stats = cache.get_stats()
        assert stats['size'] == 0
        assert stats['max_size'] == 10

    def test_cache_put_and_get(self, sample_model):
        """Test putting and getting models from cache."""
        cache = ModelCache(max_cache_size=3)

        # Load model
        from app.ml_engine.models.base import BaseModelWrapper
        model = BaseModelWrapper.load(sample_model)

        # Cache model
        cache.put(sample_model, model)

        # Retrieve from cache
        cached_model = cache.get(sample_model)
        assert cached_model is not None
        assert cached_model.is_fitted

    def test_cache_miss(self):
        """Test cache miss."""
        cache = ModelCache(max_cache_size=3)
        result = cache.get("nonexistent_path")
        assert result is None

    def test_lru_eviction(self, sample_model, temp_dir):
        """Test LRU eviction when cache is full."""
        cache = ModelCache(max_cache_size=2)

        # Create and cache 3 models
        from app.ml_engine.models.base import BaseModelWrapper

        model_paths = []
        for i in range(3):
            config = ModelConfig(
                model_type="logistic_regression",
                hyperparameters={"max_iter": 10},
                random_state=42 + i
            )

            model = ModelFactory.create_model("logistic_regression", config=config)
            X = np.random.randn(50, 3)
            y = np.random.randint(0, 2, 50)
            model.fit(X, y)

            path = str(Path(temp_dir) / f"model_{i}.joblib")
            model.save(path)
            model_paths.append(path)

            cache.put(path, model)

        # Cache size should be at max
        stats = cache.get_stats()
        assert stats['size'] == 2

        # First model should have been evicted
        assert cache.get(model_paths[0]) is None
        # Last two should be cached
        assert cache.get(model_paths[1]) is not None
        assert cache.get(model_paths[2]) is not None

    def test_cache_clear(self, sample_model):
        """Test cache clearing."""
        cache = ModelCache(max_cache_size=3)

        from app.ml_engine.models.base import BaseModelWrapper
        model = BaseModelWrapper.load(sample_model)
        cache.put(sample_model, model)

        assert cache.get_stats()['size'] == 1

        cache.clear()
        assert cache.get_stats()['size'] == 0


class TestOptimizedPredictor:
    """Tests for OptimizedPredictor class."""

    def test_predictor_initialization(self):
        """Test predictor initialization."""
        predictor = OptimizedPredictor(cache_size=10)
        assert predictor.model_cache.max_cache_size == 10

        stats = predictor.get_cache_stats()
        assert stats['size'] == 0

    def test_load_model_with_cache(self, predictor, sample_model):
        """Test model loading with caching."""
        # First load (cold)
        model1 = predictor.load_model(sample_model, use_cache=True)
        assert model1.is_fitted

        # Check cache
        stats = predictor.get_cache_stats()
        assert stats['size'] == 1

        # Second load (warm - from cache)
        model2 = predictor.load_model(sample_model, use_cache=True)
        assert model2.is_fitted

        # Should be same instance from cache
        assert model1 is model2

    def test_load_model_without_cache(self, predictor, sample_model):
        """Test model loading without caching."""
        model1 = predictor.load_model(sample_model, use_cache=False)
        assert model1.is_fitted

        # Cache should be empty
        stats = predictor.get_cache_stats()
        assert stats['size'] == 0

    def test_predict_basic(self, predictor, sample_model):
        """Test basic prediction."""
        X_test = np.random.randn(10, 5)

        result = predictor.predict(
            model_path=sample_model,
            X=X_test,
            use_cache=True
        )

        assert 'predictions' in result
        assert 'n_samples' in result
        assert result['n_samples'] == 10
        assert 'inference_time_seconds' in result
        assert 'throughput_samples_per_sec' in result

    def test_predict_with_probabilities(self, predictor, sample_model):
        """Test prediction with probabilities."""
        X_test = np.random.randn(10, 5)

        result = predictor.predict(
            model_path=sample_model,
            X=X_test,
            return_probabilities=True,
            use_cache=True
        )

        assert 'predictions' in result
        assert 'probabilities' in result
        assert 'classes' in result
        assert len(result['probabilities']) == 10

    def test_predict_with_dataframe(self, predictor, sample_model):
        """Test prediction with pandas DataFrame input."""
        X_test = pd.DataFrame(
            np.random.randn(10, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )

        result = predictor.predict(
            model_path=sample_model,
            X=X_test,
            use_cache=True
        )

        assert result['n_samples'] == 10

    def test_predict_from_array_batched(self, predictor, sample_model):
        """Test batched prediction on large array."""
        X_test = np.random.randn(1000, 5)

        result = predictor.predict_from_array(
            model_path=sample_model,
            X=X_test,
            batch_size=100,
            use_cache=True
        )

        assert result['n_samples'] == 1000
        assert len(result['predictions']) == 1000

    def test_predict_batch_csv(self, predictor, sample_model, temp_dir):
        """Test batch prediction from CSV file."""
        # Create test CSV
        test_data = pd.DataFrame(
            np.random.randn(100, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )

        data_path = Path(temp_dir) / "test_data.csv"
        test_data.to_csv(data_path, index=False)

        output_path = Path(temp_dir) / "predictions.csv"

        # Run batch prediction
        result_df = predictor.predict_batch(
            model_path=sample_model,
            data_path=str(data_path),
            output_path=str(output_path),
            batch_size=50,
            use_cache=True
        )

        assert len(result_df) == 100
        assert 'prediction' in result_df.columns
        assert output_path.exists()

    def test_predict_batch_parquet(self, predictor, sample_model, temp_dir):
        """Test batch prediction from Parquet file."""
        # Create test Parquet
        test_data = pd.DataFrame(
            np.random.randn(100, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )

        data_path = Path(temp_dir) / "test_data.parquet"
        test_data.to_parquet(data_path, index=False)

        # Run batch prediction
        result_df = predictor.predict_batch(
            model_path=sample_model,
            data_path=str(data_path),
            batch_size=50,
            use_cache=True
        )

        assert len(result_df) == 100

    def test_predict_batch_with_probabilities(self, predictor, sample_model, temp_dir):
        """Test batch prediction with probabilities."""
        # Create test data
        test_data = pd.DataFrame(
            np.random.randn(50, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )

        data_path = Path(temp_dir) / "test_data.csv"
        test_data.to_csv(data_path, index=False)

        # Run batch prediction with probabilities
        result_df = predictor.predict_batch(
            model_path=sample_model,
            data_path=str(data_path),
            return_probabilities=True,
            use_cache=True
        )

        assert 'prediction' in result_df.columns
        # Should have probability columns
        prob_cols = [col for col in result_df.columns if col.startswith('prob_')]
        assert len(prob_cols) > 0

    def test_warmup(self, predictor, sample_model):
        """Test model warmup."""
        # Initially cache is empty
        assert predictor.get_cache_stats()['size'] == 0

        # Warmup model
        warmup_time = predictor.warmup(sample_model)

        assert warmup_time > 0
        assert predictor.get_cache_stats()['size'] == 1

    def test_clear_cache(self, predictor, sample_model):
        """Test cache clearing."""
        # Load model to cache
        predictor.load_model(sample_model, use_cache=True)
        assert predictor.get_cache_stats()['size'] == 1

        # Clear cache
        predictor.clear_cache()
        assert predictor.get_cache_stats()['size'] == 0

    def test_cache_stats(self, predictor, sample_model):
        """Test cache statistics."""
        stats = predictor.get_cache_stats()
        assert 'size' in stats
        assert 'max_size' in stats
        assert 'cached_models' in stats
        assert 'utilization' in stats

        # Load model
        predictor.load_model(sample_model, use_cache=True)

        stats = predictor.get_cache_stats()
        assert stats['size'] == 1
        assert sample_model in stats['cached_models']
        assert stats['utilization'] > 0

    def test_model_not_found(self, predictor):
        """Test handling of non-existent model."""
        with pytest.raises(FileNotFoundError):
            predictor.load_model("/nonexistent/model.joblib")

    def test_performance_improvement_with_cache(self, predictor, sample_model):
        """Test that caching improves performance."""
        X_test = np.random.randn(100, 5)

        # Clear cache for fair comparison
        predictor.clear_cache()

        # Cold start
        result_cold = predictor.predict(sample_model, X_test, use_cache=True)
        cold_time = result_cold['inference_time_seconds']

        # Warm start
        result_warm = predictor.predict(sample_model, X_test, use_cache=True)
        warm_time = result_warm['inference_time_seconds']

        # Warm should be faster (or at least not slower)
        assert warm_time <= cold_time

        # Predictions should be identical
        np.testing.assert_array_equal(
            result_cold['predictions'],
            result_warm['predictions']
        )


class TestSingletonPredictor:
    """Test singleton predictor instance."""

    def test_singleton_instance(self):
        """Test that get_optimized_predictor returns singleton."""
        predictor1 = get_optimized_predictor()
        predictor2 = get_optimized_predictor()

        assert predictor1 is predictor2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data(self, predictor, sample_model):
        """Test prediction with empty data."""
        X_empty = np.array([]).reshape(0, 5)

        result = predictor.predict(sample_model, X_empty)
        assert result['n_samples'] == 0

    def test_single_sample(self, predictor, sample_model):
        """Test prediction with single sample."""
        X_single = np.random.randn(1, 5)

        result = predictor.predict(sample_model, X_single)
        assert result['n_samples'] == 1

    def test_large_batch(self, predictor, sample_model):
        """Test prediction with very large batch."""
        X_large = np.random.randn(50000, 5)

        result = predictor.predict_from_array(
            model_path=sample_model,
            X=X_large,
            batch_size=10000
        )

        assert result['n_samples'] == 50000
        assert len(result['predictions']) == 50000

    def test_unsupported_file_format(self, predictor, sample_model, temp_dir):
        """Test batch prediction with unsupported file format."""
        data_path = Path(temp_dir) / "test_data.txt"
        data_path.write_text("not a valid format")

        with pytest.raises(ValueError, match="Unsupported file format"):
            predictor.predict_batch(sample_model, str(data_path))


class TestPerformanceMetrics:
    """Test performance metrics calculation."""

    def test_throughput_calculation(self, predictor, sample_model):
        """Test throughput metrics are calculated correctly."""
        X_test = np.random.randn(1000, 5)

        result = predictor.predict(sample_model, X_test)

        # Throughput should be samples/time
        expected_throughput = result['n_samples'] / result['inference_time_seconds']
        assert abs(result['throughput_samples_per_sec'] - expected_throughput) < 1.0

    def test_inference_time_positive(self, predictor, sample_model):
        """Test that inference time is always positive."""
        X_test = np.random.randn(10, 5)

        result = predictor.predict(sample_model, X_test)

        assert result['inference_time_seconds'] > 0
        assert result['throughput_samples_per_sec'] > 0
