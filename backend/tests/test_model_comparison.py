"""
Tests for Model Comparison Service and Endpoints

Tests model comparison logic, ranking algorithms, and API endpoints.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from uuid import uuid4, UUID
from datetime import datetime
from sqlalchemy.orm import Session

from app.services.model_comparison_service import ModelComparisonService
from app.schemas.model import (
    CompareModelsRequest,
    ModelComparisonResponse,
    ModelRankingRequest,
    ModelRankingResponse
)
from app.models.model_run import ModelRun


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def db_session():
    """Mock database session."""
    return Mock(spec=Session)


@pytest.fixture
def classification_model_runs():
    """Create sample classification model runs for testing."""
    runs = []
    
    # Model 1: Random Forest (best)
    run1 = Mock(spec=ModelRun)
    run1.id = uuid4()
    run1.experiment_id = uuid4()
    run1.model_type = "random_forest_classifier"
    run1.status = "completed"
    run1.metrics = {
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.93,
        "f1_score": 0.935,
        "roc_auc": 0.96
    }
    run1.hyperparameters = {"n_estimators": 100, "max_depth": 10}
    run1.training_time = 45.5
    run1.created_at = datetime(2025, 12, 30, 10, 0, 0)
    runs.append(run1)
    
    # Model 2: Logistic Regression (fast but lower performance)
    run2 = Mock(spec=ModelRun)
    run2.id = uuid4()
    run2.experiment_id = uuid4()
    run2.model_type = "logistic_regression"
    run2.status = "completed"
    run2.metrics = {
        "accuracy": 0.90,
        "precision": 0.89,
        "recall": 0.88,
        "f1_score": 0.885,
        "roc_auc": 0.91
    }
    run2.hyperparameters = {"C": 1.0, "penalty": "l2"}
    run2.training_time = 5.2
    run2.created_at = datetime(2025, 12, 30, 10, 5, 0)
    runs.append(run2)
    
    # Model 3: SVM (medium performance)
    run3 = Mock(spec=ModelRun)
    run3.id = uuid4()
    run3.experiment_id = uuid4()
    run3.model_type = "svm_classifier"
    run3.status = "completed"
    run3.metrics = {
        "accuracy": 0.92,
        "precision": 0.91,
        "recall": 0.90,
        "f1_score": 0.905,
        "roc_auc": 0.93
    }
    run3.hyperparameters = {"C": 10.0, "kernel": "rbf"}
    run3.training_time = 30.1
    run3.created_at = datetime(2025, 12, 30, 10, 10, 0)
    runs.append(run3)
    
    return runs


@pytest.fixture
def regression_model_runs():
    """Create sample regression model runs for testing."""
    runs = []
    
    # Model 1: Random Forest (best)
    run1 = Mock(spec=ModelRun)
    run1.id = uuid4()
    run1.experiment_id = uuid4()
    run1.model_type = "random_forest_regressor"
    run1.status = "completed"
    run1.metrics = {
        "r2_score": 0.92,
        "rmse": 0.15,
        "mae": 0.12,
        "mse": 0.0225
    }
    run1.hyperparameters = {"n_estimators": 200, "max_depth": 15}
    run1.training_time = 60.0
    run1.created_at = datetime(2025, 12, 30, 10, 0, 0)
    runs.append(run1)
    
    # Model 2: Linear Regression (fast but lower performance)
    run2 = Mock(spec=ModelRun)
    run2.id = uuid4()
    run2.experiment_id = uuid4()
    run2.model_type = "linear_regression"
    run2.status = "completed"
    run2.metrics = {
        "r2_score": 0.85,
        "rmse": 0.22,
        "mae": 0.18,
        "mse": 0.0484
    }
    run2.hyperparameters = {"fit_intercept": True}
    run2.training_time = 2.5
    run2.created_at = datetime(2025, 12, 30, 10, 5, 0)
    runs.append(run2)
    
    return runs


# ============================================================================
# Service Tests - Model Comparison
# ============================================================================


class TestModelComparisonService:
    """Test ModelComparisonService methods."""
    
    def test_compare_classification_models(self, db_session, classification_model_runs):
        """Test comparing classification models."""
        service = ModelComparisonService(db_session)
        
        # Mock database query
        model_ids = [run.id for run in classification_model_runs]
        db_session.query.return_value.filter.return_value.all.return_value = classification_model_runs
        
        request = CompareModelsRequest(
            model_run_ids=model_ids,
            ranking_criteria="f1_score"
        )
        
        result = service.compare_models(request)
        
        # Assertions
        assert isinstance(result, ModelComparisonResponse)
        assert result.task_type == "classification"
        assert result.total_models == 3
        assert len(result.compared_models) == 3
        assert result.ranking_criteria == "f1_score"
        
        # Best model should be random_forest_classifier
        assert result.best_model.model_type == "random_forest_classifier"
        assert result.best_model.rank == 1
        assert result.best_model.ranking_score == 0.935
        
        # Check rankings
        assert result.compared_models[0].rank == 1  # Random Forest
        assert result.compared_models[1].rank == 2  # SVM
        assert result.compared_models[2].rank == 3  # Logistic Regression
    
    def test_compare_regression_models(self, db_session, regression_model_runs):
        """Test comparing regression models."""
        service = ModelComparisonService(db_session)
        
        model_ids = [run.id for run in regression_model_runs]
        db_session.query.return_value.filter.return_value.all.return_value = regression_model_runs
        
        request = CompareModelsRequest(
            model_run_ids=model_ids,
            ranking_criteria="r2_score"
        )
        
        result = service.compare_models(request)
        
        assert result.task_type == "regression"
        assert result.total_models == 2
        assert result.ranking_criteria == "r2_score"
        assert result.best_model.model_type == "random_forest_regressor"
        assert result.best_model.ranking_score == 0.92
    
    def test_auto_detect_metrics(self, db_session, classification_model_runs):
        """Test automatic metric detection."""
        service = ModelComparisonService(db_session)
        
        model_ids = [run.id for run in classification_model_runs]
        db_session.query.return_value.filter.return_value.all.return_value = classification_model_runs
        
        request = CompareModelsRequest(
            model_run_ids=model_ids,
            comparison_metrics=None,  # Auto-detect
            ranking_criteria=None  # Auto-detect
        )
        
        result = service.compare_models(request)
        
        # Should auto-detect classification metrics and f1_score as ranking
        assert result.ranking_criteria == "f1_score"
        assert len(result.metric_statistics) > 0
        
        # Check that classification metrics are included
        metric_names = [stat.metric_name for stat in result.metric_statistics]
        assert "accuracy" in metric_names
        assert "f1_score" in metric_names
    
    def test_metric_statistics_calculation(self, db_session, classification_model_runs):
        """Test metric statistics calculation."""
        service = ModelComparisonService(db_session)
        
        model_ids = [run.id for run in classification_model_runs]
        db_session.query.return_value.filter.return_value.all.return_value = classification_model_runs
        
        request = CompareModelsRequest(
            model_run_ids=model_ids,
            comparison_metrics=["accuracy", "f1_score"]
        )
        
        result = service.compare_models(request)
        
        # Check accuracy statistics
        accuracy_stat = next(
            (s for s in result.metric_statistics if s.metric_name == "accuracy"),
            None
        )
        assert accuracy_stat is not None
        assert accuracy_stat.min == 0.90
        assert accuracy_stat.max == 0.95
        assert 0.90 <= accuracy_stat.mean <= 0.95
        assert accuracy_stat.std >= 0
    
    def test_recommendations_generation(self, db_session, classification_model_runs):
        """Test recommendation generation."""
        service = ModelComparisonService(db_session)
        
        model_ids = [run.id for run in classification_model_runs]
        db_session.query.return_value.filter.return_value.all.return_value = classification_model_runs
        
        request = CompareModelsRequest(model_run_ids=model_ids)
        result = service.compare_models(request)
        
        # Should have recommendations
        assert len(result.recommendations) > 0
        
        # First recommendation should mention best model
        assert "random_forest_classifier" in result.recommendations[0]
        assert "f1_score" in result.recommendations[0]
    
    def test_training_time_recommendation(self, db_session, classification_model_runs):
        """Test recommendation for faster models with similar performance."""
        service = ModelComparisonService(db_session)
        
        # Modify model 2 to have similar performance but much faster
        classification_model_runs[1].metrics["f1_score"] = 0.930  # Close to best (0.935)
        classification_model_runs[1].training_time = 5.0  # Much faster
        
        model_ids = [run.id for run in classification_model_runs]
        db_session.query.return_value.filter.return_value.all.return_value = classification_model_runs
        
        request = CompareModelsRequest(model_run_ids=model_ids)
        result = service.compare_models(request)
        
        # Should recommend faster model
        recommendations_text = " ".join(result.recommendations)
        assert "faster" in recommendations_text.lower()
    
    def test_compare_with_custom_metrics(self, db_session, classification_model_runs):
        """Test comparison with custom metric list."""
        service = ModelComparisonService(db_session)
        
        model_ids = [run.id for run in classification_model_runs]
        db_session.query.return_value.filter.return_value.all.return_value = classification_model_runs
        
        request = CompareModelsRequest(
            model_run_ids=model_ids,
            comparison_metrics=["precision", "recall"],
            ranking_criteria="precision"
        )
        
        result = service.compare_models(request)
        
        # Should only have statistics for requested metrics
        metric_names = [stat.metric_name for stat in result.metric_statistics]
        assert "precision" in metric_names
        assert "recall" in metric_names
        assert len(metric_names) == 2


# ============================================================================
# Service Tests - Model Ranking
# ============================================================================


class TestModelRanking:
    """Test model ranking with weighted criteria."""
    
    def test_rank_with_equal_weights(self, db_session, classification_model_runs):
        """Test ranking with equal weights."""
        service = ModelComparisonService(db_session)
        
        model_ids = [run.id for run in classification_model_runs]
        db_session.query.return_value.filter.return_value.all.return_value = classification_model_runs
        
        request = ModelRankingRequest(
            model_run_ids=model_ids,
            ranking_weights={
                "precision": 0.5,
                "recall": 0.5
            }
        )
        
        result = service.rank_models(request)
        
        assert isinstance(result, ModelRankingResponse)
        assert len(result.ranked_models) == 3
        
        # Models should be ranked
        assert result.ranked_models[0].rank == 1
        assert result.ranked_models[1].rank == 2
        assert result.ranked_models[2].rank == 3
        
        # Best model should be first
        assert result.best_model.rank == 1
        assert result.best_model.model_run_id == result.ranked_models[0].model_run_id
    
    def test_rank_with_custom_weights(self, db_session, classification_model_runs):
        """Test ranking with custom weights favoring precision."""
        service = ModelComparisonService(db_session)
        
        model_ids = [run.id for run in classification_model_runs]
        db_session.query.return_value.filter.return_value.all.return_value = classification_model_runs
        
        request = ModelRankingRequest(
            model_run_ids=model_ids,
            ranking_weights={
                "precision": 0.7,
                "recall": 0.2,
                "accuracy": 0.1
            }
        )
        
        result = service.rank_models(request)
        
        # Check that weights are preserved
        assert result.ranking_weights == request.ranking_weights
        
        # Check composite scores
        for ranked_model in result.ranked_models:
            assert ranked_model.composite_score >= 0.0
            assert ranked_model.composite_score <= 1.0
            
            # Check weighted contributions
            assert "precision" in ranked_model.weighted_contributions
            assert "recall" in ranked_model.weighted_contributions
            assert "accuracy" in ranked_model.weighted_contributions
    
    def test_rank_weights_sum_validation(self, db_session, classification_model_runs):
        """Test that weights must sum to 1.0."""
        service = ModelComparisonService(db_session)
        
        model_ids = [run.id for run in classification_model_runs]
        db_session.query.return_value.filter.return_value.all.return_value = classification_model_runs
        
        request = ModelRankingRequest(
            model_run_ids=model_ids,
            ranking_weights={
                "precision": 0.5,
                "recall": 0.3  # Sum is 0.8, not 1.0
            }
        )
        
        with pytest.raises(ValueError, match="must sum to 1.0"):
            service.rank_models(request)
    
    def test_rank_score_range(self, db_session, classification_model_runs):
        """Test score range calculation."""
        service = ModelComparisonService(db_session)
        
        model_ids = [run.id for run in classification_model_runs]
        db_session.query.return_value.filter.return_value.all.return_value = classification_model_runs
        
        request = ModelRankingRequest(
            model_run_ids=model_ids,
            ranking_weights={"f1_score": 1.0}
        )
        
        result = service.rank_models(request)
        
        # Check score range
        assert "min" in result.score_range
        assert "max" in result.score_range
        assert "spread" in result.score_range
        
        assert result.score_range["min"] >= 0.0
        assert result.score_range["max"] <= 1.0
        assert result.score_range["spread"] >= 0.0


# ============================================================================
# Validation Tests
# ============================================================================


class TestValidation:
    """Test validation logic."""
    
    def test_insufficient_models(self, db_session):
        """Test that at least 2 models are required."""
        service = ModelComparisonService(db_session)
        
        # Only one model
        run1 = Mock(spec=ModelRun)
        run1.id = uuid4()
        run1.status = "completed"
        run1.metrics = {"accuracy": 0.95}
        
        db_session.query.return_value.filter.return_value.all.return_value = [run1]
        
        request = CompareModelsRequest(model_run_ids=[run1.id])
        
        with pytest.raises(ValueError, match="At least 2 models required"):
            service.compare_models(request)
    
    def test_incomplete_models(self, db_session):
        """Test that all models must be completed."""
        service = ModelComparisonService(db_session)
        
        run1 = Mock(spec=ModelRun)
        run1.id = uuid4()
        run1.status = "completed"
        run1.metrics = {"accuracy": 0.95}
        
        run2 = Mock(spec=ModelRun)
        run2.id = uuid4()
        run2.status = "running"  # Not completed
        run2.metrics = {"accuracy": 0.90}
        
        db_session.query.return_value.filter.return_value.all.return_value = [run1, run2]
        
        request = CompareModelsRequest(model_run_ids=[run1.id, run2.id])
        
        with pytest.raises(ValueError, match="must have status 'completed'"):
            service.compare_models(request)
    
    def test_models_without_metrics(self, db_session):
        """Test that all models must have metrics."""
        service = ModelComparisonService(db_session)
        
        run1 = Mock(spec=ModelRun)
        run1.id = uuid4()
        run1.status = "completed"
        run1.metrics = {"accuracy": 0.95}
        
        run2 = Mock(spec=ModelRun)
        run2.id = uuid4()
        run2.status = "completed"
        run2.metrics = None  # No metrics
        
        db_session.query.return_value.filter.return_value.all.return_value = [run1, run2]
        
        request = CompareModelsRequest(model_run_ids=[run1.id, run2.id])
        
        with pytest.raises(ValueError, match="must have metrics"):
            service.compare_models(request)
    
    def test_models_not_found(self, db_session):
        """Test handling of missing models."""
        service = ModelComparisonService(db_session)
        
        model_ids = [uuid4(), uuid4(), uuid4()]
        
        # Only return 2 models instead of 3
        run1 = Mock(spec=ModelRun)
        run1.id = model_ids[0]
        run1.status = "completed"
        run1.metrics = {"accuracy": 0.95}
        
        run2 = Mock(spec=ModelRun)
        run2.id = model_ids[1]
        run2.status = "completed"
        run2.metrics = {"accuracy": 0.90}
        
        db_session.query.return_value.filter.return_value.all.return_value = [run1, run2]
        
        request = CompareModelsRequest(model_run_ids=model_ids)
        
        with pytest.raises(ValueError, match="Model runs not found"):
            service.compare_models(request)
    
    def test_missing_ranking_metric(self, db_session, classification_model_runs):
        """Test ranking with metric not present in models."""
        service = ModelComparisonService(db_session)
        
        model_ids = [run.id for run in classification_model_runs]
        db_session.query.return_value.filter.return_value.all.return_value = classification_model_runs
        
        request = ModelRankingRequest(
            model_run_ids=model_ids,
            ranking_weights={"nonexistent_metric": 1.0}
        )
        
        with pytest.raises(ValueError, match="not found in model"):
            service.rank_models(request)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_identical_scores(self, db_session):
        """Test models with identical scores."""
        service = ModelComparisonService(db_session)
        
        # Create models with identical scores
        runs = []
        for i in range(3):
            run = Mock(spec=ModelRun)
            run.id = uuid4()
            run.experiment_id = uuid4()
            run.model_type = f"model_{i}"
            run.status = "completed"
            run.metrics = {"accuracy": 0.95, "f1_score": 0.94}  # Identical
            run.hyperparameters = {}
            run.training_time = 10.0
            run.created_at = datetime(2025, 12, 30, 10, i, 0)
            runs.append(run)
        
        model_ids = [run.id for run in runs]
        db_session.query.return_value.filter.return_value.all.return_value = runs
        
        request = CompareModelsRequest(model_run_ids=model_ids)
        result = service.compare_models(request)
        
        # Should still rank them (by order or other factors)
        assert len(result.compared_models) == 3
        assert all(item.rank is not None for item in result.compared_models)
    
    def test_extreme_metric_values(self, db_session):
        """Test handling of extreme metric values."""
        service = ModelComparisonService(db_session)
        
        run1 = Mock(spec=ModelRun)
        run1.id = uuid4()
        run1.experiment_id = uuid4()
        run1.model_type = "model_1"
        run1.status = "completed"
        run1.metrics = {"accuracy": 0.99999}  # Very high
        run1.hyperparameters = {}
        run1.training_time = 10.0
        run1.created_at = datetime.now()
        
        run2 = Mock(spec=ModelRun)
        run2.id = uuid4()
        run2.experiment_id = uuid4()
        run2.model_type = "model_2"
        run2.status = "completed"
        run2.metrics = {"accuracy": 0.00001}  # Very low
        run2.hyperparameters = {}
        run2.training_time = 5.0
        run2.created_at = datetime.now()
        
        model_ids = [run1.id, run2.id]
        db_session.query.return_value.filter.return_value.all.return_value = [run1, run2]
        
        request = CompareModelsRequest(model_run_ids=model_ids)
        result = service.compare_models(request)
        
        # Should handle extreme values
        assert result.best_model.model_type == "model_1"
        assert result.compared_models[1].rank == 2
    
    def test_many_models(self, db_session):
        """Test comparison with maximum number of models."""
        service = ModelComparisonService(db_session)
        
        # Create 10 models (maximum allowed)
        runs = []
        for i in range(10):
            run = Mock(spec=ModelRun)
            run.id = uuid4()
            run.experiment_id = uuid4()
            run.model_type = f"model_{i}"
            run.status = "completed"
            run.metrics = {"accuracy": 0.90 + i * 0.001}  # Slightly different
            run.hyperparameters = {}
            run.training_time = 10.0 + i
            run.created_at = datetime(2025, 12, 30, 10, i, 0)
            runs.append(run)
        
        model_ids = [run.id for run in runs]
        db_session.query.return_value.filter.return_value.all.return_value = runs
        
        request = CompareModelsRequest(model_run_ids=model_ids)
        result = service.compare_models(request)
        
        # Should handle all 10 models
        assert result.total_models == 10
        assert len(result.compared_models) == 10
        assert all(item.rank is not None for item in result.compared_models)


# ============================================================================
# Integration Tests (if needed)
# ============================================================================


class TestAPIIntegration:
    """Test API endpoint integration (requires full app setup)."""
    
    @pytest.mark.skip(reason="Requires full FastAPI test client setup")
    def test_compare_endpoint(self):
        """Test /models/compare endpoint."""
        pass
    
    @pytest.mark.skip(reason="Requires full FastAPI test client setup")
    def test_rank_endpoint(self):
        """Test /models/rank endpoint."""
        pass
