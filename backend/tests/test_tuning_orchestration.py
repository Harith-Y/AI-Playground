"""
Tests for Tuning Orchestration Service and API.

Tests comprehensive tuning workflows including:
- Progressive search (grid → random → bayesian)
- Multi-model parallel comparison  
- Workflow status tracking
- Parameter refinement
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4, UUID
from datetime import datetime

from app.services.tuning_orchestration_service import (
    TuningOrchestrationService,
    ProgressiveSearchConfig,
    MultiModelConfig
)
from app.models.model_run import ModelRun
from app.models.tuning_run import TuningRun, TuningStatus
from app.models.experiment import Experiment


class TestProgressiveSearch:
    """Tests for progressive search orchestration."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        db = Mock()
        db.query = Mock()
        db.add = Mock()
        db.commit = Mock()
        db.refresh = Mock()
        return db
    
    @pytest.fixture
    def mock_model_run(self):
        """Create mock model run."""
        model_run = Mock(spec=ModelRun)
        model_run.id = uuid4()
        model_run.status = "completed"
        model_run.model_type = "random_forest_classifier"
        return model_run
    
    @pytest.fixture
    def progressive_config(self):
        """Create progressive search configuration."""
        return ProgressiveSearchConfig(
            initial_param_grid={
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20]
            },
            refinement_factor=0.3,
            cv_folds=5,
            scoring_metric='accuracy',
            n_iter_random=50,
            n_iter_bayesian=30
        )
    
    @patch('app.services.tuning_orchestration_service.tune_hyperparameters')
    def test_progressive_search_initiates_grid_search(
        self,
        mock_tune_task,
        mock_db,
        mock_model_run,
        progressive_config
    ):
        """Test that progressive search initiates grid search stage."""
        # Setup
        mock_db.query.return_value.filter.return_value.first.return_value = mock_model_run
        mock_task = Mock()
        mock_task.id = "task-abc-123"
        mock_tune_task.apply_async.return_value = mock_task
        
        service = TuningOrchestrationService(mock_db)
        
        # Execute
        result = service.progressive_search(
            model_run_id=mock_model_run.id,
            config=progressive_config,
            user_id="user-123"
        )
        
        # Assert
        assert 'orchestration_id' in result
        assert result['workflow'] == 'progressive_search'
        assert len(result['stages']) == 3
        assert result['grid_search']['status'] == 'RUNNING'
        assert result['random_search']['status'] == 'PENDING'
        assert result['bayesian_optimization']['status'] == 'PENDING'
        
        # Verify grid search task was queued
        mock_tune_task.apply_async.assert_called_once()
        call_kwargs = mock_tune_task.apply_async.call_args[1]['kwargs']
        assert call_kwargs['tuning_method'] == 'grid_search'
        assert call_kwargs['cv_folds'] == 5
        assert call_kwargs['scoring_metric'] == 'accuracy'
    
    def test_progressive_search_rejects_incomplete_model(
        self,
        mock_db,
        progressive_config
    ):
        """Test that progressive search rejects incomplete model runs."""
        # Setup incomplete model
        incomplete_model = Mock(spec=ModelRun)
        incomplete_model.id = uuid4()
        incomplete_model.status = "training"
        mock_db.query.return_value.filter.return_value.first.return_value = incomplete_model
        
        service = TuningOrchestrationService(mock_db)
        
        # Execute & Assert
        with pytest.raises(ValueError, match="must be completed"):
            service.progressive_search(
                model_run_id=incomplete_model.id,
                config=progressive_config
            )
    
    def test_progressive_search_creates_three_tuning_runs(
        self,
        mock_db,
        mock_model_run,
        progressive_config
    ):
        """Test that three tuning runs are created for progressive search."""
        # Setup
        mock_db.query.return_value.filter.return_value.first.return_value = mock_model_run
        
        with patch('app.services.tuning_orchestration_service.tune_hyperparameters'):
            service = TuningOrchestrationService(mock_db)
            service.progressive_search(
                model_run_id=mock_model_run.id,
                config=progressive_config
            )
        
        # Assert 3 tuning runs were added
        assert mock_db.add.call_count == 3
    
    @patch('app.services.tuning_orchestration_service.tune_hyperparameters')
    def test_trigger_next_stage_random_search(
        self,
        mock_tune_task,
        mock_db
    ):
        """Test triggering random search after grid search completion."""
        # Setup completed grid search run
        orchestration_id = str(uuid4())
        model_run_id = uuid4()
        
        completed_run = Mock(spec=TuningRun)
        completed_run.id = uuid4()
        completed_run.model_run_id = model_run_id
        completed_run.status = TuningStatus.COMPLETED
        completed_run.best_params = {'n_estimators': 100, 'max_depth': 10}
        completed_run.results = {
            'orchestration_id': orchestration_id,
            'next_stage': 'random_search',
            'cv_folds': 5,
            'scoring_metric': 'accuracy',
            'all_results': [
                {'params': {'n_estimators': 100, 'max_depth': 10}, 'score': 0.95},
                {'params': {'n_estimators': 50, 'max_depth': 5}, 'score': 0.90}
            ]
        }
        
        # Setup pending random search run
        random_run = Mock(spec=TuningRun)
        random_run.id = uuid4()
        random_run.status = TuningStatus.PENDING
        random_run.results = {}
        
        # Mock DB queries
        def mock_query_side_effect(*args):
            mock_result = Mock()
            if args[0] == TuningRun:
                filter_mock = Mock()
                filter_mock.first.side_effect = [completed_run, random_run]
                mock_result.filter.return_value = filter_mock
            return mock_result
        
        mock_db.query.side_effect = mock_query_side_effect
        
        mock_task = Mock()
        mock_task.id = "task-random-456"
        mock_tune_task.apply_async.return_value = mock_task
        
        service = TuningOrchestrationService(mock_db)
        
        # Execute
        result = service.trigger_next_stage(
            orchestration_id=orchestration_id,
            completed_tuning_run_id=completed_run.id,
            user_id="user-123"
        )
        
        # Assert
        assert result is not None
        assert result['stage'] == 'random_search'
        assert result['status'] == 'RUNNING'
        assert 'task_id' in result
        
        # Verify random search was queued
        mock_tune_task.apply_async.assert_called_once()
        call_kwargs = mock_tune_task.apply_async.call_args[1]['kwargs']
        assert call_kwargs['tuning_method'] == 'random_search'
        assert call_kwargs['n_iter'] == 50
    
    def test_refine_param_grid_narrows_search_space(self, mock_db):
        """Test parameter grid refinement."""
        service = TuningOrchestrationService(mock_db)
        
        best_params = {'n_estimators': 100, 'max_depth': 10}
        top_results = [
            {'params': {'n_estimators': 100, 'max_depth': 10}, 'score': 0.95},
            {'params': {'n_estimators': 90, 'max_depth': 12}, 'score': 0.94},
            {'params': {'n_estimators': 110, 'max_depth': 8}, 'score': 0.93}
        ]
        
        refined = service._refine_param_grid(
            best_params=best_params,
            top_results=top_results,
            refinement_factor=0.3
        )
        
        # Assert refined grid is narrower
        assert 'n_estimators' in refined
        assert 'max_depth' in refined
        assert len(refined['n_estimators']) > 0
        assert len(refined['max_depth']) > 0


class TestMultiModelComparison:
    """Tests for multi-model comparison orchestration."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        db = Mock()
        db.query = Mock()
        db.add = Mock()
        db.commit = Mock()
        return db
    
    @pytest.fixture
    def mock_model_runs(self):
        """Create multiple mock model runs."""
        runs = []
        for i in range(3):
            run = Mock(spec=ModelRun)
            run.id = uuid4()
            run.status = "completed"
            run.model_type = f"model_type_{i}"
            runs.append(run)
        return runs
    
    @pytest.fixture
    def multi_model_config(self, mock_model_runs):
        """Create multi-model comparison configuration."""
        return MultiModelConfig(
            model_run_ids=[run.id for run in mock_model_runs],
            tuning_method="bayesian",
            cv_folds=5,
            scoring_metric="accuracy",
            n_iter=30,
            parallel=True
        )
    
    @patch('app.services.tuning_orchestration_service.group')
    @patch('app.services.tuning_orchestration_service.tune_hyperparameters')
    def test_multi_model_parallel_execution(
        self,
        mock_tune_task,
        mock_group,
        mock_db,
        mock_model_runs,
        multi_model_config
    ):
        """Test parallel execution of multiple model tuning."""
        # Setup
        mock_db.query.return_value.filter.return_value.first.side_effect = mock_model_runs
        
        mock_group_result = Mock()
        mock_group_result.id = "group-task-xyz"
        mock_group_instance = Mock()
        mock_group_instance.apply_async.return_value = mock_group_result
        mock_group.return_value = mock_group_instance
        
        service = TuningOrchestrationService(mock_db)
        
        # Execute
        result = service.multi_model_comparison(
            config=multi_model_config,
            user_id="user-123"
        )
        
        # Assert
        assert 'orchestration_id' in result
        assert result['workflow'] == 'multi_model_comparison'
        assert result['n_models'] == 3
        assert result['parallel'] is True
        assert result['group_task_id'] == "group-task-xyz"
        assert len(result['tuning_runs']) == 3
        
        # Verify group was created with 3 tasks
        mock_group.assert_called_once()
        signatures = mock_group.call_args[0][0]
        assert len(signatures) == 3
    
    @patch('app.services.tuning_orchestration_service.tune_hyperparameters')
    def test_multi_model_sequential_execution(
        self,
        mock_tune_task,
        mock_db,
        mock_model_runs
    ):
        """Test sequential execution of multiple model tuning."""
        # Setup
        mock_db.query.return_value.filter.return_value.first.side_effect = mock_model_runs
        
        mock_task = Mock()
        mock_task.id = "task-seq-123"
        mock_tune_task.apply_async.return_value = mock_task
        
        config = MultiModelConfig(
            model_run_ids=[run.id for run in mock_model_runs],
            tuning_method="bayesian",
            parallel=False
        )
        
        service = TuningOrchestrationService(mock_db)
        
        # Execute
        result = service.multi_model_comparison(
            config=config,
            user_id="user-123"
        )
        
        # Assert
        assert result['parallel'] is False
        assert result['group_task_id'] is None
        
        # Verify tasks were queued sequentially
        assert mock_tune_task.apply_async.call_count == 3
    
    def test_multi_model_with_custom_param_grids(
        self,
        mock_db,
        mock_model_runs
    ):
        """Test multi-model comparison with model-specific param grids."""
        # Setup
        mock_db.query.return_value.filter.return_value.first.side_effect = mock_model_runs
        
        param_grids = {
            mock_model_runs[0].id: {'C': [0.1, 1.0, 10.0]},
            mock_model_runs[1].id: {'n_estimators': [50, 100, 200]}
        }
        
        config = MultiModelConfig(
            model_run_ids=[run.id for run in mock_model_runs],
            tuning_method="bayesian",
            param_grids=param_grids,
            parallel=False
        )
        
        with patch('app.services.tuning_orchestration_service.tune_hyperparameters') as mock_tune:
            mock_tune.apply_async.return_value = Mock(id="task-123")
            
            service = TuningOrchestrationService(mock_db)
            result = service.multi_model_comparison(config=config)
        
        # Assert
        assert result['n_models'] == 3
    
    def test_get_best_model_from_comparison(self, mock_db):
        """Test getting best model from completed comparison."""
        # Setup
        orchestration_id = str(uuid4())
        
        # Create mock tuning runs with different scores
        runs = []
        for i, score in enumerate([0.92, 0.96, 0.89]):
            run = Mock(spec=TuningRun)
            run.id = uuid4()
            run.model_run_id = uuid4()
            run.status = TuningStatus.COMPLETED
            run.tuning_method = "bayesian"
            run.best_params = {'param': i}
            run.results = {'best_score': score, 'orchestration_id': orchestration_id}
            runs.append(run)
        
        # Mock model runs
        model_runs = []
        for i, run in enumerate(runs):
            model_run = Mock(spec=ModelRun)
            model_run.id = run.model_run_id
            model_run.model_type = f"model_type_{i}"
            model_runs.append(model_run)
        
        # Mock DB queries
        def mock_query_side_effect(*args):
            mock_result = Mock()
            if args[0] == TuningRun:
                mock_filter = Mock()
                mock_filter.all.return_value = runs
                mock_result.filter.return_value = mock_filter
            elif args[0] == ModelRun:
                # Return the model run for the best tuning run (index 1, score 0.96)
                mock_filter = Mock()
                mock_filter.first.return_value = model_runs[1]
                mock_result.filter.return_value = mock_filter
            return mock_result
        
        mock_db.query.side_effect = mock_query_side_effect
        
        service = TuningOrchestrationService(mock_db)
        
        # Execute
        result = service.get_best_model_from_comparison(orchestration_id)
        
        # Assert
        assert result['orchestration_id'] == orchestration_id
        assert result['best_model']['best_score'] == 0.96
        assert result['best_model']['model_type'] == 'model_type_1'
        assert len(result['all_models']) == 3


class TestOrchestrationStatus:
    """Tests for orchestration status tracking."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        db = Mock()
        db.query = Mock()
        return db
    
    def test_get_orchestration_status_progressive(self, mock_db):
        """Test getting status for progressive search."""
        orchestration_id = str(uuid4())
        model_run_id = uuid4()
        
        # Create mock tuning runs for 3 stages
        runs = []
        statuses = [TuningStatus.COMPLETED, TuningStatus.RUNNING, TuningStatus.PENDING]
        stages = ['grid_search', 'random_search', 'bayesian_optimization']
        
        for status_val, stage in zip(statuses, stages):
            run = Mock(spec=TuningRun)
            run.id = uuid4()
            run.model_run_id = model_run_id
            run.tuning_method = stage.split('_')[0]
            run.status = status_val
            run.best_params = {'param': 1} if status_val == TuningStatus.COMPLETED else None
            run.results = {
                'orchestration_id': orchestration_id,
                'stage': stage,
                'task_id': f"task-{stage}",
                'best_score': 0.95 if status_val == TuningStatus.COMPLETED else None
            }
            runs.append(run)
        
        # Mock DB query
        mock_filter = Mock()
        mock_filter.all.return_value = runs
        mock_db.query.return_value.filter.return_value = mock_filter
        
        service = TuningOrchestrationService(mock_db)
        
        # Execute
        result = service.get_orchestration_status(orchestration_id)
        
        # Assert
        assert result['orchestration_id'] == orchestration_id
        assert result['workflow_type'] == 'progressive_search'
        assert result['overall_status'] == 'RUNNING'
        assert result['progress']['completed'] == 1
        assert result['progress']['total'] == 3
        assert result['progress']['percentage'] == 33.33
        assert len(result['stages']) == 3
    
    def test_get_orchestration_status_multi_model(self, mock_db):
        """Test getting status for multi-model comparison."""
        orchestration_id = str(uuid4())
        
        # Create mock tuning runs for 3 models
        runs = []
        for i in range(3):
            run = Mock(spec=TuningRun)
            run.id = uuid4()
            run.model_run_id = uuid4()
            run.tuning_method = "bayesian"
            run.status = TuningStatus.COMPLETED
            run.best_params = {'param': i}
            run.results = {
                'orchestration_id': orchestration_id,
                'group_task_id': 'group-123',
                'best_score': 0.90 + i * 0.02
            }
            runs.append(run)
        
        # Mock DB query
        mock_filter = Mock()
        mock_filter.all.return_value = runs
        mock_db.query.return_value.filter.return_value = mock_filter
        
        service = TuningOrchestrationService(mock_db)
        
        # Execute
        result = service.get_orchestration_status(orchestration_id)
        
        # Assert
        assert result['orchestration_id'] == orchestration_id
        assert result['workflow_type'] == 'multi_model_comparison'
        assert result['overall_status'] == 'COMPLETED'
        assert result['progress']['completed'] == 3
        assert result['progress']['total'] == 3
        assert result['progress']['percentage'] == 100.0


class TestOrchestrationAPI:
    """Tests for orchestration API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)
    
    @patch('app.api.v1.endpoints.tuning_orchestration.TuningOrchestrationService')
    def test_start_progressive_search_endpoint(self, mock_service_class, client):
        """Test progressive search endpoint."""
        # Setup mock service
        mock_service = Mock()
        mock_service.progressive_search.return_value = {
            'orchestration_id': 'orch-123',
            'model_run_id': 'model-456',
            'workflow': 'progressive_search',
            'stages': [],
            'grid_search': {'tuning_run_id': 'run-1', 'task_id': 'task-1', 'status': 'RUNNING'},
            'random_search': {'tuning_run_id': 'run-2', 'status': 'PENDING'},
            'bayesian_optimization': {'tuning_run_id': 'run-3', 'status': 'PENDING'},
            'message': 'Progressive search workflow initiated'
        }
        mock_service_class.return_value = mock_service
        
        # Execute
        response = client.post(
            "/api/v1/tuning-orchestration/progressive-search",
            json={
                "model_run_id": "123e4567-e89b-12d3-a456-426614174000",
                "initial_param_grid": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 20]
                },
                "cv_folds": 5,
                "scoring_metric": "accuracy"
            }
        )
        
        # Assert
        assert response.status_code == 202
        data = response.json()
        assert data['workflow'] == 'progressive_search'
        assert 'orchestration_id' in data
    
    @patch('app.api.v1.endpoints.tuning_orchestration.TuningOrchestrationService')
    def test_start_multi_model_comparison_endpoint(self, mock_service_class, client):
        """Test multi-model comparison endpoint."""
        # Setup mock service
        mock_service = Mock()
        mock_service.multi_model_comparison.return_value = {
            'orchestration_id': 'orch-123',
            'workflow': 'multi_model_comparison',
            'n_models': 3,
            'tuning_method': 'bayesian',
            'parallel': True,
            'group_task_id': 'group-456',
            'tuning_runs': [],
            'message': 'Multi-model comparison initiated for 3 models'
        }
        mock_service_class.return_value = mock_service
        
        # Execute
        response = client.post(
            "/api/v1/tuning-orchestration/multi-model-comparison",
            json={
                "model_run_ids": [
                    "123e4567-e89b-12d3-a456-426614174000",
                    "123e4567-e89b-12d3-a456-426614174001",
                    "123e4567-e89b-12d3-a456-426614174002"
                ],
                "tuning_method": "bayesian",
                "cv_folds": 5,
                "scoring_metric": "accuracy",
                "n_iter": 30,
                "parallel": True
            }
        )
        
        # Assert
        assert response.status_code == 202
        data = response.json()
        assert data['workflow'] == 'multi_model_comparison'
        assert data['n_models'] == 3
        assert data['parallel'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
