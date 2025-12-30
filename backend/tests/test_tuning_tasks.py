"""Tests for Celery tuning tasks (BACKEND-46)."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import uuid
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from app.tasks.tuning_tasks import tune_hyperparameters, validate_model_cv
from app.models.tuning_run import TuningStatus
from app.ml_engine.tuning import GridSearchResult, RandomSearchResult, BayesianSearchResult


class TestTuneHyperparametersTask(unittest.TestCase):
    """Test the tune_hyperparameters Celery task."""

    def setUp(self):
        """Set up test fixtures."""
        self.tuning_run_id = str(uuid.uuid4())
        self.model_run_id = str(uuid.uuid4())
        self.dataset_id = str(uuid.uuid4())
        self.user_id = str(uuid.uuid4())

    @patch('app.tasks.tuning_tasks.SessionLocal')
    @patch('app.tasks.tuning_tasks.get_model_serialization_service')
    @patch('app.tasks.tuning_tasks.run_grid_search')
    @patch('app.tasks.tuning_tasks.get_logger')
    def test_grid_search_tuning(self, mock_logger, mock_grid_search, mock_get_service, mock_session):
        """Test grid search hyperparameter tuning."""
        # Mock database session
        mock_db = MagicMock()
        mock_session.return_value = mock_db

        # Mock tuning run
        mock_tuning_run = MagicMock()
        mock_tuning_run.id = uuid.UUID(self.tuning_run_id)
        mock_db.query().filter().first.return_value = mock_tuning_run

        # Mock model run
        mock_model_run = MagicMock()
        mock_model_run.id = uuid.UUID(self.model_run_id)
        mock_model_run.model_type = 'logistic_regression'
        mock_model_run.model_artifact_path = '/path/to/model.pkl'

        # Mock model
        mock_model = MagicMock()
        mock_model.model = LogisticRegression()
        mock_metadata = {
            'dataset_id': self.dataset_id,
            'target_column': 'target',
            'feature_columns': ['f1', 'f2', 'f3']
        }

        mock_service = MagicMock()
        mock_service.load_model.return_value = (mock_model, mock_metadata)
        mock_get_service.return_value = mock_service

        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.file_path = '/tmp/test_data.csv'

        # Configure query chain for different models
        def query_side_effect(model_class):
            query_mock = MagicMock()
            if model_class.__name__ == 'TuningRun':
                query_mock.filter().first.return_value = mock_tuning_run
            elif model_class.__name__ == 'ModelRun':
                query_mock.filter().first.return_value = mock_model_run
            elif model_class.__name__ == 'Dataset':
                query_mock.filter().first.return_value = mock_dataset
            else:
                query_mock.filter().order_by().all.return_value = []
            return query_mock

        mock_db.query.side_effect = query_side_effect

        # Mock dataset file
        test_data = pd.DataFrame({
            'f1': [1, 2, 3, 4, 5],
            'f2': [2, 3, 4, 5, 6],
            'f3': [3, 4, 5, 6, 7],
            'target': [0, 1, 0, 1, 0]
        })

        with patch('pandas.read_csv', return_value=test_data):
            # Mock grid search result
            mock_result = GridSearchResult(
                best_params={'C': 1.0},
                best_score=0.85,
                scoring='accuracy',
                cv_folds=5,
                n_candidates=10,
                results=[
                    {
                        'rank': 1,
                        'params': {'C': 1.0},
                        'mean_score': 0.85,
                        'std_score': 0.02,
                        'scores': [0.83, 0.84, 0.85, 0.86, 0.87]
                    }
                ]
            )
            mock_grid_search.return_value = mock_result

            # Mock task instance
            mock_task = MagicMock()
            mock_task.request.id = 'test-task-id'
            mock_task.update_state = MagicMock()

            # Call the task
            result = tune_hyperparameters(
                mock_task,
                tuning_run_id=self.tuning_run_id,
                model_run_id=self.model_run_id,
                tuning_method='grid_search',
                param_grid={'C': [0.1, 1.0, 10.0]},
                cv_folds=5,
                scoring_metric='accuracy',
                user_id=self.user_id
            )

        # Assertions
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['tuning_method'], 'grid_search')
        self.assertEqual(result['best_params'], {'C': 1.0})
        self.assertAlmostEqual(result['best_score'], 0.85)
        self.assertEqual(result['total_combinations'], 10)

        # Verify grid search was called
        mock_grid_search.assert_called_once()
        call_args = mock_grid_search.call_args
        self.assertIn('param_grid', call_args.kwargs)

        # Verify tuning run was updated
        self.assertEqual(mock_tuning_run.status, TuningStatus.COMPLETED)
        self.assertIsNotNone(mock_tuning_run.results)

    @patch('app.tasks.tuning_tasks.SessionLocal')
    @patch('app.tasks.tuning_tasks.get_model_serialization_service')
    @patch('app.tasks.tuning_tasks.run_random_search')
    @patch('app.tasks.tuning_tasks.get_logger')
    def test_random_search_tuning(self, mock_logger, mock_random_search, mock_get_service, mock_session):
        """Test random search hyperparameter tuning."""
        # Setup similar to grid search
        mock_db = MagicMock()
        mock_session.return_value = mock_db

        mock_tuning_run = MagicMock()
        mock_tuning_run.id = uuid.UUID(self.tuning_run_id)

        mock_model_run = MagicMock()
        mock_model_run.model_type = 'random_forest_classifier'
        mock_model_run.model_artifact_path = '/path/to/model.pkl'

        mock_model = MagicMock()
        mock_model.model = RandomForestClassifier()
        mock_metadata = {
            'dataset_id': self.dataset_id,
            'target_column': 'target',
            'feature_columns': ['f1', 'f2']
        }

        mock_service = MagicMock()
        mock_service.load_model.return_value = (mock_model, mock_metadata)
        mock_get_service.return_value = mock_service

        mock_dataset = MagicMock()
        mock_dataset.file_path = '/tmp/test_data.csv'

        def query_side_effect(model_class):
            query_mock = MagicMock()
            if model_class.__name__ == 'TuningRun':
                query_mock.filter().first.return_value = mock_tuning_run
            elif model_class.__name__ == 'ModelRun':
                query_mock.filter().first.return_value = mock_model_run
            elif model_class.__name__ == 'Dataset':
                query_mock.filter().first.return_value = mock_dataset
            else:
                query_mock.filter().order_by().all.return_value = []
            return query_mock

        mock_db.query.side_effect = query_side_effect

        test_data = pd.DataFrame({
            'f1': np.random.randn(50),
            'f2': np.random.randn(50),
            'target': np.random.randint(0, 2, 50)
        })

        with patch('pandas.read_csv', return_value=test_data):
            mock_result = RandomSearchResult(
                best_params={'n_estimators': 100, 'max_depth': 10},
                best_score=0.88,
                scoring='f1',
                cv_folds=5,
                n_iter=20,
                n_candidates=20,
                results=[
                    {
                        'rank': 1,
                        'params': {'n_estimators': 100, 'max_depth': 10},
                        'mean_score': 0.88,
                        'std_score': 0.03,
                        'scores': [0.85, 0.87, 0.88, 0.89, 0.91]
                    }
                ]
            )
            mock_random_search.return_value = mock_result

            mock_task = MagicMock()
            mock_task.request.id = 'test-task-id'
            mock_task.update_state = MagicMock()

            result = tune_hyperparameters(
                mock_task,
                tuning_run_id=self.tuning_run_id,
                model_run_id=self.model_run_id,
                tuning_method='random_search',
                param_grid={'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]},
                cv_folds=5,
                n_iter=20,
                user_id=self.user_id
            )

        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['tuning_method'], 'random_search')
        self.assertEqual(result['best_params']['n_estimators'], 100)
        self.assertAlmostEqual(result['best_score'], 0.88)

        mock_random_search.assert_called_once()

    @patch('app.tasks.tuning_tasks.SessionLocal')
    @patch('app.tasks.tuning_tasks.get_model_serialization_service')
    @patch('app.tasks.tuning_tasks.run_bayesian_search')
    @patch('app.tasks.tuning_tasks.get_logger')
    def test_bayesian_search_tuning(self, mock_logger, mock_bayesian_search, mock_get_service, mock_session):
        """Test Bayesian optimization hyperparameter tuning."""
        mock_db = MagicMock()
        mock_session.return_value = mock_db

        mock_tuning_run = MagicMock()
        mock_tuning_run.id = uuid.UUID(self.tuning_run_id)

        mock_model_run = MagicMock()
        mock_model_run.model_type = 'logistic_regression'
        mock_model_run.model_artifact_path = '/path/to/model.pkl'

        mock_model = MagicMock()
        mock_model.model = LogisticRegression()
        mock_metadata = {
            'dataset_id': self.dataset_id,
            'target_column': 'target',
            'feature_columns': ['f1', 'f2']
        }

        mock_service = MagicMock()
        mock_service.load_model.return_value = (mock_model, mock_metadata)
        mock_get_service.return_value = mock_service

        mock_dataset = MagicMock()
        mock_dataset.file_path = '/tmp/test_data.csv'

        def query_side_effect(model_class):
            query_mock = MagicMock()
            if model_class.__name__ == 'TuningRun':
                query_mock.filter().first.return_value = mock_tuning_run
            elif model_class.__name__ == 'ModelRun':
                query_mock.filter().first.return_value = mock_model_run
            elif model_class.__name__ == 'Dataset':
                query_mock.filter().first.return_value = mock_dataset
            else:
                query_mock.filter().order_by().all.return_value = []
            return query_mock

        mock_db.query.side_effect = query_side_effect

        test_data = pd.DataFrame({
            'f1': np.random.randn(50),
            'f2': np.random.randn(50),
            'target': np.random.randint(0, 2, 50)
        })

        with patch('pandas.read_csv', return_value=test_data):
            mock_result = BayesianSearchResult(
                best_params={'C': 5.0},
                best_score=0.90,
                scoring='accuracy',
                cv_folds=5,
                n_iter=30,
                n_candidates=30,
                results=[],
                method='bayesian'
            )
            mock_bayesian_search.return_value = mock_result

            mock_task = MagicMock()
            mock_task.request.id = 'test-task-id'
            mock_task.update_state = MagicMock()

            result = tune_hyperparameters(
                mock_task,
                tuning_run_id=self.tuning_run_id,
                model_run_id=self.model_run_id,
                tuning_method='bayesian',
                param_grid={'C': [0.1, 1.0, 10.0]},
                cv_folds=5,
                n_iter=30,
                user_id=self.user_id
            )

        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['tuning_method'], 'bayesian')
        self.assertEqual(result['method_used'], 'bayesian')
        self.assertAlmostEqual(result['best_score'], 0.90)

        mock_bayesian_search.assert_called_once()

    @patch('app.tasks.tuning_tasks.SessionLocal')
    @patch('app.tasks.tuning_tasks.get_default_search_space')
    @patch('app.tasks.tuning_tasks.get_logger')
    def test_tuning_with_default_search_space(self, mock_logger, mock_get_space, mock_session):
        """Test tuning with default search space when param_grid is None."""
        mock_get_space.return_value = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }

        mock_db = MagicMock()
        mock_session.return_value = mock_db

        mock_tuning_run = MagicMock()
        mock_model_run = MagicMock()
        mock_model_run.model_type = 'logistic_regression'

        # This test verifies that get_default_search_space is called
        # when param_grid is None
        # Full implementation would require more mocking


class TestValidateModelCVTask(unittest.TestCase):
    """Test the validate_model_cv Celery task."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_run_id = str(uuid.uuid4())
        self.dataset_id = str(uuid.uuid4())
        self.user_id = str(uuid.uuid4())

    @patch('app.tasks.tuning_tasks.SessionLocal')
    @patch('app.tasks.tuning_tasks.get_model_serialization_service')
    @patch('app.tasks.tuning_tasks.run_cross_validation')
    @patch('app.tasks.tuning_tasks.get_logger')
    def test_cross_validation_task(self, mock_logger, mock_cv, mock_get_service, mock_session):
        """Test cross-validation task execution."""
        mock_db = MagicMock()
        mock_session.return_value = mock_db

        mock_model_run = MagicMock()
        mock_model_run.id = uuid.UUID(self.model_run_id)
        mock_model_run.model_type = 'logistic_regression'
        mock_model_run.model_artifact_path = '/path/to/model.pkl'

        mock_model = MagicMock()
        mock_model.model = LogisticRegression()
        mock_metadata = {
            'dataset_id': self.dataset_id,
            'target_column': 'target',
            'feature_columns': ['f1', 'f2']
        }

        mock_service = MagicMock()
        mock_service.load_model.return_value = (mock_model, mock_metadata)
        mock_get_service.return_value = mock_service

        mock_dataset = MagicMock()
        mock_dataset.file_path = '/tmp/test_data.csv'

        def query_side_effect(model_class):
            query_mock = MagicMock()
            if model_class.__name__ == 'ModelRun':
                query_mock.filter().first.return_value = mock_model_run
            elif model_class.__name__ == 'Dataset':
                query_mock.filter().first.return_value = mock_dataset
            else:
                query_mock.filter().order_by().all.return_value = []
            return query_mock

        mock_db.query.side_effect = query_side_effect

        test_data = pd.DataFrame({
            'f1': np.random.randn(100),
            'f2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

        # Mock CV result
        from app.ml_engine.tuning import CrossValidationResult

        mock_cv_result = CrossValidationResult(
            mean_score=0.85,
            std_score=0.02,
            median_score=0.86,
            min_score=0.81,
            max_score=0.89,
            scores=np.array([0.81, 0.84, 0.86, 0.87, 0.89]),
            fit_times=np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
            score_times=np.array([0.01, 0.01, 0.01, 0.01, 0.01]),
            train_scores=np.array([0.90, 0.91, 0.90, 0.92, 0.91]),
            additional_metrics={
                'precision': {
                    'mean': 0.84,
                    'std': 0.03,
                    'scores': [0.80, 0.83, 0.85, 0.86, 0.88]
                }
            }
        )
        mock_cv.return_value = mock_cv_result

        with patch('pandas.read_csv', return_value=test_data):
            mock_task = MagicMock()
            mock_task.request.id = 'test-cv-task-id'
            mock_task.update_state = MagicMock()

            result = validate_model_cv(
                mock_task,
                model_run_id=self.model_run_id,
                cv_folds=5,
                scoring_metrics=['accuracy', 'precision'],
                user_id=self.user_id
            )

        # Assertions
        self.assertEqual(result['model_run_id'], self.model_run_id)
        self.assertEqual(result['cv_folds'], 5)
        self.assertAlmostEqual(result['mean_score'], 0.85, places=2)
        self.assertAlmostEqual(result['std_score'], 0.02, places=2)
        self.assertIn('confidence_interval_95', result)
        self.assertIn('additional_metrics', result)
        self.assertIn('precision', result['additional_metrics'])

        # Verify CV was called
        mock_cv.assert_called_once()


class TestTuningTaskErrorHandling(unittest.TestCase):
    """Test error handling in tuning tasks."""

    @patch('app.tasks.tuning_tasks.SessionLocal')
    @patch('app.tasks.tuning_tasks.get_logger')
    def test_tuning_run_not_found(self, mock_logger, mock_session):
        """Test handling when tuning run is not found."""
        mock_db = MagicMock()
        mock_session.return_value = mock_db
        mock_db.query().filter().first.return_value = None

        mock_task = MagicMock()
        mock_task.request.id = 'test-task-id'

        with self.assertRaises(ValueError) as context:
            tune_hyperparameters(
                mock_task,
                tuning_run_id=str(uuid.uuid4()),
                model_run_id=str(uuid.uuid4()),
                tuning_method='grid_search',
                param_grid={'C': [1.0]},
                user_id=str(uuid.uuid4())
            )

        self.assertIn("TuningRun with id", str(context.exception))

    @patch('app.tasks.tuning_tasks.SessionLocal')
    @patch('app.tasks.tuning_tasks.get_logger')
    def test_invalid_tuning_method(self, mock_logger, mock_session):
        """Test handling of invalid tuning method."""
        # This would require more setup to reach the tuning method validation
        # Placeholder for comprehensive error testing
        pass


if __name__ == '__main__':
    unittest.main()
