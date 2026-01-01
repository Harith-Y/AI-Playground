"""
Unit tests for experiment configuration serialization.
"""

import pytest
import json
import tempfile
from pathlib import Path
from uuid import uuid4
from datetime import datetime

from app.services.experiment_config_service import ExperimentConfigSerializer
from app.models.experiment import Experiment, ExperimentStatus
from app.models.dataset import Dataset
from app.models.preprocessing_step import PreprocessingStep
from app.models.model_run import ModelRun
from app.models.user import User


@pytest.fixture
def test_user(db_session):
    """Create a test user."""
    user = User(
        id=uuid4(),
        email="test@example.com",
        hashed_password="hashed_password"
    )
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def test_dataset(db_session, test_user):
    """Create a test dataset."""
    dataset = Dataset(
        id=uuid4(),
        user_id=test_user.id,
        name="Test Dataset",
        file_path="/path/to/dataset.csv",
        shape={"rows": 100, "columns": 5},
        dtypes={"col1": "int64", "col2": "float64", "col3": "object"},
        missing_values={"col1": 0, "col2": 5, "col3": 0}
    )
    db_session.add(dataset)
    db_session.commit()
    return dataset


@pytest.fixture
def test_experiment(db_session, test_user, test_dataset):
    """Create a test experiment."""
    experiment = Experiment(
        id=uuid4(),
        user_id=test_user.id,
        dataset_id=test_dataset.id,
        name="Test Experiment",
        status=ExperimentStatus.COMPLETED
    )
    db_session.add(experiment)
    db_session.commit()
    return experiment


@pytest.fixture
def test_preprocessing_steps(db_session, test_dataset):
    """Create test preprocessing steps."""
    steps = [
        PreprocessingStep(
            id=uuid4(),
            dataset_id=test_dataset.id,
            step_type="imputation",
            parameters={"strategy": "mean"},
            column_name="col2",
            order=0,
            is_active=True
        ),
        PreprocessingStep(
            id=uuid4(),
            dataset_id=test_dataset.id,
            step_type="scaling",
            parameters={"method": "standard"},
            column_name=None,
            order=1,
            is_active=True
        )
    ]
    for step in steps:
        db_session.add(step)
    db_session.commit()
    return steps


@pytest.fixture
def test_model_runs(db_session, test_experiment):
    """Create test model runs."""
    runs = [
        ModelRun(
            id=uuid4(),
            experiment_id=test_experiment.id,
            model_type="random_forest",
            hyperparameters={"n_estimators": 100, "max_depth": 10},
            metrics={"accuracy": 0.95, "f1_score": 0.93},
            training_time=45.2,
            status="completed"
        ),
        ModelRun(
            id=uuid4(),
            experiment_id=test_experiment.id,
            model_type="logistic_regression",
            hyperparameters={"C": 1.0, "penalty": "l2"},
            metrics={"accuracy": 0.92, "f1_score": 0.90},
            training_time=12.5,
            status="completed"
        )
    ]
    for run in runs:
        db_session.add(run)
    db_session.commit()
    return runs


class TestExperimentConfigSerializer:
    """Tests for ExperimentConfigSerializer class."""
    
    def test_serialize_experiment_basic(
        self,
        db_session,
        test_experiment,
        test_preprocessing_steps,
        test_model_runs
    ):
        """Test basic experiment serialization."""
        serializer = ExperimentConfigSerializer(db_session)
        
        config = serializer.serialize_experiment(test_experiment.id)
        
        assert config["version"] == "1.0.0"
        assert config["experiment"]["name"] == "Test Experiment"
        assert config["experiment"]["status"] == "completed"
        assert len(config["preprocessing"]) == 2
        assert len(config["models"]) == 2
    
    def test_serialize_experiment_without_results(
        self,
        db_session,
        test_experiment,
        test_model_runs
    ):
        """Test serialization without results."""
        serializer = ExperimentConfigSerializer(db_session)
        
        config = serializer.serialize_experiment(
            test_experiment.id,
            include_results=False
        )
        
        assert "metrics" not in config["models"][0]
        assert "training_time" not in config["models"][0]
    
    def test_serialize_experiment_with_artifacts(
        self,
        db_session,
        test_experiment,
        test_model_runs
    ):
        """Test serialization with artifact paths."""
        # Add artifact path to model run
        test_model_runs[0].model_artifact_path = "/path/to/model.pkl"
        db_session.commit()
        
        serializer = ExperimentConfigSerializer(db_session)
        
        config = serializer.serialize_experiment(
            test_experiment.id,
            include_artifacts=True
        )
        
        assert config["models"][0]["model_artifact_path"] == "/path/to/model.pkl"
    
    def test_serialize_nonexistent_experiment(self, db_session):
        """Test serialization of nonexistent experiment."""
        serializer = ExperimentConfigSerializer(db_session)
        
        with pytest.raises(ValueError, match="not found"):
            serializer.serialize_experiment(uuid4())
    
    def test_save_to_file(
        self,
        db_session,
        test_experiment,
        test_preprocessing_steps,
        test_model_runs
    ):
        """Test saving configuration to file."""
        serializer = ExperimentConfigSerializer(db_session)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "config.json"
            
            result_path = serializer.save_to_file(
                test_experiment.id,
                file_path
            )
            
            assert result_path.exists()
            
            # Verify file content
            with open(result_path, 'r') as f:
                config = json.load(f)
            
            assert config["experiment"]["name"] == "Test Experiment"
            assert len(config["models"]) == 2
    
    def test_load_from_file(
        self,
        db_session,
        test_experiment,
        test_preprocessing_steps,
        test_model_runs
    ):
        """Test loading configuration from file."""
        serializer = ExperimentConfigSerializer(db_session)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "config.json"
            
            # Save first
            serializer.save_to_file(test_experiment.id, file_path)
            
            # Load
            loaded_config = serializer.load_from_file(file_path)
            
            assert loaded_config["experiment"]["name"] == "Test Experiment"
            assert len(loaded_config["preprocessing"]) == 2
    
    def test_load_nonexistent_file(self, db_session):
        """Test loading from nonexistent file."""
        serializer = ExperimentConfigSerializer(db_session)
        
        with pytest.raises(FileNotFoundError):
            serializer.load_from_file(Path("/nonexistent/file.json"))
    
    def test_compare_experiments(
        self,
        db_session,
        test_user,
        test_dataset,
        test_preprocessing_steps
    ):
        """Test comparing two experiments."""
        # Create two experiments
        exp1 = Experiment(
            id=uuid4(),
            user_id=test_user.id,
            dataset_id=test_dataset.id,
            name="Experiment 1",
            status=ExperimentStatus.COMPLETED
        )
        exp2 = Experiment(
            id=uuid4(),
            user_id=test_user.id,
            dataset_id=test_dataset.id,
            name="Experiment 2",
            status=ExperimentStatus.COMPLETED
        )
        db_session.add_all([exp1, exp2])
        
        # Add different models
        model1 = ModelRun(
            id=uuid4(),
            experiment_id=exp1.id,
            model_type="random_forest",
            hyperparameters={"n_estimators": 100},
            status="completed"
        )
        model2 = ModelRun(
            id=uuid4(),
            experiment_id=exp2.id,
            model_type="logistic_regression",
            hyperparameters={"C": 1.0},
            status="completed"
        )
        db_session.add_all([model1, model2])
        db_session.commit()
        
        serializer = ExperimentConfigSerializer(db_session)
        
        comparison = serializer.compare_experiments(exp1.id, exp2.id)
        
        assert comparison["experiments"]["experiment_1"] == "Experiment 1"
        assert comparison["experiments"]["experiment_2"] == "Experiment 2"
        assert comparison["differences"]["models"]["model_types_1"] == ["random_forest"]
        assert comparison["differences"]["models"]["model_types_2"] == ["logistic_regression"]
    
    def test_export_for_reproduction(
        self,
        db_session,
        test_experiment,
        test_preprocessing_steps,
        test_model_runs
    ):
        """Test exporting experiment package."""
        serializer = ExperimentConfigSerializer(db_session)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            exported_files = serializer.export_for_reproduction(
                test_experiment.id,
                output_dir
            )
            
            # Check all files exist
            assert exported_files["full_config"].exists()
            assert exported_files["preprocessing"].exists()
            assert exported_files["models"].exists()
            assert exported_files["readme"].exists()
            
            # Verify README content
            readme_content = exported_files["readme"].read_text()
            assert "Test Experiment" in readme_content
            assert "Reproduction Steps" in readme_content
    
    def test_preprocessing_comparison(
        self,
        db_session,
        test_user,
        test_dataset
    ):
        """Test preprocessing pipeline comparison."""
        # Create two experiments with different preprocessing
        exp1 = Experiment(
            id=uuid4(),
            user_id=test_user.id,
            dataset_id=test_dataset.id,
            name="Exp1",
            status=ExperimentStatus.COMPLETED
        )
        
        # Create second dataset for second experiment
        dataset2 = Dataset(
            id=uuid4(),
            user_id=test_user.id,
            name="Dataset 2",
            file_path="/path/to/dataset2.csv",
            shape={"rows": 100, "columns": 5},
            dtypes={"col1": "int64"},
            missing_values={}
        )
        
        exp2 = Experiment(
            id=uuid4(),
            user_id=test_user.id,
            dataset_id=dataset2.id,
            name="Exp2",
            status=ExperimentStatus.COMPLETED
        )
        
        db_session.add_all([exp1, dataset2, exp2])
        
        # Add different preprocessing steps
        step1 = PreprocessingStep(
            id=uuid4(),
            dataset_id=test_dataset.id,
            step_type="imputation",
            parameters={"strategy": "mean"},
            order=0,
            is_active=True
        )
        step2 = PreprocessingStep(
            id=uuid4(),
            dataset_id=dataset2.id,
            step_type="scaling",
            parameters={"method": "standard"},
            order=0,
            is_active=True
        )
        
        db_session.add_all([step1, step2])
        db_session.commit()
        
        serializer = ExperimentConfigSerializer(db_session)
        
        comparison = serializer.compare_experiments(exp1.id, exp2.id)
        
        assert not comparison["summary"]["same_preprocessing"]
        assert comparison["differences"]["preprocessing"]["step_types_1"] == ["imputation"]
        assert comparison["differences"]["preprocessing"]["step_types_2"] == ["scaling"]


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_serialize_experiment_to_file(
        self,
        db_session,
        test_experiment,
        test_preprocessing_steps,
        test_model_runs
    ):
        """Test convenience function for serialization."""
        from app.services.experiment_config_service import serialize_experiment_to_file
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "config.json"
            
            result_path = serialize_experiment_to_file(
                db_session,
                test_experiment.id,
                file_path
            )
            
            assert result_path.exists()
    
    def test_load_experiment_from_file(
        self,
        db_session,
        test_experiment,
        test_preprocessing_steps,
        test_model_runs
    ):
        """Test convenience function for loading."""
        from app.services.experiment_config_service import (
            serialize_experiment_to_file,
            load_experiment_from_file
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "config.json"
            
            # Save
            serialize_experiment_to_file(
                db_session,
                test_experiment.id,
                file_path
            )
            
            # Load
            config = load_experiment_from_file(db_session, file_path)
            
            assert config["experiment"]["name"] == "Test Experiment"
    
    def test_export_experiment_package(
        self,
        db_session,
        test_experiment,
        test_preprocessing_steps,
        test_model_runs
    ):
        """Test convenience function for export."""
        from app.services.experiment_config_service import export_experiment_package
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            exported_files = export_experiment_package(
                db_session,
                test_experiment.id,
                output_dir
            )
            
            assert len(exported_files) == 4
            assert all(path.exists() for path in exported_files.values())


class TestConfigurationContent:
    """Tests for configuration content and structure."""
    
    def test_preprocessing_config_structure(
        self,
        db_session,
        test_experiment,
        test_preprocessing_steps
    ):
        """Test preprocessing configuration structure."""
        serializer = ExperimentConfigSerializer(db_session)
        config = serializer.serialize_experiment(test_experiment.id)
        
        preprocessing = config["preprocessing"]
        
        assert len(preprocessing) == 2
        assert preprocessing[0]["step_type"] == "imputation"
        assert preprocessing[0]["parameters"] == {"strategy": "mean"}
        assert preprocessing[0]["column_name"] == "col2"
        assert preprocessing[0]["order"] == 0
        
        assert preprocessing[1]["step_type"] == "scaling"
        assert preprocessing[1]["order"] == 1
    
    def test_model_config_structure(
        self,
        db_session,
        test_experiment,
        test_model_runs
    ):
        """Test model configuration structure."""
        serializer = ExperimentConfigSerializer(db_session)
        config = serializer.serialize_experiment(test_experiment.id)
        
        models = config["models"]
        
        assert len(models) == 2
        
        rf_model = next(m for m in models if m["model_type"] == "random_forest")
        assert rf_model["hyperparameters"]["n_estimators"] == 100
        assert rf_model["metrics"]["accuracy"] == 0.95
        assert rf_model["training_time"] == 45.2
        assert rf_model["status"] == "completed"
    
    def test_metadata_structure(
        self,
        db_session,
        test_experiment,
        test_preprocessing_steps
    ):
        """Test metadata structure."""
        serializer = ExperimentConfigSerializer(db_session)
        config = serializer.serialize_experiment(test_experiment.id)
        
        metadata = config["metadata"]
        
        assert "serialized_at" in metadata
        assert metadata["serializer_version"] == "1.0.0"
        
        # Verify timestamp format
        datetime.fromisoformat(metadata["serialized_at"])
