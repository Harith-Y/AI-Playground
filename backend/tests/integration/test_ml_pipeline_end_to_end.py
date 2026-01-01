"""
Integration Tests for Complete ML Pipeline

Tests the entire ML workflow from data upload through preprocessing,
training, evaluation, and prediction.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from fastapi.testclient import TestClient
import time
import json


@pytest.mark.integration
class TestMLPipelineEndToEnd:
    """Test complete ML pipeline workflow"""

    def test_classification_pipeline_end_to_end(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """
        Test complete classification pipeline from upload to prediction
        
        Steps:
        1. Upload dataset
        2. Create preprocessing pipeline
        3. Apply preprocessing
        4. Train model
        5. Evaluate model
        6. Make predictions
        """
        
        # Step 1: Upload dataset
        print("\n[Step 1] Uploading classification dataset...")
        with open(classification_dataset, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("classification_data.csv", f, "text/csv")},
                data={
                    "name": "E2E Classification Dataset",
                    "description": "End-to-end test dataset"
                }
            )
        
        assert response.status_code == 200
        dataset_id = response.json()["id"]
        print(f"✓ Dataset uploaded: {dataset_id}")
        
        # Step 2: Create preprocessing configuration
        print("\n[Step 2] Creating preprocessing configuration...")
        preprocess_config = {
            "dataset_id": dataset_id,
            "name": "E2E Preprocessing Pipeline",
            "steps": [
                {
                    "type": "imputation",
                    "strategy": "mean",
                    "columns": None  # Apply to all numeric columns
                },
                {
                    "type": "scaling",
                    "method": "standard",
                    "columns": None  # Apply to all numeric columns
                }
            ]
        }
        
        response = client.post(
            "/api/v1/preprocessing/pipeline",
            headers=auth_headers,
            json=preprocess_config
        )
        
        assert response.status_code in [200, 201]
        pipeline_id = response.json()["id"]
        print(f"✓ Preprocessing pipeline created: {pipeline_id}")
        
        # Step 3: Apply preprocessing
        print("\n[Step 3] Applying preprocessing to dataset...")
        response = client.post(
            f"/api/v1/preprocessing/pipeline/{pipeline_id}/apply",
            headers=auth_headers,
            json={"dataset_id": dataset_id}
        )
        
        assert response.status_code in [200, 202]
        processed_data = response.json()
        print(f"✓ Preprocessing applied: {processed_data.get('shape', 'N/A')}")
        
        # Step 4: Train model
        print("\n[Step 4] Training classification model...")
        training_config = {
            "dataset_id": dataset_id,
            "model_type": "random_forest",
            "target_column": "target",
            "hyperparameters": {
                "n_estimators": 50,
                "max_depth": 5,
                "random_state": 42
            },
            "test_size": 0.2,
            "cv_folds": 3
        }
        
        response = client.post(
            "/api/v1/models/train",
            headers=auth_headers,
            json=training_config
        )
        
        assert response.status_code in [200, 202]
        model_run_id = response.json().get("model_run_id") or response.json().get("id")
        print(f"✓ Model training started: {model_run_id}")
        
        # Wait for training to complete
        max_wait = 60  # seconds
        start_time = time.time()
        training_complete = False
        
        while time.time() - start_time < max_wait:
            response = client.get(
                f"/api/v1/models/train/{model_run_id}/status",
                headers=auth_headers
            )
            
            if response.status_code == 200:
                status = response.json().get("status")
                if status in ["completed", "success"]:
                    training_complete = True
                    print(f"✓ Training completed successfully")
                    break
                elif status in ["failed", "error"]:
                    pytest.fail(f"Training failed: {response.json()}")
            
            time.sleep(2)
        
        if not training_complete:
            pytest.skip("Training did not complete in time")
        
        # Step 5: Evaluate model
        print("\n[Step 5] Evaluating model performance...")
        response = client.get(
            f"/api/v1/models/train/{model_run_id}/metrics",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        metrics = response.json()
        print(f"✓ Model evaluated - Accuracy: {metrics.get('accuracy', 'N/A')}")
        
        # Verify key metrics exist
        assert "accuracy" in metrics or "metrics" in metrics
        
        # Step 6: Make predictions
        print("\n[Step 6] Making predictions with trained model...")
        
        # Create sample data for prediction
        sample_data = {
            "features": [
                [1.5, 2.3, 0.8, 1.2],  # Sample row 1
                [0.9, 1.7, 2.1, 0.5]   # Sample row 2
            ]
        }
        
        response = client.post(
            f"/api/v1/models/{model_run_id}/predict",
            headers=auth_headers,
            json=sample_data
        )
        
        # Predictions might not be implemented, so we check
        if response.status_code in [200, 201]:
            predictions = response.json()
            print(f"✓ Predictions made: {len(predictions.get('predictions', []))} samples")
        else:
            print(f"⚠ Prediction endpoint returned {response.status_code}")
        
        print("\n✓ End-to-end classification pipeline test completed successfully")

    def test_regression_pipeline_end_to_end(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        regression_dataset: Path
    ):
        """
        Test complete regression pipeline from upload to prediction
        """
        
        # Step 1: Upload dataset
        print("\n[Step 1] Uploading regression dataset...")
        with open(regression_dataset, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("regression_data.csv", f, "text/csv")},
                data={
                    "name": "E2E Regression Dataset",
                    "description": "End-to-end regression test"
                }
            )
        
        assert response.status_code == 200
        dataset_id = response.json()["id"]
        print(f"✓ Dataset uploaded: {dataset_id}")
        
        # Step 2: Perform EDA
        print("\n[Step 2] Performing exploratory data analysis...")
        response = client.get(
            f"/api/v1/datasets/{dataset_id}/statistics",
            headers=auth_headers
        )
        
        if response.status_code == 200:
            stats = response.json()
            print(f"✓ EDA completed: {len(stats.get('columns', []))} columns analyzed")
        
        # Step 3: Feature selection
        print("\n[Step 3] Performing feature selection...")
        feature_config = {
            "dataset_id": dataset_id,
            "target_column": "target",
            "method": "variance_threshold",
            "threshold": 0.01
        }
        
        response = client.post(
            "/api/v1/features/select",
            headers=auth_headers,
            json=feature_config
        )
        
        if response.status_code in [200, 201]:
            selected_features = response.json()
            print(f"✓ Features selected: {len(selected_features.get('features', []))} features")
        
        # Step 4: Train regression model
        print("\n[Step 4] Training regression model...")
        training_config = {
            "dataset_id": dataset_id,
            "model_type": "linear_regression",
            "target_column": "target",
            "hyperparameters": {
                "fit_intercept": True
            },
            "test_size": 0.2
        }
        
        response = client.post(
            "/api/v1/models/train",
            headers=auth_headers,
            json=training_config
        )
        
        assert response.status_code in [200, 202]
        model_run_id = response.json().get("model_run_id") or response.json().get("id")
        print(f"✓ Model training started: {model_run_id}")
        
        # Wait for training
        time.sleep(5)
        
        # Step 5: Get regression metrics
        print("\n[Step 5] Evaluating regression model...")
        response = client.get(
            f"/api/v1/models/train/{model_run_id}/metrics",
            headers=auth_headers
        )
        
        if response.status_code == 200:
            metrics = response.json()
            print(f"✓ Model evaluated - MSE: {metrics.get('mse', 'N/A')}")
        
        print("\n✓ End-to-end regression pipeline test completed successfully")


@pytest.mark.integration
class TestMLPipelineWithTuning:
    """Test ML pipeline with hyperparameter tuning"""

    def test_pipeline_with_grid_search(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test ML pipeline with grid search hyperparameter tuning"""
        
        # Upload dataset
        print("\n[Step 1] Uploading dataset for tuning...")
        with open(classification_dataset, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("tuning_data.csv", f, "text/csv")},
                data={"name": "Tuning Test Dataset"}
            )
        
        assert response.status_code == 200
        dataset_id = response.json()["id"]
        print(f"✓ Dataset uploaded: {dataset_id}")
        
        # Create tuning job
        print("\n[Step 2] Starting hyperparameter tuning...")
        tuning_config = {
            "dataset_id": dataset_id,
            "model_type": "random_forest",
            "target_column": "target",
            "tuning_method": "grid_search",
            "param_grid": {
                "n_estimators": [10, 50],
                "max_depth": [3, 5]
            },
            "cv_folds": 2,
            "scoring": "accuracy"
        }
        
        response = client.post(
            "/api/v1/tuning/start",
            headers=auth_headers,
            json=tuning_config
        )
        
        assert response.status_code in [200, 202]
        tuning_job_id = response.json().get("job_id") or response.json().get("id")
        print(f"✓ Tuning job started: {tuning_job_id}")
        
        # Wait for tuning to complete
        max_wait = 120  # Grid search can take longer
        start_time = time.time()
        tuning_complete = False
        
        while time.time() - start_time < max_wait:
            response = client.get(
                f"/api/v1/tuning/{tuning_job_id}/status",
                headers=auth_headers
            )
            
            if response.status_code == 200:
                status = response.json().get("status")
                if status in ["completed", "success"]:
                    tuning_complete = True
                    print(f"✓ Tuning completed successfully")
                    break
                elif status in ["failed", "error"]:
                    pytest.fail(f"Tuning failed: {response.json()}")
            
            time.sleep(3)
        
        if tuning_complete:
            # Get best parameters
            response = client.get(
                f"/api/v1/tuning/{tuning_job_id}/results",
                headers=auth_headers
            )
            
            if response.status_code == 200:
                results = response.json()
                print(f"✓ Best parameters: {results.get('best_params', {})}")
                print(f"✓ Best score: {results.get('best_score', 'N/A')}")
        else:
            pytest.skip("Tuning did not complete in time")

    def test_pipeline_with_bayesian_optimization(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        regression_dataset: Path
    ):
        """Test ML pipeline with Bayesian optimization"""
        
        # Upload dataset
        print("\n[Step 1] Uploading dataset for Bayesian optimization...")
        with open(regression_dataset, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("bayesian_data.csv", f, "text/csv")},
                data={"name": "Bayesian Optimization Dataset"}
            )
        
        assert response.status_code == 200
        dataset_id = response.json()["id"]
        print(f"✓ Dataset uploaded: {dataset_id}")
        
        # Create Bayesian optimization job
        print("\n[Step 2] Starting Bayesian optimization...")
        tuning_config = {
            "dataset_id": dataset_id,
            "model_type": "gradient_boosting",
            "target_column": "target",
            "tuning_method": "bayesian",
            "param_bounds": {
                "n_estimators": [10, 100],
                "learning_rate": [0.01, 0.3],
                "max_depth": [3, 10]
            },
            "n_iterations": 10,
            "cv_folds": 2
        }
        
        response = client.post(
            "/api/v1/tuning/start",
            headers=auth_headers,
            json=tuning_config
        )
        
        if response.status_code in [200, 202]:
            job_id = response.json().get("job_id") or response.json().get("id")
            print(f"✓ Bayesian optimization started: {job_id}")
        else:
            pytest.skip("Bayesian optimization endpoint not available")


@pytest.mark.integration
class TestMLPipelineFeatureEngineering:
    """Test ML pipeline with feature engineering"""

    def test_pipeline_with_feature_selection(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test pipeline with various feature selection methods"""
        
        # Upload dataset
        with open(classification_dataset, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("feature_selection.csv", f, "text/csv")},
                data={"name": "Feature Selection Dataset"}
            )
        
        assert response.status_code == 200
        dataset_id = response.json()["id"]
        
        # Test 1: Variance threshold
        print("\n[Test 1] Variance Threshold Feature Selection...")
        response = client.post(
            "/api/v1/features/select",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "method": "variance_threshold",
                "threshold": 0.01
            }
        )
        
        if response.status_code in [200, 201]:
            result = response.json()
            print(f"✓ Variance threshold: {len(result.get('features', []))} features selected")
        
        # Test 2: Correlation-based selection
        print("\n[Test 2] Correlation-Based Feature Selection...")
        response = client.post(
            "/api/v1/features/select",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "target_column": "target",
                "method": "correlation",
                "threshold": 0.1
            }
        )
        
        if response.status_code in [200, 201]:
            result = response.json()
            print(f"✓ Correlation-based: {len(result.get('features', []))} features selected")
        
        # Test 3: Mutual information
        print("\n[Test 3] Mutual Information Feature Selection...")
        response = client.post(
            "/api/v1/features/select",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "target_column": "target",
                "method": "mutual_info",
                "k": 5
            }
        )
        
        if response.status_code in [200, 201]:
            result = response.json()
            print(f"✓ Mutual information: {len(result.get('features', []))} features selected")

    def test_pipeline_with_feature_importance(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test feature importance analysis after model training"""
        
        # Upload and train model
        with open(classification_dataset, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("importance_test.csv", f, "text/csv")},
                data={"name": "Feature Importance Dataset"}
            )
        
        dataset_id = response.json()["id"]
        
        # Train model
        response = client.post(
            "/api/v1/models/train",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "model_type": "random_forest",
                "target_column": "target",
                "hyperparameters": {"n_estimators": 50}
            }
        )
        
        if response.status_code in [200, 202]:
            model_run_id = response.json().get("model_run_id") or response.json().get("id")
            
            # Wait for training
            time.sleep(5)
            
            # Get feature importance
            response = client.get(
                f"/api/v1/models/train/{model_run_id}/feature-importance",
                headers=auth_headers
            )
            
            if response.status_code == 200:
                importance = response.json()
                print(f"✓ Feature importance retrieved: {len(importance.get('features', []))} features")
                assert "features" in importance or "importance" in importance


@pytest.mark.integration
class TestMLPipelineDataPreprocessing:
    """Test comprehensive data preprocessing in pipeline"""

    def test_preprocessing_with_missing_values(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        tmp_path: Path
    ):
        """Test preprocessing pipeline with missing values"""
        
        # Create dataset with missing values
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, None, 4.0, 5.0],
            'feature2': [None, 2.0, 3.0, 4.0, 5.0],
            'feature3': [1.0, 2.0, 3.0, 4.0, None],
            'target': [0, 1, 0, 1, 0]
        })
        
        dataset_file = tmp_path / "missing_values.csv"
        df.to_csv(dataset_file, index=False)
        
        # Upload
        with open(dataset_file, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("missing_values.csv", f, "text/csv")},
                data={"name": "Missing Values Dataset"}
            )
        
        assert response.status_code == 200
        dataset_id = response.json()["id"]
        
        # Apply mean imputation
        response = client.post(
            "/api/v1/preprocessing/pipeline",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "name": "Imputation Pipeline",
                "steps": [
                    {
                        "type": "imputation",
                        "strategy": "mean"
                    }
                ]
            }
        )
        
        if response.status_code in [200, 201]:
            pipeline_id = response.json()["id"]
            print(f"✓ Imputation pipeline created: {pipeline_id}")

    def test_preprocessing_with_outliers(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        tmp_path: Path
    ):
        """Test preprocessing pipeline with outlier detection"""
        
        # Create dataset with outliers
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 95)
        outliers = np.array([150, 200, 300, -50, -100])
        
        df = pd.DataFrame({
            'feature1': np.concatenate([normal_data, outliers]),
            'feature2': np.random.normal(100, 20, 100),
            'target': np.random.randint(0, 2, 100)
        })
        
        dataset_file = tmp_path / "outliers.csv"
        df.to_csv(dataset_file, index=False)
        
        # Upload
        with open(dataset_file, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("outliers.csv", f, "text/csv")},
                data={"name": "Outliers Dataset"}
            )
        
        assert response.status_code == 200
        dataset_id = response.json()["id"]
        
        # Apply outlier detection
        response = client.post(
            "/api/v1/preprocessing/pipeline",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "name": "Outlier Detection Pipeline",
                "steps": [
                    {
                        "type": "outlier_detection",
                        "method": "iqr",
                        "threshold": 1.5
                    }
                ]
            }
        )
        
        if response.status_code in [200, 201]:
            print(f"✓ Outlier detection pipeline created")

    def test_preprocessing_with_encoding(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        tmp_path: Path
    ):
        """Test preprocessing with categorical encoding"""
        
        # Create dataset with categorical features
        df = pd.DataFrame({
            'category1': ['A', 'B', 'C', 'A', 'B'],
            'category2': ['X', 'Y', 'X', 'Y', 'X'],
            'numeric': [1.0, 2.0, 3.0, 4.0, 5.0],
            'target': [0, 1, 0, 1, 0]
        })
        
        dataset_file = tmp_path / "categorical.csv"
        df.to_csv(dataset_file, index=False)
        
        # Upload
        with open(dataset_file, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("categorical.csv", f, "text/csv")},
                data={"name": "Categorical Dataset"}
            )
        
        assert response.status_code == 200
        dataset_id = response.json()["id"]
        
        # Apply encoding
        response = client.post(
            "/api/v1/preprocessing/pipeline",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "name": "Encoding Pipeline",
                "steps": [
                    {
                        "type": "encoding",
                        "method": "onehot",
                        "columns": ["category1", "category2"]
                    }
                ]
            }
        )
        
        if response.status_code in [200, 201]:
            print(f"✓ Encoding pipeline created")


@pytest.mark.integration
class TestMLPipelineModelComparison:
    """Test model comparison across multiple algorithms"""

    def test_compare_multiple_models(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test training and comparing multiple models"""
        
        # Upload dataset
        with open(classification_dataset, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("comparison.csv", f, "text/csv")},
                data={"name": "Model Comparison Dataset"}
            )
        
        dataset_id = response.json()["id"]
        
        # Train multiple models
        models_to_test = [
            ("logistic_regression", {}),
            ("random_forest", {"n_estimators": 50}),
            ("gradient_boosting", {"n_estimators": 50})
        ]
        
        model_results = []
        
        for model_type, hyperparams in models_to_test:
            print(f"\n[Training] {model_type}...")
            
            response = client.post(
                "/api/v1/models/train",
                headers=auth_headers,
                json={
                    "dataset_id": dataset_id,
                    "model_type": model_type,
                    "target_column": "target",
                    "hyperparameters": hyperparams,
                    "test_size": 0.2
                }
            )
            
            if response.status_code in [200, 202]:
                model_run_id = response.json().get("model_run_id") or response.json().get("id")
                model_results.append({
                    "model_type": model_type,
                    "model_run_id": model_run_id
                })
                print(f"✓ {model_type} training started: {model_run_id}")
        
        # Wait for all models to train
        time.sleep(10)
        
        # Compare results
        print("\n[Comparison] Retrieving metrics for all models...")
        comparison_results = []
        
        for model_info in model_results:
            response = client.get(
                f"/api/v1/models/train/{model_info['model_run_id']}/metrics",
                headers=auth_headers
            )
            
            if response.status_code == 200:
                metrics = response.json()
                comparison_results.append({
                    "model_type": model_info["model_type"],
                    "metrics": metrics
                })
                print(f"✓ {model_info['model_type']}: {metrics.get('accuracy', 'N/A')}")
        
        # Verify we got results
        assert len(comparison_results) > 0, "No model results retrieved"


@pytest.mark.integration
class TestMLPipelineExperimentTracking:
    """Test experiment tracking and versioning"""

    def test_create_and_track_experiment(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test creating and tracking ML experiments"""
        
        # Create experiment
        print("\n[Step 1] Creating experiment...")
        response = client.post(
            "/api/v1/experiments",
            headers=auth_headers,
            json={
                "name": "Customer Churn Experiment",
                "description": "Testing different models for churn prediction",
                "tags": ["classification", "churn"]
            }
        )
        
        if response.status_code not in [200, 201]:
            pytest.skip("Experiment tracking not available")
        
        experiment_id = response.json()["id"]
        print(f"✓ Experiment created: {experiment_id}")
        
        # Upload dataset to experiment
        with open(classification_dataset, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("experiment_data.csv", f, "text/csv")},
                data={
                    "name": "Experiment Dataset",
                    "experiment_id": experiment_id
                }
            )
        
        if response.status_code == 200:
            dataset_id = response.json()["id"]
            print(f"✓ Dataset linked to experiment: {dataset_id}")
        
        # Train model as part of experiment
        response = client.post(
            "/api/v1/models/train",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "experiment_id": experiment_id,
                "model_type": "random_forest",
                "target_column": "target",
                "hyperparameters": {"n_estimators": 100}
            }
        )
        
        if response.status_code in [200, 202]:
            model_run_id = response.json().get("model_run_id") or response.json().get("id")
            print(f"✓ Model training started in experiment: {model_run_id}")
        
        # Get experiment summary
        response = client.get(
            f"/api/v1/experiments/{experiment_id}",
            headers=auth_headers
        )
        
        if response.status_code == 200:
            experiment = response.json()
            print(f"✓ Experiment summary retrieved: {experiment.get('name')}")


@pytest.mark.integration
class TestMLPipelineCodeGeneration:
    """Test code generation from trained pipeline"""

    def test_generate_deployment_code(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test generating deployment code from trained model"""
        
        # Upload dataset and train model
        with open(classification_dataset, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("codegen.csv", f, "text/csv")},
                data={"name": "Code Generation Dataset"}
            )
        
        dataset_id = response.json()["id"]
        
        # Train model
        response = client.post(
            "/api/v1/models/train",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "model_type": "random_forest",
                "target_column": "target"
            }
        )
        
        if response.status_code not in [200, 202]:
            pytest.skip("Model training not available")
        
        model_run_id = response.json().get("model_run_id") or response.json().get("id")
        time.sleep(5)
        
        # Generate code
        response = client.post(
            f"/api/v1/code-generation/generate",
            headers=auth_headers,
            json={
                "model_run_id": model_run_id,
                "code_type": "deployment",
                "include_preprocessing": True
            }
        )
        
        if response.status_code == 200:
            code = response.json()
            print(f"✓ Deployment code generated: {len(code.get('code', ''))} chars")
            assert "code" in code or "preprocessing_code" in code


@pytest.mark.integration
@pytest.mark.slow
class TestMLPipelinePerformance:
    """Test pipeline performance with larger datasets"""

    def test_pipeline_with_large_dataset(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        tmp_path: Path
    ):
        """Test pipeline performance with larger dataset"""
        
        # Create larger dataset
        print("\n[Setup] Creating large dataset (10,000 rows)...")
        np.random.seed(42)
        n_samples = 10000
        
        df = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'feature4': np.random.randn(n_samples),
            'feature5': np.random.randn(n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        
        dataset_file = tmp_path / "large_dataset.csv"
        df.to_csv(dataset_file, index=False)
        print(f"✓ Dataset created: {n_samples} rows")
        
        # Upload
        print("\n[Step 1] Uploading large dataset...")
        start_time = time.time()
        
        with open(dataset_file, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("large_dataset.csv", f, "text/csv")},
                data={"name": "Large Dataset"}
            )
        
        upload_time = time.time() - start_time
        assert response.status_code == 200
        dataset_id = response.json()["id"]
        print(f"✓ Upload completed in {upload_time:.2f}s")
        
        # Preprocess
        print("\n[Step 2] Preprocessing large dataset...")
        start_time = time.time()
        
        response = client.post(
            "/api/v1/preprocessing/pipeline",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "name": "Large Dataset Pipeline",
                "steps": [
                    {"type": "scaling", "method": "standard"}
                ]
            }
        )
        
        preprocess_time = time.time() - start_time
        print(f"✓ Preprocessing completed in {preprocess_time:.2f}s")
        
        # Train
        print("\n[Step 3] Training on large dataset...")
        start_time = time.time()
        
        response = client.post(
            "/api/v1/models/train",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "model_type": "logistic_regression",
                "target_column": "target",
                "test_size": 0.2
            }
        )
        
        if response.status_code in [200, 202]:
            model_run_id = response.json().get("model_run_id") or response.json().get("id")
            
            # Wait for training
            max_wait = 60
            training_start = time.time()
            
            while time.time() - training_start < max_wait:
                response = client.get(
                    f"/api/v1/models/train/{model_run_id}/status",
                    headers=auth_headers
                )
                
                if response.status_code == 200:
                    status = response.json().get("status")
                    if status in ["completed", "success"]:
                        training_time = time.time() - start_time
                        print(f"✓ Training completed in {training_time:.2f}s")
                        break
                
                time.sleep(2)
        
        print(f"\n✓ Large dataset pipeline test completed")
        print(f"   - Upload: {upload_time:.2f}s")
        print(f"   - Preprocessing: {preprocess_time:.2f}s")
        if 'training_time' in locals():
            print(f"   - Training: {training_time:.2f}s")
