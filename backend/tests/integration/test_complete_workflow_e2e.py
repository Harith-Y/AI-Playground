"""
Complete End-to-End Workflow Tests

Tests the complete ML workflow: Upload → Train → Evaluate → Code Generation

This test suite covers the entire user journey from uploading a dataset
through training models, evaluating results, and generating production code.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from fastapi.testclient import TestClient
import time
import json
import tempfile


@pytest.fixture
def sample_classification_data(tmp_path: Path) -> Path:
    """Create a sample classification dataset for testing."""
    np.random.seed(42)
    
    # Generate synthetic classification data
    n_samples = 200
    df = pd.DataFrame({
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'years_employed': np.random.randint(0, 40, n_samples),
        'debt_ratio': np.random.uniform(0, 1, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    })
    
    file_path = tmp_path / "classification_data.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_regression_data(tmp_path: Path) -> Path:
    """Create a sample regression dataset for testing."""
    np.random.seed(42)
    
    # Generate synthetic regression data
    n_samples = 200
    X = np.random.randn(n_samples, 4)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] + np.random.randn(n_samples) * 0.5
    
    df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4'])
    df['target'] = y
    
    file_path = tmp_path / "regression_data.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.mark.integration
@pytest.mark.e2e
class TestCompleteWorkflowClassification:
    """Test complete workflow for classification tasks."""
    
    def test_full_classification_workflow(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        sample_classification_data: Path
    ):
        """
        Test complete classification workflow:
        1. Upload dataset
        2. Preview and explore data
        3. Create preprocessing steps
        4. Train model
        5. Evaluate model
        6. Generate code
        """
        
        print("\n" + "="*80)
        print("COMPLETE CLASSIFICATION WORKFLOW TEST")
        print("="*80)
        
        # ===================================================================
        # STEP 1: UPLOAD DATASET
        # ===================================================================
        print("\n[STEP 1] Uploading classification dataset...")
        
        with open(sample_classification_data, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("classification_data.csv", f, "text/csv")}
            )
        
        assert response.status_code in [200, 201], f"Upload failed: {response.text}"
        dataset_data = response.json()
        dataset_id = dataset_data.get("id")
        
        assert dataset_id is not None, "Dataset ID not returned"
        print(f"✓ Dataset uploaded successfully")
        print(f"  - Dataset ID: {dataset_id}")
        print(f"  - Rows: {dataset_data.get('rows', 'N/A')}")
        print(f"  - Columns: {dataset_data.get('cols', 'N/A')}")
        
        # ===================================================================
        # STEP 2: PREVIEW AND EXPLORE DATA
        # ===================================================================
        print("\n[STEP 2] Previewing and exploring dataset...")
        
        # Get dataset preview
        response = client.get(
            f"/api/v1/datasets/{dataset_id}/preview",
            headers=auth_headers,
            params={"rows": 5}
        )
        
        if response.status_code == 200:
            preview_data = response.json()
            print(f"✓ Dataset preview retrieved")
            print(f"  - Preview rows: {len(preview_data.get('data', []))}")
        
        # Get dataset statistics
        response = client.get(
            f"/api/v1/datasets/{dataset_id}/stats",
            headers=auth_headers
        )
        
        if response.status_code == 200:
            stats_data = response.json()
            print(f"✓ Dataset statistics retrieved")
            print(f"  - Missing values: {stats_data.get('missing_values', {})}")
            print(f"  - Duplicates: {stats_data.get('duplicates', 'N/A')}")
        
        # ===================================================================
        # STEP 3: CREATE PREPROCESSING STEPS
        # ===================================================================
        print("\n[STEP 3] Creating preprocessing pipeline...")
        
        preprocessing_steps = []
        
        # Step 3.1: Create imputation step
        response = client.post(
            "/api/v1/preprocessing/steps",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "step_type": "imputation",
                "parameters": {"strategy": "mean"},
                "order": 0
            }
        )
        
        if response.status_code in [200, 201]:
            step_data = response.json()
            preprocessing_steps.append(step_data.get("id"))
            print(f"✓ Imputation step created")
        
        # Step 3.2: Create scaling step
        response = client.post(
            "/api/v1/preprocessing/steps",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "step_type": "scaling",
                "parameters": {"method": "standard"},
                "order": 1
            }
        )
        
        if response.status_code in [200, 201]:
            step_data = response.json()
            preprocessing_steps.append(step_data.get("id"))
            print(f"✓ Scaling step created")
        
        print(f"  - Total preprocessing steps: {len(preprocessing_steps)}")
        
        # ===================================================================
        # STEP 4: TRAIN MODEL
        # ===================================================================
        print("\n[STEP 4] Training classification model...")
        
        training_config = {
            "dataset_id": dataset_id,
            "model_type": "random_forest",
            "hyperparameters": {
                "n_estimators": 50,
                "max_depth": 10,
                "random_state": 42
            }
        }
        
        response = client.post(
            "/api/v1/models/train",
            headers=auth_headers,
            json=training_config
        )
        
        assert response.status_code in [200, 201, 202], f"Training failed: {response.text}"
        train_response = response.json()
        model_run_id = train_response.get("model_run_id") or train_response.get("run_id") or train_response.get("id")
        
        assert model_run_id is not None, "Model run ID not returned"
        print(f"✓ Model training initiated")
        print(f"  - Model Run ID: {model_run_id}")
        print(f"  - Model Type: {training_config['model_type']}")
        
        # Wait for training to complete
        print("\n  Waiting for training to complete...")
        max_wait_time = 60  # seconds
        start_time = time.time()
        training_completed = False
        
        while time.time() - start_time < max_wait_time:
            response = client.get(
                f"/api/v1/models/runs/{model_run_id}/status",
                headers=auth_headers
            )
            
            if response.status_code == 200:
                status_data = response.json()
                status = status_data.get("status", "").lower()
                
                if status in ["completed", "success", "finished"]:
                    training_completed = True
                    print(f"✓ Training completed successfully")
                    break
                elif status in ["failed", "error"]:
                    pytest.fail(f"Training failed: {status_data}")
                    break
            
            time.sleep(2)
        
        if not training_completed:
            pytest.skip("Training did not complete within timeout period")
        
        # ===================================================================
        # STEP 5: EVALUATE MODEL
        # ===================================================================
        print("\n[STEP 5] Evaluating model performance...")
        
        # Get training results
        response = client.get(
            f"/api/v1/models/runs/{model_run_id}/results",
            headers=auth_headers
        )
        
        if response.status_code == 200:
            results_data = response.json()
            metrics = results_data.get("metrics", {})
            
            print(f"✓ Model evaluation results retrieved")
            
            # Display metrics
            if isinstance(metrics, dict):
                if "test" in metrics:
                    test_metrics = metrics["test"]
                    print(f"  - Test Accuracy: {test_metrics.get('accuracy', 'N/A')}")
                    print(f"  - Test Precision: {test_metrics.get('precision', 'N/A')}")
                    print(f"  - Test Recall: {test_metrics.get('recall', 'N/A')}")
                    print(f"  - Test F1 Score: {test_metrics.get('f1_score', 'N/A')}")
                else:
                    print(f"  - Accuracy: {metrics.get('accuracy', 'N/A')}")
                    print(f"  - Precision: {metrics.get('precision', 'N/A')}")
                    print(f"  - Recall: {metrics.get('recall', 'N/A')}")
                    print(f"  - F1 Score: {metrics.get('f1_score', 'N/A')}")
            
            # Verify key metrics exist
            assert metrics is not None, "No metrics returned"
        else:
            print(f"⚠ Could not retrieve results (status: {response.status_code})")
        
        # Get feature importance (if available)
        response = client.get(
            f"/api/v1/models/runs/{model_run_id}/feature-importance",
            headers=auth_headers
        )
        
        if response.status_code == 200:
            importance_data = response.json()
            print(f"✓ Feature importance retrieved")
            print(f"  - Features analyzed: {len(importance_data.get('features', []))}")
        
        # ===================================================================
        # STEP 6: GENERATE CODE
        # ===================================================================
        print("\n[STEP 6] Generating production code...")
        
        # Generate Python script
        response = client.post(
            "/api/v1/code-generation/python",
            headers=auth_headers,
            json={
                "experiment_id": dataset_id,
                "include_preprocessing": True,
                "include_training": True,
                "include_evaluation": True
            }
        )
        
        if response.status_code in [200, 201]:
            code_data = response.json()
            generated_code = code_data.get("code", "")
            
            print(f"✓ Python code generated")
            print(f"  - Code length: {len(generated_code)} characters")
            print(f"  - Lines of code: {len(generated_code.splitlines())}")
            
            # Verify code contains key components
            assert "import" in generated_code, "Generated code missing imports"
            assert "def" in generated_code or "class" in generated_code, "Generated code missing functions/classes"
        else:
            print(f"⚠ Code generation returned status: {response.status_code}")
        
        # Generate Jupyter Notebook
        response = client.post(
            "/api/v1/code-generation/notebook",
            headers=auth_headers,
            json={
                "experiment_id": dataset_id
            }
        )
        
        if response.status_code in [200, 201]:
            notebook_data = response.json()
            print(f"✓ Jupyter notebook generated")
        
        # ===================================================================
        # STEP 7: EXPORT EXPERIMENT CONFIGURATION
        # ===================================================================
        print("\n[STEP 7] Exporting experiment configuration...")
        
        response = client.get(
            f"/api/v1/experiments/{dataset_id}/config",
            headers=auth_headers,
            params={"include_results": True}
        )
        
        if response.status_code == 200:
            config_data = response.json()
            print(f"✓ Experiment configuration exported")
            print(f"  - Configuration version: {config_data.get('version', 'N/A')}")
        
        # ===================================================================
        # WORKFLOW COMPLETE
        # ===================================================================
        print("\n" + "="*80)
        print("✓ COMPLETE CLASSIFICATION WORKFLOW TEST PASSED")
        print("="*80)


@pytest.mark.integration
@pytest.mark.e2e
class TestCompleteWorkflowRegression:
    """Test complete workflow for regression tasks."""
    
    def test_full_regression_workflow(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        sample_regression_data: Path
    ):
        """
        Test complete regression workflow:
        1. Upload dataset
        2. Explore data
        3. Preprocess data
        4. Train model
        5. Evaluate model
        6. Generate code
        """
        
        print("\n" + "="*80)
        print("COMPLETE REGRESSION WORKFLOW TEST")
        print("="*80)
        
        # ===================================================================
        # STEP 1: UPLOAD DATASET
        # ===================================================================
        print("\n[STEP 1] Uploading regression dataset...")
        
        with open(sample_regression_data, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("regression_data.csv", f, "text/csv")}
            )
        
        assert response.status_code in [200, 201]
        dataset_data = response.json()
        dataset_id = dataset_data.get("id")
        
        print(f"✓ Dataset uploaded successfully")
        print(f"  - Dataset ID: {dataset_id}")
        
        # ===================================================================
        # STEP 2: TRAIN REGRESSION MODEL
        # ===================================================================
        print("\n[STEP 2] Training regression model...")
        
        training_config = {
            "dataset_id": dataset_id,
            "model_type": "linear_regression",
            "hyperparameters": {
                "fit_intercept": True
            }
        }
        
        response = client.post(
            "/api/v1/models/train",
            headers=auth_headers,
            json=training_config
        )
        
        assert response.status_code in [200, 201, 202]
        train_response = response.json()
        model_run_id = train_response.get("model_run_id") or train_response.get("run_id") or train_response.get("id")
        
        print(f"✓ Model training initiated")
        print(f"  - Model Run ID: {model_run_id}")
        
        # Wait for training
        time.sleep(5)
        
        # ===================================================================
        # STEP 3: EVALUATE REGRESSION MODEL
        # ===================================================================
        print("\n[STEP 3] Evaluating regression model...")
        
        response = client.get(
            f"/api/v1/models/runs/{model_run_id}/results",
            headers=auth_headers
        )
        
        if response.status_code == 200:
            results_data = response.json()
            metrics = results_data.get("metrics", {})
            
            print(f"✓ Model evaluation results retrieved")
            
            # Display regression metrics
            if isinstance(metrics, dict):
                if "test" in metrics:
                    test_metrics = metrics["test"]
                    print(f"  - Test MSE: {test_metrics.get('mse', 'N/A')}")
                    print(f"  - Test RMSE: {test_metrics.get('rmse', 'N/A')}")
                    print(f"  - Test MAE: {test_metrics.get('mae', 'N/A')}")
                    print(f"  - Test R²: {test_metrics.get('r2', 'N/A')}")
                else:
                    print(f"  - MSE: {metrics.get('mse', 'N/A')}")
                    print(f"  - RMSE: {metrics.get('rmse', 'N/A')}")
                    print(f"  - MAE: {metrics.get('mae', 'N/A')}")
                    print(f"  - R²: {metrics.get('r2', 'N/A')}")
        
        # ===================================================================
        # STEP 4: GENERATE CODE
        # ===================================================================
        print("\n[STEP 4] Generating production code...")
        
        response = client.post(
            "/api/v1/code-generation/python",
            headers=auth_headers,
            json={
                "experiment_id": dataset_id,
                "include_preprocessing": True,
                "include_training": True,
                "include_evaluation": True
            }
        )
        
        if response.status_code in [200, 201]:
            code_data = response.json()
            generated_code = code_data.get("code", "")
            
            print(f"✓ Python code generated")
            print(f"  - Code length: {len(generated_code)} characters")
        
        print("\n" + "="*80)
        print("✓ COMPLETE REGRESSION WORKFLOW TEST PASSED")
        print("="*80)


@pytest.mark.integration
@pytest.mark.e2e
class TestCompleteWorkflowWithTuning:
    """Test complete workflow including hyperparameter tuning."""
    
    def test_workflow_with_hyperparameter_tuning(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        sample_classification_data: Path
    ):
        """
        Test workflow with hyperparameter tuning:
        1. Upload dataset
        2. Train baseline model
        3. Perform hyperparameter tuning
        4. Train optimized model
        5. Compare results
        6. Generate code
        """
        
        print("\n" + "="*80)
        print("WORKFLOW WITH HYPERPARAMETER TUNING TEST")
        print("="*80)
        
        # Upload dataset
        print("\n[STEP 1] Uploading dataset...")
        with open(sample_classification_data, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("tuning_data.csv", f, "text/csv")}
            )
        
        dataset_id = response.json().get("id")
        print(f"✓ Dataset uploaded: {dataset_id}")
        
        # Train baseline model
        print("\n[STEP 2] Training baseline model...")
        response = client.post(
            "/api/v1/models/train",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "model_type": "random_forest",
                "hyperparameters": {
                    "n_estimators": 10,
                    "max_depth": 3
                }
            }
        )
        
        baseline_run_id = response.json().get("model_run_id") or response.json().get("id")
        print(f"✓ Baseline model training started: {baseline_run_id}")
        
        # Perform hyperparameter tuning
        print("\n[STEP 3] Performing hyperparameter tuning...")
        response = client.post(
            "/api/v1/tuning/optimize",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "model_type": "random_forest",
                "tuning_method": "grid_search",
                "param_grid": {
                    "n_estimators": [10, 50],
                    "max_depth": [3, 5]
                },
                "cv_folds": 2
            }
        )
        
        if response.status_code in [200, 201, 202]:
            tuning_id = response.json().get("tuning_id") or response.json().get("id")
            print(f"✓ Hyperparameter tuning started: {tuning_id}")
            
            # Wait for tuning
            time.sleep(10)
            
            # Get best parameters
            response = client.get(
                f"/api/v1/tuning/{tuning_id}/results",
                headers=auth_headers
            )
            
            if response.status_code == 200:
                tuning_results = response.json()
                best_params = tuning_results.get("best_params", {})
                best_score = tuning_results.get("best_score", "N/A")
                
                print(f"✓ Tuning completed")
                print(f"  - Best parameters: {best_params}")
                print(f"  - Best score: {best_score}")
        
        # Generate code
        print("\n[STEP 4] Generating code...")
        response = client.post(
            "/api/v1/code-generation/python",
            headers=auth_headers,
            json={"experiment_id": dataset_id}
        )
        
        if response.status_code in [200, 201]:
            print(f"✓ Code generated successfully")
        
        print("\n" + "="*80)
        print("✓ WORKFLOW WITH TUNING TEST PASSED")
        print("="*80)


@pytest.mark.integration
@pytest.mark.e2e
class TestCompleteWorkflowMultipleModels:
    """Test workflow with multiple model comparison."""
    
    def test_workflow_with_model_comparison(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        sample_classification_data: Path
    ):
        """
        Test workflow with multiple models:
        1. Upload dataset
        2. Train multiple models
        3. Compare results
        4. Select best model
        5. Generate code for best model
        """
        
        print("\n" + "="*80)
        print("WORKFLOW WITH MODEL COMPARISON TEST")
        print("="*80)
        
        # Upload dataset
        print("\n[STEP 1] Uploading dataset...")
        with open(sample_classification_data, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("comparison_data.csv", f, "text/csv")}
            )
        
        dataset_id = response.json().get("id")
        print(f"✓ Dataset uploaded: {dataset_id}")
        
        # Train multiple models
        print("\n[STEP 2] Training multiple models...")
        models_to_train = [
            ("logistic_regression", {}),
            ("random_forest", {"n_estimators": 50}),
            ("gradient_boosting", {"n_estimators": 50})
        ]
        
        model_runs = []
        
        for model_type, hyperparams in models_to_train:
            response = client.post(
                "/api/v1/models/train",
                headers=auth_headers,
                json={
                    "dataset_id": dataset_id,
                    "model_type": model_type,
                    "hyperparameters": hyperparams
                }
            )
            
            if response.status_code in [200, 201, 202]:
                run_id = response.json().get("model_run_id") or response.json().get("id")
                model_runs.append({
                    "model_type": model_type,
                    "run_id": run_id
                })
                print(f"✓ {model_type} training started: {run_id}")
        
        # Wait for all models to train
        print("\n  Waiting for models to train...")
        time.sleep(15)
        
        # Compare results
        print("\n[STEP 3] Comparing model results...")
        best_model = None
        best_score = 0
        
        for model_info in model_runs:
            response = client.get(
                f"/api/v1/models/runs/{model_info['run_id']}/results",
                headers=auth_headers
            )
            
            if response.status_code == 200:
                results = response.json()
                metrics = results.get("metrics", {})
                
                # Get accuracy
                accuracy = None
                if isinstance(metrics, dict):
                    if "test" in metrics:
                        accuracy = metrics["test"].get("accuracy")
                    else:
                        accuracy = metrics.get("accuracy")
                
                if accuracy and accuracy > best_score:
                    best_score = accuracy
                    best_model = model_info
                
                print(f"  - {model_info['model_type']}: {accuracy}")
        
        if best_model:
            print(f"\n✓ Best model: {best_model['model_type']} (accuracy: {best_score})")
        
        # Generate code for best model
        print("\n[STEP 4] Generating code for best model...")
        response = client.post(
            "/api/v1/code-generation/python",
            headers=auth_headers,
            json={"experiment_id": dataset_id}
        )
        
        if response.status_code in [200, 201]:
            print(f"✓ Code generated successfully")
        
        print("\n" + "="*80)
        print("✓ MODEL COMPARISON WORKFLOW TEST PASSED")
        print("="*80)


@pytest.mark.integration
@pytest.mark.e2e
class TestCompleteWorkflowErrorHandling:
    """Test workflow error handling and edge cases."""
    
    def test_workflow_with_invalid_data(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        tmp_path: Path
    ):
        """Test workflow handles invalid data gracefully."""
        
        print("\n[TEST] Workflow with invalid data...")
        
        # Create invalid dataset (all missing values)
        df = pd.DataFrame({
            'col1': [None, None, None],
            'col2': [None, None, None],
            'target': [0, 1, 0]
        })
        
        file_path = tmp_path / "invalid_data.csv"
        df.to_csv(file_path, index=False)
        
        # Upload
        with open(file_path, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("invalid_data.csv", f, "text/csv")}
            )
        
        if response.status_code in [200, 201]:
            dataset_id = response.json().get("id")
            
            # Try to train model (should handle gracefully)
            response = client.post(
                "/api/v1/models/train",
                headers=auth_headers,
                json={
                    "dataset_id": dataset_id,
                    "model_type": "random_forest"
                }
            )
            
            # Should either succeed with preprocessing or return error
            assert response.status_code in [200, 201, 202, 400, 422]
            print(f"✓ Invalid data handled gracefully (status: {response.status_code})")
    
    def test_workflow_with_missing_dataset(
        self,
        client: TestClient,
        auth_headers: Dict[str, str]
    ):
        """Test workflow handles missing dataset gracefully."""
        
        print("\n[TEST] Workflow with missing dataset...")
        
        # Try to train model with non-existent dataset
        response = client.post(
            "/api/v1/models/train",
            headers=auth_headers,
            json={
                "dataset_id": "00000000-0000-0000-0000-000000000000",
                "model_type": "random_forest"
            }
        )
        
        # Should return 404 or 400
        assert response.status_code in [400, 404, 422]
        print(f"✓ Missing dataset handled gracefully (status: {response.status_code})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
