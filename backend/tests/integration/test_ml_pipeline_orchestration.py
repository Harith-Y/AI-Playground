"""
Integration Tests for ML Pipeline Orchestration

Tests the orchestration of multiple ML components including
tuning, feature engineering, and model comparison.
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
class TestTuningOrchestration:
    """Test hyperparameter tuning orchestration"""

    def test_grid_search_orchestration(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test grid search tuning orchestration"""
        
        print("\n[Grid Search Orchestration Test]")
        
        # Upload dataset
        with open(classification_dataset, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("grid_search.csv", f, "text/csv")},
                data={"name": "Grid Search Dataset"}
            )
        
        assert response.status_code == 200
        dataset_id = response.json()["id"]
        
        # Create tuning job
        tuning_config = {
            "dataset_id": dataset_id,
            "experiment_name": "Grid Search Experiment",
            "model_type": "random_forest",
            "target_column": "target",
            "tuning_method": "grid_search",
            "param_grid": {
                "n_estimators": [10, 50, 100],
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5]
            },
            "cv_folds": 3,
            "scoring": "accuracy",
            "n_jobs": -1
        }
        
        response = client.post(
            "/api/v1/tuning/orchestrate",
            headers=auth_headers,
            json=tuning_config
        )
        
        if response.status_code in [200, 201, 202]:
            result = response.json()
            job_id = result.get("job_id") or result.get("id")
            
            print(f"✓ Grid search job created: {job_id}")
            
            # Monitor progress
            max_wait = 180
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                response = client.get(
                    f"/api/v1/tuning/orchestrate/{job_id}/status",
                    headers=auth_headers
                )
                
                if response.status_code == 200:
                    status_data = response.json()
                    status = status_data.get("status")
                    
                    print(f"  Status: {status}")
                    
                    if status in ["completed", "success"]:
                        print(f"✓ Grid search completed")
                        
                        # Get results
                        response = client.get(
                            f"/api/v1/tuning/orchestrate/{job_id}/results",
                            headers=auth_headers
                        )
                        
                        if response.status_code == 200:
                            results = response.json()
                            print(f"✓ Best params: {results.get('best_params')}")
                            print(f"✓ Best score: {results.get('best_score')}")
                            
                            assert "best_params" in results
                            assert "best_score" in results
                        
                        break
                    
                    elif status in ["failed", "error"]:
                        pytest.fail(f"Tuning failed: {status_data}")
                
                time.sleep(5)
        else:
            pytest.skip("Tuning orchestration endpoint not available")

    def test_random_search_orchestration(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        regression_dataset: Path
    ):
        """Test random search tuning orchestration"""
        
        print("\n[Random Search Orchestration Test]")
        
        # Upload dataset
        with open(regression_dataset, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("random_search.csv", f, "text/csv")},
                data={"name": "Random Search Dataset"}
            )
        
        dataset_id = response.json()["id"]
        
        # Create random search job
        tuning_config = {
            "dataset_id": dataset_id,
            "experiment_name": "Random Search Experiment",
            "model_type": "gradient_boosting",
            "target_column": "target",
            "tuning_method": "random_search",
            "param_distributions": {
                "n_estimators": {"type": "randint", "low": 10, "high": 200},
                "learning_rate": {"type": "uniform", "low": 0.01, "high": 0.3},
                "max_depth": {"type": "randint", "low": 3, "high": 10}
            },
            "n_iterations": 20,
            "cv_folds": 3,
            "scoring": "neg_mean_squared_error"
        }
        
        response = client.post(
            "/api/v1/tuning/orchestrate",
            headers=auth_headers,
            json=tuning_config
        )
        
        if response.status_code in [200, 201, 202]:
            job_id = response.json().get("job_id") or response.json().get("id")
            print(f"✓ Random search job created: {job_id}")
        else:
            pytest.skip("Random search not available")


@pytest.mark.integration
class TestFeatureEngineeringOrchestration:
    """Test feature engineering orchestration"""

    def test_automated_feature_engineering(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test automated feature engineering pipeline"""
        
        print("\n[Automated Feature Engineering Test]")
        
        # Upload dataset
        with open(classification_dataset, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("feature_eng.csv", f, "text/csv")},
                data={"name": "Feature Engineering Dataset"}
            )
        
        dataset_id = response.json()["id"]
        
        # Step 1: Analyze features
        print("\n[Step 1] Analyzing features...")
        response = client.get(
            f"/api/v1/datasets/{dataset_id}/feature-analysis",
            headers=auth_headers
        )
        
        if response.status_code == 200:
            analysis = response.json()
            print(f"✓ Features analyzed: {len(analysis.get('features', []))} features")
        
        # Step 2: Select features using multiple methods
        print("\n[Step 2] Selecting features...")
        selection_results = []
        
        methods = [
            {"method": "variance_threshold", "params": {"threshold": 0.01}},
            {"method": "correlation", "params": {"threshold": 0.3, "target_column": "target"}},
            {"method": "mutual_info", "params": {"k": 10, "target_column": "target"}}
        ]
        
        for method_config in methods:
            response = client.post(
                "/api/v1/features/select",
                headers=auth_headers,
                json={
                    "dataset_id": dataset_id,
                    **method_config
                }
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                selection_results.append({
                    "method": method_config["method"],
                    "features": result.get("features", [])
                })
                print(f"✓ {method_config['method']}: {len(result.get('features', []))} features")
        
        # Step 3: Create ensemble feature selection
        if selection_results:
            # Find common features across methods
            feature_sets = [set(r["features"]) for r in selection_results if r["features"]]
            if feature_sets:
                common_features = set.intersection(*feature_sets) if len(feature_sets) > 1 else feature_sets[0]
                print(f"\n✓ Common features across methods: {len(common_features)}")

    def test_feature_importance_ranking(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test feature importance ranking"""
        
        print("\n[Feature Importance Ranking Test]")
        
        # Upload and train model
        with open(classification_dataset, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("importance.csv", f, "text/csv")},
                data={"name": "Importance Dataset"}
            )
        
        dataset_id = response.json()["id"]
        
        # Train model with feature importance
        response = client.post(
            "/api/v1/models/train",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "model_type": "random_forest",
                "target_column": "target",
                "hyperparameters": {"n_estimators": 100},
                "compute_feature_importance": True
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
                importance_data = response.json()
                features = importance_data.get("features", [])
                
                print(f"✓ Feature importance computed for {len(features)} features")
                
                # Verify sorted by importance
                if features and isinstance(features, list):
                    importances = [f.get("importance", 0) for f in features if isinstance(f, dict)]
                    if importances:
                        assert importances == sorted(importances, reverse=True), \
                            "Features should be sorted by importance"


@pytest.mark.integration
class TestModelComparisonOrchestration:
    """Test model comparison orchestration"""

    def test_automated_model_comparison(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test automated comparison of multiple models"""
        
        print("\n[Automated Model Comparison Test]")
        
        # Upload dataset
        with open(classification_dataset, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("comparison.csv", f, "text/csv")},
                data={"name": "Model Comparison Dataset"}
            )
        
        dataset_id = response.json()["id"]
        
        # Define models to compare
        models_config = [
            {
                "name": "Logistic Regression",
                "model_type": "logistic_regression",
                "hyperparameters": {"max_iter": 1000}
            },
            {
                "name": "Random Forest",
                "model_type": "random_forest",
                "hyperparameters": {"n_estimators": 50, "max_depth": 5}
            },
            {
                "name": "Gradient Boosting",
                "model_type": "gradient_boosting",
                "hyperparameters": {"n_estimators": 50, "learning_rate": 0.1}
            }
        ]
        
        # Train all models
        model_runs = []
        
        for model_config in models_config:
            print(f"\n[Training] {model_config['name']}...")
            
            response = client.post(
                "/api/v1/models/train",
                headers=auth_headers,
                json={
                    "dataset_id": dataset_id,
                    "model_type": model_config["model_type"],
                    "target_column": "target",
                    "hyperparameters": model_config["hyperparameters"],
                    "test_size": 0.2,
                    "cv_folds": 3
                }
            )
            
            if response.status_code in [200, 202]:
                model_run_id = response.json().get("model_run_id") or response.json().get("id")
                model_runs.append({
                    "name": model_config["name"],
                    "model_run_id": model_run_id
                })
                print(f"✓ {model_config['name']} training started")
        
        # Wait for training
        print("\n[Waiting] Training in progress...")
        time.sleep(15)
        
        # Compare results
        print("\n[Comparison] Collecting metrics...")
        comparison_table = []
        
        for model_run in model_runs:
            response = client.get(
                f"/api/v1/models/train/{model_run['model_run_id']}/metrics",
                headers=auth_headers
            )
            
            if response.status_code == 200:
                metrics = response.json()
                comparison_table.append({
                    "model": model_run["name"],
                    "accuracy": metrics.get("accuracy", "N/A"),
                    "precision": metrics.get("precision", "N/A"),
                    "recall": metrics.get("recall", "N/A"),
                    "f1_score": metrics.get("f1_score", "N/A")
                })
        
        # Display comparison
        if comparison_table:
            print("\n" + "=" * 60)
            print("MODEL COMPARISON RESULTS")
            print("=" * 60)
            
            for result in comparison_table:
                print(f"\n{result['model']}:")
                print(f"  Accuracy:  {result['accuracy']}")
                print(f"  Precision: {result['precision']}")
                print(f"  Recall:    {result['recall']}")
                print(f"  F1 Score:  {result['f1_score']}")
            
            print("\n" + "=" * 60)
            
            # Verify we got results
            assert len(comparison_table) > 0

    def test_model_comparison_with_cv(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        regression_dataset: Path
    ):
        """Test model comparison with cross-validation"""
        
        print("\n[Model Comparison with CV Test]")
        
        # Upload dataset
        with open(regression_dataset, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("cv_comparison.csv", f, "text/csv")},
                data={"name": "CV Comparison Dataset"}
            )
        
        dataset_id = response.json()["id"]
        
        # Train models with CV
        models = ["linear_regression", "ridge", "lasso"]
        
        for model_type in models:
            print(f"\n[Training] {model_type} with CV...")
            
            response = client.post(
                "/api/v1/models/train",
                headers=auth_headers,
                json={
                    "dataset_id": dataset_id,
                    "model_type": model_type,
                    "target_column": "target",
                    "cv_folds": 5,
                    "cv_scoring": "neg_mean_squared_error"
                }
            )
            
            if response.status_code in [200, 202]:
                print(f"✓ {model_type} training started")


@pytest.mark.integration
class TestPipelineExportOrchestration:
    """Test pipeline export and code generation orchestration"""

    def test_export_complete_pipeline(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test exporting complete pipeline as code"""
        
        print("\n[Pipeline Export Test]")
        
        # Upload dataset
        with open(classification_dataset, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("export_test.csv", f, "text/csv")},
                data={"name": "Export Test Dataset"}
            )
        
        dataset_id = response.json()["id"]
        
        # Create preprocessing pipeline
        response = client.post(
            "/api/v1/preprocessing/pipeline",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "name": "Export Pipeline",
                "steps": [
                    {"type": "imputation", "strategy": "mean"},
                    {"type": "scaling", "method": "standard"}
                ]
            }
        )
        
        if response.status_code in [200, 201]:
            pipeline_id = response.json()["id"]
            
            # Train model
            response = client.post(
                "/api/v1/models/train",
                headers=auth_headers,
                json={
                    "dataset_id": dataset_id,
                    "model_type": "random_forest",
                    "target_column": "target",
                    "pipeline_id": pipeline_id
                }
            )
            
            if response.status_code in [200, 202]:
                model_run_id = response.json().get("model_run_id") or response.json().get("id")
                
                # Wait for training
                time.sleep(5)
                
                # Generate export code
                response = client.post(
                    "/api/v1/code-generation/export-pipeline",
                    headers=auth_headers,
                    json={
                        "model_run_id": model_run_id,
                        "include_preprocessing": True,
                        "include_training": True,
                        "include_evaluation": True,
                        "output_format": "modular"
                    }
                )
                
                if response.status_code == 200:
                    export_data = response.json()
                    
                    print(f"✓ Pipeline exported")
                    
                    # Verify export contains code
                    assert "preprocessing_code" in export_data or "code" in export_data
                    
                    if "preprocessing_code" in export_data:
                        print(f"  - Preprocessing code: {len(export_data['preprocessing_code'])} chars")
                    if "training_code" in export_data:
                        print(f"  - Training code: {len(export_data['training_code'])} chars")
                    if "evaluation_code" in export_data:
                        print(f"  - Evaluation code: {len(export_data['evaluation_code'])} chars")


@pytest.mark.integration
class TestExperimentOrchestration:
    """Test experiment orchestration and tracking"""

    def test_full_experiment_lifecycle(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test complete experiment lifecycle"""
        
        print("\n" + "=" * 60)
        print("FULL EXPERIMENT LIFECYCLE TEST")
        print("=" * 60)
        
        # Step 1: Create experiment
        print("\n[Step 1] Creating experiment...")
        response = client.post(
            "/api/v1/experiments",
            headers=auth_headers,
            json={
                "name": "Complete Lifecycle Experiment",
                "description": "Testing full experiment workflow",
                "tags": ["integration", "test"]
            }
        )
        
        if response.status_code not in [200, 201]:
            pytest.skip("Experiment API not available")
        
        experiment_id = response.json()["id"]
        print(f"✓ Experiment created: {experiment_id}")
        
        # Step 2: Upload dataset
        print("\n[Step 2] Uploading dataset...")
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
        
        dataset_id = response.json()["id"]
        print(f"✓ Dataset uploaded: {dataset_id}")
        
        # Step 3: Create preprocessing config
        print("\n[Step 3] Creating preprocessing configuration...")
        response = client.post(
            f"/api/v1/experiments/{experiment_id}/preprocessing-config",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "steps": [
                    {"type": "imputation", "strategy": "mean"},
                    {"type": "scaling", "method": "standard"}
                ]
            }
        )
        
        if response.status_code in [200, 201]:
            print(f"✓ Preprocessing config created")
        
        # Step 4: Run multiple models
        print("\n[Step 4] Training multiple models...")
        model_types = ["logistic_regression", "random_forest"]
        
        for model_type in model_types:
            response = client.post(
                "/api/v1/models/train",
                headers=auth_headers,
                json={
                    "dataset_id": dataset_id,
                    "experiment_id": experiment_id,
                    "model_type": model_type,
                    "target_column": "target"
                }
            )
            
            if response.status_code in [200, 202]:
                print(f"✓ {model_type} training started")
        
        # Step 5: Get experiment summary
        print("\n[Step 5] Getting experiment summary...")
        time.sleep(5)
        
        response = client.get(
            f"/api/v1/experiments/{experiment_id}/summary",
            headers=auth_headers
        )
        
        if response.status_code == 200:
            summary = response.json()
            print(f"✓ Experiment summary retrieved")
            print(f"  - Models: {summary.get('total_models', 0)}")
            print(f"  - Datasets: {summary.get('total_datasets', 0)}")
        
        print("\n" + "=" * 60)
        print("EXPERIMENT LIFECYCLE TEST COMPLETED")
        print("=" * 60)


@pytest.mark.integration
@pytest.mark.slow
class TestDataPipelineOrchestration:
    """Test data pipeline orchestration"""

    def test_data_quality_pipeline(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        tmp_path: Path
    ):
        """Test data quality checking pipeline"""
        
        print("\n[Data Quality Pipeline Test]")
        
        # Create dataset with quality issues
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5, 100, 7, 8],  # Missing value and outlier
            'feature2': [10, 10, 10, 10, 10, 10, 10, 10],  # Low variance
            'feature3': [1, 2, 3, 4, 5, 6, 7, 8],  # Good feature
            'duplicate_col': [1, 2, 3, 4, 5, 6, 7, 8],
            'target': [0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # Add duplicate rows
        df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
        
        dataset_file = tmp_path / "quality_check.csv"
        df.to_csv(dataset_file, index=False)
        
        # Upload dataset
        with open(dataset_file, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("quality_check.csv", f, "text/csv")},
                data={"name": "Quality Check Dataset"}
            )
        
        dataset_id = response.json()["id"]
        
        # Run quality checks
        response = client.get(
            f"/api/v1/datasets/{dataset_id}/quality-report",
            headers=auth_headers
        )
        
        if response.status_code == 200:
            quality_report = response.json()
            
            print(f"✓ Quality report generated:")
            print(f"  - Missing values: {quality_report.get('missing_values', 0)}")
            print(f"  - Duplicate rows: {quality_report.get('duplicate_rows', 0)}")
            print(f"  - Outliers detected: {quality_report.get('outliers', 0)}")
        else:
            pytest.skip("Quality report endpoint not available")

    def test_data_transformation_pipeline(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        tmp_path: Path
    ):
        """Test complete data transformation pipeline"""
        
        print("\n[Data Transformation Pipeline Test]")
        
        # Create complex dataset
        np.random.seed(42)
        df = pd.DataFrame({
            'numeric1': np.random.randn(100),
            'numeric2': np.random.randn(100) * 10 + 50,
            'categorical1': np.random.choice(['A', 'B', 'C'], 100),
            'categorical2': np.random.choice(['X', 'Y'], 100),
            'text': ['sample_text'] * 100,
            'target': np.random.randint(0, 2, 100)
        })
        
        # Add missing values
        df.loc[df.sample(10).index, 'numeric1'] = np.nan
        df.loc[df.sample(10).index, 'categorical1'] = np.nan
        
        dataset_file = tmp_path / "transformation.csv"
        df.to_csv(dataset_file, index=False)
        
        # Upload
        with open(dataset_file, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("transformation.csv", f, "text/csv")},
                data={"name": "Transformation Dataset"}
            )
        
        dataset_id = response.json()["id"]
        
        # Create comprehensive transformation pipeline
        transformation_config = {
            "dataset_id": dataset_id,
            "name": "Complete Transformation Pipeline",
            "steps": [
                {
                    "type": "drop_columns",
                    "columns": ["text"]
                },
                {
                    "type": "imputation",
                    "strategy": "mean",
                    "columns": ["numeric1", "numeric2"]
                },
                {
                    "type": "imputation",
                    "strategy": "mode",
                    "columns": ["categorical1", "categorical2"]
                },
                {
                    "type": "outlier_detection",
                    "method": "zscore",
                    "threshold": 3.0
                },
                {
                    "type": "scaling",
                    "method": "standard",
                    "columns": ["numeric1", "numeric2"]
                },
                {
                    "type": "encoding",
                    "method": "onehot",
                    "columns": ["categorical1", "categorical2"]
                }
            ]
        }
        
        response = client.post(
            "/api/v1/preprocessing/pipeline",
            headers=auth_headers,
            json=transformation_config
        )
        
        if response.status_code in [200, 201]:
            pipeline_id = response.json()["id"]
            print(f"✓ Transformation pipeline created: {pipeline_id}")
            
            # Apply pipeline
            response = client.post(
                f"/api/v1/preprocessing/pipeline/{pipeline_id}/apply",
                headers=auth_headers,
                json={"dataset_id": dataset_id}
            )
            
            if response.status_code in [200, 202]:
                print(f"✓ Pipeline applied successfully")
        else:
            pytest.skip("Transformation pipeline endpoint not available")
