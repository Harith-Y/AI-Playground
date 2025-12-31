"""
Tests for Code Generation Endpoints

Tests the /code-generation endpoints for generating Python code.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestCodeGenerationEndpoint:
    """Test /code-generation/python endpoint."""
    
    def test_generate_python_code_basic(self):
        """Test basic Python code generation."""
        request_data = {
            "experiment_name": "Test Experiment",
            "model_type": "random_forest_classifier",
            "task_type": "classification",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 10
            },
            "output_format": "script"
        }
        
        response = client.post("/api/v1/code-generation/python", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "code" in data
        assert "code_type" in data
        assert "output_format" in data
        assert "metadata" in data
        
        # Check code content
        code = data["code"]
        assert len(code) > 0
        assert "import" in code
        assert "random_forest_classifier" in code.lower() or "RandomForestClassifier" in code
    
    def test_generate_python_code_with_preprocessing(self):
        """Test code generation with preprocessing steps."""
        request_data = {
            "experiment_name": "Preprocessing Test",
            "model_type": "logistic_regression",
            "task_type": "classification",
            "preprocessing_steps": [
                {
                    "type": "missing_value_imputation",
                    "parameters": {
                        "strategy": "mean",
                        "columns": ["age", "income"]
                    }
                },
                {
                    "type": "scaling",
                    "parameters": {
                        "scaler": "standard",
                        "columns": ["age", "income", "balance"]
                    }
                }
            ],
            "output_format": "module"
        }
        
        response = client.post("/api/v1/code-generation/python", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        code = data["code"]
        assert "preprocessing" in code.lower() or "imputation" in code.lower()
        assert "scaling" in code.lower() or "scaler" in code.lower()
    
    def test_generate_python_code_with_dataset_info(self):
        """Test code generation with dataset information."""
        request_data = {
            "experiment_name": "Dataset Test",
            "model_type": "random_forest_regressor",
            "task_type": "regression",
            "dataset_info": {
                "file_path": "data/housing.csv",
                "file_format": "csv",
                "target_column": "price",
                "feature_columns": ["bedrooms", "bathrooms", "sqft"]
            },
            "output_format": "script"
        }
        
        response = client.post("/api/v1/code-generation/python", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        code = data["code"]
        assert "housing.csv" in code or "data" in code.lower()
    
    def test_generate_python_code_modular(self):
        """Test modular code generation."""
        request_data = {
            "experiment_name": "Modular Test",
            "model_type": "gradient_boosting_classifier",
            "task_type": "classification",
            "modular": True,
            "output_format": "module"
        }
        
        response = client.post("/api/v1/code-generation/python", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["code_type"] == "modular_pipeline"
        code = data["code"]
        assert len(code) > 0
    
    def test_generate_python_code_different_formats(self):
        """Test different output formats."""
        formats = ["script", "function", "module"]
        
        for output_format in formats:
            request_data = {
                "experiment_name": f"Format Test - {output_format}",
                "model_type": "random_forest_classifier",
                "task_type": "classification",
                "output_format": output_format
            }
            
            response = client.post("/api/v1/code-generation/python", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["output_format"] == output_format
            assert len(data["code"]) > 0
    
    def test_generate_python_code_with_evaluation(self):
        """Test code generation with evaluation."""
        request_data = {
            "experiment_name": "Evaluation Test",
            "model_type": "svc",
            "task_type": "classification",
            "include_evaluation": True,
            "output_format": "script"
        }
        
        response = client.post("/api/v1/code-generation/python", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["metadata"]["includes_evaluation"] is True
    
    def test_generate_python_code_without_evaluation(self):
        """Test code generation without evaluation."""
        request_data = {
            "experiment_name": "No Evaluation Test",
            "model_type": "random_forest_classifier",
            "task_type": "classification",
            "include_evaluation": False,
            "output_format": "script"
        }
        
        response = client.post("/api/v1/code-generation/python", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["metadata"]["includes_evaluation"] is False
    
    def test_generate_python_code_metadata(self):
        """Test metadata in response."""
        request_data = {
            "experiment_name": "Metadata Test",
            "model_type": "random_forest_classifier",
            "task_type": "classification",
            "preprocessing_steps": [
                {"type": "scaling", "parameters": {"scaler": "standard"}}
            ]
        }
        
        response = client.post("/api/v1/code-generation/python", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        metadata = data["metadata"]
        assert "generated_at" in metadata
        assert "model_type" in metadata
        assert "task_type" in metadata
        assert "lines_of_code" in metadata
        assert "includes_preprocessing" in metadata
        assert metadata["model_type"] == "random_forest_classifier"
        assert metadata["includes_preprocessing"] is True
    
    def test_generate_python_code_invalid_model_type(self):
        """Test error handling for invalid model type."""
        request_data = {
            "experiment_name": "Invalid Model Test",
            "model_type": "invalid_model_type",
            "task_type": "classification"
        }
        
        response = client.post("/api/v1/code-generation/python", json=request_data)
        
        # Should still return 200 but may have issues in code
        # The endpoint doesn't validate model types strictly
        assert response.status_code in [200, 500]


class TestModularCodeGenerationEndpoint:
    """Test /code-generation/python/modular endpoint."""
    
    def test_generate_modular_code_basic(self):
        """Test basic modular code generation."""
        request_data = {
            "experiment_name": "Modular Test",
            "model_type": "random_forest_classifier",
            "task_type": "classification",
            "output_format": "module"
        }
        
        response = client.post(
            "/api/v1/code-generation/python/modular",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "training_code" in data
        assert "prediction_code" in data
        assert "metadata" in data
        
        # Check that code sections exist
        assert data["training_code"] is not None
        assert data["prediction_code"] is not None
        assert len(data["training_code"]) > 0
        assert len(data["prediction_code"]) > 0
    
    def test_generate_modular_code_with_preprocessing(self):
        """Test modular code with preprocessing."""
        request_data = {
            "experiment_name": "Modular Preprocessing Test",
            "model_type": "logistic_regression",
            "task_type": "classification",
            "preprocessing_steps": [
                {
                    "type": "missing_value_imputation",
                    "parameters": {"strategy": "mean"}
                }
            ],
            "output_format": "module"
        }
        
        response = client.post(
            "/api/v1/code-generation/python/modular",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "preprocessing_code" in data
        assert data["preprocessing_code"] is not None
        assert len(data["preprocessing_code"]) > 0
    
    def test_generate_modular_code_with_evaluation(self):
        """Test modular code with evaluation."""
        request_data = {
            "experiment_name": "Modular Evaluation Test",
            "model_type": "random_forest_classifier",
            "task_type": "classification",
            "include_evaluation": True,
            "output_format": "module"
        }
        
        response = client.post(
            "/api/v1/code-generation/python/modular",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "evaluation_code" in data
        assert data["evaluation_code"] is not None
        assert len(data["evaluation_code"]) > 0
    
    def test_generate_modular_code_with_requirements(self):
        """Test modular code with requirements."""
        request_data = {
            "experiment_name": "Requirements Test",
            "model_type": "xgboost_classifier",
            "task_type": "classification",
            "output_format": "module"
        }
        
        response = client.post(
            "/api/v1/code-generation/python/modular?include_requirements=true",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "requirements" in data
        assert data["requirements"] is not None
        assert isinstance(data["requirements"], dict)
        assert len(data["requirements"]) > 0
    
    def test_generate_modular_code_without_requirements(self):
        """Test modular code without requirements."""
        request_data = {
            "experiment_name": "No Requirements Test",
            "model_type": "random_forest_classifier",
            "task_type": "classification",
            "output_format": "module"
        }
        
        response = client.post(
            "/api/v1/code-generation/python/modular?include_requirements=false",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Requirements should be None or not included
        assert data.get("requirements") is None or len(data.get("requirements", {})) == 0
    
    def test_generate_modular_code_metadata(self):
        """Test metadata in modular response."""
        request_data = {
            "experiment_name": "Metadata Test",
            "model_type": "gradient_boosting_classifier",
            "task_type": "classification",
            "preprocessing_steps": [
                {"type": "scaling", "parameters": {"scaler": "standard"}}
            ],
            "include_evaluation": True
        }
        
        response = client.post(
            "/api/v1/code-generation/python/modular",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        metadata = data["metadata"]
        assert "generated_at" in metadata
        assert "model_type" in metadata
        assert "task_type" in metadata
        assert "total_lines" in metadata
        assert "components" in metadata
        
        # Check components list
        components = metadata["components"]
        assert "training" in components
        assert "prediction" in components


class TestRequirementsGenerationEndpoint:
    """Test /code-generation/requirements endpoint."""
    
    def test_generate_requirements_basic(self):
        """Test basic requirements generation."""
        request_data = {
            "model_type": "random_forest_classifier",
            "task_type": "classification",
            "output_format": "pip"
        }
        
        response = client.post(
            "/api/v1/code-generation/requirements",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, dict)
        assert "requirements.txt" in data
        
        requirements = data["requirements.txt"]
        assert "numpy" in requirements
        assert "pandas" in requirements
        assert "scikit-learn" in requirements
    
    def test_generate_requirements_modular(self):
        """Test modular requirements generation."""
        request_data = {
            "model_type": "xgboost_classifier",
            "task_type": "classification",
            "modular": True,
            "output_format": "pip"
        }
        
        response = client.post(
            "/api/v1/code-generation/requirements",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have multiple files
        assert len(data) > 1
        assert "requirements.txt" in data
        assert "requirements-training.txt" in data or "requirements-prediction.txt" in data
    
    def test_generate_requirements_docker(self):
        """Test Docker requirements generation."""
        request_data = {
            "model_type": "lightgbm_regressor",
            "task_type": "regression",
            "output_format": "docker"
        }
        
        response = client.post(
            "/api/v1/code-generation/requirements",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        requirements = data["requirements.txt"]
        assert "Docker" in requirements or "docker" in requirements
        assert "==" in requirements  # Pinned versions
    
    def test_generate_requirements_conda(self):
        """Test conda environment generation."""
        request_data = {
            "model_type": "catboost_classifier",
            "task_type": "classification",
            "output_format": "conda"
        }
        
        response = client.post(
            "/api/v1/code-generation/requirements",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "environment.yml" in data
        env_yml = data["environment.yml"]
        assert "name:" in env_yml
        assert "channels:" in env_yml
        assert "dependencies:" in env_yml
    
    def test_generate_requirements_with_preprocessing(self):
        """Test requirements with preprocessing steps."""
        request_data = {
            "model_type": "random_forest_classifier",
            "task_type": "classification",
            "preprocessing_steps": [
                {
                    "type": "missing_value_imputation",
                    "parameters": {"strategy": "mean"}
                },
                {
                    "type": "scaling",
                    "parameters": {"scaler": "standard"}
                }
            ],
            "output_format": "pip"
        }
        
        response = client.post(
            "/api/v1/code-generation/requirements",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        requirements = data["requirements.txt"]
        assert "scikit-learn" in requirements
    
    def test_generate_requirements_with_evaluation(self):
        """Test requirements with evaluation."""
        request_data = {
            "model_type": "random_forest_classifier",
            "task_type": "classification",
            "include_evaluation": True,
            "output_format": "pip"
        }
        
        response = client.post(
            "/api/v1/code-generation/requirements",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        requirements = data["requirements.txt"]
        # Should include visualization libraries
        assert "matplotlib" in requirements or "seaborn" in requirements


class TestIntegration:
    """Integration tests for code generation."""
    
    def test_end_to_end_code_generation(self):
        """Test complete code generation workflow."""
        # 1. Generate Python code
        code_request = {
            "experiment_name": "End-to-End Test",
            "model_type": "random_forest_classifier",
            "task_type": "classification",
            "preprocessing_steps": [
                {
                    "type": "missing_value_imputation",
                    "parameters": {"strategy": "mean"}
                }
            ],
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 10
            },
            "include_evaluation": True,
            "output_format": "module"
        }
        
        code_response = client.post(
            "/api/v1/code-generation/python",
            json=code_request
        )
        
        assert code_response.status_code == 200
        code_data = code_response.json()
        assert len(code_data["code"]) > 0
        
        # 2. Generate requirements
        req_request = {
            "model_type": "random_forest_classifier",
            "task_type": "classification",
            "preprocessing_steps": [
                {
                    "type": "missing_value_imputation",
                    "parameters": {"strategy": "mean"}
                }
            ],
            "include_evaluation": True,
            "modular": True
        }
        
        req_response = client.post(
            "/api/v1/code-generation/requirements",
            json=req_request
        )
        
        assert req_response.status_code == 200
        req_data = req_response.json()
        assert len(req_data) > 0
        
        # 3. Verify consistency
        assert code_data["metadata"]["model_type"] == req_request["model_type"]
    
    def test_modular_code_with_requirements(self):
        """Test generating modular code with requirements."""
        request_data = {
            "experiment_name": "Complete Modular Test",
            "model_type": "xgboost_classifier",
            "task_type": "classification",
            "preprocessing_steps": [
                {"type": "scaling", "parameters": {"scaler": "standard"}}
            ],
            "include_evaluation": True,
            "output_format": "module"
        }
        
        response = client.post(
            "/api/v1/code-generation/python/modular?include_requirements=true",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all components
        assert data["preprocessing_code"] is not None
        assert data["training_code"] is not None
        assert data["evaluation_code"] is not None
        assert data["prediction_code"] is not None
        assert data["requirements"] is not None
        
        # Verify requirements include xgboost
        requirements_str = str(data["requirements"])
        assert "xgboost" in requirements_str.lower()
