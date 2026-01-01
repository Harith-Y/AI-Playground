"""
Fixtures for ML pipeline integration tests

Provides realistic datasets and configurations for testing the complete ML workflow
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any


@pytest.fixture
def classification_dataset(tmp_path: Path) -> Path:
    """
    Create a classification dataset (Iris-like) for testing
    """
    np.random.seed(42)
    n_samples = 500

    # Generate features
    data = {
        'sepal_length': np.random.normal(5.8, 0.8, n_samples),
        'sepal_width': np.random.normal(3.0, 0.4, n_samples),
        'petal_length': np.random.normal(3.7, 1.8, n_samples),
        'petal_width': np.random.normal(1.2, 0.8, n_samples),
        'species': np.random.choice(['setosa', 'versicolor', 'virginica'], n_samples),
    }

    df = pd.DataFrame(data)

    # Add some missing values
    df.loc[np.random.choice(df.index, 20, replace=False), 'sepal_length'] = np.nan
    df.loc[np.random.choice(df.index, 15, replace=False), 'petal_width'] = np.nan

    # Add some outliers
    df.loc[np.random.choice(df.index, 5, replace=False), 'sepal_length'] = 15.0

    csv_file = tmp_path / "classification_dataset.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def regression_dataset(tmp_path: Path) -> Path:
    """
    Create a regression dataset (house prices) for testing
    """
    np.random.seed(42)
    n_samples = 400

    # Generate features
    sqft = np.random.normal(2000, 500, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    age = np.random.randint(0, 50, n_samples)
    location = np.random.choice(['urban', 'suburban', 'rural'], n_samples)

    # Target with some relationship to features
    price = (
        sqft * 150
        + bedrooms * 20000
        + bathrooms * 15000
        - age * 1000
        + np.random.normal(0, 50000, n_samples)
    )

    data = {
        'sqft': sqft,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age,
        'location': location,
        'price': price,
    }

    df = pd.DataFrame(data)

    # Add missing values
    df.loc[np.random.choice(df.index, 25, replace=False), 'sqft'] = np.nan
    df.loc[np.random.choice(df.index, 10, replace=False), 'age'] = np.nan

    csv_file = tmp_path / "regression_dataset.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def imbalanced_dataset(tmp_path: Path) -> Path:
    """
    Create an imbalanced classification dataset for testing
    """
    np.random.seed(42)

    # Majority class: 400 samples
    majority_data = {
        'feature1': np.random.normal(0, 1, 400),
        'feature2': np.random.normal(0, 1, 400),
        'feature3': np.random.normal(0, 1, 400),
        'target': ['normal'] * 400
    }

    # Minority class: 50 samples
    minority_data = {
        'feature1': np.random.normal(2, 1, 50),
        'feature2': np.random.normal(2, 1, 50),
        'feature3': np.random.normal(2, 1, 50),
        'target': ['anomaly'] * 50
    }

    df_majority = pd.DataFrame(majority_data)
    df_minority = pd.DataFrame(minority_data)
    df = pd.concat([df_majority, df_minority], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    csv_file = tmp_path / "imbalanced_dataset.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def multiclass_dataset(tmp_path: Path) -> Path:
    """
    Create a multiclass classification dataset
    """
    np.random.seed(42)
    n_samples_per_class = 100

    data_list = []
    for i, class_name in enumerate(['class_a', 'class_b', 'class_c', 'class_d', 'class_e']):
        class_data = {
            'x1': np.random.normal(i * 2, 0.8, n_samples_per_class),
            'x2': np.random.normal(i * 1.5, 0.8, n_samples_per_class),
            'x3': np.random.normal(i, 0.5, n_samples_per_class),
            'x4': np.random.choice(['type_1', 'type_2', 'type_3'], n_samples_per_class),
            'target': [class_name] * n_samples_per_class
        }
        data_list.append(pd.DataFrame(class_data))

    df = pd.concat(data_list, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    csv_file = tmp_path / "multiclass_dataset.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def time_series_dataset(tmp_path: Path) -> Path:
    """
    Create a time series dataset for testing
    """
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=365, freq='D')

    # Trend + seasonality + noise
    trend = np.linspace(100, 150, 365)
    seasonality = 10 * np.sin(np.linspace(0, 4 * np.pi, 365))
    noise = np.random.normal(0, 5, 365)

    data = {
        'date': dates,
        'value': trend + seasonality + noise,
        'feature1': np.random.normal(0, 1, 365),
        'feature2': np.random.choice(['weekend', 'weekday'], 365),
    }

    df = pd.DataFrame(data)
    csv_file = tmp_path / "time_series_dataset.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def preprocessing_config() -> Dict[str, Any]:
    """
    Standard preprocessing configuration for tests
    """
    return {
        "steps": [
            {
                "type": "handle_missing_values",
                "config": {
                    "strategy": "mean",
                    "columns": None  # All numeric columns
                }
            },
            {
                "type": "remove_outliers",
                "config": {
                    "method": "iqr",
                    "threshold": 1.5,
                    "columns": None
                }
            },
            {
                "type": "encode_categorical",
                "config": {
                    "method": "onehot",
                    "columns": None  # All categorical columns
                }
            },
            {
                "type": "scale_features",
                "config": {
                    "method": "standard",
                    "columns": None
                }
            }
        ]
    }


@pytest.fixture
def model_training_config_classification() -> Dict[str, Any]:
    """
    Model training configuration for classification
    """
    return {
        "model_type": "random_forest",
        "task_type": "classification",
        "target_column": "species",
        "test_size": 0.2,
        "random_state": 42,
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        }
    }


@pytest.fixture
def model_training_config_regression() -> Dict[str, Any]:
    """
    Model training configuration for regression
    """
    return {
        "model_type": "random_forest",
        "task_type": "regression",
        "target_column": "price",
        "test_size": 0.2,
        "random_state": 42,
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 5,
        }
    }


@pytest.fixture
def tuning_config() -> Dict[str, Any]:
    """
    Hyperparameter tuning configuration
    """
    return {
        "search_method": "random",
        "n_iterations": 10,
        "cv_folds": 3,
        "scoring": "accuracy",
        "param_space": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    }


@pytest.fixture
def feature_engineering_config() -> Dict[str, Any]:
    """
    Feature engineering configuration
    """
    return {
        "steps": [
            {
                "type": "polynomial_features",
                "config": {
                    "degree": 2,
                    "include_bias": False,
                    "interaction_only": False
                }
            },
            {
                "type": "feature_selection",
                "config": {
                    "method": "variance_threshold",
                    "threshold": 0.01
                }
            }
        ]
    }


@pytest.fixture
def pipeline_test_user(db) -> Dict[str, Any]:
    """
    Create a test user for pipeline tests
    """
    from app.models.user import User
    from app.core.security import get_password_hash
    import uuid

    user = User(
        id=uuid.uuid4(),
        email="pipeline_test@example.com",
        password_hash=get_password_hash("TestPassword123"),
        is_active=True,
        is_admin=False
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "id": str(user.id),
        "email": user.email,
        "password": "TestPassword123"
    }


@pytest.fixture
def auth_headers(client, pipeline_test_user) -> Dict[str, str]:
    """
    Get authentication headers for API requests
    """
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": pipeline_test_user["email"],
            "password": pipeline_test_user["password"]
        }
    )

    assert response.status_code == 200
    token_data = response.json()

    return {
        "Authorization": f"Bearer {token_data['access_token']}"
    }
