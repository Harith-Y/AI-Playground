"""
Test configuration and fixtures for pytest.
"""

import os
import pytest
import tempfile
from pathlib import Path
from typing import Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.main import app
from app.db.base import Base
from app.db.session import get_db
from app.core.config import settings


# Use test database - defaults to NeonDB from settings, can be overridden
test_db_url = os.getenv("TEST_DATABASE_URL", settings.DATABASE_URL)

# Create engine with appropriate settings
engine = create_engine(
    test_db_url,
    pool_pre_ping=True,
    pool_size=2,  # Small pool for testing
    max_overflow=5
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db() -> Generator[Session, None, None]:
    """
    Create a fresh database for each test.
    """
    # Create tables
    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        # Drop tables after test
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db: Session) -> Generator[TestClient, None, None]:
    """
    Create a test client with database dependency override.
    """
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def temp_upload_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for file uploads during testing.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Override the upload directory setting
        original_upload_dir = settings.UPLOAD_DIR
        settings.UPLOAD_DIR = str(temp_path)

        yield temp_path

        # Restore original setting
        settings.UPLOAD_DIR = original_upload_dir


@pytest.fixture
def sample_csv_file(tmp_path: Path) -> Path:
    """
    Create a sample CSV file for testing.
    """
    csv_content = """name,age,city,salary
John Doe,30,New York,75000
Jane Smith,25,San Francisco,85000
Bob Johnson,35,Chicago,65000
Alice Williams,28,Boston,72000
Charlie Brown,32,Seattle,78000
"""

    csv_file = tmp_path / "test_dataset.csv"
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def sample_excel_file(tmp_path: Path) -> Path:
    """
    Create a sample Excel file for testing.
    """
    import pandas as pd

    data = {
        'product': ['A', 'B', 'C', 'D', 'E'],
        'price': [10.5, 20.0, 15.75, 8.99, 12.50],
        'quantity': [100, 50, 75, 200, 150],
        'category': ['Electronics', 'Clothing', 'Electronics', 'Food', 'Clothing']
    }

    df = pd.DataFrame(data)
    excel_file = tmp_path / "test_dataset.xlsx"
    df.to_excel(excel_file, index=False)
    return excel_file


@pytest.fixture
def sample_json_file(tmp_path: Path) -> Path:
    """
    Create a sample JSON file for testing.
    """
    import json

    data = [
        {"id": 1, "name": "Item 1", "value": 100, "active": True},
        {"id": 2, "name": "Item 2", "value": 200, "active": False},
        {"id": 3, "name": "Item 3", "value": 150, "active": True},
        {"id": 4, "name": "Item 4", "value": 300, "active": True},
    ]

    json_file = tmp_path / "test_dataset.json"
    json_file.write_text(json.dumps(data))
    return json_file


@pytest.fixture
def large_csv_file(tmp_path: Path) -> Path:
    """
    Create a large CSV file for testing file size limits.
    """
    import pandas as pd
    import numpy as np

    # Create a large dataset (simulate ~150MB file)
    rows = 1000000
    data = {
        'col1': np.random.randint(0, 1000, rows),
        'col2': np.random.rand(rows),
        'col3': [f'text_{i}' for i in range(rows)],
    }

    df = pd.DataFrame(data)
    large_file = tmp_path / "large_dataset.csv"
    df.to_csv(large_file, index=False)
    return large_file


@pytest.fixture
def invalid_file(tmp_path: Path) -> Path:
    """
    Create an invalid file (not CSV/Excel/JSON).
    """
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("This is not a valid dataset file")
    return invalid_file


# ============================================================================
# ML Pipeline Integration Test Fixtures
# ============================================================================

@pytest.fixture
def classification_dataset(tmp_path: Path) -> Path:
    """
    Create a classification dataset for ML pipeline testing.
    """
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=500,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42,
        flip_y=0.1
    )
    
    df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4'])
    df['target'] = y
    
    csv_file = tmp_path / "classification_data.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def regression_dataset(tmp_path: Path) -> Path:
    """
    Create a regression dataset for ML pipeline testing.
    """
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_regression
    
    X, y = make_regression(
        n_samples=400,
        n_features=5,
        n_informative=4,
        noise=10,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])
    df['target'] = y
    
    csv_file = tmp_path / "regression_data.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def auth_headers(client: TestClient, db: Session) -> dict:
    """
    Create authentication headers for API requests.
    """
    from app.models.user import User
    from app.core.security import get_password_hash
    
    # Create test user
    test_user = User(
        email="test@example.com",
        hashed_password=get_password_hash("testpassword123"),
        full_name="Test User",
        is_active=True
    )
    db.add(test_user)
    db.commit()
    db.refresh(test_user)
    
    # Login to get token
    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": "test@example.com",
            "password": "testpassword123"
        }
    )
    
    if response.status_code == 200:
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    else:
        # If login endpoint not available, return empty headers
        return {}


@pytest.fixture
def sample_preprocessed_data(tmp_path: Path) -> Path:
    """
    Create a preprocessed dataset for testing model training.
    """
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.randn(300),
        'feature2': np.random.randn(300),
        'feature3': np.random.randn(300),
        'feature4': np.random.randn(300),
        'feature5': np.random.randn(300),
        'target': np.random.randint(0, 2, 300)
    })
    
    csv_file = tmp_path / "preprocessed_data.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def multiclass_dataset(tmp_path: Path) -> Path:
    """
    Create a multiclass classification dataset.
    """
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(6)])
    df['target'] = y
    
    csv_file = tmp_path / "multiclass_data.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def imbalanced_dataset(tmp_path: Path) -> Path:
    """
    Create an imbalanced classification dataset.
    """
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    
    # Create imbalanced dataset (90% class 0, 10% class 1)
    n_majority = 450
    n_minority = 50
    
    X_majority = np.random.randn(n_majority, 5)
    y_majority = np.zeros(n_majority)
    
    X_minority = np.random.randn(n_minority, 5) + 2
    y_minority = np.ones(n_minority)
    
    X = np.vstack([X_majority, X_minority])
    y = np.concatenate([y_majority, y_minority])
    
    df = pd.DataFrame(X, columns=[f'var_{i}' for i in range(5)])
    df['target'] = y
    
    csv_file = tmp_path / "imbalanced_data.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def time_series_dataset(tmp_path: Path) -> Path:
    """
    Create a time series dataset for testing.
    """
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    
    # Create trend + seasonality + noise
    trend = np.linspace(100, 150, 365)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(365) / 365)
    noise = np.random.randn(365) * 5
    
    df = pd.DataFrame({
        'date': dates,
        'value': trend + seasonality + noise,
        'feature1': np.random.randn(365),
        'feature2': np.random.randn(365)
    })
    
    csv_file = tmp_path / "time_series_data.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def dataset_with_missing_values(tmp_path: Path) -> Path:
    """
    Create a dataset with various types of missing values.
    """
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
        'feature2': [np.nan, 20, 30, np.nan, 50, 60, 70, np.nan, 90, 100],
        'feature3': [1.5, 2.5, 3.5, 4.5, np.nan, 6.5, 7.5, 8.5, 9.5, np.nan],
        'category': ['A', np.nan, 'B', 'A', 'C', 'B', np.nan, 'A', 'B', 'C'],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    
    csv_file = tmp_path / "missing_values_data.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def dataset_with_outliers(tmp_path: Path) -> Path:
    """
    Create a dataset with outliers for testing outlier detection.
    """
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    normal_data = np.random.normal(50, 10, 95)
    outliers = np.array([200, -100, 300, 250, -50])
    
    df = pd.DataFrame({
        'feature1': np.concatenate([normal_data, outliers]),
        'feature2': np.random.normal(100, 20, 100),
        'feature3': np.random.normal(0, 1, 100),
        'target': np.random.randint(0, 2, 100)
    })
    
    csv_file = tmp_path / "outliers_data.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def categorical_dataset(tmp_path: Path) -> Path:
    """
    Create a dataset with categorical features for encoding testing.
    """
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    df = pd.DataFrame({
        'color': np.random.choice(['red', 'green', 'blue'], 100),
        'size': np.random.choice(['small', 'medium', 'large'], 100),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'numeric1': np.random.randn(100),
        'numeric2': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })
    
    csv_file = tmp_path / "categorical_data.csv"
    df.to_csv(csv_file, index=False)
    return csv_file
