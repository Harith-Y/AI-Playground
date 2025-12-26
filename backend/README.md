# Backend API - AI-Playground

FastAPI backend for the AI-Playground ML platform. Provides RESTful APIs for dataset management, preprocessing pipeline configuration, feature engineering, model training, and code generation.

## 🏗️ Architecture

```
backend/
├── app/
│   ├── main.py                # FastAPI application entry point
│   ├── celery_app.py          # Celery task queue configuration
│   ├── api/                   # API routes and endpoints
│   ├── core/                  # Core configuration and utilities
│   ├── db/                    # Database session and connection
│   ├── models/                # SQLAlchemy ORM models
│   ├── schemas/               # Pydantic validation schemas
│   ├── services/              # Business logic layer
│   ├── ml_engine/             # Machine learning modules
│   ├── tasks/                 # Celery async tasks
│   └── utils/                 # Utility functions
├── tests/                     # Unit and integration tests
├── alembic/                   # Database migrations
├── requirements.txt           # Python dependencies
└── .env                      # Environment variables
```

## ✨ Features

### ✅ Implemented

#### Dataset Management
- **Upload datasets** - CSV, Excel (XLSX, XLS), JSON formats
- **Preview data** - Get first N rows of dataset
- **Statistics** - Compute comprehensive dataset statistics
- **Metadata extraction** - Automatic dtype detection, missing values, duplicates
- **File storage** - Organized by user/dataset hierarchy
- **CRUD operations** - Full create, read, update, delete functionality

#### Preprocessing Pipeline
- **Step management** - Create, read, update, delete preprocessing steps
- **Step reordering** - Change execution order of pipeline steps
- **Configuration persistence** - Store step parameters in PostgreSQL (JSONB)
- **Authorization** - User ownership verification for all operations
- **Modular architecture** - Each step is independently configurable

**Supported Step Types:**
- Missing Value Imputation (mean, median, mode, constant)
- Outlier Detection (IQR, Z-score)
- Scaling (standard, minmax, robust)
- Encoding (onehot, label, ordinal)
- Feature Selection (variance, correlation, mutual information)

#### ML Engine Modules
- **EDA Statistics** - 10+ analysis methods for exploratory data analysis
- **Correlation Analysis** - Pearson, Spearman, Kendall with heatmap data, clustering, multicollinearity detection
- **Variance Threshold Selector** - Remove low-variance features
- **Correlation Selector** - Select features based on correlation with target
- **Mutual Information Selector** - MI-based feature selection
- **Mode Imputer** - Fill categorical missing values
- **Mean/Median Imputer** - Fill numeric missing values
- **IQR/Z-Score Outlier Detectors** - Outlier detection and removal
- **Scalers** - Standard, MinMax, Robust normalization
- **Encoders** - OneHot, Label, Ordinal encoding

#### Infrastructure
- **NeonDB Integration** - Optimized connection pooling for serverless PostgreSQL
- **Redis Caching** - Fast data access and session management
- **Celery Tasks** - Async processing for long-running operations
- **Comprehensive Tests** - 80+ unit tests with pytest
- **OpenAPI Documentation** - Auto-generated Swagger UI and ReDoc
- **Error Handling** - Structured error responses with validation
- **Logging** - Structured logging with configurable levels

### 🚧 Coming Soon

- Preprocessing apply endpoint (execute pipeline on dataset)
- Model training endpoints (train, status, results)
- Hyperparameter tuning endpoints (optimize, status, results)
- Code generation endpoints (Python, Jupyter, FastAPI export)
- WebSocket support for real-time updates
- User authentication and JWT tokens

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 15+ (or NeonDB account)
- Redis 7+

### Installation

```powershell
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the backend directory:

```env
# Database (NeonDB recommended)
DATABASE_URL=postgresql://user:password@host.neon.tech/aiplayground?sslmode=require

# Redis
REDIS_URL=redis://localhost:6379/0

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Security
SECRET_KEY=your-super-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# File Storage
UPLOAD_DIR=./uploads
MAX_UPLOAD_SIZE=104857600  # 100MB

# Environment
ENVIRONMENT=development
DEBUG=True

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs
```

### Database Setup

```powershell
# Run migrations
alembic upgrade head

# Create a new migration (if needed)
alembic revision --autogenerate -m "description"
```

### Running the Application

```powershell
# Start FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# In a separate terminal, start Celery worker
celery -A app.celery_app worker --loglevel=info

# In a separate terminal, start Celery beat (for scheduled tasks)
celery -A app.celery_app beat --loglevel=info
```

### Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 📝 API Endpoints

### Dataset Management

#### Upload Dataset
```http
POST /api/v1/datasets/upload
Content-Type: multipart/form-data

file: <dataset-file.csv>
```

**Response:**
```json
{
  "id": "uuid",
  "user_id": "uuid",
  "name": "dataset_name",
  "file_path": "/uploads/user_id/dataset_id/file.csv",
  "rows": 1000,
  "cols": 10,
  "dtypes": {"col1": "int64", "col2": "float64"},
  "missing_values": {"col1": 5, "col2": 10},
  "uploaded_at": "2025-01-15T10:30:00Z"
}
```

#### List Datasets
```http
GET /api/v1/datasets
```

#### Get Dataset
```http
GET /api/v1/datasets/{dataset_id}
```

#### Get Dataset Statistics
```http
GET /api/v1/datasets/{dataset_id}/stats
```

**Response:**
```json
{
  "stats": {
    "rowCount": 1000,
    "columnCount": 10,
    "numericColumns": 7,
    "categoricalColumns": 3,
    "missingValues": 15,
    "duplicateRows": 2,
    "memoryUsage": 80000
  },
  "columns": [
    {
      "name": "age",
      "dataType": "int64",
      "uniqueCount": 50,
      "nullCount": 5,
      "min": 18.0,
      "max": 65.0,
      "mean": 35.2,
      "median": 34.0
    }
  ]
}
```

#### Get Dataset Preview
```http
GET /api/v1/datasets/{dataset_id}/preview?rows=10
```

#### Delete Dataset
```http
DELETE /api/v1/datasets/{dataset_id}
```

### Preprocessing Pipeline

#### Create Preprocessing Step
```http
POST /api/v1/preprocessing/
Content-Type: application/json

{
  "dataset_id": "uuid",
  "step_type": "missing_value_imputation",
  "parameters": {"strategy": "mean"},
  "column_name": "age",
  "order": 0
}
```

**Step Types:**
- `missing_value_imputation` - Fill missing values
- `outlier_detection` - Detect/remove outliers
- `scaling` - Scale numeric features
- `encoding` - Encode categorical features
- `feature_selection` - Select important features
- `transformation` - Transform features (log, sqrt, etc.)

**Response:**
```json
{
  "id": "uuid",
  "dataset_id": "uuid",
  "step_type": "missing_value_imputation",
  "parameters": {"strategy": "mean"},
  "column_name": "age",
  "order": 0
}
```

#### List Preprocessing Steps
```http
GET /api/v1/preprocessing/?dataset_id=uuid
```

#### Get Preprocessing Step
```http
GET /api/v1/preprocessing/{step_id}
```

#### Update Preprocessing Step
```http
PUT /api/v1/preprocessing/{step_id}
Content-Type: application/json

{
  "parameters": {"strategy": "median"}
}
```

#### Delete Preprocessing Step
```http
DELETE /api/v1/preprocessing/{step_id}
```

#### Reorder Preprocessing Steps
```http
POST /api/v1/preprocessing/reorder?dataset_id=uuid&step_ids=id1&step_ids=id2&step_ids=id3
```

## 🗄️ Database Models

### Dataset
```python
class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID, primary_key=True)
    user_id = Column(UUID, ForeignKey("users.id"))
    name = Column(String)
    file_path = Column(String)
    rows = Column(Integer)
    cols = Column(Integer)
    dtypes = Column(JSONB)  # {"col1": "int64", ...}
    missing_values = Column(JSONB)  # {"col1": 5, ...}
    uploaded_at = Column(DateTime)
```

### PreprocessingStep
```python
class PreprocessingStep(Base):
    __tablename__ = "preprocessing_steps"

    id = Column(UUID, primary_key=True)
    dataset_id = Column(UUID, ForeignKey("datasets.id"))
    step_type = Column(String)  # "missing_value_imputation", "scaling", etc.
    parameters = Column(JSONB)  # {"strategy": "mean", ...}
    column_name = Column(String)  # Column to apply step to (null for all)
    order = Column(Integer)  # Execution order
```

### User
```python
class User(Base):
    __tablename__ = "users"

    id = Column(UUID, primary_key=True)
    email = Column(String, unique=True)
    created_at = Column(DateTime)
```

## 🧪 Testing

### Run All Tests
```powershell
pytest
```

### Run with Coverage
```powershell
pytest --cov=app tests/
```

### Run Specific Test File
```powershell
pytest tests/test_preprocessing_endpoints.py -v
```

### Run with HTML Coverage Report
```powershell
pytest --cov=app --cov-report=html tests/
# Open htmlcov/index.html in browser
```

### Test Coverage

- **Dataset Endpoints**: 100% coverage (test_datasets.py)
- **Preprocessing Endpoints**: 80+ tests (test_preprocessing_endpoints.py)
- **ML Engine Modules**:
  - Variance Threshold: 100% coverage
  - Correlation Selector: 100% coverage
  - Mutual Information Selector: 100% coverage
  - Mode Imputer: 100% coverage
  - Mean/Median Imputer: 100% coverage
  - IQR Outlier Detector: 100% coverage
  - Z-Score Outlier Detector: 100% coverage
  - Scalers: 100% coverage
  - Encoders: 100% coverage
  - EDA Statistics: 100% coverage
  - Correlation Analysis: 100% coverage

## 🛠️ ML Engine

### Preprocessing Modules

Located in `app/ml_engine/preprocessing/`:

- **base.py** (186 lines) - Base transformer class with sklearn interface
- **cleaner.py** (458 lines) - IQR and Z-score outlier detectors
- **encoder.py** (449 lines) - OneHot, Label, Ordinal encoders
- **imputer.py** (300 lines) - Mean, Median, Mode imputers
- **scaler.py** (464 lines) - Standard, MinMax, Robust scalers

**Example Usage:**
```python
from app.ml_engine.preprocessing.imputer import MeanImputer
from app.ml_engine.preprocessing.scaler import StandardScaler
import pandas as pd

# Load data
df = pd.DataFrame({"age": [25, None, 35, 40], "salary": [50000, 60000, None, 80000]})

# Impute missing values
imputer = MeanImputer()
df_imputed = imputer.fit_transform(df)

# Scale features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_imputed)
```

### Feature Selection Modules

Located in `app/ml_engine/feature_selection/`:

- **variance_threshold.py** (319 lines) - Variance-based feature selection
- **correlation_selector.py** (386 lines) - Correlation-based feature selection
- **mutual_information_selector.py** (390 lines) - Mutual information-based selection

**Example Usage:**
```python
from app.ml_engine.feature_selection.variance_threshold import VarianceThresholdSelector
import pandas as pd

df = pd.DataFrame({
    "constant": [1, 1, 1, 1],
    "low_var": [1, 1, 1, 2],
    "high_var": [1, 5, 10, 15]
})

selector = VarianceThresholdSelector(threshold=0.1)
selected_features = selector.fit_transform(df)
# Removes "constant" and "low_var", keeps "high_var"
```

### EDA Modules

- **eda_statistics.py** (514 lines) - Comprehensive EDA analysis
- **correlation_analysis.py** (535 lines) - Correlation matrix analysis

**Example Usage:**
```python
from app.ml_engine.eda_statistics import EDAStatistics
from app.ml_engine.correlation_analysis import CorrelationMatrix
import pandas as pd

df = pd.DataFrame({"age": [25, 30, 35], "salary": [50000, 60000, 70000]})

# EDA Statistics
eda = EDAStatistics(df)
summary = eda.quick_summary()
print(summary)

# Correlation Analysis
corr_matrix = CorrelationMatrix(df)
corr = corr_matrix.compute_correlation(method='pearson')
heatmap_data = corr_matrix.get_heatmap_data()
```

## 🔧 Configuration

### NeonDB Optimization

The database session is optimized for NeonDB serverless PostgreSQL:

```python
# app/db/session.py
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,           # Test connections before use
    pool_size=5,                  # Smaller pool for serverless
    max_overflow=10,              # Reduced overflow
    pool_recycle=300,             # Recycle after 5 minutes
    connect_args={"connect_timeout": 10}
)
```

### Logging Configuration

```python
# app/utils/logger.py
LOG_LEVEL=INFO        # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_DIR=./logs        # Log file directory
```

### File Upload Limits

```python
MAX_UPLOAD_SIZE=104857600  # 100MB
UPLOAD_DIR=./uploads       # Upload directory
```

## 📦 Dependencies

**Core:**
- FastAPI 0.126.0 - Web framework
- SQLAlchemy 2.0.45 - ORM
- Alembic 1.17.2 - Database migrations
- Pydantic 2.x - Data validation
- Uvicorn 0.34.2 - ASGI server

**ML & Data:**
- scikit-learn 1.8.0
- pandas 2.3.3
- numpy 2.4.0
- scipy 1.16.3

**Task Queue:**
- Celery 5.6.0
- Redis 5.1.2

**Testing:**
- pytest 8.3.4
- pytest-cov 7.0.0
- httpx 0.30.0

## 🐛 Troubleshooting

### Database Connection Issues

**Problem:** `OperationalError: could not connect to server`

**Solution:**
1. Verify DATABASE_URL in .env
2. Check NeonDB project is active
3. Ensure `?sslmode=require` is in connection string
4. Test connection: `psql $DATABASE_URL`

### Redis Connection Failed

**Problem:** `ConnectionError: Error 10061 connecting to localhost:6379`

**Solution:**
1. Ensure Redis server is running: `redis-server`
2. Check REDIS_URL in .env
3. Windows: Use WSL or Redis for Windows

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'app'`

**Solution:**
```powershell
# Ensure you're in backend directory
cd backend

# Activate virtual environment
.\venv\Scripts\Activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Alembic Migration Errors

**Problem:** `Target database is not up to date`

**Solution:**
```powershell
# Check current revision
alembic current

# Upgrade to latest
alembic upgrade head

# Rollback one revision
alembic downgrade -1
```

## 📚 Additional Resources

- **[Main README](../README.md)** - Project overview
- **[SETUP.md](../SETUP.md)** - Detailed setup guide
- **[ML Engine Docs](app/ml_engine/README.md)** - ML modules documentation
- **[Store README](../frontend/src/store/README.md)** - Redux state management
- **[API Documentation](http://localhost:8000/docs)** - Interactive Swagger UI

## 🤝 Contributing

1. Write tests for new features
2. Follow PEP 8 style guide
3. Use Black for code formatting
4. Run tests before committing: `pytest`
5. Update documentation

## 📄 License

This project is for educational purposes.
