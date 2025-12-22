# AI-Playground 🚀

A full-stack machine learning platform for automated ML workflows, from data ingestion to model deployment and code generation. Built with FastAPI, React, and modern ML libraries.

## 🎯 Overview

AI-Playground is an end-to-end ML platform that enables:
- **Dataset Management** - Upload, explore, and visualize datasets
- **Automated Preprocessing** - Smart data cleaning, encoding, and scaling
- **Feature Engineering** - Feature selection, importance analysis, correlation matrices
- **Model Training** - Support for regression, classification, and clustering
- **Hyperparameter Tuning** - Grid search, random search, and Bayesian optimization
- **Model Evaluation** - Comprehensive metrics and visualizations
- **Code Generation** - Export production-ready ML pipeline code
- **Experiment Tracking** - Track and compare model experiments

## 🏗️ Complete Project Structure

```
AI-Playground/
├── README.md                 # This file
├── SETUP.md                  # Detailed setup instructions
├── APPROACH.md               # Project approach & architecture
├── ML-PIPELINE.md            # ML pipeline documentation
├── MODELS.md                 # Available models & algorithms
├── docker-compose.yml        # Docker orchestration
│
├── backend/                  # FastAPI Backend (Python 3.11+)
│   ├── alembic.ini          # Database migration configuration
│   ├── celery_worker.py     # Celery worker entry point
│   ├── Dockerfile           # Backend container definition
│   ├── requirements.txt     # Python dependencies
│   ├── pyproject.toml       # Python project metadata
│   │
│   ├── alembic/             # Database Migrations
│   │   ├── env.py
│   │   ├── script.py.mako
│   │   └── versions/        # Migration files
│   │
│   ├── app/                 # Main Application
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI app entry point
│   │   ├── celery_app.py    # Celery configuration
│   │   │
│   │   ├── api/             # API Layer
│   │   │   └── v1/
│   │   │       ├── api.py   # API router aggregator
│   │   │       └── endpoints/
│   │   │           ├── datasets.py          # Dataset CRUD operations
│   │   │           ├── preprocessing.py     # Data preprocessing endpoints
│   │   │           ├── features.py          # Feature engineering endpoints
│   │   │           ├── models.py            # Model training endpoints
│   │   │           ├── tuning.py            # Hyperparameter tuning endpoints
│   │   │           ├── experiments.py       # Experiment tracking endpoints
│   │   │           └── code_generation.py   # Code generation endpoints
│   │   │
│   │   ├── core/            # Core Configuration
│   │   │   ├── config.py    # Application settings
│   │   │   ├── security.py  # Authentication & authorization
│   │   │   └── exceptions.py # Custom exceptions
│   │   │
│   │   ├── db/              # Database Layer
│   │   │   ├── base.py      # SQLAlchemy base
│   │   │   └── session.py   # Database session management
│   │   │
│   │   ├── models/          # Database Models (SQLAlchemy ORM)
│   │   │   ├── dataset.py              # Dataset model
│   │   │   ├── experiment.py           # Experiment model
│   │   │   ├── model_run.py            # Model run model
│   │   │   ├── preprocessing_step.py   # Preprocessing step model
│   │   │   └── user.py                 # User model
│   │   │
│   │   ├── schemas/         # Pydantic Schemas (API validation)
│   │   │   ├── dataset.py       # Dataset schemas
│   │   │   ├── experiment.py    # Experiment schemas
│   │   │   ├── model.py         # Model schemas
│   │   │   └── preprocessing.py # Preprocessing schemas
│   │   │
│   │   ├── services/        # Business Logic Layer
│   │   │   ├── dataset_service.py        # Dataset operations
│   │   │   ├── model_service.py          # Model operations
│   │   │   ├── preprocessing_service.py  # Preprocessing operations
│   │   │   └── storage_service.py        # File storage operations
│   │   │
│   │   ├── ml_engine/       # Machine Learning Pipeline
│   │   │   ├── models/      # ML Model Implementations
│   │   │   │   ├── regression.py      # Linear, Ridge, Lasso, ElasticNet, SVR, etc.
│   │   │   │   ├── classification.py  # Logistic, SVC, Random Forest, XGBoost, etc.
│   │   │   │   ├── clustering.py      # K-Means, DBSCAN, Hierarchical, etc.
│   │   │   │   └── registry.py        # Model registry & factory
│   │   │   │
│   │   │   ├── preprocessing/  # Data Preprocessing
│   │   │   │   ├── cleaner.py   # Missing value handling, outlier removal
│   │   │   │   ├── encoder.py   # Label encoding, one-hot encoding
│   │   │   │   ├── scaler.py    # Standard, MinMax, Robust scaling
│   │   │   │   └── pipeline.py  # Preprocessing pipeline orchestration
│   │   │   │
│   │   │   ├── training/    # Model Training
│   │   │   │   ├── trainer.py            # Training orchestration
│   │   │   │   └── cross_validation.py   # K-fold, stratified CV
│   │   │   │
│   │   │   ├── tuning/      # Hyperparameter Optimization
│   │   │   │   ├── grid_search.py     # Exhaustive grid search
│   │   │   │   ├── random_search.py   # Random search
│   │   │   │   └── bayesian.py        # Bayesian optimization
│   │   │   │
│   │   │   ├── evaluation/  # Model Evaluation
│   │   │   │   ├── metrics.py          # Accuracy, precision, recall, RMSE, etc.
│   │   │   │   └── visualizations.py   # Confusion matrix, ROC curves, etc.
│   │   │   │
│   │   │   └── code_generation/  # Code Export
│   │   │       ├── generator.py   # Code generation logic
│   │   │       └── templates.py   # Code templates
│   │   │
│   │   ├── tasks/           # Async Task Queue (Celery)
│   │   │   ├── training_tasks.py  # Async model training
│   │   │   └── tuning_tasks.py    # Async hyperparameter tuning
│   │   │
│   │   └── utils/           # Utility Functions
│   │       ├── file_handler.py  # File operations
│   │       └── logger.py        # Logging configuration
│   │
│   └── tests/               # Backend Tests
│
├── frontend/                # React Frontend (TypeScript)
│   ├── Dockerfile
│   ├── package.json         # Node dependencies
│   ├── tsconfig.json        # TypeScript configuration
│   ├── vite.config.ts       # Vite configuration
│   ├── index.html           # Entry HTML
│   │
│   ├── public/              # Static Assets
│   │
│   └── src/                 # Source Code
│       ├── main.tsx         # React entry point
│       ├── App.tsx          # Main App component
│       ├── App.css
│       ├── index.css
│       │
│       ├── components/      # React Components
│       │   ├── common/      # Shared Components
│       │   │   ├── Layout.tsx         # Main layout wrapper
│       │   │   ├── Header.tsx         # Top navigation
│       │   │   ├── Sidebar.tsx        # Side navigation
│       │   │   ├── Loading.tsx        # Loading spinner
│       │   │   └── ErrorBoundary.tsx  # Error handling
│       │   │
│       │   ├── dataset/     # Dataset Components
│       │   │   ├── DatasetUpload.tsx        # File upload
│       │   │   ├── DatasetPreview.tsx       # Data table preview
│       │   │   ├── DatasetStats.tsx         # Statistical summary
│       │   │   └── VisualizationGallery.tsx # Charts & plots
│       │   │
│       │   ├── preprocessing/  # Preprocessing Components
│       │   │   ├── StepBuilder.tsx   # Build preprocessing steps
│       │   │   ├── PreviewPanel.tsx  # Preview transformations
│       │   │   └── StepHistory.tsx   # Step history
│       │   │
│       │   ├── features/    # Feature Engineering Components
│       │   │   # (Feature selection, importance, correlation)
│       │   │
│       │   ├── modeling/    # Model Training Components
│       │   │   ├── ModelSelector.tsx        # Choose ML algorithm
│       │   │   ├── HyperparameterForm.tsx   # Set parameters
│       │   │   ├── TrainingProgress.tsx     # Training progress
│       │   │   └── MetricsDisplay.tsx       # Show metrics
│       │   │
│       │   ├── tuning/      # Hyperparameter Tuning Components
│       │   │   # (Method selector, parameter ranges, results)
│       │   │
│       │   ├── evaluation/  # Evaluation Components
│       │   │   # (Metrics, confusion matrix, residual plots)
│       │   │
│       │   └── code/        # Code Generation Components
│       │       # (Code preview, download)
│       │
│       ├── pages/           # Page Components (React Router)
│       │   ├── HomePage.tsx                 # Landing page
│       │   ├── DatasetUploadPage.tsx        # Dataset upload
│       │   ├── ExplorationPage.tsx          # Data exploration
│       │   ├── PreprocessingPage.tsx        # Data preprocessing
│       │   ├── FeatureEngineeringPage.tsx   # Feature engineering
│       │   ├── ModelingPage.tsx             # Model training
│       │   ├── TuningPage.tsx               # Hyperparameter tuning
│       │   └── CodeGenerationPage.tsx       # Code export
│       │
│       ├── store/           # Redux State Management
│       │   ├── index.ts     # Store configuration
│       │   └── slices/
│       │       ├── datasetSlice.ts         # Dataset state
│       │       ├── preprocessingSlice.ts   # Preprocessing state
│       │       ├── featureSlice.ts         # Feature state
│       │       ├── modelingSlice.ts        # Modeling state
│       │       ├── tuningSlice.ts          # Tuning state
│       │       └── evaluationSlice.ts      # Evaluation state
│       │
│       ├── services/        # API & WebSocket Services
│       │   ├── api.ts               # Axios configuration
│       │   ├── datasetService.ts    # Dataset API calls
│       │   ├── modelService.ts      # Model API calls
│       │   └── websocketService.ts  # WebSocket connection
│       │
│       ├── hooks/           # Custom React Hooks
│       │   ├── useDataset.ts    # Dataset operations
│       │   ├── useModel.ts      # Model operations
│       │   └── useWebSocket.ts  # WebSocket hook
│       │
│       ├── types/           # TypeScript Type Definitions
│       │   ├── index.ts      # Type exports
│       │   ├── dataset.ts    # Dataset types
│       │   ├── model.ts      # Model types
│       │   └── api.ts        # API types
│       │
│       ├── utils/           # Utility Functions
│       │   ├── constants.ts   # App constants
│       │   ├── helpers.ts     # Helper functions
│       │   └── validators.ts  # Validation functions
│       │
│       └── styles/          # Styling
│           ├── global.css    # Global styles
│           └── theme.ts      # MUI theme configuration
│
└── docker/                  # Docker Configuration
    ├── postgres/
    │   └── init.sql         # PostgreSQL initialization
    └── redis/
        └── redis.conf       # Redis configuration
```

## 🛠️ Technology Stack

### Backend
- **Framework**: FastAPI 0.126.0
- **ORM**: SQLAlchemy 2.0.45
- **Database**: PostgreSQL (via Neon or self-hosted)
- **Caching/Queue**: Redis 7.1.0
- **Task Queue**: Celery 5.6.0
- **Migrations**: Alembic 1.17.2
- **Authentication**: python-jose, passlib, bcrypt

### Machine Learning
- **Core**: scikit-learn 1.8.0, pandas 2.3.3, numpy 2.4.0
- **Boosting**: XGBoost 3.1.2, LightGBM 4.6.0, CatBoost 1.2.8
- **Visualization**: matplotlib 3.10.8, seaborn 0.13.2, plotly 6.5.0
- **Stats**: scipy 1.16.3

### Frontend
- **Framework**: React 19.2.0 with TypeScript 5.9.3
- **Build Tool**: Vite 7.2.4
- **UI Library**: Material-UI (@mui/material) 7.3.6
- **State Management**: Redux Toolkit 2.11.2
- **Routing**: React Router 7.11.0
- **HTTP Client**: Axios 1.13.2
- **Charts**: Plotly.js 3.3.1, Recharts 3.6.0
- **Forms**: React Hook Form 7.69.0 + Yup 1.7.1
- **Testing**: Vitest 4.0.16, Testing Library

### Development Tools
- **Code Quality**: Black, Flake8, mypy (Python) | ESLint (TypeScript)
- **Testing**: pytest, pytest-cov, pytest-asyncio
- **API Testing**: httpx
- **Containerization**: Docker, Docker Compose

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **Node.js 18+** and npm
- **PostgreSQL** (or [Neon](https://neon.tech) account - recommended)
- **Redis 7+**
- **Docker** (optional, for containerized setup)

### Option 1: Local Development

#### Backend Setup
```powershell
# Navigate to backend
cd backend

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your database URL and Redis URL

# Run migrations
alembic upgrade head

# Start FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# In a separate terminal, start Celery worker
celery -A app.celery_app worker --loglevel=info
```

#### Frontend Setup
```powershell
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Option 2: Docker Setup
```powershell
# Build and start all services
docker-compose build
docker-compose up -d

# View logs
docker-compose logs -f
```

Access the application:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📖 Documentation

- **[SETUP.md](SETUP.md)** - Comprehensive setup guide with database configuration, environment variables, and troubleshooting
- **[APPROACH.md](APPROACH.md)** - Project architecture, design decisions, and development approach
- **[ML-PIPELINE.md](ML-PIPELINE.md)** - Detailed ML pipeline documentation
- **[MODELS.md](MODELS.md)** - Available ML models and algorithms

## 🗄️ Database: Neon PostgreSQL (Recommended)

This project is optimized for **[Neon](https://neon.tech)** - serverless PostgreSQL:

✅ **Zero Setup** - No local installation, instant provisioning  
✅ **Autoscaling** - Automatically scales with your workload  
✅ **Database Branching** - Instant dev/staging/prod branches  
✅ **Generous Free Tier** - Perfect for development  
✅ **High Performance** - Optimized for ML workloads  
✅ **Simple Configuration** - Just copy connection string

**Alternative**: Local PostgreSQL 15+ is also supported (see [SETUP.md](SETUP.md))

## 🧪 Testing

### Backend Tests
```powershell
cd backend
pytest
pytest --cov=app tests/  # With coverage
```

### Frontend Tests
```powershell
cd frontend
npm test
npm run test:coverage
```

## 📝 API Endpoints

### Core Endpoints
- `POST /api/v1/datasets/upload` - Upload dataset
- `GET /api/v1/datasets/{id}` - Get dataset details
- `POST /api/v1/preprocessing/apply` - Apply preprocessing
- `POST /api/v1/features/select` - Feature selection
- `POST /api/v1/models/train` - Train model
- `POST /api/v1/tuning/optimize` - Hyperparameter tuning
- `GET /api/v1/experiments/{id}` - Get experiment results
- `POST /api/v1/code-generation/generate` - Generate code

Full API documentation available at `/docs` (Swagger UI) and `/redoc` (ReDoc)

## 🔧 Configuration

### Environment Variables

**Backend (.env)**
```env
DATABASE_URL=postgresql://user:password@localhost:5432/ai_playground
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
UPLOAD_DIR=./uploads
```

**Frontend (.env)**
```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
```

See [SETUP.md](SETUP.md) for complete configuration details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is for educational and development purposes.

## 🐛 Troubleshooting

See [SETUP.md](SETUP.md) for common issues and solutions.

## 📧 Support

For issues and questions, please open an issue on GitHub.