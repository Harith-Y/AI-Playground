# AI-Playground

## 📂 Structure Created

### **Backend** (FastAPI + Python)
- **api/v1/endpoints/** - Modular API endpoints (datasets, preprocessing, features, models, tuning, experiments, code generation)
- **core/** - Configuration, security, exceptions
- **models/** - Database models (SQLAlchemy)
- **schemas/** - Pydantic schemas for validation
- **services/** - Business logic layer
- **ml_engine/** - ML pipeline modules:
  - `models/` - Model registry (regression, classification, clustering)
  - `preprocessing/` - Data cleaning, encoding, scaling
  - `training/` - Model training, cross-validation
  - `tuning/` - Grid search, random search, Bayesian optimization
  - `evaluation/` - Metrics, visualizations
  - `code_generation/` - Code generation templates
- **db/** - Database session management
- **utils/** - File handling, logging
- **tasks/** - Celery async tasks

### **Frontend** (React + TypeScript)
- **components/** - Modular UI components organized by feature:
  - `common/` - Layout, Header, Sidebar, Loading, ErrorBoundary
  - `dataset/` - Upload, Preview, Stats, Visualizations
  - `preprocessing/` - Step Builder, Preview, History
  - `features/` - Selection, Importance, Correlation
  - `modeling/` - Model Selector, Hyperparameters, Training Progress, Metrics
  - `tuning/` - Method Selector, Parameter Ranges, Results
  - `evaluation/` - Metrics, Confusion Matrix, Residual Plots
  - `code/` - Code Preview, Download
- **pages/** - Page-level components for routing
- **store/slices/** - Redux state management (dataset, preprocessing, features, modeling, tuning, evaluation)
- **services/** - API calls, WebSocket connection
- **hooks/** - Custom React hooks
- **types/** - TypeScript type definitions
- **utils/** - Helper functions, validators, constants

### **Infrastructure**
- Docker configurations
- Docker Compose setup
- Environment templates

## 🚀 Quick Start

All initialization commands are in [SETUP.md](SETUP.md). You can:

**Option 1: Local Development**
```powershell
# Backend
cd backend
python -m venv venv
.\venv\Scripts\Activate
pip install fastapi uvicorn sqlalchemy psycopg2-binary alembic celery redis
pip install scikit-learn pandas numpy matplotlib seaborn plotly xgboost

# Frontend
cd frontend
npm create vite@latest . -- --template react-ts
npm install
npm install @reduxjs/toolkit react-redux react-router-dom axios @mui/material plotly.js
```

**Option 2: Docker**
```powershell
docker-compose build
docker-compose up -d
```

Check [SETUP.md](SETUP.md) for complete details including database setup (Neon recommended), environment configuration, testing setup, and troubleshooting!

## 🗄️ Database: Why Neon?

This project uses **[Neon](https://neon.tech)** (serverless PostgreSQL) as the recommended database:

✅ **Instant Setup** - No local installation, get started in 30 seconds
✅ **Autoscaling** - Scales automatically with your workload
✅ **Branching** - Create dev/staging/prod branches instantly
✅ **Free Tier** - Generous free tier for development
✅ **ML-Optimized** - Great performance for large datasets
✅ **Zero Config** - Just copy connection string and go

**Alternative:** You can also use local PostgreSQL if you prefer (see [SETUP.md](SETUP.md))