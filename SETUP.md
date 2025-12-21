# AI-Playground Setup Guide

## Prerequisites

Before starting, ensure you have the following installed:
- **Python 3.11+** (for backend)
- **Node.js 18+** and **npm/yarn** (for frontend)
- **PostgreSQL 15+** (database)
- **Redis 7+** (caching & task queue)
- **Docker & Docker Compose** (optional, for containerized setup)
- **Git** (version control)

---

## 📁 Project Structure

```
AI-Playground/
├── backend/              # FastAPI backend
├── frontend/             # React frontend
├── docker/               # Docker configurations
├── docker-compose.yml    # Docker Compose setup
├── APPROACH.md           # Project approach
├── ML-PIPELINE.md        # ML pipeline details
├── MODELS.md             # Available models
├── README.md             # Project overview
└── SETUP.md              # This file
```

---

## 🚀 Initialization Commands

### Option 1: Local Development Setup (Without Docker)

#### **1. Backend Setup**

```powershell
# Navigate to backend directory
cd backend

# Create Python virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\Activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies (will be populated later)
pip install fastapi uvicorn sqlalchemy psycopg2-binary alembic celery redis python-multipart pydantic-settings

# Install ML libraries
pip install scikit-learn pandas numpy matplotlib seaborn plotly xgboost lightgbm catboost

# Install additional tools
pip install pytest black flake8 mypy python-jose passlib bcrypt

# Generate requirements.txt
pip freeze > requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your settings (database URL, Redis URL, etc.)
# Use your preferred editor: notepad, VSCode, etc.

# Initialize Alembic for database migrations
alembic init alembic

# Run migrations (once configured)
# alembic upgrade head

# Start the backend server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### **2. Frontend Setup**

```powershell
# Navigate to frontend directory (in new terminal)
cd frontend

# Initialize React app with TypeScript (using Vite)
npm create vite@latest . -- --template react-ts

# Or if directory already exists and you want to use CRA:
# npx create-react-app . --template typescript

# Install dependencies
npm install

# Install additional packages
npm install @reduxjs/toolkit react-redux react-router-dom axios
npm install @mui/material @emotion/react @emotion/styled
npm install plotly.js react-plotly.js
npm install recharts
npm install react-hook-form yup @hookform/resolvers

# Install dev dependencies
npm install -D @types/node @types/react @types/react-dom
npm install -D @types/plotly.js

# Copy environment template
cp .env.example .env

# Edit .env with backend API URL
# REACT_APP_API_URL=http://localhost:8000

# Start the development server
npm run dev
# Or for CRA: npm start
```

#### **3. Database Setup (PostgreSQL)**

```powershell
# Start PostgreSQL service (Windows)
# If using PostgreSQL installed locally:
# Services -> PostgreSQL -> Start

# Create database using psql
psql -U postgres

# In psql:
CREATE DATABASE aiplayground;
CREATE USER aiplayground_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE aiplayground TO aiplayground_user;
\q
```

#### **4. Redis Setup**

```powershell
# Windows: Install Redis using WSL or download Redis for Windows
# https://github.com/microsoftarchive/redis/releases

# Start Redis server
redis-server

# Test Redis connection (in new terminal)
redis-cli ping
# Should return: PONG
```

#### **5. Celery Worker Setup**

```powershell
# Navigate to backend directory (in new terminal)
cd backend

# Activate virtual environment
.\venv\Scripts\Activate

# Start Celery worker
celery -A app.tasks worker --loglevel=info

# Optional: Start Celery Beat (for scheduled tasks)
# celery -A app.tasks beat --loglevel=info
```

---

### Option 2: Docker Setup (Recommended for Development)

#### **1. Build and Start All Services**

```powershell
# From project root directory
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

#### **2. Individual Service Commands**

```powershell
# Start only backend
docker-compose up -d backend

# Start only frontend
docker-compose up -d frontend

# Start only database
docker-compose up -d postgres

# Start only Redis
docker-compose up -d redis

# Restart a specific service
docker-compose restart backend

# View logs for specific service
docker-compose logs -f backend
```

#### **3. Database Migrations with Docker**

```powershell
# Run migrations inside Docker container
docker-compose exec backend alembic upgrade head

# Create new migration
docker-compose exec backend alembic revision --autogenerate -m "Initial migration"
```

---

## 📦 Package Installation Summary

### Backend (Python)

**Core Framework:**
- `fastapi` - Web framework
- `uvicorn[standard]` - ASGI server
- `pydantic` - Data validation
- `pydantic-settings` - Settings management

**Database:**
- `sqlalchemy` - ORM
- `psycopg2-binary` - PostgreSQL adapter
- `alembic` - Database migrations

**Task Queue:**
- `celery` - Async task queue
- `redis` - Message broker

**ML Libraries:**
- `scikit-learn` - ML algorithms
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `plotly` - Interactive plots
- `xgboost` - Gradient boosting
- `lightgbm` - Gradient boosting
- `catboost` - Gradient boosting
- `scikit-optimize` - Bayesian optimization (optional)

**Utilities:**
- `python-multipart` - File uploads
- `python-jose[cryptography]` - JWT tokens
- `passlib[bcrypt]` - Password hashing
- `python-dotenv` - Environment variables

**Development:**
- `pytest` - Testing
- `black` - Code formatting
- `flake8` - Linting
- `mypy` - Type checking

### Frontend (JavaScript/TypeScript)

**Core:**
- `react` - UI library
- `react-dom` - React DOM
- `typescript` - Type safety
- `vite` - Build tool (or `react-scripts` for CRA)

**State Management:**
- `@reduxjs/toolkit` - State management
- `react-redux` - React bindings for Redux

**Routing:**
- `react-router-dom` - Client-side routing

**HTTP Client:**
- `axios` - API requests

**UI Components:**
- `@mui/material` - Material-UI components
- `@emotion/react` - CSS-in-JS
- `@emotion/styled` - Styled components

**Data Visualization:**
- `plotly.js` - Interactive charts
- `react-plotly.js` - React wrapper for Plotly
- `recharts` - React charts library

**Forms:**
- `react-hook-form` - Form handling
- `yup` - Schema validation
- `@hookform/resolvers` - Form validators

**Development:**
- `@types/react` - React types
- `@types/react-dom` - React DOM types
- `@types/node` - Node types
- `eslint` - Linting
- `prettier` - Code formatting

---

## 🔧 Configuration Files

### Backend `.env` Configuration

```env
# Database
DATABASE_URL=postgresql://aiplayground_user:your_password@localhost:5432/aiplayground

# Redis
REDIS_URL=redis://localhost:6379/0

# API Settings
API_V1_PREFIX=/api/v1
SECRET_KEY=your-secret-key-here-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# File Storage
UPLOAD_DIR=./uploads
MAX_UPLOAD_SIZE=104857600  # 100MB

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Environment
ENVIRONMENT=development
DEBUG=True
```

### Frontend `.env` Configuration

```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000/ws
REACT_APP_ENV=development
```

---

## 🧪 Testing Setup

### Backend Tests

```powershell
cd backend

# Install test dependencies
pip install pytest pytest-cov pytest-asyncio httpx

# Run tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_dataset.py
```

### Frontend Tests

```powershell
cd frontend

# Install test dependencies
npm install -D @testing-library/react @testing-library/jest-dom @testing-library/user-event vitest jsdom

# Run tests
npm test

# Run with coverage
npm test -- --coverage
```

---

## 📊 Database Migrations

```powershell
# Create new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# View migration history
alembic history

# View current version
alembic current
```

---

## 🐛 Development Tools

### Code Formatting (Backend)

```powershell
cd backend

# Format code with Black
black app/

# Check code quality with Flake8
flake8 app/

# Type checking with MyPy
mypy app/
```

### Code Formatting (Frontend)

```powershell
cd frontend

# Format code with Prettier
npx prettier --write src/

# Lint with ESLint
npm run lint
```

---

## 🚀 Running the Complete Stack

### Development Mode

```powershell
# Terminal 1: Backend
cd backend
.\venv\Scripts\Activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev

# Terminal 3: Celery Worker
cd backend
.\venv\Scripts\Activate
celery -A app.tasks worker --loglevel=info

# Terminal 4: Redis (if not running as service)
redis-server

# Terminal 5: PostgreSQL (if not running as service)
# Usually runs as a system service
```

### Access Points

- **Frontend**: http://localhost:5173 (Vite) or http://localhost:3000 (CRA)
- **Backend API**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs
- **API Docs (ReDoc)**: http://localhost:8000/redoc

---

## 🐳 Docker Commands Reference

```powershell
# Build images
docker-compose build

# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f [service_name]

# Execute commands in container
docker-compose exec backend bash
docker-compose exec postgres psql -U postgres

# Restart service
docker-compose restart [service_name]

# Remove volumes
docker-compose down -v

# Rebuild without cache
docker-compose build --no-cache
```

---

## 📝 Next Steps

After initialization:

1. ✅ **Verify all services are running**
2. ✅ **Run database migrations**
3. ✅ **Test API endpoints** (visit http://localhost:8000/docs)
4. ✅ **Test frontend** (visit http://localhost:5173)
5. ✅ **Create first user** (if authentication implemented)
6. ✅ **Upload test dataset**
7. ✅ **Start coding!**

---

## 🆘 Troubleshooting

### Common Issues

**Backend won't start:**
- Check if PostgreSQL is running
- Verify DATABASE_URL in .env
- Check port 8000 is not in use

**Frontend won't start:**
- Delete node_modules and reinstall: `rm -rf node_modules && npm install`
- Clear npm cache: `npm cache clean --force`
- Check port 5173/3000 is not in use

**Database connection failed:**
- Verify PostgreSQL service is running
- Check database credentials
- Ensure database exists

**Celery worker not processing tasks:**
- Check if Redis is running
- Verify CELERY_BROKER_URL in .env
- Check worker logs for errors

**Docker issues:**
- Restart Docker Desktop
- Check Docker logs: `docker-compose logs`
- Clean Docker system: `docker system prune -a`

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Redux Toolkit Documentation](https://redux-toolkit.js.org/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Celery Documentation](https://docs.celeryproject.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

**Happy Coding! 🎉**
