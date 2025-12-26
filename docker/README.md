# Docker Configuration for AI-Playground

This directory contains Docker configuration files for the AI-Playground project.

## 📁 Directory Structure

```
AI-Playground/
├── docker-compose.yml         # Multi-service orchestration
├── docker/
│   ├── README.md              # This file
│   ├── postgres/
│   │   └── init.sql          # PostgreSQL initialization script
│   └── redis/
│       └── redis.conf        # Redis configuration
├── backend/
│   └── Dockerfile            # FastAPI & Celery worker image
└── frontend/
    └── Dockerfile            # React + Nginx image
```

## 🐳 Docker Services

The AI-Playground stack consists of **4 containerized services**:

### 1. Redis (aiplayground-redis)
- **Purpose**: Caching and Celery message broker
- **Image**: redis:7-alpine
- **Port**: 6379
- **Container Name**: aiplayground-redis
- **Configuration**: Custom `docker/redis/redis.conf`
- **Features**:
  - LRU eviction policy (allkeys-lru)
  - 256MB memory limit
  - RDB persistence (snapshots every 15min/5min/1min)
  - Health check with redis-cli ping
  - Optimized for ML workload caching

### 2. Backend API (aiplayground-backend)
- **Purpose**: FastAPI REST API server
- **Build**: `backend/Dockerfile` (Python 3.11-slim)
- **Port**: 8000
- **Container Name**: aiplayground-backend
- **Depends On**: Redis (with health check)
- **Features**:
  - Auto database migrations (Alembic)
  - Hot reload enabled (development)
  - Health check endpoint at /health
  - Persistent uploads and logs
  - NeonDB connection (serverless PostgreSQL)

### 3. Celery Worker (aiplayground-celery-worker)
- **Purpose**: Async task processing
- **Build**: `backend/Dockerfile` (same as backend)
- **Container Name**: aiplayground-celery-worker
- **Depends On**: Redis, Backend
- **Features**:
  - Processes long-running ML tasks
  - 2 concurrent workers
  - Shared volumes with backend (uploads, logs)
  - Auto-restart on failure

### 4. Frontend (aiplayground-frontend)
- **Purpose**: React web application
- **Build**: `frontend/Dockerfile` (Node 20-alpine + Nginx)
- **Port**: 80
- **Container Name**: aiplayground-frontend
- **Depends On**: Backend
- **Features**:
  - Multi-stage build (build + production)
  - Vite build optimization
  - Nginx reverse proxy
  - Health check at /health

## 🚀 Quick Start

### Prerequisites

1. **Docker & Docker Compose** installed
2. **NeonDB Connection String** - Get from [neon.tech](https://neon.tech)
3. **Environment Variables** - Set DATABASE_URL

### Environment Setup

**Option 1: Using .env file (Recommended)**

Create a `.env` file in the project root:

```env
# NeonDB Connection String (REQUIRED)
DATABASE_URL=postgresql://user:password@ep-xxx-xxx.us-east-2.aws.neon.tech/aiplayground?sslmode=require

# Optional: Override other settings
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
```

**Option 2: Export environment variable**

**Windows (PowerShell):**
```powershell
$env:DATABASE_URL="postgresql://user:password@ep-xxx-xxx.us-east-2.aws.neon.tech/aiplayground?sslmode=require"
docker-compose up -d
```

**Linux/Mac:**
```bash
export DATABASE_URL="postgresql://user:password@ep-xxx-xxx.us-east-2.aws.neon.tech/aiplayground?sslmode=require"
docker-compose up -d
```

### Starting the Stack

```bash
# Build all services (first time or after Dockerfile changes)
docker-compose build

# Start all services in detached mode
docker-compose up -d

# View status
docker-compose ps

# View logs (all services)
docker-compose logs -f

# View logs (specific service)
docker-compose logs -f backend
docker-compose logs -f celery-worker

# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes all data!)
docker-compose down -v
```

### Access the Application

Once started:
- **Frontend**: http://localhost (or http://localhost:80)
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## ⚙️ Configuration Files

### docker-compose.yml

Main orchestration file defining all 4 services with:

**Key Configuration:**
- **Networks**: Single bridge network (aiplayground-network)
- **Volumes**: 3 persistent volumes (redis_data, backend_uploads, backend_logs)
- **Health Checks**: All services have health monitoring
- **Dependencies**: Proper service startup order
- **Restart Policy**: unless-stopped (auto-restart on failure)

**Environment Variables (Backend & Celery):**
```yaml
DATABASE_URL: ${DATABASE_URL}              # NeonDB connection (from .env)
REDIS_URL: redis://redis:6379/0           # Internal Redis connection
CELERY_BROKER_URL: redis://redis:6379/0   # Celery message broker
CELERY_RESULT_BACKEND: redis://redis:6379/0
SECRET_KEY: docker-dev-secret-key-change-in-production
UPLOAD_DIR: /app/uploads
MAX_UPLOAD_SIZE: 104857600                # 100MB
ENVIRONMENT: development
DEBUG: "True"
```

### backend/Dockerfile

Multi-stage build for Python backend:

**Stages:**
1. **base** - Dependencies layer (Python 3.11-slim + requirements.txt)
2. **development** - Dev image with hot reload
3. **production** - Optimized image with non-root user + 4 workers

**System Dependencies:**
- gcc, g++ (for numpy/pandas)
- curl (health checks)
- postgresql-client (database management)

**Image Size Optimization:**
- Slim base image
- Cached dependency layer
- No cache directories
- Production: minimal files only

### frontend/Dockerfile

Multi-stage build for React frontend:

**Stage 1: Build** (Node 20-alpine)
- Install dependencies with `npm ci`
- Build Vite app with environment variables
- Output to `/app/dist`

**Stage 2: Production** (Nginx-alpine)
- Serve static files from `/usr/share/nginx/html`
- Custom nginx configuration
- Lightweight (< 50MB total)

### docker/postgres/init.sql

PostgreSQL initialization script (runs once on first container creation):

**Features:**
- Enables UUID extension (uuid-ossp)
- Enables text search optimization (pg_trgm)
- Grants all privileges to aiplayground user
- Sets default privileges for future tables

**Note**: This is for NeonDB alternative. The main setup uses NeonDB (no local PostgreSQL container).

### docker/redis/redis.conf

Redis configuration optimized for ML workloads:

**Memory Management:**
- 256MB max memory limit
- allkeys-lru eviction policy (evict least recently used when full)

**Persistence:**
- RDB snapshots: save 900 1, save 300 10, save 60 10000
- AOF disabled (development mode)

**Performance:**
- 10000 max clients
- Lazy freeing disabled
- Hash/list optimizations enabled

**Security:**
- No password (development)
- Bind to 0.0.0.0 (container network)
- Protected mode disabled

## 🔧 Common Tasks

### Backend API

**View backend logs:**
```bash
docker-compose logs -f backend
```

**Access backend shell:**
```bash
docker-compose exec backend bash
```

**Run database migrations:**
```bash
docker-compose exec backend alembic upgrade head
```

**Create new migration:**
```bash
docker-compose exec backend alembic revision --autogenerate -m "description"
```

**Run Python shell:**
```bash
docker-compose exec backend python
```

**Restart backend:**
```bash
docker-compose restart backend
```

### Celery Worker

**View Celery logs:**
```bash
docker-compose logs -f celery-worker
```

**Inspect active tasks:**
```bash
docker-compose exec celery-worker celery -A celery_worker.celery_app inspect active
```

**Inspect scheduled tasks:**
```bash
docker-compose exec celery-worker celery -A celery_worker.celery_app inspect scheduled
```

**Purge all tasks:**
```bash
docker-compose exec celery-worker celery -A celery_worker.celery_app purge
```

**Restart worker:**
```bash
docker-compose restart celery-worker
```

### Frontend

**View frontend logs:**
```bash
docker-compose logs -f frontend
```

**Rebuild frontend (after code changes):**
```bash
docker-compose build frontend
docker-compose up -d frontend
```

**Access Nginx logs:**
```bash
docker-compose exec frontend cat /var/log/nginx/access.log
docker-compose exec frontend cat /var/log/nginx/error.log
```

### Redis

**Access Redis CLI:**
```bash
docker-compose exec redis redis-cli
```

**Check memory usage:**
```bash
docker-compose exec redis redis-cli INFO memory
```

**View all keys:**
```bash
docker-compose exec redis redis-cli KEYS '*'
```

**Flush all data:**
```bash
docker-compose exec redis redis-cli FLUSHALL
```

**Monitor commands:**
```bash
docker-compose exec redis redis-cli MONITOR
```

## 📊 Monitoring

### Check Service Health

```bash
# All services
docker-compose ps

# Redis health
docker-compose exec redis redis-cli ping

# Backend health
curl http://localhost:8000/health

# Frontend health
curl http://localhost/health
```

### View Resource Usage

```bash
# All containers
docker stats

# Specific containers
docker stats aiplayground-backend
docker stats aiplayground-celery-worker
docker stats aiplayground-redis
docker stats aiplayground-frontend
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f celery-worker
docker-compose logs -f redis
docker-compose logs -f frontend

# Last 100 lines
docker-compose logs --tail=100 backend

# Since specific time
docker-compose logs --since 30m backend
```

### Debugging

**Check container status:**
```bash
docker-compose ps
```

**Inspect container:**
```bash
docker inspect aiplayground-backend
```

**View container processes:**
```bash
docker-compose top backend
```

**Execute commands in running container:**
```bash
docker-compose exec backend ls -la /app
docker-compose exec backend env
```

## 🔒 Security Considerations

### Development (Current Setup)
- Default SECRET_KEY (change in production!)
- No authentication required for Redis
- DEBUG mode enabled
- All services exposed on localhost
- No SSL/TLS encryption

### Production Recommendations

1. **Change SECRET_KEY:**
```yaml
environment:
  SECRET_KEY: ${SECRET_KEY}  # Load from secure source
```

2. **Enable Redis authentication:**
```conf
# In docker/redis/redis.conf
requirepass your_strong_redis_password_here
```

Then update docker-compose.yml:
```yaml
REDIS_URL: redis://:your_strong_redis_password_here@redis:6379/0
CELERY_BROKER_URL: redis://:your_strong_redis_password_here@redis:6379/0
```

3. **Restrict network access:**
```yaml
# Don't expose ports publicly in production
ports:
  - "127.0.0.1:8000:8000"  # Only localhost can access
```

4. **Use Docker secrets:**
```yaml
secrets:
  database_url:
    file: ./secrets/database_url.txt
  secret_key:
    file: ./secrets/secret_key.txt

services:
  backend:
    secrets:
      - database_url
      - secret_key
```

5. **Disable DEBUG mode:**
```yaml
environment:
  DEBUG: "False"
  ENVIRONMENT: production
```

6. **Use production Dockerfile target:**
```yaml
backend:
  build:
    target: production  # Non-root user, 4 workers, no reload
```

7. **Enable SSL/TLS:**
- Configure Nginx with SSL certificates (Let's Encrypt)
- Use NeonDB SSL connection (`?sslmode=require`)
- Enable Redis TLS mode

8. **Add rate limiting:**
- Configure Nginx rate limiting
- Add backend rate limiting middleware
- Use Redis for distributed rate limiting

9. **Security headers:**
```nginx
# In nginx.conf
add_header X-Frame-Options "SAMEORIGIN";
add_header X-Content-Type-Options "nosniff";
add_header X-XSS-Protection "1; mode=block";
add_header Strict-Transport-Security "max-age=31536000";
```

## 🐛 Troubleshooting

### Backend Won't Start

**Problem:** Backend container exits immediately

**Solution:**
```bash
# Check logs for errors
docker-compose logs backend

# Common issues:
# 1. Missing DATABASE_URL - ensure .env file exists or env var is set
# 2. Database connection failed - verify NeonDB connection string
# 3. Alembic migration failed - check database schema

# Manual migration check
docker-compose exec backend alembic current
docker-compose exec backend alembic upgrade head
```

**Problem:** Import errors or module not found

**Solution:**
```bash
# Rebuild with no cache
docker-compose build --no-cache backend

# Check Python path
docker-compose exec backend python -c "import sys; print(sys.path)"
```

### Celery Worker Issues

**Problem:** Tasks not processing

**Solution:**
```bash
# Check worker status
docker-compose logs celery-worker

# Verify Redis connection
docker-compose exec celery-worker python -c "import redis; r = redis.from_url('redis://redis:6379/0'); print(r.ping())"

# Inspect active tasks
docker-compose exec celery-worker celery -A celery_worker.celery_app inspect active

# Restart worker
docker-compose restart celery-worker
```

**Problem:** High memory usage

**Solution:**
```bash
# Check worker memory
docker stats aiplayground-celery-worker

# Reduce concurrency in docker-compose.yml
command: celery -A celery_worker.celery_app worker --loglevel=info --concurrency=1

# Restart with new config
docker-compose up -d celery-worker
```

### Redis Issues

**Problem:** High memory usage

**Solution:**
```bash
# Check memory info
docker-compose exec redis redis-cli INFO memory

# Flush unnecessary data
docker-compose exec redis redis-cli FLUSHDB

# Adjust maxmemory in docker/redis/redis.conf
maxmemory 512mb  # Increase limit

# Restart Redis
docker-compose restart redis
```

**Problem:** Connection refused

**Solution:**
```bash
# Test connection
docker-compose exec redis redis-cli ping

# Check Redis logs
docker-compose logs redis

# Verify Redis is running
docker-compose ps redis

# Check network connectivity
docker-compose exec backend ping redis
```

### Frontend Issues

**Problem:** Blank page or 404 errors

**Solution:**
```bash
# Check frontend logs
docker-compose logs frontend

# Verify build completed successfully
docker-compose build frontend

# Check Nginx configuration
docker-compose exec frontend cat /etc/nginx/conf.d/default.conf

# Restart frontend
docker-compose restart frontend
```

**Problem:** API calls failing (CORS errors)

**Solution:**
- Verify VITE_API_URL in docker-compose.yml matches backend URL
- Check browser console for CORS errors
- Ensure backend is running: `curl http://localhost:8000/health`

### Database Connection Issues

**Problem:** NeonDB connection timeout

**Solution:**
```bash
# Verify connection string format
# postgresql://user:password@host.neon.tech/dbname?sslmode=require

# Test connection from container
docker-compose exec backend python -c "from sqlalchemy import create_engine; engine = create_engine('$DATABASE_URL'); print(engine.connect())"

# Check NeonDB project status at neon.tech
# Ensure project is not suspended

# Increase connection timeout in connection string
?sslmode=require&connect_timeout=30
```

### Volume Permission Issues

**Problem:** Permission denied errors

**Solution:**
```bash
# Check volume permissions
docker-compose exec backend ls -la /app/uploads
docker-compose exec backend ls -la /app/logs

# Fix permissions
docker-compose exec backend chown -R $(id -u):$(id -g) /app/uploads /app/logs

# Or remove volumes and recreate
docker-compose down -v
docker-compose up -d
```

### Port Conflicts

**Problem:** Port already in use

**Solution:**
```bash
# Check what's using the port (Windows)
netstat -ano | findstr :8000
netstat -ano | findstr :80
netstat -ano | findstr :6379

# Check what's using the port (Linux/Mac)
lsof -i :8000
lsof -i :80
lsof -i :6379

# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use different host port
```

### Complete Reset

**Problem:** Everything is broken

**Solution:**
```bash
# Stop all containers
docker-compose down

# Remove all volumes (WARNING: deletes all data!)
docker-compose down -v

# Remove all images
docker-compose down --rmi all

# Rebuild everything from scratch
docker-compose build --no-cache
docker-compose up -d

# Check logs
docker-compose logs -f
```

## 📚 Additional Resources

### Documentation
- **[Main README](../README.md)** - Project overview and features
- **[Backend README](../backend/README.md)** - API documentation and ML engine
- **[Frontend README](../frontend/README.md)** - React components and Redux store
- **[ML Engine README](../backend/app/ml_engine/README.md)** - Preprocessing and feature selection

### Docker
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Multi-stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [Docker Networking](https://docs.docker.com/network/)

### Technologies
- [FastAPI Docker](https://fastapi.tiangolo.com/deployment/docker/)
- [Redis Docker Hub](https://hub.docker.com/_/redis)
- [Redis Documentation](https://redis.io/docs/)
- [Celery Documentation](https://docs.celeryq.dev/)
- [NeonDB Documentation](https://neon.tech/docs/introduction)
- [Nginx Docker](https://hub.docker.com/_/nginx)

## 🔄 Updates and Maintenance

### Updating Images

**Backend/Celery:**
```bash
# Update Python version in backend/Dockerfile
FROM python:3.12-slim as base  # Change from 3.11

# Rebuild
docker-compose build --no-cache backend celery-worker
docker-compose up -d backend celery-worker
```

**Frontend:**
```bash
# Update Node version in frontend/Dockerfile
FROM node:22-alpine as build  # Change from 20

# Rebuild
docker-compose build --no-cache frontend
docker-compose up -d frontend
```

**Redis:**
```bash
# Update image version in docker-compose.yml
image: redis:8-alpine  # Change from 7-alpine

# Restart service
docker-compose pull redis
docker-compose up -d redis
```

### Updating Dependencies

**Backend Python packages:**
```bash
# Update requirements.txt
# Then rebuild
docker-compose build --no-cache backend celery-worker
docker-compose up -d
```

**Frontend npm packages:**
```bash
# Update package.json
# Then rebuild
docker-compose build --no-cache frontend
docker-compose up -d frontend
```

### Database Migrations

```bash
# Create new migration
docker-compose exec backend alembic revision --autogenerate -m "description"

# Apply migrations
docker-compose exec backend alembic upgrade head

# Rollback one version
docker-compose exec backend alembic downgrade -1

# View migration history
docker-compose exec backend alembic history
```

## 📊 Performance Optimization

### Backend Optimization

**Production settings (docker-compose.yml):**
```yaml
backend:
  build:
    target: production  # Use production stage
  environment:
    DEBUG: "False"
  command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Gunicorn alternative (better for production):**
```yaml
command: gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Redis Optimization

**For larger datasets:**
```conf
# In docker/redis/redis.conf
maxmemory 1gb              # Increase from 256mb
maxmemory-policy allkeys-lru
```

**Enable AOF persistence (production):**
```conf
appendonly yes
appendfsync everysec
```

### Celery Optimization

**Increase workers for CPU-bound tasks:**
```yaml
celery-worker:
  command: celery -A celery_worker.celery_app worker --loglevel=info --concurrency=4
```

**Add Celery beat for scheduled tasks:**
```yaml
celery-beat:
  build:
    context: ./backend
    dockerfile: Dockerfile
  container_name: aiplayground-celery-beat
  depends_on:
    - redis
  command: celery -A celery_worker.celery_app beat --loglevel=info
```

## 📝 Notes

### Important Considerations

- **NeonDB Connection**: The stack uses NeonDB (serverless PostgreSQL) instead of a local PostgreSQL container
- **Environment Variables**: DATABASE_URL must be set via .env file or environment
- **Volumes**: 3 persistent volumes (redis_data, backend_uploads, backend_logs) survive container restarts
- **Configuration**: Redis config (`docker/redis/redis.conf`) is mounted as volume - changes require restart
- **Migrations**: Alembic migrations run automatically on backend startup
- **Development Mode**: Hot reload enabled for backend, DEBUG=True, no authentication
- **Health Checks**: All services have health checks - use `docker-compose ps` to verify

### Service Dependencies

```
Frontend → Backend → Redis
                ↓
         Celery Worker → Redis
```

- Frontend depends on Backend
- Backend depends on Redis (health check)
- Celery Worker depends on both Backend and Redis (health checks)

### Data Persistence

**What's persisted:**
- Redis data (`redis_data` volume)
- Backend uploads (`backend_uploads` volume)
- Backend logs (`backend_logs` volume)
- Database data (in NeonDB - external)

**What's NOT persisted:**
- Container logs (use `docker-compose logs` or export to volume)
- Temporary files in containers
- Build cache (cleared with `docker-compose build --no-cache`)

### Common Workflows

**Development workflow:**
1. Edit code locally (backend/ or frontend/)
2. Backend: Changes auto-reload (volume mount + --reload flag)
3. Frontend: Rebuild container (`docker-compose build frontend && docker-compose up -d frontend`)

**Production deployment:**
1. Set environment variables (DATABASE_URL, SECRET_KEY, etc.)
2. Build with production target (`target: production`)
3. Disable DEBUG mode (`DEBUG: "False"`)
4. Use strong SECRET_KEY and Redis password
5. Enable SSL/TLS certificates
6. Configure firewall and rate limiting

---

**Last Updated**: 2025-01-15
**Docker Compose Version**: 3.8
**Services**: Redis 7, FastAPI (Python 3.11), Celery, React (Node 20 + Nginx)
