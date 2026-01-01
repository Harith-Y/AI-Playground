# Docker Setup Guide for AI-Playground Backend

This document provides comprehensive instructions for running the AI-Playground backend using Docker.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Docker Files Overview](#docker-files-overview)
3. [Development Setup](#development-setup)
4. [Production Setup](#production-setup)
5. [Environment Variables](#environment-variables)
6. [Common Commands](#common-commands)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Development (with local PostgreSQL)

```bash
# Start all services including PostgreSQL, Redis, Backend, Celery, pgAdmin
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f backend

# Access services:
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Flower (Celery): http://localhost:5555
# - pgAdmin: http://localhost:5050
```

### Production (with external database like Neon)

```bash
# Set DATABASE_URL environment variable
export DATABASE_URL="postgresql://user:pass@your-db-host:5432/dbname"

# Start services (uses docker-compose.yml)
docker-compose up -d

# Access:
# - Backend API: http://localhost:8000
# - Frontend: http://localhost (port 80)
```

---

## Docker Files Overview

### Dockerfiles

| File | Purpose | Use Case |
|------|---------|----------|
| `Dockerfile` | Production-optimized multi-stage build | Render deployment, production |
| `Dockerfile.dev` | Development with all tools | Local development with hot reload |

### Docker Compose Files

| File | Purpose | Services Included |
|------|---------|-------------------|
| `docker-compose.yml` | Production setup | Redis, Backend, Celery Worker, Frontend |
| `docker-compose.dev.yml` | Development setup | PostgreSQL, Redis, Backend, Celery, Flower, pgAdmin |

### Configuration Files

| File | Purpose |
|------|---------|
| `.dockerignore` | Excludes files from Docker build context |
| `.env.docker` | Docker environment template |
| `requirements.txt` | Full Python dependencies |
| `requirements.render.txt` | Minimal dependencies for Render free tier |

---

## Development Setup

### Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Docker Compose v2.0+
- At least 4GB RAM available for Docker

### Step 1: Clone and Navigate

```bash
cd AI-Playground/backend
```

### Step 2: Start Development Environment

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# Check status
docker-compose -f docker-compose.dev.yml ps

# View logs
docker-compose -f docker-compose.dev.yml logs -f
```

### Step 3: Access Services

- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Flower (Celery Monitor)**: http://localhost:5555
- **pgAdmin (Database UI)**: http://localhost:5050
  - Email: `admin@aiplayground.local`
  - Password: `admin`

### Step 4: Run Database Migrations

```bash
# Migrations run automatically on startup, but you can run manually:
docker-compose -f docker-compose.dev.yml exec backend alembic upgrade head
```

### Step 5: Create Test User (Optional)

```bash
docker-compose -f docker-compose.dev.yml exec backend python init_db.py
```

### Development Features

✅ **Hot Reload**: Code changes automatically reload the server
✅ **Volume Mounts**: Source code is mounted, no rebuild needed
✅ **Full Dependencies**: All ML libraries included
✅ **Database Included**: PostgreSQL running in container
✅ **Monitoring Tools**: Flower for Celery, pgAdmin for database

---

## Production Setup

### For Render Deployment

The `Dockerfile` is optimized for Render's free tier:

```bash
# Build for Render (minimal dependencies)
docker build -t aiplayground-backend .

# Build with full dependencies
docker build --build-arg REQUIREMENTS_FILE=requirements.txt -t aiplayground-backend .
```

### For Self-Hosted Production

1. **Set up external database** (PostgreSQL - recommended: Neon, Supabase)

2. **Create `.env` file**:

```bash
cp .env.docker .env
# Edit .env with your production values
```

3. **Start production stack**:

```bash
# Export database URL
export DATABASE_URL="postgresql://user:pass@host:5432/db"

# Start services
docker-compose up -d

# Check health
docker-compose ps
```

### Production Environment Variables

Required:
- `DATABASE_URL` - PostgreSQL connection string
- `SECRET_KEY` - JWT secret (generate with `openssl rand -hex 32`)
- `REDIS_URL` - Redis connection string

Optional:
- `ENVIRONMENT=production`
- `DEBUG=False`
- `LOG_LEVEL=INFO`

---

## Environment Variables

### Complete List

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection | - | ✅ |
| `REDIS_URL` | Redis connection | `redis://redis:6379/0` | ✅ |
| `SECRET_KEY` | JWT secret key | - | ✅ |
| `CELERY_BROKER_URL` | Celery broker | Same as REDIS_URL | ✅ |
| `CELERY_RESULT_BACKEND` | Celery results | Same as REDIS_URL | ✅ |
| `ENVIRONMENT` | Environment name | `development` | ❌ |
| `DEBUG` | Debug mode | `True` | ❌ |
| `LOG_LEVEL` | Logging level | `INFO` | ❌ |
| `UPLOAD_DIR` | File uploads directory | `/app/uploads` | ❌ |
| `MAX_UPLOAD_SIZE` | Max file size (bytes) | `104857600` | ❌ |

### Setting Environment Variables

**Development** (`.env` file):
```bash
DATABASE_URL=postgresql://aiplayground:password@postgres:5432/aiplayground
REDIS_URL=redis://redis:6379/0
SECRET_KEY=dev-secret-key
```

**Production** (export or docker-compose.yml):
```bash
export DATABASE_URL="postgresql://..."
export SECRET_KEY="$(openssl rand -hex 32)"
```

---

## Common Commands

### Building Images

```bash
# Development image
docker build -f Dockerfile.dev -t aiplayground-backend:dev .

# Production image (Render-optimized)
docker build -t aiplayground-backend:prod .

# Production image (full dependencies)
docker build --build-arg REQUIREMENTS_FILE=requirements.txt -t aiplayground-backend:full .
```

### Managing Services

```bash
# Start services
docker-compose -f docker-compose.dev.yml up -d

# Stop services
docker-compose -f docker-compose.dev.yml down

# Restart a service
docker-compose -f docker-compose.dev.yml restart backend

# Rebuild and restart
docker-compose -f docker-compose.dev.yml up -d --build backend

# View logs
docker-compose -f docker-compose.dev.yml logs -f backend

# Execute command in container
docker-compose -f docker-compose.dev.yml exec backend bash
```

### Database Operations

```bash
# Run migrations
docker-compose -f docker-compose.dev.yml exec backend alembic upgrade head

# Create new migration
docker-compose -f docker-compose.dev.yml exec backend alembic revision --autogenerate -m "description"

# Rollback migration
docker-compose -f docker-compose.dev.yml exec backend alembic downgrade -1

# Access PostgreSQL CLI
docker-compose -f docker-compose.dev.yml exec postgres psql -U aiplayground -d aiplayground

# Backup database
docker-compose -f docker-compose.dev.yml exec postgres pg_dump -U aiplayground aiplayground > backup.sql

# Restore database
docker-compose -f docker-compose.dev.yml exec -T postgres psql -U aiplayground aiplayground < backup.sql
```

### Celery Operations

```bash
# View Celery worker logs
docker-compose -f docker-compose.dev.yml logs -f celery-worker

# Inspect active tasks
docker-compose -f docker-compose.dev.yml exec celery-worker celery -A celery_worker.celery_app inspect active

# Purge all tasks
docker-compose -f docker-compose.dev.yml exec celery-worker celery -A celery_worker.celery_app purge
```

### Cleanup

```bash
# Stop and remove containers
docker-compose -f docker-compose.dev.yml down

# Remove volumes (WARNING: deletes data!)
docker-compose -f docker-compose.dev.yml down -v

# Remove unused images
docker image prune -a

# Full cleanup (containers, volumes, images)
docker-compose -f docker-compose.dev.yml down -v --rmi all
```

---

## Troubleshooting

### Backend Won't Start

**Problem**: Container exits immediately

```bash
# Check logs
docker-compose -f docker-compose.dev.yml logs backend

# Common causes:
# 1. Database not ready - increase sleep time in command
# 2. Port 8000 already in use - change port mapping
# 3. Missing environment variables - check .env file
```

**Solution**:
```bash
# Check if database is ready
docker-compose -f docker-compose.dev.yml exec postgres pg_isready

# Restart backend
docker-compose -f docker-compose.dev.yml restart backend
```

### Database Connection Failed

**Problem**: `sqlalchemy.exc.OperationalError: could not connect`

```bash
# Check if PostgreSQL is running
docker-compose -f docker-compose.dev.yml ps postgres

# Check connection string
docker-compose -f docker-compose.dev.yml exec backend env | grep DATABASE_URL

# Test connection
docker-compose -f docker-compose.dev.yml exec postgres psql -U aiplayground -d aiplayground -c "SELECT 1;"
```

### Port Already in Use

**Problem**: `Error: port is already allocated`

```bash
# Find process using port 8000
# Windows:
netstat -ano | findstr :8000

# Linux/Mac:
lsof -i :8000

# Change port in docker-compose.yml:
ports:
  - "8001:8000"  # Changed from 8000:8000
```

### Out of Memory

**Problem**: Container killed due to OOM

**Solution**: Increase Docker memory limit
```bash
# Docker Desktop: Settings → Resources → Memory → 4GB+

# Or reduce workers in docker-compose.yml:
command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Hot Reload Not Working

**Problem**: Code changes don't reload server

```bash
# Check volume mounts
docker-compose -f docker-compose.dev.yml exec backend ls -la /app/app

# Restart with rebuild
docker-compose -f docker-compose.dev.yml up -d --build backend

# Check uvicorn is running with --reload
docker-compose -f docker-compose.dev.yml exec backend ps aux | grep uvicorn
```

### Celery Tasks Not Processing

**Problem**: Tasks stuck in queue

```bash
# Check Celery worker status
docker-compose -f docker-compose.dev.yml logs celery-worker

# Check Redis connection
docker-compose -f docker-compose.dev.yml exec redis redis-cli ping

# Restart Celery
docker-compose -f docker-compose.dev.yml restart celery-worker

# View tasks in Flower
# Open http://localhost:5555
```

### Permission Denied Errors

**Problem**: Cannot write to mounted volumes

```bash
# Check ownership
docker-compose -f docker-compose.dev.yml exec backend ls -la /app/uploads

# Fix permissions (run on host)
sudo chown -R 1000:1000 backend/uploads
sudo chown -R 1000:1000 backend/logs
```

---

## Advanced Topics

### Multi-Platform Builds

Build for different architectures (useful for ARM-based systems):

```bash
# Build for AMD64 and ARM64
docker buildx build --platform linux/amd64,linux/arm64 -t aiplayground-backend .
```

### Custom Networks

Connect to external services:

```bash
# Create external network
docker network create aiplayground-external

# Reference in docker-compose.yml:
networks:
  default:
    external: true
    name: aiplayground-external
```

### Health Checks

Monitor container health:

```bash
# Check health status
docker inspect --format='{{json .State.Health}}' aiplayground-backend-dev

# View health check logs
docker inspect aiplayground-backend-dev | jq '.[0].State.Health'
```

---

## Best Practices

### Development
- ✅ Use `docker-compose.dev.yml` for local development
- ✅ Mount source code as volumes for hot reload
- ✅ Use named volumes for data persistence
- ✅ Enable debug logging
- ✅ Use tools like pgAdmin and Flower

### Production
- ✅ Use multi-stage builds to reduce image size
- ✅ Run containers as non-root user
- ✅ Use external managed databases (Neon, RDS)
- ✅ Enable health checks
- ✅ Set resource limits
- ✅ Use production-grade secrets management
- ✅ Monitor with proper logging/metrics

### Security
- ❌ Never commit `.env` files with real credentials
- ❌ Don't use default passwords in production
- ✅ Use strong SECRET_KEY (32+ random bytes)
- ✅ Enable SSL/TLS for database connections
- ✅ Regularly update base images
- ✅ Scan images for vulnerabilities

---

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [FastAPI Docker Guide](https://fastapi.tiangolo.com/deployment/docker/)
- [PostgreSQL Docker Hub](https://hub.docker.com/_/postgres)
- [Redis Docker Hub](https://hub.docker.com/_/redis)

---

## Support

For issues or questions:
1. Check logs: `docker-compose logs`
2. Review this troubleshooting guide
3. Check GitHub issues
4. Create new issue with logs and environment details
