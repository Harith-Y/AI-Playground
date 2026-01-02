# Infrastructure Documentation

Comprehensive documentation for AI-Playground infrastructure architecture, services, deployment, and operations.

**Version**: 1.0.0  
**Last Updated**: January 2, 2026  
**Status**: Production-Ready Development Environment

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Services](#core-services)
4. [Infrastructure Components](#infrastructure-components)
5. [Networking](#networking)
6. [Data Storage](#data-storage)
7. [Build & Deployment](#build--deployment)
8. [Monitoring & Logging](#monitoring--logging)
9. [Security](#security)
10. [Scalability](#scalability)
11. [Disaster Recovery](#disaster-recovery)
12. [Environments](#environments)
13. [Cost Considerations](#cost-considerations)
14. [Operations & Maintenance](#operations--maintenance)
15. [Troubleshooting](#troubleshooting)
16. [References](#references)

---

## Overview

### System Architecture

AI-Playground is a containerized machine learning platform built with a microservices architecture. The system consists of:

- **Frontend**: React + Vite SPA served via Nginx
- **Backend API**: FastAPI Python application
- **Async Workers**: Celery for ML task processing
- **Cache/Broker**: Redis for caching and message queuing
- **Database**: NeonDB (serverless PostgreSQL)

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18, Vite 5, Redux Toolkit | User interface & state management |
| **Backend** | FastAPI, Python 3.11, SQLAlchemy | REST API & business logic |
| **ML Engine** | scikit-learn, pandas, numpy | Machine learning pipelines |
| **Task Queue** | Celery, Redis | Asynchronous task processing |
| **Database** | PostgreSQL 15 (NeonDB) | Persistent data storage |
| **Cache** | Redis 7 | Caching & message broker |
| **Container** | Docker, Docker Compose | Containerization & orchestration |
| **CI/CD** | GitHub Actions | Automated builds & testing |
| **Monitoring** | Prometheus, AlertManager | Metrics & alerting |
| **Logging** | ELK Stack (Elasticsearch, Logstash, Kibana) | Centralized logging |

### Design Principles

1. **Containerization**: All services run in Docker containers for consistency
2. **Statelessness**: Backend and frontend are stateless for horizontal scaling
3. **Separation of Concerns**: Clear boundaries between services
4. **Async Processing**: Heavy ML tasks run asynchronously via Celery
5. **Observability**: Comprehensive logging, metrics, and monitoring
6. **Security**: Multi-layer security with secrets management
7. **Resilience**: Health checks, auto-restart, and error recovery
8. **Developer Experience**: Hot reload, easy setup, comprehensive documentation

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           Internet                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────▼───────────┐
                │   Nginx (Port 80)      │
                │   Frontend Container   │
                └────────────┬───────────┘
                             │
                ┌────────────▼───────────┐
                │  FastAPI (Port 8000)   │
                │   Backend Container    │
                └─────┬──────────────┬───┘
                      │              │
        ┌─────────────▼───┐    ┌────▼──────────────┐
        │  Redis (6379)   │    │  NeonDB           │
        │  Cache/Broker   │    │  PostgreSQL       │
        └─────┬───────────┘    │  (External)       │
              │                └───────────────────┘
        ┌─────▼───────────┐
        │  Celery Worker  │
        │  ML Processing  │
        └─────────────────┘
```

### Service Communication

```
User Browser → Nginx (Frontend)
              ↓
        FastAPI Backend ← → NeonDB (PostgreSQL)
              ↓
            Redis ← → Celery Workers
              ↓
        Task Results
```

### Container Network

All services communicate via Docker bridge network: `aiplayground-network`

**Service Discovery**: Docker DNS resolves service names to container IPs
- `redis` → Redis container
- `backend` → Backend API container
- `celery-worker` → Celery worker container
- `frontend` → Nginx frontend container

---

## Core Services

### 1. Frontend Service

**Container**: `aiplayground-frontend`  
**Image**: Custom (built from `frontend/Dockerfile`)  
**Port**: 80  
**Base**: nginx:alpine + Node 20-alpine (build stage)

**Responsibilities**:
- Serve React SPA
- Handle routing (SPA mode)
- Proxy API requests to backend
- Static asset serving

**Key Features**:
- Multi-stage build (95% size reduction)
- Vite build optimization
- Gzip compression
- HTTP/2 support
- Health check endpoint: `/health`

**Resource Limits**:
- Memory: 256 MB
- CPU: 0.5 cores

**Build Process**:
```bash
# Stage 1: Build React app (Node 20-alpine)
npm ci → npm run build → /app/dist

# Stage 2: Serve with Nginx (nginx:alpine)
Copy dist → Configure nginx → Final image (~25MB)
```

### 2. Backend API Service

**Container**: `aiplayground-backend`  
**Image**: Custom (built from `backend/Dockerfile`)  
**Port**: 8000  
**Base**: python:3.11-slim

**Responsibilities**:
- REST API endpoints
- Request validation & authentication
- Business logic orchestration
- ML pipeline coordination
- Database operations
- Celery task dispatching

**Key Features**:
- Multi-stage build (38% size reduction)
- Hot reload in development
- Health checks
- OpenAPI documentation (Swagger/ReDoc)
- Non-root user execution
- Alembic database migrations

**Resource Limits**:
- Memory: 1 GB
- CPU: 1 core
- Workers: 4 (production), 1 (development)

**Endpoints**:
- `/health` - Health check
- `/docs` - OpenAPI documentation
- `/redoc` - ReDoc documentation
- `/api/v1/*` - API endpoints

### 3. Celery Worker Service

**Container**: `aiplayground-celery-worker`  
**Image**: Same as backend  
**Port**: None (internal)

**Responsibilities**:
- Asynchronous ML training
- Data preprocessing
- Model evaluation
- Hyperparameter tuning
- Feature selection
- Long-running computations

**Key Features**:
- Shares codebase with backend
- Configurable concurrency (default: 2)
- Auto-retry on failure
- Result storage in Redis
- Task monitoring support

**Resource Limits**:
- Memory: 2 GB
- CPU: 2 cores
- Concurrency: 2 workers

**Task Types**:
- `train_model_task` - Model training
- `preprocess_data_task` - Data preprocessing
- `evaluate_model_task` - Model evaluation
- `tune_hyperparameters_task` - Hyperparameter optimization
- `select_features_task` - Feature selection

### 4. Redis Service

**Container**: `aiplayground-redis`  
**Image**: redis:7-alpine  
**Port**: 6379  

**Responsibilities**:
- Celery message broker
- Task result backend
- Application caching
- Session storage

**Key Features**:
- LRU eviction policy
- RDB persistence (snapshots)
- Health checks
- Custom configuration

**Configuration**:
- Max Memory: 256 MB (configurable to 1 GB)
- Eviction: allkeys-lru
- Persistence: RDB snapshots (900s, 300s, 60s)
- AOF: Disabled (development) / Enabled (production)

**Resource Limits**:
- Memory: 256 MB (dev) / 1 GB (prod)
- CPU: 0.5 cores

---

## Infrastructure Components

### Database (NeonDB)

**Type**: Serverless PostgreSQL  
**Provider**: Neon.tech  
**Version**: PostgreSQL 15  
**Connection**: SSL/TLS required

**Features**:
- Serverless auto-scaling
- Point-in-time restore
- Branching support
- Automatic backups
- Connection pooling

**Connection String Format**:
```
postgresql://user:password@ep-xxx-xxx.region.aws.neon.tech/aiplayground?sslmode=require
```

**Schema Management**:
- Migrations: Alembic
- Version Control: Git
- Auto-migration: On backend startup

**Tables**:
- `users` - User accounts
- `datasets` - Uploaded datasets
- `preprocessings` - Preprocessing configurations
- `models` - Trained ML models
- `evaluations` - Model evaluation results
- `hyperparameter_tunings` - Tuning configurations

### Docker Orchestration

**Tool**: Docker Compose v3.8  
**File**: `docker-compose.yml`

**Features**:
- Multi-service orchestration
- Dependency management
- Health checks
- Auto-restart policies
- Volume management
- Network isolation

**Compose Files**:
- `docker-compose.yml` - Main services
- `docker-compose.dev.yml` - Development overrides
- `docker-compose.production.yml` - Production overrides
- `docker-compose.logging.yml` - ELK stack
- `docker-compose.alerting.yml` - Prometheus + AlertManager

### Build System

**Scripts**:
- `docker/build/docker-build.sh` - Linux/Mac build script
- `docker/build/docker-build.ps1` - Windows PowerShell script
- `Makefile` - Convenient build commands

**Features**:
- Multi-platform builds (amd64, arm64)
- Environment selection (dev, prod)
- Registry push support
- Cache management
- Custom tagging
- Build validation

**Build Targets**:
- Backend: Production (~500MB), Development (~800MB)
- Frontend: Production (~25MB), Development (~500MB)

### CI/CD Pipeline

**Platform**: GitHub Actions  
**Workflow**: `.github/workflows/docker-build.yml`

**Triggers**:
- Push to `main` or `develop`
- Pull requests
- Manual workflow dispatch
- Git tags (`v*`)

**Stages**:
1. **Checkout** - Clone repository
2. **Setup** - Configure Docker Buildx
3. **Build** - Multi-platform builds
4. **Test** - Integration tests
5. **Scan** - Security scanning (Trivy)
6. **Push** - Upload to GHCR
7. **Notify** - Status notifications

**Build Matrix**:
- Services: backend, frontend
- Platforms: linux/amd64, linux/arm64
- Environments: dev, prod

---

## Networking

### Network Architecture

**Network Name**: `aiplayground-network`  
**Type**: Bridge  
**Driver**: bridge  

**Subnet**: Auto-assigned by Docker  
**DNS**: Docker internal DNS

### Port Mapping

| Service | Internal Port | External Port | Protocol |
|---------|--------------|---------------|----------|
| Frontend | 80 | 80 | HTTP |
| Backend | 8000 | 8000 | HTTP |
| Redis | 6379 | 6379 | TCP |
| Celery | - | - | Internal |

### Service URLs

**Development**:
- Frontend: http://localhost
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

**Internal Communication**:
- Backend → Redis: `redis://redis:6379/0`
- Backend → NeonDB: `postgresql://...neon.tech/aiplayground`
- Celery → Redis: `redis://redis:6379/0`
- Frontend → Backend: `http://backend:8000`

### Firewall Rules

**Development** (Localhost only):
- Allow: 80, 8000, 6379 from localhost
- Deny: All from external networks

**Production**:
- Allow: 80, 443 from internet (via reverse proxy)
- Allow: 8000 from internal network only
- Deny: 6379 from external networks
- Restrict: Database access to backend only

---

## Data Storage

### Persistent Volumes

| Volume | Purpose | Size | Backup |
|--------|---------|------|--------|
| `redis_data` | Redis persistence | 1 GB | Daily |
| `backend_uploads` | Uploaded datasets | 10 GB | Daily |
| `backend_logs` | Application logs | 5 GB | Weekly |

**Volume Locations** (Docker managed):
- Linux: `/var/lib/docker/volumes/`
- Windows: `C:\ProgramData\Docker\volumes\`

### Data Lifecycle

**Uploaded Files**:
1. User uploads CSV/JSON
2. Saved to `backend_uploads` volume
3. Processed and stored in database
4. Original file retained for 30 days
5. Automatic cleanup of old files

**Logs**:
1. Application logs to `backend_logs` volume
2. Rotated daily (keep 7 days)
3. Compressed after rotation
4. Exported to ELK stack (if enabled)

**Redis Data**:
1. In-memory with RDB snapshots
2. Snapshots saved to `redis_data` volume
3. Automatic snapshots (15min, 5min, 1min)
4. Restore on container restart

### Backup Strategy

**Database (NeonDB)**:
- Automatic backups by Neon
- Point-in-time restore (30 days)
- Manual snapshots via Neon console

**Volumes**:
```bash
# Backup script
docker run --rm -v backend_uploads:/data -v $(pwd):/backup \
  alpine tar czf /backup/uploads-$(date +%Y%m%d).tar.gz /data
```

**Frequency**:
- Database: Continuous (Neon)
- Uploads: Daily at 2 AM
- Logs: Weekly
- Redis: On-demand before maintenance

---

## Build & Deployment

### Local Development Setup

1. **Prerequisites**:
   ```bash
   # Install Docker & Docker Compose
   # Clone repository
   git clone https://github.com/yourusername/AI-Playground.git
   cd AI-Playground
   ```

2. **Environment Configuration**:
   ```bash
   # Copy environment template
   cp .env.example .env.docker
   
   # Set DATABASE_URL
   export DATABASE_URL="postgresql://user:pass@host.neon.tech/aiplayground?sslmode=require"
   ```

3. **Build & Start**:
   ```bash
   # Option 1: Using docker-compose
   docker-compose up -d
   
   # Option 2: Using Makefile
   make build
   docker-compose up -d
   
   # Option 3: Using build scripts
   ./docker/build/docker-build.sh --service all --env dev
   docker-compose up -d
   ```

4. **Verify**:
   ```bash
   docker-compose ps
   curl http://localhost:8000/health
   curl http://localhost/health
   ```

### Production Deployment

#### Option 1: Docker Compose (Single Server)

1. **Server Setup**:
   ```bash
   # Update system
   apt update && apt upgrade -y
   
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   
   # Install Docker Compose
   apt install docker-compose-plugin
   ```

2. **Deploy Application**:
   ```bash
   # Clone repository
   git clone https://github.com/yourusername/AI-Playground.git
   cd AI-Playground
   
   # Configure environment
   cp .env.example .env.docker
   nano .env.docker  # Set production values
   
   # Build and start
   docker-compose -f docker-compose.yml -f docker-compose.production.yml up -d
   ```

3. **Configure Reverse Proxy** (Nginx/Caddy):
   ```nginx
   server {
       listen 80;
       server_name aiplayground.example.com;
       
       location / {
           proxy_pass http://localhost:80;
       }
       
       location /api/ {
           proxy_pass http://localhost:8000;
       }
   }
   ```

#### Option 2: Cloud Deployment (AWS/GCP/Azure)

1. **Container Registry**:
   ```bash
   # Push to GHCR
   ./docker/build/docker-build.sh --push --registry ghcr.io --tag v1.0.0
   ```

2. **Deploy to Cloud**:
   - **AWS ECS**: Use task definitions with Docker images
   - **GCP Cloud Run**: Deploy containers directly
   - **Azure Container Instances**: Deploy via Azure CLI

3. **Configure Load Balancer**:
   - Route traffic to frontend container
   - Health checks on `/health`
   - SSL/TLS termination

#### Option 3: Kubernetes

See `docs/KUBERNETES_DEPLOYMENT.md` (future)

### Deployment Checklist

- [ ] Environment variables configured
- [ ] Database connection tested
- [ ] Secrets properly managed
- [ ] SSL certificates installed
- [ ] Firewall rules configured
- [ ] Monitoring enabled
- [ ] Backups configured
- [ ] Health checks passing
- [ ] Logs collection enabled
- [ ] Error tracking configured

---

## Monitoring & Logging

### Monitoring Stack

**Components**:
- **Prometheus**: Metrics collection
- **AlertManager**: Alert routing
- **Grafana**: Visualization (optional)

**Setup**:
```bash
docker-compose -f docker-compose.alerting.yml up -d
```

**Access**:
- Prometheus: http://localhost:9090
- AlertManager: http://localhost:9093

**Metrics Collected**:
- Request rate and latency
- Error rates
- CPU and memory usage
- Task queue length
- Task success/failure rates
- Database connection pool
- Redis memory usage

### Logging Stack

**Components**:
- **Elasticsearch**: Log storage
- **Logstash**: Log processing
- **Kibana**: Log visualization

**Setup**:
```bash
docker-compose -f docker-compose.logging.yml up -d
```

**Access**:
- Kibana: http://localhost:5601

**Log Sources**:
- Backend application logs
- Nginx access/error logs
- Celery worker logs
- Redis logs

**Log Retention**:
- Hot data: 7 days
- Warm data: 30 days
- Archive: 90 days

### Health Checks

**Backend** (`/health`):
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database": "connected",
  "redis": "connected",
  "uptime": 3600
}
```

**Frontend** (`/health`):
```json
{
  "status": "healthy",
  "service": "frontend"
}
```

**Redis**:
```bash
docker-compose exec redis redis-cli ping
# Response: PONG
```

### Alerting Rules

**Critical Alerts**:
- Service down (5xx errors > 10% for 5 minutes)
- Database connection lost
- Redis unavailable
- Disk usage > 90%
- Memory usage > 90%

**Warning Alerts**:
- Response time > 2s (p95)
- Error rate > 5%
- Task queue > 100 pending
- CPU usage > 80%

**Notification Channels**:
- Email: alerts@example.com
- Slack: #alerts channel
- PagerDuty: Critical only

---

## Security

### Application Security

**Authentication**:
- JWT tokens (HS256 algorithm)
- Token expiration: 30 minutes
- Refresh tokens: 7 days
- Secure cookie storage

**Authorization**:
- Role-based access control (RBAC)
- API endpoint permissions
- Resource ownership validation

**Input Validation**:
- Pydantic models for request validation
- SQL injection prevention (SQLAlchemy ORM)
- XSS protection
- CSRF tokens

### Container Security

**Base Images**:
- Official images only
- Pinned versions (no `latest`)
- Slim/Alpine variants
- Regular updates

**Non-Root Users**:
- Backend: runs as `appuser` (uid 1000)
- Frontend: runs as nginx user
- No privileged containers

**Image Scanning**:
- Trivy security scanning in CI/CD
- Vulnerability database updates
- Block builds with critical CVEs

### Network Security

**Firewall**:
- Only necessary ports exposed
- Internal services isolated
- No direct database access from internet

**SSL/TLS**:
- NeonDB: SSL required
- Production frontend: HTTPS only
- Certificate management: Let's Encrypt

**Secrets Management**:
- Environment variables for secrets
- No secrets in code or Dockerfiles
- `.env` files in `.gitignore`
- Consider: HashiCorp Vault for production

### Data Security

**Encryption**:
- Data in transit: TLS 1.2+
- Data at rest: NeonDB encryption
- Backup encryption: enabled

**Access Control**:
- Database: minimum required privileges
- Redis: password authentication (production)
- File uploads: size and type validation

**Audit Logging**:
- All authentication attempts
- Data access and modifications
- Administrative actions
- Failed authorization attempts

---

## Scalability

### Horizontal Scaling

**Frontend**:
- Stateless design enables easy scaling
- Deploy multiple containers behind load balancer
- No session storage in containers
- CDN for static assets

**Backend**:
- Stateless API design
- Scale to N instances
- Load balancer distributes requests
- Database connection pooling

**Celery Workers**:
- Add workers for more throughput
- Task distribution via Redis
- Each worker handles 2 concurrent tasks
- Scale based on queue length

**Redis**:
- Single instance sufficient for moderate load
- Redis Cluster for high availability
- Sentinel for automatic failover

### Vertical Scaling

**Current Limits**:
- Backend: 1 CPU, 1 GB RAM
- Worker: 2 CPU, 2 GB RAM
- Redis: 0.5 CPU, 256 MB RAM
- Frontend: 0.5 CPU, 256 MB RAM

**Scaling Up**:
```yaml
# docker-compose.yml
backend:
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 4G
```

### Auto-Scaling

**Kubernetes** (future):
- Horizontal Pod Autoscaler (HPA)
- Metrics: CPU, memory, request rate
- Min replicas: 2
- Max replicas: 10

**Cloud Platforms**:
- AWS ECS: Target tracking scaling
- GCP Cloud Run: Automatic instance management
- Azure: Container scaling rules

### Database Scaling

**NeonDB**:
- Automatic scaling included
- Compute scales with load
- Storage scales automatically
- Connection pooling built-in

**Read Replicas**:
- Consider for read-heavy workloads
- Route read queries to replicas
- Write to primary only

### Performance Optimization

**Caching Strategy**:
- API responses: 5-60 minutes
- ML predictions: 24 hours
- Static assets: 1 year
- Database queries: selective caching

**Code Optimization**:
- Async I/O operations
- Database query optimization
- Pagination for large datasets
- Lazy loading

---

## Disaster Recovery

### Backup Strategy

**Database**:
- Provider: NeonDB automatic backups
- Frequency: Continuous
- Retention: 30 days
- Recovery Point Objective (RPO): < 1 minute
- Recovery Time Objective (RTO): < 5 minutes

**Application Data**:
- Uploads volume: Daily backups
- Logs volume: Weekly backups
- Configuration: Git repository

### Recovery Procedures

**Database Recovery**:
```bash
# Point-in-time restore via Neon console
# 1. Go to NeonDB console
# 2. Select project
# 3. Choose "Restore" > "Point in time"
# 4. Select timestamp
# 5. Create new branch or restore to existing
```

**Volume Recovery**:
```bash
# Restore uploads
docker volume create backend_uploads
docker run --rm -v backend_uploads:/data -v $(pwd):/backup \
  alpine tar xzf /backup/uploads-20260102.tar.gz -C /
```

**Full System Recovery**:
1. Provision new server
2. Install Docker & Docker Compose
3. Clone repository
4. Restore database from backup
5. Restore volumes from backups
6. Configure environment variables
7. Start services: `docker-compose up -d`
8. Verify health checks
9. Update DNS records

### High Availability

**Current Setup**: Single server (development)

**Production HA** (recommended):
- Load balancer: 2+ frontend instances
- API servers: 3+ backend instances
- Workers: 4+ Celery workers
- Redis: Redis Sentinel (3 nodes)
- Database: NeonDB (built-in HA)

**Availability Targets**:
- Development: 95% (best effort)
- Production: 99.9% (3-nines)
- Enterprise: 99.99% (4-nines)

### Monitoring & Alerting

**Critical Alerts**:
- Immediate notification: PagerDuty
- Response time: < 15 minutes
- Escalation: After 30 minutes

**Health Checks**:
- Frequency: Every 30 seconds
- Timeout: 5 seconds
- Retries: 3 attempts
- Action: Auto-restart container

---

## Environments

### Development

**Purpose**: Local development and testing

**Configuration**:
- Debug mode: Enabled
- Hot reload: Enabled
- Authentication: Optional
- Database: NeonDB (branch)
- Redis: No persistence

**Access**:
- Frontend: http://localhost
- Backend: http://localhost:8000
- Docs: http://localhost:8000/docs

**Resource Usage**: Minimal (< 2 GB RAM)

### Staging

**Purpose**: Pre-production testing

**Configuration**:
- Debug mode: Disabled
- Hot reload: Disabled
- Authentication: Required
- Database: NeonDB (staging branch)
- Redis: RDB persistence

**Access**:
- Frontend: https://staging.example.com
- Backend: https://staging-api.example.com

**Resource Usage**: Moderate (4 GB RAM)

### Production

**Purpose**: Live user-facing application

**Configuration**:
- Debug mode: Disabled
- Hot reload: Disabled
- Authentication: Required
- Database: NeonDB (main branch)
- Redis: RDB + AOF persistence
- SSL: Required
- Monitoring: Enabled
- Alerting: Enabled

**Access**:
- Frontend: https://aiplayground.example.com
- Backend: https://api.aiplayground.example.com

**Resource Usage**: Scalable (8+ GB RAM)

### Environment Variables

**Common Variables**:
```bash
# Database
DATABASE_URL=postgresql://...

# Redis
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# API
API_V1_PREFIX=/api/v1
PROJECT_NAME=AI-Playground
VERSION=1.0.0

# Security
SECRET_KEY=<strong-secret-key>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Environment
ENVIRONMENT=development|staging|production
DEBUG=True|False
```

**Environment-Specific**:
```bash
# Development
DEBUG=True
CORS_ORIGINS=http://localhost:5173,http://localhost

# Production
DEBUG=False
CORS_ORIGINS=https://aiplayground.example.com
SENTRY_DSN=https://...
```

---

## Cost Considerations

### Current Costs (Monthly Estimates)

**Development** (Localhost):
- Infrastructure: $0
- NeonDB: $0 (Free tier: 1 project)
- Total: **$0/month**

**Production** (Cloud Hosting):

| Component | Service | Cost |
|-----------|---------|------|
| Compute | DigitalOcean Droplet (4GB) | $24 |
| Database | NeonDB Pro | $19 |
| Storage | Block Storage (50GB) | $5 |
| Bandwidth | 1TB included | $0 |
| Monitoring | Prometheus (self-hosted) | $0 |
| **Total** | | **$48/month** |

### Scaling Costs

**100 Users/Day**:
- Current setup sufficient
- Cost: $48/month

**1,000 Users/Day**:
- Upgrade to 8GB droplet: $48/month
- NeonDB Scale: $69/month
- Total: **$117/month**

**10,000 Users/Day**:
- Load balanced (2x 8GB): $96/month
- NeonDB Scale: $229/month
- CDN (Cloudflare): $20/month
- Total: **$345/month**

### Cost Optimization

**Strategies**:
1. Use NeonDB free tier for development
2. Optimize Docker images (smaller = faster)
3. Implement caching (reduce database load)
4. Use CDN for static assets
5. Right-size compute resources
6. Schedule scaling (scale down at night)
7. Reserved instances for predictable load
8. Monitor and eliminate waste

---

## Operations & Maintenance

### Daily Operations

**Morning Checks** (5 minutes):
```bash
# Check service health
docker-compose ps
curl http://localhost:8000/health

# Check logs for errors
docker-compose logs --tail=100 backend | grep ERROR

# Check resource usage
docker stats --no-stream
```

**Monitoring**:
- Dashboard review: Prometheus/Grafana
- Alert review: AlertManager
- Log review: Kibana (if errors reported)

### Weekly Maintenance

**Tasks**:
1. Review metrics and trends
2. Check disk usage: `docker system df`
3. Clean old images: `docker system prune -a`
4. Update dependencies (security patches)
5. Review and archive old logs
6. Backup verification
7. Performance tuning if needed

**Scripts**:
```bash
# Weekly maintenance script
#!/bin/bash
echo "=== Weekly Maintenance ==="

# System cleanup
docker system prune -f
docker volume prune -f

# Backup uploads
./scripts/backup/backup-volumes.sh

# Check for updates
docker-compose pull
docker images --filter "dangling=true"

echo "=== Maintenance Complete ==="
```

### Monthly Maintenance

**Tasks**:
1. Security updates: OS and packages
2. Database maintenance: vacuum, analyze
3. Review and optimize queries
4. Capacity planning review
5. Cost analysis and optimization
6. Documentation updates
7. Disaster recovery test

### Update Procedures

**Application Updates**:
```bash
# 1. Pull latest code
git pull origin main

# 2. Rebuild images
docker-compose build --no-cache

# 3. Restart services
docker-compose down
docker-compose up -d

# 4. Verify health
curl http://localhost:8000/health
```

**Dependency Updates**:
```bash
# Backend Python packages
cd backend
pip install -U -r requirements.txt
pip freeze > requirements.txt

# Frontend npm packages
cd frontend
npm update
npm audit fix
```

**Database Migrations**:
```bash
# Create migration
docker-compose exec backend alembic revision --autogenerate -m "description"

# Review migration file
# Edit if needed

# Apply migration
docker-compose exec backend alembic upgrade head

# Verify
docker-compose exec backend alembic current
```

### Rollback Procedures

**Application Rollback**:
```bash
# 1. Stop services
docker-compose down

# 2. Revert code
git checkout <previous-commit>

# 3. Rebuild (if needed)
docker-compose build

# 4. Start services
docker-compose up -d

# 5. Verify
curl http://localhost:8000/health
```

**Database Rollback**:
```bash
# Rollback one migration
docker-compose exec backend alembic downgrade -1

# Rollback to specific version
docker-compose exec backend alembic downgrade <revision>

# Nuclear option: restore from backup
# Use NeonDB point-in-time restore
```

---

## Troubleshooting

### Common Issues

#### Issue: Backend Won't Start

**Symptoms**: Backend container exits immediately

**Diagnosis**:
```bash
docker-compose logs backend
```

**Solutions**:
1. Check DATABASE_URL is set
2. Verify NeonDB connection
3. Check for port conflicts
4. Review environment variables
5. Check disk space

#### Issue: High Memory Usage

**Symptoms**: Containers consuming too much RAM

**Diagnosis**:
```bash
docker stats
docker-compose exec backend ps aux
```

**Solutions**:
1. Check for memory leaks in code
2. Reduce worker concurrency
3. Increase container memory limits
4. Restart containers
5. Scale horizontally instead of vertically

#### Issue: Slow API Response

**Symptoms**: Requests taking > 2 seconds

**Diagnosis**:
```bash
# Check database queries
docker-compose logs backend | grep "slow query"

# Check Redis
docker-compose exec redis redis-cli INFO stats

# Check system resources
docker stats
```

**Solutions**:
1. Add database indexes
2. Implement caching
3. Optimize queries
4. Scale workers
5. Add Redis persistence

#### Issue: Celery Tasks Failing

**Symptoms**: Tasks stuck or failing

**Diagnosis**:
```bash
docker-compose logs celery-worker
docker-compose exec celery-worker celery -A celery_worker.celery_app inspect active
```

**Solutions**:
1. Check Redis connection
2. Review task code for errors
3. Increase worker timeout
4. Check resource limits
5. Purge stuck tasks

### Debug Commands

**Service Status**:
```bash
docker-compose ps
docker-compose top
```

**Logs**:
```bash
docker-compose logs -f backend
docker-compose logs --tail=100 celery-worker
docker-compose logs --since 30m redis
```

**Container Shell**:
```bash
docker-compose exec backend bash
docker-compose exec redis redis-cli
```

**Network Testing**:
```bash
docker-compose exec backend ping redis
docker-compose exec backend curl http://frontend
```

**Database Testing**:
```bash
docker-compose exec backend python -c "
from sqlalchemy import create_engine
import os
engine = create_engine(os.getenv('DATABASE_URL'))
print(engine.connect())
"
```

### Emergency Procedures

**Total System Failure**:
```bash
# 1. Stop everything
docker-compose down

# 2. Clean system
docker system prune -a --volumes

# 3. Restore from backup
# Follow disaster recovery procedures

# 4. Rebuild and restart
docker-compose build --no-cache
docker-compose up -d

# 5. Verify
./docker/build/health-check.ps1
```

**Database Emergency**:
```bash
# Use NeonDB console for:
# - Point-in-time restore
# - Create new branch
# - Restore from snapshot
```

---

## References

### Documentation

**Project Documentation**:
- [Main README](../README.md) - Project overview
- [Docker README](../docker/README.md) - Docker comprehensive guide
- [Backend README](../backend/README.md) - API documentation
- [Frontend README](../frontend/README.md) - Frontend documentation
- [Build Quick Start](../docker/build/QUICK_START.md) - Build system

**Operational Guides**:
- [Deployment Guide](../docker/DEPLOYMENT_GUIDE.md) - Production deployment
- [Monitoring Guide](../docker/MONITORING_QUICK_START.md) - Monitoring setup
- [Logging Setup](../docker/logging/README.md) - ELK stack configuration

### External Resources

**Technologies**:
- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Redis Documentation](https://redis.io/docs/)
- [Celery Documentation](https://docs.celeryq.dev/)
- [NeonDB Documentation](https://neon.tech/docs/)

**Best Practices**:
- [12-Factor App](https://12factor.net/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices)

### Support

**Getting Help**:
- GitHub Issues: [github.com/yourusername/AI-Playground/issues](https://github.com)
- Documentation: This file and linked docs
- Community: Discord/Slack (if applicable)

**Reporting Issues**:
1. Check documentation first
2. Search existing issues
3. Provide reproduction steps
4. Include logs and screenshots
5. Specify environment details

---

## Appendix

### Architecture Decisions

**ADR-001**: Use NeonDB instead of self-hosted PostgreSQL
- Reason: Serverless, auto-scaling, automatic backups
- Trade-off: External dependency, monthly cost
- Status: Accepted

**ADR-002**: Use Docker Compose instead of Kubernetes
- Reason: Simpler for small to medium deployments
- Trade-off: Limited orchestration features
- Status: Accepted (revisit at scale)

**ADR-003**: Use Celery with Redis broker
- Reason: Proven, reliable, Python-native
- Trade-off: Another service to manage
- Status: Accepted

### Glossary

- **Container**: Isolated runtime environment
- **Image**: Template for creating containers
- **Volume**: Persistent data storage
- **Network**: Communication layer between containers
- **Service**: Logical grouping of containers
- **Stack**: Complete application (all services)
- **Orchestration**: Automated deployment and scaling
- **Health Check**: Automated service verification
- **Reverse Proxy**: Routes external requests to internal services

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-02 | Initial infrastructure documentation |

---

**Document Maintained By**: Infrastructure Team  
**Last Review**: January 2, 2026  
**Next Review**: April 2, 2026
