# 🚀 AI-Playground Backend - Complete Deployment Guide

**Version:** 1.0.0  
**Last Updated:** January 2, 2026  
**Maintainer:** AI-Playground Team

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Deployment Options](#deployment-options)
4. [Quick Start](#quick-start)
5. [Production Deployment](#production-deployment)
6. [Cloud Platform Guides](#cloud-platform-guides)
7. [Configuration](#configuration)
8. [Database Setup](#database-setup)
9. [Monitoring & Logging](#monitoring--logging)
10. [Security Best Practices](#security-best-practices)
11. [Troubleshooting](#troubleshooting)
12. [Maintenance](#maintenance)

---

## 🎯 Overview

This guide covers deploying the AI-Playground backend API to various environments:

- **Local Development** - Docker Compose with all services
- **Production** - Docker with external managed services
- **Cloud Platforms** - Render, Railway, DigitalOcean, AWS, GCP, Azure
- **Kubernetes** - Scalable container orchestration

### Architecture Components

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer                        │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐       ┌───────▼────────┐
│  FastAPI App   │       │  FastAPI App   │
│   (Backend)    │       │   (Backend)    │
└───────┬────────┘       └───────┬────────┘
        │                         │
        └────────────┬────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐       ┌───────▼────────┐
│   PostgreSQL   │       │     Redis      │
│   (Database)   │       │    (Cache)     │
└────────────────┘       └────────────────┘
        │
┌───────▼────────┐
│ Celery Workers │
│  (Async Tasks) │
└────────────────┘
```


---

## 📦 Prerequisites

### Required Services

| Service | Purpose | Recommended Providers |
|---------|---------|----------------------|
| **PostgreSQL 15+** | Primary database | NeonDB, Supabase, AWS RDS, DigitalOcean |
| **Redis 7+** | Cache & message broker | Upstash, Redis Cloud, AWS ElastiCache |
| **Object Storage** | File uploads (optional) | Cloudflare R2, AWS S3, DigitalOcean Spaces |

### Required Tools

- **Docker** 24.0+ and Docker Compose 2.0+
- **Python** 3.11+ (for local development)
- **Git** for version control
- **OpenSSL** for generating secrets

### System Requirements

**Minimum (Development):**
- 2 CPU cores
- 4GB RAM
- 10GB disk space

**Recommended (Production):**
- 4+ CPU cores
- 8GB+ RAM
- 50GB+ disk space
- SSD storage

---

## 🎯 Deployment Options

### Option 1: Docker Compose (Recommended for Getting Started)

**Best for:** Local development, small teams, testing

**Pros:**
- ✅ All services in one place
- ✅ Easy to set up and tear down
- ✅ Consistent across environments
- ✅ Includes monitoring tools

**Cons:**
- ❌ Not suitable for high-traffic production
- ❌ Single point of failure
- ❌ Limited scalability

**Setup Time:** 10 minutes

### Option 2: Cloud Platform (Recommended for Production)

**Best for:** Production deployments, scalability, managed services

**Pros:**
- ✅ Managed infrastructure
- ✅ Auto-scaling
- ✅ High availability
- ✅ Built-in monitoring

**Cons:**
- ❌ Monthly costs
- ❌ Platform lock-in
- ❌ Learning curve

**Setup Time:** 30-60 minutes

### Option 3: Kubernetes

**Best for:** Large-scale deployments, microservices, enterprise

**Pros:**
- ✅ Highly scalable
- ✅ Self-healing
- ✅ Platform-agnostic
- ✅ Advanced orchestration

**Cons:**
- ❌ Complex setup
- ❌ Requires DevOps expertise
- ❌ Higher operational overhead

**Setup Time:** 2-4 hours


---

## 🚀 Quick Start

### Local Development with Docker Compose

**Step 1: Clone Repository**
```bash
git clone https://github.com/your-org/AI-Playground.git
cd AI-Playground
```

**Step 2: Configure Environment**
```bash
# Copy environment template
cp .env.docker .env

# Edit .env with your settings (optional for development)
# Default values work out of the box
```

**Step 3: Start Services**
```bash
# Start all services (PostgreSQL, Redis, Backend, Celery, Frontend)
docker-compose up -d

# View logs
docker-compose logs -f backend

# Check status
docker-compose ps
```

**Step 4: Verify Deployment**
```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs

# Frontend
open http://localhost
```

**Step 5: Initialize Database**
```bash
# Run migrations
docker-compose exec backend alembic upgrade head

# Create test user (optional)
docker-compose exec backend python init_db.py
```

✅ **Success!** Your development environment is ready.

### Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```


---

## 🏭 Production Deployment

### Step 1: Set Up External Services

#### 1.1 PostgreSQL Database (NeonDB Recommended)

**Why NeonDB?**
- ✅ Serverless PostgreSQL
- ✅ Free tier available
- ✅ Auto-scaling
- ✅ Built-in connection pooling

**Setup:**
1. Go to [neon.tech](https://neon.tech)
2. Create account and new project
3. Copy connection string:
   ```
   postgresql://user:password@ep-xxx.neon.tech/aiplayground?sslmode=require
   ```

**Alternative Providers:**
- **Supabase** - Free tier, includes auth
- **AWS RDS** - Enterprise-grade, expensive
- **DigitalOcean Managed Database** - Simple, affordable
- **Google Cloud SQL** - Good for GCP deployments

#### 1.2 Redis Cache (Upstash Recommended)

**Why Upstash?**
- ✅ Serverless Redis
- ✅ Free tier (10k commands/day)
- ✅ Global edge network
- ✅ REST API support

**Setup:**
1. Go to [upstash.com](https://upstash.com)
2. Create Redis database
3. Copy connection string:
   ```
   rediss://default:xxxxx@us1-xxx.upstash.io:6379
   ```

**Alternative Providers:**
- **Redis Cloud** - Official Redis hosting
- **AWS ElastiCache** - Managed Redis on AWS
- **DigitalOcean Managed Redis** - Simple setup

#### 1.3 Object Storage (Optional - Cloudflare R2 Recommended)

**Why Cloudflare R2?**
- ✅ S3-compatible API
- ✅ No egress fees
- ✅ 10GB free storage
- ✅ Fast global CDN

**Setup:**
1. Go to Cloudflare Dashboard → R2
2. Create bucket: `aiplayground-storage`
3. Create API token with read/write permissions
4. Note: Account ID, Access Key, Secret Key

**Alternative Providers:**
- **AWS S3** - Industry standard, expensive egress
- **DigitalOcean Spaces** - S3-compatible, affordable
- **Backblaze B2** - Cheapest option

### Step 2: Generate Secrets

```bash
# Generate SECRET_KEY (32+ characters)
openssl rand -hex 32

# Output example:
# a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6

# Save this for environment variables
```

### Step 3: Configure Environment Variables

Create `.env.production` file:

```bash
# Database
DATABASE_URL=postgresql://user:password@ep-xxx.neon.tech/aiplayground?sslmode=require

# Redis
REDIS_URL=rediss://default:xxxxx@us1-xxx.upstash.io:6379
CELERY_BROKER_URL=${REDIS_URL}
CELERY_RESULT_BACKEND=${REDIS_URL}

# Security
SECRET_KEY=<your-generated-secret-key>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# API Configuration
API_V1_PREFIX=/api/v1
PROJECT_NAME=AI-Playground
VERSION=1.0.0

# Environment
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO

# File Storage (if using R2)
R2_ACCOUNT_ID=your-account-id
R2_ACCESS_KEY_ID=your-access-key
R2_SECRET_ACCESS_KEY=your-secret-key
R2_BUCKET_NAME=aiplayground-storage
R2_PUBLIC_URL=https://pub-xxxxx.r2.dev

# CORS (update with your frontend URL)
ALLOWED_ORIGINS=https://your-frontend.vercel.app,https://your-domain.com

# Limits
MAX_UPLOAD_SIZE=104857600  # 100MB
UPLOAD_DIR=/app/uploads

# Monitoring (optional)
SENTRY_DSN=https://xxxxx@sentry.io/xxxxx
```


### Step 4: Build Production Docker Image

```bash
cd backend

# Build with full dependencies
docker build -t aiplayground-backend:latest .

# Or build with minimal dependencies (for memory-constrained environments)
docker build --build-arg REQUIREMENTS_FILE=requirements.render.txt \
  -t aiplayground-backend:lite .

# Test locally
docker run -p 8000:8000 --env-file .env.production aiplayground-backend:latest

# Verify
curl http://localhost:8000/health
```

### Step 5: Deploy to Production

Choose your deployment method:

#### Option A: Docker Compose (Simple Production)

```bash
# Use production docker-compose
docker-compose -f docker-compose.yml up -d

# Run migrations
docker-compose exec backend alembic upgrade head

# Check logs
docker-compose logs -f backend
```

#### Option B: Cloud Platform (Recommended)

See [Cloud Platform Guides](#cloud-platform-guides) section below.

#### Option C: Kubernetes

See [Kubernetes Deployment](#kubernetes-deployment) section below.

### Step 6: Post-Deployment Verification

```bash
# 1. Health check
curl https://your-domain.com/health

# Expected response:
# {"status": "healthy", "version": "1.0.0"}

# 2. API documentation
open https://your-domain.com/docs

# 3. Test dataset upload
curl -X POST https://your-domain.com/api/v1/datasets/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@test_data.csv"

# 4. Check database connection
docker exec backend python -c "from app.db.session import engine; print(engine.connect())"

# 5. Check Redis connection
docker exec backend python -c "from app.utils.cache import redis_client; print(redis_client.ping())"

# 6. Check Celery workers
docker exec celery-worker celery -A celery_worker.celery_app inspect active
```

✅ **Production deployment complete!**


---

## ☁️ Cloud Platform Guides

### Render (Easiest - Free Tier Available)

**Cost:** Free tier available (512MB RAM)  
**Best for:** Small projects, MVPs, demos

#### Setup Steps:

**1. Prepare Repository**
```bash
# Ensure these files exist:
# - backend/Dockerfile
# - backend/requirements.render.txt (lightweight)
# - render.yaml (optional, for auto-config)

git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

**2. Create Render Account**
- Go to [render.com](https://render.com)
- Sign up with GitHub

**3. Create Web Service**
- Click "New +" → "Web Service"
- Connect your GitHub repository
- Configure:
  - **Name:** `aiplayground-backend`
  - **Region:** Choose closest to users
  - **Branch:** `main`
  - **Root Directory:** `backend`
  - **Environment:** `Docker`
  - **Plan:** Free (or Starter for $7/month)

**4. Add Environment Variables**

In Render dashboard → Environment:
```bash
DATABASE_URL=postgresql://...  # From NeonDB
REDIS_URL=rediss://...         # From Upstash
SECRET_KEY=<generated-secret>
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO
ALLOWED_ORIGINS=https://your-frontend.vercel.app
```

**5. Deploy**
- Click "Create Web Service"
- Wait for build (5-10 minutes)
- Check logs for errors

**6. Verify**
```bash
curl https://aiplayground-backend.onrender.com/health
```

#### Render Limitations (Free Tier):
- ⚠️ 512MB RAM (XGBoost, CatBoost, LightGBM disabled)
- ⚠️ Spins down after 15 minutes of inactivity
- ⚠️ Cold start takes 30-60 seconds
- ✅ Upgrade to Starter ($7/mo) for 2GB RAM and no spin-down

**See:** [RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md) for detailed guide

---

### Railway (Developer-Friendly)

**Cost:** $5/month credit, pay-as-you-go  
**Best for:** Startups, growing projects

#### Setup Steps:

**1. Install Railway CLI**
```bash
npm install -g @railway/cli
railway login
```

**2. Initialize Project**
```bash
cd backend
railway init
```

**3. Add Services**
```bash
# Add PostgreSQL
railway add --database postgres

# Add Redis
railway add --database redis

# Deploy backend
railway up
```

**4. Configure Environment**
```bash
# Railway auto-generates DATABASE_URL and REDIS_URL
# Add remaining variables:
railway variables set SECRET_KEY=<your-secret>
railway variables set ENVIRONMENT=production
railway variables set DEBUG=False
railway variables set ALLOWED_ORIGINS=https://your-frontend.com
```

**5. Deploy**
```bash
railway up
```

**6. Get URL**
```bash
railway domain
# Generates: https://aiplayground-backend.up.railway.app
```

#### Railway Advantages:
- ✅ Generous free tier ($5/month credit)
- ✅ No cold starts
- ✅ Built-in PostgreSQL and Redis
- ✅ Easy CLI workflow
- ✅ Automatic HTTPS

---

### DigitalOcean App Platform

**Cost:** $12/month (Basic)  
**Best for:** Production apps, predictable pricing

#### Setup Steps:

**1. Create App**
- Go to [DigitalOcean](https://cloud.digitalocean.com)
- Apps → Create App
- Connect GitHub repository

**2. Configure App**
- **Source:** `backend/`
- **Type:** Docker
- **Plan:** Basic ($12/mo) or Professional ($24/mo)
- **Region:** Choose closest

**3. Add Managed Databases**
```bash
# Create PostgreSQL database
doctl databases create aiplayground-db --engine pg --region nyc1

# Create Redis cluster
doctl databases create aiplayground-redis --engine redis --region nyc1

# Get connection strings
doctl databases connection aiplayground-db
doctl databases connection aiplayground-redis
```

**4. Set Environment Variables**

In App Platform → Settings → Environment Variables:
```bash
DATABASE_URL=${aiplayground-db.DATABASE_URL}
REDIS_URL=${aiplayground-redis.REDIS_URL}
SECRET_KEY=<your-secret>
ENVIRONMENT=production
DEBUG=False
```

**5. Deploy**
- Click "Deploy"
- Wait for build

**6. Custom Domain (Optional)**
- Settings → Domains
- Add your domain
- Update DNS records

#### DigitalOcean Advantages:
- ✅ Predictable pricing
- ✅ Managed databases included
- ✅ Good performance
- ✅ Easy scaling
- ✅ 99.99% uptime SLA


---

### AWS (Enterprise-Grade)

**Cost:** Variable (starts ~$50/month)  
**Best for:** Enterprise, high-traffic, compliance requirements

#### Architecture:

```
┌─────────────────────────────────────────────────────────┐
│                Application Load Balancer                │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐       ┌───────▼────────┐
│   ECS Fargate  │       │   ECS Fargate  │
│   (Backend)    │       │   (Backend)    │
└───────┬────────┘       └───────┬────────┘
        │                         │
        └────────────┬────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐       ┌───────▼────────┐
│   RDS Postgres │       │  ElastiCache   │
│   (Database)   │       │    (Redis)     │
└────────────────┘       └────────────────┘
```

#### Setup Steps:

**1. Create ECR Repository**
```bash
# Install AWS CLI
aws configure

# Create repository
aws ecr create-repository --repository-name aiplayground-backend

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t aiplayground-backend .
docker tag aiplayground-backend:latest \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com/aiplayground-backend:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/aiplayground-backend:latest
```

**2. Create RDS PostgreSQL**
```bash
aws rds create-db-instance \
  --db-instance-identifier aiplayground-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --engine-version 15.4 \
  --master-username admin \
  --master-user-password <strong-password> \
  --allocated-storage 20 \
  --vpc-security-group-ids sg-xxxxx \
  --db-subnet-group-name default \
  --backup-retention-period 7 \
  --publicly-accessible false
```

**3. Create ElastiCache Redis**
```bash
aws elasticache create-cache-cluster \
  --cache-cluster-id aiplayground-redis \
  --cache-node-type cache.t3.micro \
  --engine redis \
  --num-cache-nodes 1 \
  --security-group-ids sg-xxxxx
```

**4. Create ECS Cluster**
```bash
aws ecs create-cluster --cluster-name aiplayground-cluster
```

**5. Create Task Definition**

Create `task-definition.json`:
```json
{
  "family": "aiplayground-backend",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/aiplayground-backend:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "DEBUG", "value": "False"},
        {"name": "LOG_LEVEL", "value": "INFO"}
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:xxxxx:secret:database-url"
        },
        {
          "name": "SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:xxxxx:secret:secret-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/aiplayground-backend",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

Register task:
```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

**6. Create ECS Service**
```bash
aws ecs create-service \
  --cluster aiplayground-cluster \
  --service-name aiplayground-backend-service \
  --task-definition aiplayground-backend \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxx],securityGroups=[sg-xxxxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=backend,containerPort=8000"
```

**7. Create Application Load Balancer**
```bash
# Create ALB
aws elbv2 create-load-balancer \
  --name aiplayground-alb \
  --subnets subnet-xxxxx subnet-yyyyy \
  --security-groups sg-xxxxx

# Create target group
aws elbv2 create-target-group \
  --name aiplayground-tg \
  --protocol HTTP \
  --port 8000 \
  --vpc-id vpc-xxxxx \
  --target-type ip \
  --health-check-path /health

# Create listener
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:elasticloadbalancing:... \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...
```

#### AWS Cost Estimate (Monthly):
- ECS Fargate (2 tasks): ~$30
- RDS PostgreSQL (db.t3.micro): ~$15
- ElastiCache Redis (cache.t3.micro): ~$12
- ALB: ~$20
- Data transfer: ~$10
- **Total: ~$87/month**

#### AWS Advantages:
- ✅ Enterprise-grade reliability
- ✅ Extensive service ecosystem
- ✅ Advanced security features
- ✅ Global infrastructure
- ✅ Compliance certifications


---

### Google Cloud Platform (GCP)

**Cost:** Variable (starts ~$40/month)  
**Best for:** Data-intensive apps, ML workloads

#### Setup Steps:

**1. Build and Push to Container Registry**
```bash
# Install gcloud CLI
gcloud init

# Configure Docker
gcloud auth configure-docker

# Build and push
docker build -t gcr.io/PROJECT_ID/aiplayground-backend .
docker push gcr.io/PROJECT_ID/aiplayground-backend
```

**2. Create Cloud SQL PostgreSQL**
```bash
gcloud sql instances create aiplayground-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=us-central1

gcloud sql databases create aiplayground \
  --instance=aiplayground-db

gcloud sql users create admin \
  --instance=aiplayground-db \
  --password=<strong-password>
```

**3. Create Memorystore Redis**
```bash
gcloud redis instances create aiplayground-redis \
  --size=1 \
  --region=us-central1 \
  --redis-version=redis_7_0
```

**4. Deploy to Cloud Run**
```bash
gcloud run deploy aiplayground-backend \
  --image gcr.io/PROJECT_ID/aiplayground-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="ENVIRONMENT=production,DEBUG=False" \
  --set-secrets="DATABASE_URL=database-url:latest,SECRET_KEY=secret-key:latest" \
  --add-cloudsql-instances=PROJECT_ID:us-central1:aiplayground-db \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=1 \
  --max-instances=10
```

**5. Set Up Cloud Scheduler (for Celery)**
```bash
# Create Pub/Sub topic
gcloud pubsub topics create celery-tasks

# Create Cloud Function for Celery worker
# Deploy function that processes Pub/Sub messages
```

#### GCP Advantages:
- ✅ Excellent for ML workloads
- ✅ BigQuery integration
- ✅ Competitive pricing
- ✅ Cloud Run auto-scaling
- ✅ Strong data analytics tools

---

### Azure

**Cost:** Variable (starts ~$45/month)  
**Best for:** Enterprise, Microsoft ecosystem

#### Setup Steps:

**1. Create Container Registry**
```bash
# Install Azure CLI
az login

# Create resource group
az group create --name aiplayground-rg --location eastus

# Create container registry
az acr create --resource-group aiplayground-rg \
  --name aiplaygroundacr --sku Basic

# Login to ACR
az acr login --name aiplaygroundacr

# Build and push
docker build -t aiplaygroundacr.azurecr.io/backend .
docker push aiplaygroundacr.azurecr.io/backend
```

**2. Create PostgreSQL**
```bash
az postgres flexible-server create \
  --resource-group aiplayground-rg \
  --name aiplayground-db \
  --location eastus \
  --admin-user admin \
  --admin-password <strong-password> \
  --sku-name Standard_B1ms \
  --version 15
```

**3. Create Redis Cache**
```bash
az redis create \
  --resource-group aiplayground-rg \
  --name aiplayground-redis \
  --location eastus \
  --sku Basic \
  --vm-size c0
```

**4. Deploy to Container Instances**
```bash
az container create \
  --resource-group aiplayground-rg \
  --name aiplayground-backend \
  --image aiplaygroundacr.azurecr.io/backend \
  --cpu 2 \
  --memory 4 \
  --registry-login-server aiplaygroundacr.azurecr.io \
  --registry-username <username> \
  --registry-password <password> \
  --dns-name-label aiplayground-backend \
  --ports 8000 \
  --environment-variables \
    ENVIRONMENT=production \
    DEBUG=False \
  --secure-environment-variables \
    DATABASE_URL=<connection-string> \
    SECRET_KEY=<secret>
```

#### Azure Advantages:
- ✅ Strong enterprise support
- ✅ Active Directory integration
- ✅ Hybrid cloud capabilities
- ✅ Compliance certifications
- ✅ Microsoft ecosystem integration


---

## ⚙️ Configuration

### Environment Variables Reference

#### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host:5432/db` |
| `REDIS_URL` | Redis connection string | `redis://host:6379/0` |
| `SECRET_KEY` | JWT signing key (32+ chars) | `a1b2c3d4e5f6...` |
| `CELERY_BROKER_URL` | Celery message broker | Same as REDIS_URL |
| `CELERY_RESULT_BACKEND` | Celery results storage | Same as REDIS_URL |

#### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment name | `development` |
| `DEBUG` | Enable debug mode | `True` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `API_V1_PREFIX` | API route prefix | `/api/v1` |
| `PROJECT_NAME` | Project name | `AI-Playground` |
| `VERSION` | API version | `1.0.0` |
| `ALLOWED_ORIGINS` | CORS origins (comma-separated) | `*` |
| `UPLOAD_DIR` | File upload directory | `/app/uploads` |
| `MAX_UPLOAD_SIZE` | Max file size (bytes) | `104857600` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT expiration | `30` |
| `ALGORITHM` | JWT algorithm | `HS256` |

#### Object Storage (Optional)

| Variable | Description |
|----------|-------------|
| `R2_ACCOUNT_ID` | Cloudflare R2 account ID |
| `R2_ACCESS_KEY_ID` | R2 access key |
| `R2_SECRET_ACCESS_KEY` | R2 secret key |
| `R2_BUCKET_NAME` | R2 bucket name |
| `R2_PUBLIC_URL` | R2 public URL |

#### Monitoring (Optional)

| Variable | Description |
|----------|-------------|
| `SENTRY_DSN` | Sentry error tracking DSN |
| `PROMETHEUS_PORT` | Prometheus metrics port |

### Configuration Files

#### `.env` - Local Development
```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/aiplayground
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=dev-secret-key
DEBUG=True
LOG_LEVEL=DEBUG
```

#### `.env.production` - Production
```bash
DATABASE_URL=postgresql://user:pass@prod-host:5432/aiplayground?sslmode=require
REDIS_URL=rediss://default:pass@prod-redis:6379
SECRET_KEY=<strong-random-key>
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO
ALLOWED_ORIGINS=https://your-domain.com
```

#### `.env.test` - Testing
```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/aiplayground_test
REDIS_URL=redis://localhost:6379/1
SECRET_KEY=test-secret-key
ENVIRONMENT=test
DEBUG=True
LOG_LEVEL=DEBUG
```

### Security Best Practices

#### 1. Secret Management

**Never commit secrets to Git:**
```bash
# Add to .gitignore
.env
.env.production
.env.local
*.pem
*.key
```

**Use secret management services:**
- **AWS Secrets Manager** - For AWS deployments
- **GCP Secret Manager** - For GCP deployments
- **Azure Key Vault** - For Azure deployments
- **HashiCorp Vault** - Platform-agnostic
- **Doppler** - Developer-friendly

**Example with AWS Secrets Manager:**
```bash
# Store secret
aws secretsmanager create-secret \
  --name aiplayground/database-url \
  --secret-string "postgresql://..."

# Retrieve in application
import boto3
client = boto3.client('secretsmanager')
response = client.get_secret_value(SecretId='aiplayground/database-url')
DATABASE_URL = response['SecretString']
```

#### 2. Generate Strong Secrets

```bash
# SECRET_KEY (32 bytes)
openssl rand -hex 32

# Database password (24 chars)
openssl rand -base64 24

# API key (32 chars)
openssl rand -base64 32 | tr -d "=+/" | cut -c1-32
```

#### 3. SSL/TLS Configuration

**Always use SSL for database connections:**
```bash
# PostgreSQL with SSL
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require

# Redis with TLS
REDIS_URL=rediss://default:pass@host:6379
```

**Enable HTTPS for API:**
- Use reverse proxy (Nginx, Caddy)
- Use cloud load balancer with SSL certificate
- Use Let's Encrypt for free certificates


---

## 🗄️ Database Setup

### Initial Setup

#### 1. Create Database

**PostgreSQL:**
```sql
-- Connect as superuser
psql -U postgres

-- Create database
CREATE DATABASE aiplayground;

-- Create user
CREATE USER aiplayground_user WITH PASSWORD 'strong_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE aiplayground TO aiplayground_user;

-- Connect to database
\c aiplayground

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO aiplayground_user;
```

#### 2. Run Migrations

```bash
# Check current version
alembic current

# Upgrade to latest
alembic upgrade head

# Verify tables created
psql $DATABASE_URL -c "\dt"
```

#### 3. Seed Initial Data (Optional)

```bash
# Run seed script
python backend/init_db.py

# Or manually:
psql $DATABASE_URL <<EOF
INSERT INTO users (id, email, created_at)
VALUES (
  gen_random_uuid(),
  'admin@aiplayground.com',
  NOW()
);
EOF
```

### Database Migrations

#### Create New Migration

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "Add new column to datasets table"

# Manually create migration
alembic revision -m "Custom migration"
```

#### Apply Migrations

```bash
# Upgrade to latest
alembic upgrade head

# Upgrade to specific revision
alembic upgrade abc123

# Upgrade one version
alembic upgrade +1
```

#### Rollback Migrations

```bash
# Downgrade one version
alembic downgrade -1

# Downgrade to specific revision
alembic downgrade abc123

# Downgrade to base (WARNING: drops all tables)
alembic downgrade base
```

#### Check Migration Status

```bash
# Show current version
alembic current

# Show migration history
alembic history

# Show pending migrations
alembic heads
```

### Database Backup & Restore

#### Backup

```bash
# Full database backup
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql

# Compressed backup
pg_dump $DATABASE_URL | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Schema only
pg_dump --schema-only $DATABASE_URL > schema_backup.sql

# Data only
pg_dump --data-only $DATABASE_URL > data_backup.sql

# Specific tables
pg_dump --table=datasets --table=users $DATABASE_URL > tables_backup.sql
```

#### Restore

```bash
# Restore from backup
psql $DATABASE_URL < backup.sql

# Restore compressed backup
gunzip -c backup.sql.gz | psql $DATABASE_URL

# Restore with progress
pv backup.sql | psql $DATABASE_URL
```

#### Automated Backups

**Using cron (Linux):**
```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * pg_dump $DATABASE_URL | gzip > /backups/aiplayground_$(date +\%Y\%m\%d).sql.gz

# Keep only last 7 days
0 3 * * * find /backups -name "aiplayground_*.sql.gz" -mtime +7 -delete
```

**Using AWS S3:**
```bash
#!/bin/bash
# backup-to-s3.sh

BACKUP_FILE="backup_$(date +%Y%m%d_%H%M%S).sql.gz"
pg_dump $DATABASE_URL | gzip > /tmp/$BACKUP_FILE
aws s3 cp /tmp/$BACKUP_FILE s3://aiplayground-backups/
rm /tmp/$BACKUP_FILE
```

### Database Optimization

#### Connection Pooling

**SQLAlchemy Configuration:**
```python
# app/db/session.py
engine = create_engine(
    DATABASE_URL,
    pool_size=10,              # Number of connections to maintain
    max_overflow=20,           # Max connections beyond pool_size
    pool_pre_ping=True,        # Test connections before use
    pool_recycle=3600,         # Recycle connections after 1 hour
    echo=False,                # Disable SQL logging in production
    connect_args={
        "connect_timeout": 10,
        "options": "-c timezone=utc"
    }
)
```

#### Indexes

```sql
-- Add indexes for frequently queried columns
CREATE INDEX idx_datasets_user_id ON datasets(user_id);
CREATE INDEX idx_datasets_created_at ON datasets(created_at);
CREATE INDEX idx_model_runs_dataset_id ON model_runs(dataset_id);
CREATE INDEX idx_model_runs_status ON model_runs(status);

-- Composite indexes
CREATE INDEX idx_datasets_user_created ON datasets(user_id, created_at DESC);

-- Partial indexes
CREATE INDEX idx_active_model_runs ON model_runs(dataset_id) 
WHERE status IN ('running', 'pending');
```

#### Query Optimization

```sql
-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM datasets WHERE user_id = 'xxx';

-- Update statistics
ANALYZE datasets;

-- Vacuum database
VACUUM ANALYZE;
```

### Database Monitoring

#### Check Database Size

```sql
-- Database size
SELECT pg_size_pretty(pg_database_size('aiplayground'));

-- Table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Index sizes
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY pg_relation_size(indexrelid) DESC;
```

#### Check Active Connections

```sql
-- Active connections
SELECT count(*) FROM pg_stat_activity;

-- Connections by state
SELECT state, count(*) 
FROM pg_stat_activity 
GROUP BY state;

-- Long-running queries
SELECT 
    pid,
    now() - query_start AS duration,
    query
FROM pg_stat_activity
WHERE state = 'active'
ORDER BY duration DESC;

-- Kill long-running query
SELECT pg_terminate_backend(pid);
```


---

## 📊 Monitoring & Logging

### Application Logging

#### Log Levels

```python
# app/core/logging_config.py

# Development
LOG_LEVEL=DEBUG  # All logs including debug info

# Staging
LOG_LEVEL=INFO   # Info, warnings, errors

# Production
LOG_LEVEL=WARNING  # Only warnings and errors
```

#### Log Files

```bash
# Application logs
backend/logs/app.log          # General application logs
backend/logs/error.log        # Error logs only
backend/logs/celery.log       # Celery task logs
backend/logs/access.log       # API access logs
```

#### Viewing Logs

```bash
# Tail application logs
tail -f backend/logs/app.log

# Search for errors
grep "ERROR" backend/logs/app.log

# Filter by date
grep "2026-01-02" backend/logs/app.log

# Count errors
grep -c "ERROR" backend/logs/app.log

# Docker logs
docker-compose logs -f backend
docker-compose logs --tail=100 backend
```

### Centralized Logging

#### Sentry (Error Tracking)

**Setup:**
```bash
# Install Sentry SDK
pip install sentry-sdk[fastapi]

# Configure in app/main.py
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    environment=os.getenv("ENVIRONMENT", "development"),
    traces_sample_rate=0.1,  # 10% of transactions
    integrations=[FastApiIntegration()],
)
```

**Environment Variable:**
```bash
SENTRY_DSN=https://xxxxx@o123456.ingest.sentry.io/123456
```

#### Datadog (Full Observability)

**Setup:**
```bash
# Install Datadog agent
DD_API_KEY=<your-api-key> DD_SITE="datadoghq.com" \
  bash -c "$(curl -L https://s3.amazonaws.com/dd-agent/scripts/install_script.sh)"

# Configure Python integration
pip install ddtrace

# Run with Datadog
ddtrace-run uvicorn app.main:app
```

#### ELK Stack (Self-Hosted)

**docker-compose.elk.yml:**
```yaml
services:
  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"

  logstash:
    image: logstash:8.11.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: kibana:8.11.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

### Performance Monitoring

#### Prometheus + Grafana

**1. Add Prometheus Metrics**

```python
# app/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
from prometheus_client import make_asgi_app

# Metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

active_model_training = Gauge(
    'active_model_training',
    'Number of active model training tasks'
)

# Expose metrics endpoint
metrics_app = make_asgi_app()
```

**2. Mount in FastAPI**

```python
# app/main.py
from app.monitoring.metrics import metrics_app

app.mount("/metrics", metrics_app)
```

**3. Configure Prometheus**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'aiplayground-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
```

**4. Set Up Grafana Dashboard**

```bash
# Access Grafana
open http://localhost:3000

# Add Prometheus data source
# Import dashboard ID: 1860 (Node Exporter)
# Create custom dashboard for application metrics
```

#### Health Checks

**Endpoint:**
```python
# app/api/v1/endpoints/health.py
@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/health/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    checks = {
        "database": False,
        "redis": False,
        "celery": False
    }
    
    # Check database
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
    
    # Check Redis
    try:
        redis_client.ping()
        checks["redis"] = True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
    
    # Check Celery
    try:
        from app.celery_app import celery_app
        stats = celery_app.control.inspect().stats()
        checks["celery"] = bool(stats)
    except Exception as e:
        logger.error(f"Celery health check failed: {e}")
    
    status = "healthy" if all(checks.values()) else "unhealthy"
    
    return {
        "status": status,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }
```

### Alerting

#### Sentry Alerts

Configure in Sentry dashboard:
- Error rate threshold
- New issue notifications
- Performance degradation
- Slack/Email/PagerDuty integration

#### Prometheus Alertmanager

```yaml
# alertmanager.yml
route:
  receiver: 'team-email'
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h

receivers:
  - name: 'team-email'
    email_configs:
      - to: 'team@aiplayground.com'
        from: 'alerts@aiplayground.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'alerts@aiplayground.com'
        auth_password: '<password>'

# Alert rules
groups:
  - name: aiplayground
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          
      - alert: HighResponseTime
        expr: http_request_duration_seconds{quantile="0.95"} > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
```

#### Uptime Monitoring

**Services:**
- **UptimeRobot** - Free, simple uptime monitoring
- **Pingdom** - Advanced monitoring with RUM
- **StatusCake** - Affordable uptime monitoring
- **Better Uptime** - Modern, developer-friendly

**Setup:**
1. Add health check URL: `https://your-api.com/health`
2. Set check interval: 1-5 minutes
3. Configure alerts: Email, Slack, SMS
4. Create status page for users


---

## 🔒 Security Best Practices

### 1. Authentication & Authorization

#### JWT Token Security

```python
# Use strong secret key (32+ bytes)
SECRET_KEY = os.getenv("SECRET_KEY")  # Never hardcode!

# Set appropriate expiration
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # Short-lived tokens

# Use secure algorithm
ALGORITHM = "HS256"  # Or RS256 for asymmetric

# Implement token refresh
REFRESH_TOKEN_EXPIRE_DAYS = 7
```

#### API Key Management

```python
# Store API keys hashed
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

hashed_key = pwd_context.hash(api_key)

# Verify API keys
pwd_context.verify(provided_key, hashed_key)
```

### 2. Input Validation

```python
# Use Pydantic for validation
from pydantic import BaseModel, validator, Field

class DatasetUpload(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(None, max_length=1000)
    
    @validator('name')
    def validate_name(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Name must be alphanumeric')
        return v

# Validate file uploads
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.json'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

def validate_file(file: UploadFile):
    # Check extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Invalid file type")
    
    # Check size
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")
```

### 3. SQL Injection Prevention

```python
# Always use parameterized queries
# ✅ GOOD
db.execute(
    text("SELECT * FROM datasets WHERE user_id = :user_id"),
    {"user_id": user_id}
)

# ❌ BAD - Never do this!
db.execute(f"SELECT * FROM datasets WHERE user_id = '{user_id}'")

# Use SQLAlchemy ORM (automatically parameterized)
db.query(Dataset).filter(Dataset.user_id == user_id).all()
```

### 4. CORS Configuration

```python
# app/main.py
from fastapi.middleware.cors import CORSMiddleware

# Production: Specify exact origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Never use ["*"] in production!
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600,
)
```

### 5. Rate Limiting

```python
# Install slowapi
pip install slowapi

# app/main.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Apply to endpoints
@app.get("/api/v1/datasets")
@limiter.limit("100/minute")
async def list_datasets(request: Request):
    ...
```

### 6. Secrets Management

**Never commit secrets:**
```bash
# .gitignore
.env
.env.*
!.env.example
*.pem
*.key
secrets/
```

**Use environment variables:**
```python
# ✅ GOOD
SECRET_KEY = os.getenv("SECRET_KEY")

# ❌ BAD
SECRET_KEY = "hardcoded-secret-key"
```

**Use secret management services:**
- AWS Secrets Manager
- GCP Secret Manager
- Azure Key Vault
- HashiCorp Vault

### 7. HTTPS/TLS

**Always use HTTPS in production:**

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name api.aiplayground.com;
    
    ssl_certificate /etc/letsencrypt/live/api.aiplayground.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.aiplayground.com/privkey.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    location / {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.aiplayground.com;
    return 301 https://$server_name$request_uri;
}
```

### 8. Security Headers

```python
# app/middleware/security.py
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response

# Add to app
app.add_middleware(SecurityHeadersMiddleware)
```

### 9. Dependency Scanning

```bash
# Check for vulnerabilities
pip install safety
safety check

# Update dependencies
pip list --outdated
pip install --upgrade <package>

# Use Dependabot (GitHub)
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/backend"
    schedule:
      interval: "weekly"
```

### 10. Container Security

```dockerfile
# Use official base images
FROM python:3.11-slim

# Run as non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Don't install unnecessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Scan images
docker scan aiplayground-backend
```

### Security Checklist

- [ ] Strong SECRET_KEY (32+ random bytes)
- [ ] HTTPS enabled in production
- [ ] CORS configured with specific origins
- [ ] Rate limiting implemented
- [ ] Input validation on all endpoints
- [ ] SQL injection prevention (parameterized queries)
- [ ] Secrets stored in environment variables
- [ ] Security headers configured
- [ ] Dependencies regularly updated
- [ ] Container runs as non-root user
- [ ] Database connections use SSL
- [ ] API authentication required
- [ ] File upload validation
- [ ] Error messages don't leak sensitive info
- [ ] Logging doesn't include secrets


---

## 🐛 Troubleshooting

### Common Issues

#### 1. Database Connection Failed

**Symptoms:**
```
sqlalchemy.exc.OperationalError: could not connect to server
```

**Solutions:**

```bash
# Check database is running
docker-compose ps postgres

# Test connection
psql $DATABASE_URL -c "SELECT 1"

# Check connection string format
# Correct: postgresql://user:pass@host:5432/db
# With SSL: postgresql://user:pass@host:5432/db?sslmode=require

# Check firewall rules
telnet db-host 5432

# Check database logs
docker-compose logs postgres

# Verify credentials
echo $DATABASE_URL
```

#### 2. Redis Connection Failed

**Symptoms:**
```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solutions:**

```bash
# Check Redis is running
docker-compose ps redis

# Test connection
redis-cli -h localhost -p 6379 ping

# Check Redis logs
docker-compose logs redis

# Verify URL format
# Correct: redis://host:6379/0
# With password: redis://:password@host:6379/0
# With TLS: rediss://default:password@host:6379

# Check if Redis is accepting connections
docker-compose exec redis redis-cli ping
```

#### 3. Celery Tasks Not Processing

**Symptoms:**
- Tasks stuck in "pending" state
- No worker logs

**Solutions:**

```bash
# Check Celery worker is running
docker-compose ps celery-worker

# View worker logs
docker-compose logs -f celery-worker

# Check active tasks
docker-compose exec celery-worker celery -A celery_worker.celery_app inspect active

# Check registered tasks
docker-compose exec celery-worker celery -A celery_worker.celery_app inspect registered

# Purge all tasks (WARNING: deletes pending tasks)
docker-compose exec celery-worker celery -A celery_worker.celery_app purge

# Restart worker
docker-compose restart celery-worker
```

#### 4. Out of Memory (OOM)

**Symptoms:**
```
Container killed due to out of memory
Process died with code 137
```

**Solutions:**

```bash
# Check memory usage
docker stats

# Increase Docker memory limit
# Docker Desktop: Settings → Resources → Memory → 8GB

# Reduce workers
# In Dockerfile or docker-compose.yml:
command: uvicorn app.main:app --workers 1

# Use lighter dependencies
# Use requirements.render.txt instead of requirements.txt

# Enable memory optimization
ENV MALLOC_TRIM_THRESHOLD_=100000

# Monitor memory in application
import psutil
memory = psutil.virtual_memory()
print(f"Memory usage: {memory.percent}%")
```

#### 5. Port Already in Use

**Symptoms:**
```
Error: port is already allocated
```

**Solutions:**

```bash
# Find process using port
# Windows:
netstat -ano | findstr :8000
taskkill /PID <pid> /F

# Linux/Mac:
lsof -i :8000
kill -9 <pid>

# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use different host port

# Or stop conflicting service
docker-compose down
```

#### 6. Migration Errors

**Symptoms:**
```
alembic.util.exc.CommandError: Target database is not up to date
```

**Solutions:**

```bash
# Check current version
alembic current

# Check migration history
alembic history

# Stamp database to specific version
alembic stamp head

# Downgrade and re-upgrade
alembic downgrade -1
alembic upgrade head

# Reset database (WARNING: deletes all data)
alembic downgrade base
alembic upgrade head

# Check for conflicts
alembic branches
```

#### 7. File Upload Fails

**Symptoms:**
```
413 Request Entity Too Large
422 Unprocessable Entity
```

**Solutions:**

```bash
# Check MAX_UPLOAD_SIZE
echo $MAX_UPLOAD_SIZE

# Increase limit
MAX_UPLOAD_SIZE=209715200  # 200MB

# Check Nginx limit (if using)
# nginx.conf:
client_max_body_size 200M;

# Check file permissions
ls -la backend/uploads
chmod 755 backend/uploads

# Check disk space
df -h
```

#### 8. CORS Errors

**Symptoms:**
```
Access to fetch at 'http://api.com' from origin 'http://frontend.com' 
has been blocked by CORS policy
```

**Solutions:**

```python
# Check ALLOWED_ORIGINS
print(os.getenv("ALLOWED_ORIGINS"))

# Update CORS configuration
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://your-frontend.vercel.app"
]

# Verify middleware is added
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check browser console for exact error
# Check preflight OPTIONS request
```

#### 9. Slow API Response

**Symptoms:**
- Requests taking > 5 seconds
- Timeout errors

**Solutions:**

```bash
# Check database query performance
EXPLAIN ANALYZE SELECT * FROM datasets WHERE user_id = 'xxx';

# Add indexes
CREATE INDEX idx_datasets_user_id ON datasets(user_id);

# Enable query logging
# app/db/session.py
engine = create_engine(DATABASE_URL, echo=True)

# Check for N+1 queries
# Use eager loading
db.query(Dataset).options(joinedload(Dataset.user)).all()

# Enable caching
from app.utils.cache import cache_result

@cache_result(ttl=300)
def get_dataset_stats(dataset_id):
    ...

# Monitor with Prometheus
http_request_duration_seconds{quantile="0.95"}
```

#### 10. SSL Certificate Errors

**Symptoms:**
```
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solutions:**

```bash
# For development, disable SSL verification (NOT for production!)
DATABASE_URL=postgresql://...?sslmode=disable

# For production, use proper SSL
DATABASE_URL=postgresql://...?sslmode=require

# Download CA certificate
curl https://your-db-host/ca-certificate.crt -o ca.crt

# Use CA certificate
DATABASE_URL=postgresql://...?sslmode=verify-full&sslrootcert=ca.crt

# Check certificate expiration
openssl s_client -connect db-host:5432 -starttls postgres
```

### Debug Mode

Enable debug mode for detailed error messages:

```bash
# .env
DEBUG=True
LOG_LEVEL=DEBUG

# Restart application
docker-compose restart backend

# View detailed logs
docker-compose logs -f backend
```

### Getting Help

1. **Check logs first:**
   ```bash
   docker-compose logs backend
   docker-compose logs celery-worker
   ```

2. **Check health endpoints:**
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/health/detailed
   ```

3. **Test individual components:**
   ```bash
   # Database
   psql $DATABASE_URL -c "SELECT 1"
   
   # Redis
   redis-cli ping
   
   # Celery
   celery -A celery_worker.celery_app inspect ping
   ```

4. **Search existing issues:**
   - GitHub Issues
   - Stack Overflow
   - FastAPI Discussions

5. **Create detailed bug report:**
   - Environment details
   - Error messages
   - Steps to reproduce
   - Logs
   - Configuration (without secrets!)


---

## 🔧 Maintenance

### Regular Tasks

#### Daily

- [ ] Check error logs for critical issues
- [ ] Monitor API response times
- [ ] Verify Celery workers are processing tasks
- [ ] Check disk space usage

```bash
# Quick health check script
#!/bin/bash
echo "=== Health Check ==="
curl -s http://localhost:8000/health | jq
echo ""
echo "=== Disk Usage ==="
df -h | grep -E "/$|/app"
echo ""
echo "=== Memory Usage ==="
free -h
echo ""
echo "=== Active Containers ==="
docker-compose ps
```

#### Weekly

- [ ] Review error rates and trends
- [ ] Check database size and growth
- [ ] Verify backups are running
- [ ] Update dependencies (security patches)
- [ ] Review slow queries

```bash
# Weekly maintenance script
#!/bin/bash

# Database vacuum
docker-compose exec postgres psql -U aiplayground -d aiplayground -c "VACUUM ANALYZE;"

# Clear old logs (keep last 7 days)
find backend/logs -name "*.log" -mtime +7 -delete

# Check for outdated dependencies
docker-compose exec backend pip list --outdated

# Database backup
pg_dump $DATABASE_URL | gzip > backups/weekly_$(date +%Y%m%d).sql.gz
```

#### Monthly

- [ ] Review and optimize database indexes
- [ ] Analyze application performance metrics
- [ ] Update all dependencies
- [ ] Review and rotate secrets
- [ ] Test disaster recovery procedures
- [ ] Review and update documentation

```bash
# Monthly maintenance script
#!/bin/bash

# Full database backup
pg_dump $DATABASE_URL | gzip > backups/monthly_$(date +%Y%m).sql.gz

# Upload to S3
aws s3 cp backups/monthly_$(date +%Y%m).sql.gz s3://aiplayground-backups/

# Update dependencies
docker-compose exec backend pip install --upgrade -r requirements.txt

# Rebuild containers
docker-compose build --no-cache
docker-compose up -d

# Run tests
docker-compose exec backend pytest
```

### Scaling

#### Horizontal Scaling (Multiple Instances)

**1. Load Balancer Configuration**

```nginx
# nginx.conf
upstream backend {
    least_conn;  # Load balancing method
    server backend-1:8000 max_fails=3 fail_timeout=30s;
    server backend-2:8000 max_fails=3 fail_timeout=30s;
    server backend-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Health check
        proxy_next_upstream error timeout http_500 http_502 http_503;
    }
}
```

**2. Docker Compose Scaling**

```bash
# Scale backend to 3 instances
docker-compose up -d --scale backend=3

# Scale Celery workers to 5 instances
docker-compose up -d --scale celery-worker=5
```

**3. Kubernetes Scaling**

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aiplayground-backend
spec:
  replicas: 3  # Number of instances
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: aiplayground-backend:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: aiplayground-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Vertical Scaling (More Resources)

**Docker Compose:**
```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

**Cloud Platforms:**
- Render: Upgrade plan (Starter → Professional)
- Railway: Increase resource limits
- DigitalOcean: Resize droplet/app
- AWS: Change instance type

### Database Maintenance

#### Vacuum and Analyze

```bash
# Manual vacuum
docker-compose exec postgres psql -U aiplayground -d aiplayground -c "VACUUM ANALYZE;"

# Automated vacuum (cron)
0 2 * * 0 docker-compose exec postgres psql -U aiplayground -d aiplayground -c "VACUUM ANALYZE;"
```

#### Reindex

```bash
# Reindex all tables
docker-compose exec postgres psql -U aiplayground -d aiplayground -c "REINDEX DATABASE aiplayground;"

# Reindex specific table
docker-compose exec postgres psql -U aiplayground -d aiplayground -c "REINDEX TABLE datasets;"
```

#### Archive Old Data

```sql
-- Archive datasets older than 1 year
CREATE TABLE datasets_archive AS 
SELECT * FROM datasets 
WHERE created_at < NOW() - INTERVAL '1 year';

DELETE FROM datasets 
WHERE created_at < NOW() - INTERVAL '1 year';

-- Vacuum after deletion
VACUUM ANALYZE datasets;
```

### Log Rotation

**Using logrotate (Linux):**

```bash
# /etc/logrotate.d/aiplayground
/app/backend/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 appuser appuser
    postrotate
        docker-compose restart backend
    endscript
}
```

**Manual rotation:**

```bash
#!/bin/bash
# rotate-logs.sh

LOG_DIR="/app/backend/logs"
DATE=$(date +%Y%m%d)

# Rotate logs
for log in $LOG_DIR/*.log; do
    if [ -f "$log" ]; then
        mv "$log" "$log.$DATE"
        gzip "$log.$DATE"
    fi
done

# Restart to create new logs
docker-compose restart backend

# Delete logs older than 30 days
find $LOG_DIR -name "*.log.*.gz" -mtime +30 -delete
```

### Disaster Recovery

#### Backup Strategy

**3-2-1 Rule:**
- 3 copies of data
- 2 different storage types
- 1 offsite backup

**Implementation:**

```bash
#!/bin/bash
# backup-all.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/$DATE"
mkdir -p $BACKUP_DIR

# Database backup
pg_dump $DATABASE_URL | gzip > $BACKUP_DIR/database.sql.gz

# File uploads backup
tar -czf $BACKUP_DIR/uploads.tar.gz backend/uploads/

# Configuration backup
cp .env.production $BACKUP_DIR/
cp docker-compose.yml $BACKUP_DIR/

# Upload to S3
aws s3 sync $BACKUP_DIR s3://aiplayground-backups/$DATE/

# Upload to different region
aws s3 sync $BACKUP_DIR s3://aiplayground-backups-eu/$DATE/ --region eu-west-1

# Keep local backups for 7 days
find /backups -maxdepth 1 -type d -mtime +7 -exec rm -rf {} \;

echo "Backup completed: $DATE"
```

#### Recovery Procedures

**1. Database Recovery:**

```bash
# Stop application
docker-compose down

# Restore database
gunzip -c backup.sql.gz | psql $DATABASE_URL

# Run migrations
docker-compose up -d backend
docker-compose exec backend alembic upgrade head

# Verify
docker-compose exec backend python -c "from app.db.session import engine; print(engine.connect())"
```

**2. Full System Recovery:**

```bash
# 1. Provision new infrastructure
# 2. Install Docker and dependencies
# 3. Clone repository
git clone https://github.com/your-org/AI-Playground.git
cd AI-Playground

# 4. Restore configuration
cp backup/.env.production .env

# 5. Restore database
gunzip -c backup/database.sql.gz | psql $DATABASE_URL

# 6. Restore uploads
tar -xzf backup/uploads.tar.gz -C backend/

# 7. Start services
docker-compose up -d

# 8. Verify
curl http://localhost:8000/health
```

### Performance Optimization

#### Database Query Optimization

```python
# Use select_related for foreign keys
datasets = db.query(Dataset).options(
    joinedload(Dataset.user)
).all()

# Use pagination
from fastapi_pagination import Page, paginate

@app.get("/datasets", response_model=Page[DatasetSchema])
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(Dataset).all()
    return paginate(datasets)

# Add database indexes
CREATE INDEX CONCURRENTLY idx_datasets_user_created 
ON datasets(user_id, created_at DESC);
```

#### Caching Strategy

```python
# Cache expensive operations
from app.utils.cache import cache_result

@cache_result(ttl=300)  # 5 minutes
def get_dataset_statistics(dataset_id: str):
    # Expensive computation
    return compute_stats(dataset_id)

# Invalidate cache on updates
from app.utils.cache import invalidate_cache

def update_dataset(dataset_id: str, data: dict):
    # Update database
    db.query(Dataset).filter(Dataset.id == dataset_id).update(data)
    
    # Invalidate cache
    invalidate_cache(f"dataset_stats_{dataset_id}")
```

#### Connection Pooling

```python
# Optimize connection pool
engine = create_engine(
    DATABASE_URL,
    pool_size=20,           # Increase for high traffic
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)
```


---

## 📚 Additional Resources

### Documentation

- **[Backend README](../README.md)** - Project overview and quick start
- **[Docker Setup Guide](../DOCKER.md)** - Detailed Docker instructions
- **[Render Deployment](./RENDER_DEPLOYMENT.md)** - Render-specific guide
- **[API Documentation](../API_ENDPOINTS.md)** - Complete API reference
- **[Database Setup](./DATABASE_SETUP.md)** - Database configuration
- **[Testing Guide](./TESTING.md)** - Running tests
- **[Logging Guide](./LOGGING.md)** - Logging configuration

### External Resources

#### FastAPI
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices)

#### Docker
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

#### PostgreSQL
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [NeonDB Documentation](https://neon.tech/docs)

#### Redis
- [Redis Documentation](https://redis.io/documentation)
- [Redis Best Practices](https://redis.io/docs/manual/patterns/)
- [Upstash Documentation](https://docs.upstash.com/)

#### Celery
- [Celery Documentation](https://docs.celeryq.dev/)
- [Celery Best Practices](https://docs.celeryq.dev/en/stable/userguide/tasks.html#best-practices)

#### Cloud Platforms
- [Render Documentation](https://render.com/docs)
- [Railway Documentation](https://docs.railway.app/)
- [DigitalOcean App Platform](https://docs.digitalocean.com/products/app-platform/)
- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [GCP Cloud Run](https://cloud.google.com/run/docs)
- [Azure Container Instances](https://docs.microsoft.com/en-us/azure/container-instances/)

### Community

- **GitHub Issues** - Report bugs and request features
- **Discussions** - Ask questions and share ideas
- **Discord/Slack** - Real-time community support
- **Stack Overflow** - Tag: `ai-playground`

---

## 📋 Deployment Checklist

### Pre-Deployment

- [ ] Code reviewed and tested
- [ ] All tests passing
- [ ] Dependencies updated
- [ ] Security vulnerabilities addressed
- [ ] Environment variables documented
- [ ] Database migrations created
- [ ] Backup strategy in place
- [ ] Monitoring configured
- [ ] Error tracking setup (Sentry)
- [ ] Documentation updated

### Deployment

- [ ] External services provisioned (Database, Redis)
- [ ] Secrets generated and stored securely
- [ ] Environment variables configured
- [ ] Docker image built and tested
- [ ] Database migrations applied
- [ ] Application deployed
- [ ] Health checks passing
- [ ] SSL/HTTPS configured
- [ ] CORS configured correctly
- [ ] Rate limiting enabled

### Post-Deployment

- [ ] Smoke tests completed
- [ ] API endpoints responding
- [ ] Database connections working
- [ ] Celery workers processing tasks
- [ ] File uploads working
- [ ] Logs being collected
- [ ] Metrics being recorded
- [ ] Alerts configured
- [ ] Backup job running
- [ ] Documentation updated with URLs

### Monitoring

- [ ] Health check endpoint monitored
- [ ] Error rate alerts configured
- [ ] Performance metrics tracked
- [ ] Database performance monitored
- [ ] Disk space monitored
- [ ] Memory usage monitored
- [ ] API response times tracked
- [ ] Uptime monitoring active

---

## 🎯 Quick Reference

### Essential Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Run migrations
docker-compose exec backend alembic upgrade head

# Access database
psql $DATABASE_URL

# Check health
curl http://localhost:8000/health

# Run tests
docker-compose exec backend pytest

# Backup database
pg_dump $DATABASE_URL | gzip > backup.sql.gz

# Stop services
docker-compose down
```

### Essential URLs

```
Health Check:     http://localhost:8000/health
API Docs:         http://localhost:8000/docs
ReDoc:            http://localhost:8000/redoc
Metrics:          http://localhost:8000/metrics
```

### Essential Environment Variables

```bash
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379/0
SECRET_KEY=<32-char-random-string>
ENVIRONMENT=production
DEBUG=False
```

---

## 📞 Support

### Getting Help

1. **Check Documentation** - Most questions are answered here
2. **Search Issues** - Someone may have had the same problem
3. **Check Logs** - Error messages provide valuable clues
4. **Ask Community** - Discord, Slack, or GitHub Discussions
5. **Create Issue** - Provide detailed information

### Creating a Bug Report

Include:
- Environment details (OS, Docker version, etc.)
- Steps to reproduce
- Expected vs actual behavior
- Error messages and logs
- Configuration (without secrets!)
- Screenshots if applicable

### Feature Requests

Include:
- Use case description
- Expected behavior
- Why it's important
- Proposed implementation (optional)

---

## 📄 License

This project is for educational purposes.

---

## 🙏 Acknowledgments

- FastAPI team for the excellent framework
- PostgreSQL and Redis communities
- Docker for containerization
- All contributors and users

---

**Last Updated:** January 2, 2026  
**Version:** 1.0.0  
**Maintainer:** AI-Playground Team

For questions or issues, please open a GitHub issue or contact the team.

