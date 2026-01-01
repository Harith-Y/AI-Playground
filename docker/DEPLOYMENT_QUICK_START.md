# 🚀 Quick Start Deployment Guide

**For detailed instructions, see [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)**

---

## 🎯 Choose Your Deployment

### 1. Local Development (5 minutes)

```bash
# Clone and start
git clone https://github.com/your-org/AI-Playground.git
cd AI-Playground
docker-compose up -d

# Verify
curl http://localhost:8000/health
open http://localhost:8000/docs
```

✅ **Done!** All services running locally.

---

### 2. Render (Free Tier - 15 minutes)

**Best for:** MVPs, demos, small projects

**Steps:**
1. Push code to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect repository, select `backend/` directory
4. Add environment variables:
   ```bash
   DATABASE_URL=<neondb-url>
   REDIS_URL=<upstash-url>
   SECRET_KEY=<generate-with-openssl-rand-hex-32>
   ENVIRONMENT=production
   DEBUG=False
   ```
5. Deploy!

**Limitations:** 512MB RAM, spins down after 15 min inactivity

**See:** [RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md)

---

### 3. Railway ($5/month - 20 minutes)

**Best for:** Growing projects, no cold starts

**Steps:**
```bash
# Install CLI
npm install -g @railway/cli
railway login

# Deploy
cd backend
railway init
railway add --database postgres
railway add --database redis
railway up

# Set variables
railway variables set SECRET_KEY=<your-secret>
railway variables set ENVIRONMENT=production
railway variables set DEBUG=False

# Get URL
railway domain
```

✅ **Done!** No cold starts, built-in databases.

---

### 4. DigitalOcean ($12/month - 30 minutes)

**Best for:** Production apps, predictable pricing

**Steps:**
1. Go to [DigitalOcean](https://cloud.digitalocean.com)
2. Apps → Create App → Connect GitHub
3. Configure:
   - Source: `backend/`
   - Type: Docker
   - Plan: Basic ($12/mo)
4. Add managed databases (PostgreSQL + Redis)
5. Set environment variables
6. Deploy!

✅ **Done!** Production-ready with managed databases.

---

### 5. AWS (Enterprise - 2 hours)

**Best for:** Enterprise, high-traffic, compliance

**Quick Setup:**
```bash
# Build and push to ECR
aws ecr create-repository --repository-name aiplayground-backend
docker build -t aiplayground-backend .
docker tag aiplayground-backend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/aiplayground-backend:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/aiplayground-backend:latest

# Create RDS PostgreSQL
aws rds create-db-instance --db-instance-identifier aiplayground-db ...

# Create ElastiCache Redis
aws elasticache create-cache-cluster --cache-cluster-id aiplayground-redis ...

# Deploy to ECS Fargate
aws ecs create-cluster --cluster-name aiplayground-cluster
aws ecs create-service ...
```

**See:** [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md#aws-enterprise-grade)

---

## 🔑 Required Services

### PostgreSQL Database

**Recommended:** [NeonDB](https://neon.tech) (Free tier, serverless)

**Alternatives:**
- Supabase (Free tier)
- AWS RDS (Enterprise)
- DigitalOcean Managed Database (Simple)

### Redis Cache

**Recommended:** [Upstash](https://upstash.com) (Free tier, serverless)

**Alternatives:**
- Redis Cloud (Official)
- AWS ElastiCache (Enterprise)
- DigitalOcean Managed Redis (Simple)

---

## ⚙️ Essential Environment Variables

```bash
# Required
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require
REDIS_URL=rediss://default:pass@host:6379
SECRET_KEY=<openssl rand -hex 32>
CELERY_BROKER_URL=${REDIS_URL}
CELERY_RESULT_BACKEND=${REDIS_URL}

# Recommended
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO
ALLOWED_ORIGINS=https://your-frontend.com

# Optional
MAX_UPLOAD_SIZE=104857600
SENTRY_DSN=https://...
```

---

## 🔒 Security Checklist

- [ ] Generate strong SECRET_KEY: `openssl rand -hex 32`
- [ ] Use HTTPS in production
- [ ] Configure CORS with specific origins (not `*`)
- [ ] Use SSL for database: `?sslmode=require`
- [ ] Use TLS for Redis: `rediss://`
- [ ] Never commit `.env` files
- [ ] Set `DEBUG=False` in production
- [ ] Enable rate limiting

---

## ✅ Post-Deployment Verification

```bash
# 1. Health check
curl https://your-api.com/health
# Expected: {"status": "healthy", "version": "1.0.0"}

# 2. API docs
open https://your-api.com/docs

# 3. Test endpoint
curl https://your-api.com/api/v1/datasets

# 4. Check logs
# (Platform-specific: Render dashboard, Railway CLI, etc.)
```

---

## 🐛 Common Issues

### Database Connection Failed
```bash
# Check connection string format
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require

# Test connection
psql $DATABASE_URL -c "SELECT 1"
```

### Out of Memory
```bash
# Use lightweight dependencies
# In Dockerfile:
ARG REQUIREMENTS_FILE=requirements.render.txt

# Reduce workers
command: uvicorn app.main:app --workers 1
```

### CORS Errors
```python
# Set specific origins
ALLOWED_ORIGINS=https://your-frontend.com,https://www.your-frontend.com
```

---

## 📚 Full Documentation

For detailed instructions, troubleshooting, and advanced topics:

**[→ Complete Deployment Guide](./DEPLOYMENT_GUIDE.md)**

Topics covered:
- Detailed platform guides (Render, Railway, DO, AWS, GCP, Azure)
- Kubernetes deployment
- Database setup and migrations
- Monitoring and logging
- Security best practices
- Scaling strategies
- Disaster recovery
- Performance optimization

---

## 📞 Need Help?

1. Check [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
2. Check [Troubleshooting section](./DEPLOYMENT_GUIDE.md#troubleshooting)
3. Search GitHub Issues
4. Create new issue with details

---

**Quick Links:**
- [Main README](../README.md)
- [Docker Guide](../DOCKER.md)
- [API Documentation](../API_ENDPOINTS.md)
- [Testing Guide](./TESTING.md)

