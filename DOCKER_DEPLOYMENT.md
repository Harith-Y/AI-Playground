# Docker Deployment Guide

This guide explains how to deploy the AI Playground application using Docker Compose with proper environment variable configuration.

## Overview

The project includes:
- **[docker-compose.yml](docker-compose.yml)** - Base configuration for development
- **[docker-compose.production.yml](docker-compose.production.yml)** - Production overrides with environment variables
- **[.env.docker.example](.env.docker.example)** - Development environment template
- **[.env.docker.production](.env.docker.production)** - Production environment template

## Quick Start

### Development Environment

1. **Create your environment file:**
   ```bash
   cp .env.docker.example .env.docker
   ```

2. **Configure your environment:**
   Edit `.env.docker` and set your actual values, especially:
   - `DATABASE_URL` - Your Neon database connection string
   - Other service configurations as needed

3. **Start the application:**
   ```bash
   docker-compose up -d
   ```

4. **View logs:**
   ```bash
   docker-compose logs -f
   ```

### Production Environment

1. **Create your production environment file:**
   ```bash
   cp .env.docker.production .env.docker.production.local
   ```

2. **Configure production settings:**
   Edit `.env.docker.production.local` and set:
   - **Database:** `DATABASE_URL` with your production database
   - **Security:** Strong `SECRET_KEY` and `JWT_SECRET`
     ```bash
     python -c "import secrets; print(secrets.token_urlsafe(64))"
     ```
   - **Redis:** `REDIS_PASSWORD` if using password-protected Redis
   - **CORS:** `CORS_ORIGINS` with your production domain(s)
   - **API URLs:** `VITE_API_URL` and `VITE_WS_URL` for frontend
   - **Resource Limits:** Adjust CPU and memory limits based on your server

3. **Start in production mode:**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.production.yml --env-file .env.docker.production up -d
   ```

4. **Build with production settings:**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.production.yml --env-file .env.docker.production up -d --build
   ```

## Environment Variables

### Critical Variables to Configure

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host/db` |
| `SECRET_KEY` | Backend security key (64+ chars) | Generate with secrets module |
| `JWT_SECRET` | JWT token signing key (64+ chars) | Generate with secrets module |
| `REDIS_PASSWORD` | Redis authentication password | Strong random password |
| `CORS_ORIGINS` | Allowed frontend origins | `https://yourdomain.com` |
| `VITE_API_URL` | Frontend API endpoint | `https://api.yourdomain.com` |
| `VITE_WS_URL` | Frontend WebSocket endpoint | `wss://api.yourdomain.com/ws` |

### Backend Configuration

- **Database:** Connection pooling, migrations
- **Redis:** Caching and Celery broker
- **Security:** Keys, tokens, CORS, rate limiting
- **Storage:** File uploads, size limits
- **Features:** Model availability, tier limits
- **Monitoring:** Sentry, metrics, logging

### Frontend Configuration

- **API:** Backend URLs, timeout settings
- **Features:** WebSockets, analytics, model limits
- **Security:** CSP, HTTPS enforcement
- **UI:** Theme, pagination settings

### Resource Limits (Production)

Default production resource limits:
- **Backend:** 4 CPUs, 4GB RAM
- **Celery Worker:** 4 CPUs, 4GB RAM
- **Redis:** 1 CPU, 1GB RAM
- **Frontend:** 2 CPUs, 1GB RAM

Adjust in [.env.docker.production](.env.docker.production):
```env
BACKEND_CPU_LIMIT=4
BACKEND_MEM_LIMIT=4g
```

## Services

The application consists of:

1. **Redis** - Cache and message broker (port 6379)
2. **Backend** - FastAPI application (port 8000)
3. **Celery Worker** - Background task processor
4. **Frontend** - React application (port 80)

## Common Commands

### Development

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Rebuild and start
docker-compose up -d --build

# View logs
docker-compose logs -f [service_name]

# Execute command in container
docker-compose exec backend bash
```

### Production

```bash
# Start production
docker-compose -f docker-compose.yml -f docker-compose.production.yml --env-file .env.docker.production up -d

# Stop production
docker-compose -f docker-compose.yml -f docker-compose.production.yml down

# Update and restart
docker-compose -f docker-compose.yml -f docker-compose.production.yml --env-file .env.docker.production up -d --build --force-recreate
```

### Maintenance

```bash
# Check service health
docker-compose ps

# View resource usage
docker stats

# Remove unused volumes
docker volume prune

# Clean up everything
docker-compose down -v
```

## Security Checklist

Before deploying to production:

- [ ] Generate strong `SECRET_KEY` (64+ characters)
- [ ] Generate strong `JWT_SECRET` (64+ characters)
- [ ] Set strong `REDIS_PASSWORD`
- [ ] Configure proper `CORS_ORIGINS`
- [ ] Set `DEBUG=false`
- [ ] Set `ENVIRONMENT=production`
- [ ] Configure SSL/TLS certificates
- [ ] Enable rate limiting
- [ ] Set up monitoring (Sentry, metrics)
- [ ] Configure backup strategy
- [ ] Review resource limits
- [ ] Never commit `.env` files to git

## Troubleshooting

### Backend won't start
- Check `DATABASE_URL` is correct
- Ensure database is accessible
- Verify Redis is running: `docker-compose logs redis`

### Frontend can't connect to backend
- Check `VITE_API_URL` matches backend URL
- Verify CORS settings allow frontend domain
- Check firewall/network settings

### Database migrations fail
- Ensure database exists and is accessible
- Check database credentials
- Run manually: `docker-compose exec backend alembic upgrade head`

### Redis connection errors
- Verify Redis is healthy: `docker-compose ps`
- Check Redis URL format
- Verify password if authentication is enabled

## Monitoring

### Health Checks

All services include health checks:
- Backend: `http://localhost:8000/health`
- Frontend: `http://localhost/health`
- Redis: `redis-cli ping`

### Logs

View service logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend
```

## Backup and Recovery

### Database Backup

```bash
# Backup
docker-compose exec backend pg_dump $DATABASE_URL > backup.sql

# Restore
docker-compose exec -T backend psql $DATABASE_URL < backup.sql
```

### Volume Backup

```bash
# Backup uploads
docker run --rm -v ai-playground_backend_uploads:/data -v $(pwd):/backup alpine tar czf /backup/uploads-backup.tar.gz /data

# Restore uploads
docker run --rm -v ai-playground_backend_uploads:/data -v $(pwd):/backup alpine tar xzf /backup/uploads-backup.tar.gz -C /
```

## Scaling

### Horizontal Scaling

Scale specific services:
```bash
# Scale Celery workers
docker-compose up -d --scale celery-worker=3

# Scale backend (requires load balancer)
docker-compose up -d --scale backend=2
```

### Vertical Scaling

Adjust resource limits in [.env.docker.production](.env.docker.production):
```env
BACKEND_CPU_LIMIT=8
BACKEND_MEM_LIMIT=8g
```

## Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Nginx Configuration](./docker/nginx/) (if applicable)
- [Redis Configuration](./docker/redis/)

## Support

For issues or questions:
1. Check logs: `docker-compose logs -f`
2. Verify environment variables are set correctly
3. Review this documentation
4. Open an issue on GitHub
