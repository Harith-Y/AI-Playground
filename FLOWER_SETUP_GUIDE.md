# Flower Celery Monitoring - Complete Setup Guide

## Overview

This guide covers setting up Flower, a powerful web-based tool for monitoring and administrating Celery clusters. Flower provides real-time monitoring, task management, and worker control.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Authentication](#authentication)
5. [Docker Deployment](#docker-deployment)
6. [Advanced Configuration](#advanced-configuration)
7. [Features Overview](#features-overview)
8. [Production Deployment](#production-deployment)
9. [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# 1. Install Flower
pip install -r requirements.monitoring.txt

# 2. Start Flower (development, no auth)
python start_flower.py

# 3. Access dashboard
open http://localhost:5555/flower
```

## Installation

### Option 1: Using pip

```bash
# Install Flower only
pip install flower

# Or install all monitoring dependencies
pip install -r requirements.monitoring.txt
```

### Option 2: Using Docker

```bash
# Build and start with Docker Compose
docker-compose -f docker-compose.flower.yml up -d
```

## Basic Usage

### Development Mode (No Authentication)

```bash
python start_flower.py
```

**Access:** http://localhost:5555/flower

**⚠️ Warning:** No authentication - suitable for development only!

### Production Mode (With Authentication)

```bash
# Using default credentials
python start_flower.py --auth

# Using custom credentials
python start_flower.py --auth --basic_auth=admin:secretpassword

# Using environment variable
export FLOWER_BASIC_AUTH=admin:secretpassword
python start_flower.py
```

### Custom Port and Address

```bash
# Custom port
python start_flower.py --port=8888

# Bind to specific address
python start_flower.py --address=127.0.0.1

# Both
python start_flower.py --port=8888 --address=0.0.0.0
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--port` | Port to run on | 5555 |
| `--address` | Address to bind to | 0.0.0.0 |
| `--auth` | Enable authentication | False |
| `--basic_auth` | Basic auth credentials (user:pass) | admin:admin123 |
| `--debug` | Enable debug mode | False |
| `--max_tasks` | Maximum tasks to keep | 10000 |
| `--db` | Database file path | flower.db |
| `--url_prefix` | URL prefix | flower |
| `--no-persistent` | Disable persistence | False |

## Authentication

### Basic Authentication

**Development:**
```bash
python start_flower.py --auth
# Credentials: admin:admin123
```

**Production:**
```bash
python start_flower.py --auth --basic_auth=myuser:mypassword
```

**Environment Variable:**
```bash
export FLOWER_BASIC_AUTH=myuser:mypassword
python start_flower.py
```

### OAuth2 Authentication (Advanced)

Flower supports OAuth2 providers (Google, GitHub, GitLab, etc.):

```bash
celery -A celery_worker.celery_app flower \
    --auth=".*@example\.com" \
    --auth_provider=google \
    --oauth2_key=your_client_id \
    --oauth2_secret=your_client_secret \
    --oauth2_redirect_uri=http://localhost:5555/flower/login
```

### Multiple Users

```bash
# Comma-separated user:pass pairs
python start_flower.py --basic_auth=admin:pass1,user:pass2,viewer:pass3
```

## Docker Deployment

### Using Docker Compose

**1. Configure environment:**

Create `.env` file:
```env
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
FLOWER_BASIC_AUTH=admin:yoursecretpassword
```

**2. Start services:**

```bash
docker-compose -f docker-compose.flower.yml up -d
```

**3. Access Flower:**

```
http://localhost:5555/flower
```

**4. View logs:**

```bash
docker-compose -f docker-compose.flower.yml logs -f flower
```

**5. Stop services:**

```bash
docker-compose -f docker-compose.flower.yml down
```

### Building Custom Image

```bash
cd backend
docker build -f Dockerfile.flower -t aiplayground-flower .
docker run -p 5555:5555 \
    -e CELERY_BROKER_URL=redis://host.docker.internal:6379/0 \
    -e FLOWER_BASIC_AUTH=admin:password \
    aiplayground-flower
```

## Advanced Configuration

### Configuration File

Create `flowerconfig.py`:

```python
# Broker URL
broker_url = 'redis://localhost:6379/0'

# Result backend
result_backend = 'redis://localhost:6379/0'

# Port
port = 5555

# Address
address = '0.0.0.0'

# URL prefix
url_prefix = 'flower'

# Max tasks to keep in memory
max_tasks = 10000

# Basic authentication
basic_auth = ['admin:password', 'user:password']

# Enable debug mode
debug = False

# Persistent mode
persistent = True
db = 'flower.db'

# Natural time
natural_time = True

# Task columns to display
task_columns = [
    'name',
    'uuid',
    'state',
    'args',
    'kwargs',
    'result',
    'received',
    'started',
    'runtime',
    'worker',
]
```

**Usage:**
```bash
celery -A celery_worker.celery_app flower --conf=flowerconfig
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLOWER_BASIC_AUTH` | Authentication credentials | None |
| `FLOWER_PORT` | Port to run on | 5555 |
| `FLOWER_DEBUG` | Debug mode | false |
| `CELERY_BROKER_URL` | Broker URL | Required |
| `CELERY_RESULT_BACKEND` | Result backend URL | Required |

### Persistent Storage

Flower can persist task history to a database:

```bash
# SQLite (default)
python start_flower.py --db=flower.db

# PostgreSQL
celery -A celery_worker.celery_app flower \
    --db=postgresql://user:pass@localhost/flower

# MySQL
celery -A celery_worker.celery_app flower \
    --db=mysql://user:pass@localhost/flower
```

## Features Overview

### Dashboard

- **Real-time monitoring** of tasks and workers
- **Task statistics** (success rate, duration, etc.)
- **Worker status** and load
- **Queue lengths** and backlogs
- **Broker monitoring**

### Tasks View

- **Filter tasks** by state, name, worker
- **Search tasks** by ID or name
- **View task details** (args, kwargs, result, traceback)
- **Task timeline** visualization
- **Retry failed tasks**
- **Revoke running tasks**

### Workers View

- **Worker list** with status
- **Pool size** and active tasks
- **Resource usage** (CPU, memory, load average)
- **Registered tasks** per worker
- **Shutdown workers** remotely
- **Pool grow/shrink** operations
- **Restart workers**

### Broker View

- **Broker connection status**
- **Queue statistics**
- **Message counts**
- **Connection details**

### Monitor View

- **Real-time task execution**
- **Task rate graphs**
- **Success/failure trends**
- **Worker activity timeline**

### API

Flower provides a RESTful API:

```bash
# Get all tasks
curl http://localhost:5555/flower/api/tasks

# Get specific task
curl http://localhost:5555/flower/api/task/info/{task-id}

# Get workers
curl http://localhost:5555/flower/api/workers

# Revoke task
curl -X POST http://localhost:5555/flower/api/task/revoke/{task-id}

# Shutdown worker
curl -X POST http://localhost:5555/flower/api/worker/shutdown/{worker-name}
```

## Production Deployment

### 1. Security Checklist

- ✅ **Enable authentication** (basic auth or OAuth2)
- ✅ **Use HTTPS** (via reverse proxy)
- ✅ **Strong passwords** (not default credentials)
- ✅ **Firewall rules** (restrict access)
- ✅ **Regular updates** (keep Flower updated)

### 2. Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name flower.example.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name flower.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location /flower {
        proxy_pass http://localhost:5555/flower;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts
        proxy_connect_timeout 600;
        proxy_send_timeout 600;
        proxy_read_timeout 600;
        send_timeout 600;
    }
}
```

### 3. Systemd Service

Create `/etc/systemd/system/flower.service`:

```ini
[Unit]
Description=Flower Celery Monitoring
After=network.target redis.service

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/var/www/aiplayground/backend
Environment="FLOWER_BASIC_AUTH=admin:yourpassword"
Environment="PATH=/var/www/aiplayground/venv/bin"
ExecStart=/var/www/aiplayground/venv/bin/python start_flower.py --port=5555
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Start service:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable flower
sudo systemctl start flower
sudo systemctl status flower
```

### 4. Docker Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  flower:
    build:
      context: ./backend
      dockerfile: Dockerfile.flower
    restart: always
    ports:
      - "127.0.0.1:5555:5555"  # Bind to localhost only
    environment:
      - CELERY_BROKER_URL=${CELERY_BROKER_URL}
      - CELERY_RESULT_BACKEND=${CELERY_RESULT_BACKEND}
      - FLOWER_BASIC_AUTH=${FLOWER_BASIC_AUTH}
    volumes:
      - flower-data:/data
    networks:
      - internal
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
```

### 5. Monitoring Flower Itself

**Health Check:**
```bash
curl http://localhost:5555/flower/healthcheck
```

**Prometheus Metrics:**
```bash
# Flower exposes some metrics at /metrics
curl http://localhost:5555/flower/metrics
```

## Troubleshooting

### Flower Won't Start

**Error:** `ModuleNotFoundError: No module named 'flower'`

**Solution:**
```bash
pip install flower
```

---

**Error:** `Connection refused to broker`

**Solution:**
- Check broker is running: `redis-cli ping`
- Verify broker URL in configuration
- Check network connectivity

---

**Error:** `Permission denied on port 5555`

**Solution:**
- Use port > 1024 or run with sudo
- Or use `--port=8888`

### Authentication Issues

**Can't login with credentials**

**Solution:**
- Check format: `user:password` (no spaces)
- Verify environment variable is set
- Try URL encoding special characters

### Tasks Not Appearing

**No tasks show in dashboard**

**Solution:**
- Enable task events: `celery_app.conf.worker_send_task_events = True`
- Restart Celery workers
- Check broker connection

### High Memory Usage

**Flower using too much memory**

**Solution:**
- Reduce `--max_tasks` value
- Enable task event expiration
- Restart Flower periodically

### WebSocket Connection Failed

**Real-time updates not working**

**Solution:**
- Check reverse proxy WebSocket support
- Verify firewall allows WebSocket connections
- Try accessing without reverse proxy

## Best Practices

1. **Always use authentication** in production
2. **Enable HTTPS** via reverse proxy
3. **Limit access** with firewall rules
4. **Monitor Flower** itself (memory, CPU)
5. **Regular backups** of flower.db
6. **Update regularly** for security patches
7. **Use persistent storage** for task history
8. **Set appropriate max_tasks** limit
9. **Configure log rotation**
10. **Document access credentials** securely

## Useful Commands

```bash
# Start with custom settings
python start_flower.py --port=8888 --max_tasks=5000 --auth

# View Flower help
python start_flower.py --help

# Check Flower version
celery -A celery_worker.celery_app flower --version

# Run in background
nohup python start_flower.py &> flower.log &

# View logs
tail -f flower.log

# Stop Flower
pkill -f "python start_flower.py"
```

## Additional Resources

- [Flower Documentation](https://flower.readthedocs.io/)
- [Flower GitHub](https://github.com/mher/flower)
- [Celery Documentation](https://docs.celeryproject.org/)
- [Celery Monitoring Guide](https://docs.celeryproject.org/en/stable/userguide/monitoring.html)

## Support

For issues:
1. Check logs: `tail -f flower.log`
2. Verify broker connectivity
3. Check Celery worker status
4. Review Flower GitHub issues
5. Consult Celery documentation
