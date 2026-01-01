# AI Playground Logging Guide

Comprehensive guide for logging, monitoring, and troubleshooting the AI Playground application.

## Table of Contents

- [Overview](#overview)
- [Log Levels](#log-levels)
- [Log Types](#log-types)
- [Configuration](#configuration)
- [Production Logging](#production-logging)
- [Log Aggregation](#log-aggregation)
- [Viewing Logs](#viewing-logs)
- [Log Rotation](#log-rotation)
- [Monitoring & Alerts](#monitoring--alerts)
- [Troubleshooting](#troubleshooting)

## Overview

### Logging Architecture

```
Application
    ├── Console Logs (stdout/stderr)
    ├── File Logs (/var/log/aiplayground/)
    │   ├── application.log
    │   ├── error.log
    │   ├── access.log
    │   ├── security.log
    │   ├── performance.log
    │   ├── celery.log
    │   └── preprocessing.log
    ├── Fluentd (log aggregator)
    └── Storage
        ├── Elasticsearch (searchable)
        ├── Loki (time-series)
        └── S3 (long-term)
```

### Features

✅ **Structured JSON logging** for easy parsing
✅ **Multiple log levels** (DEBUG, INFO, WARNING, ERROR, CRITICAL)
✅ **Separate log files** by type and severity
✅ **Automatic log rotation** with compression
✅ **Request/response logging** with unique IDs
✅ **Security event logging** (auth, access)
✅ **Performance monitoring** (slow queries)
✅ **Log aggregation** (ELK Stack, Loki)
✅ **Real-time visualization** (Kibana, Grafana)

## Log Levels

### Development

Default: `DEBUG`

```env
LOG_LEVEL=DEBUG
```

**Logs everything** including debug information, SQL queries, and detailed tracing.

### Staging

Default: `INFO`

```env
LOG_LEVEL=INFO
```

**Logs normal operations**, warnings, and errors. Excludes debug information.

### Production

Default: `WARNING` or `INFO`

```env
LOG_LEVEL=WARNING  # Recommended for high-traffic
# or
LOG_LEVEL=INFO  # For detailed monitoring
```

**Logs warnings and errors only** to reduce noise.

## Log Types

### Application Log (`application.log`)

General application events and operations.

**Example:**
```json
{
  "timestamp": "2026-01-01T12:00:00.000Z",
  "level": "INFO",
  "logger": "app.main",
  "message": "Application started",
  "environment": "production",
  "source": {
    "module": "main",
    "function": "startup",
    "line": 42
  }
}
```

### Error Log (`error.log`)

Application errors and exceptions.

**Example:**
```json
{
  "timestamp": "2026-01-01T12:00:00.000Z",
  "level": "ERROR",
  "logger": "app.api.datasets",
  "message": "Failed to process dataset",
  "error": "ValueError: Invalid format",
  "traceback": "...",
  "dataset_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

### Access Log (`access.log`)

HTTP requests and responses.

**Example:**
```json
{
  "timestamp": "2026-01-01T12:00:00.000Z",
  "level": "INFO",
  "event": "api_request",
  "method": "POST",
  "path": "/api/v1/datasets/upload",
  "status_code": 201,
  "response_time_ms": 245.67,
  "request_id": "abc123-def456",
  "user_id": "user-uuid",
  "ip_address": "192.168.1.100"
}
```

### Security Log (`security.log`)

Authentication, authorization, and security events.

**Example:**
```json
{
  "timestamp": "2026-01-01T12:00:00.000Z",
  "level": "INFO",
  "event": "security_event",
  "event_type": "login",
  "success": true,
  "user_id": "user-uuid",
  "ip_address": "192.168.1.100",
  "details": {"method": "jwt"}
}
```

### Performance Log (`performance.log`)

Slow operations and performance metrics.

**Example:**
```json
{
  "timestamp": "2026-01-01T12:00:00.000Z",
  "level": "WARNING",
  "event": "performance_metric",
  "operation": "POST /api/v1/models/train",
  "duration_ms": 5234.12,
  "details": {
    "dataset_rows": 100000,
    "model_type": "random_forest"
  }
}
```

### Celery Log (`celery.log`)

Background task execution.

**Example:**
```json
{
  "timestamp": "2026-01-01T12:00:00.000Z",
  "level": "INFO",
  "logger": "celery.worker",
  "message": "Task completed",
  "task_id": "task-uuid",
  "task_name": "preprocessing.apply_pipeline",
  "execution_time": 45.2
}
```

### Preprocessing Log (`preprocessing.log`)

Data preprocessing operations.

**Example:**
```json
{
  "timestamp": "2026-01-01T12:00:00.000Z",
  "level": "INFO",
  "event": "preprocessing_step_completed",
  "dataset_id": "dataset-uuid",
  "step_type": "handle_missing_values",
  "execution_time_seconds": 2.5,
  "rows_before": 1000,
  "rows_after": 980,
  "cols_before": 10,
  "cols_after": 10
}
```

## Configuration

### Environment Variables

Add to [.env.docker](.env.docker) or [.env.docker.production](.env.docker.production):

```env
# Logging Configuration
LOG_LEVEL=INFO
LOG_DIR=/var/log/aiplayground
ENABLE_JSON_LOGGING=true
ENABLE_FILE_LOGGING=true

# Log Rotation
LOG_MAX_BYTES=52428800  # 50 MB
LOG_BACKUP_COUNT=30     # Keep 30 files

# Sentry (optional)
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=0.1

# Log Aggregation (optional)
ELASTICSEARCH_HOST=elasticsearch
ELASTICSEARCH_PORT=9200
LOKI_URL=http://loki:3100
```

### Code Configuration

#### Basic Setup

```python
from app.core.logging_production import setup_production_logging

# Initialize logging
setup_production_logging(
    log_level="INFO",
    log_dir="/var/log/aiplayground",
    enable_json=True,
    enable_compression=True
)
```

#### Custom Logger

```python
import logging

logger = logging.getLogger(__name__)
logger.info("Application started")
logger.error("An error occurred", exc_info=True)
```

#### Request Logging

```python
from app.core.logging_production import log_api_request

log_api_request(
    method="POST",
    path="/api/v1/datasets/upload",
    status_code=201,
    response_time_ms=245.67,
    request_id="abc123",
    user_id="user-uuid",
    ip_address="192.168.1.100"
)
```

#### Security Logging

```python
from app.core.logging_production import log_security_event

log_security_event(
    event_type="login",
    user_id="user-uuid",
    ip_address="192.168.1.100",
    success=True,
    details={"method": "jwt"}
)
```

#### Performance Logging

```python
from app.core.logging_production import log_performance_metric

log_performance_metric(
    operation="database_query",
    duration_ms=1250.5,
    details={"query": "SELECT * FROM users"}
)
```

## Production Logging

### Enable Production Logging

#### 1. Configure main.py

Add logging middleware to [backend/app/main.py](backend/app/main.py):

```python
from app.core.logging_production import setup_production_logging
from app.middleware import (
    RequestLoggingMiddleware,
    SecurityLoggingMiddleware,
    ErrorLoggingMiddleware
)

# Initialize production logging
setup_production_logging(
    log_level=settings.LOG_LEVEL,
    log_dir=settings.LOG_DIR,
    enable_sentry=True,
    sentry_dsn=settings.SENTRY_DSN
)

# Add middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SecurityLoggingMiddleware)
app.add_middleware(ErrorLoggingMiddleware)
```

#### 2. Configure Docker Logging

Update [docker-compose.production.yml](docker-compose.production.yml):

```yaml
services:
  backend:
    logging:
      driver: "fluentd"
      options:
        fluentd-address: "localhost:24224"
        tag: "aiplayground.backend"
```

#### 3. Start with Logging Stack

```bash
# Start application with logging infrastructure
docker-compose \
  -f docker-compose.yml \
  -f docker-compose.production.yml \
  -f docker-compose.logging.yml \
  up -d
```

## Log Aggregation

### ELK Stack (Elasticsearch, Logstash, Kibana)

#### Start ELK Stack

```bash
docker-compose -f docker-compose.logging.yml up -d elasticsearch kibana
```

#### Access Kibana

1. Open http://localhost:5601
2. Go to Management → Stack Management → Index Patterns
3. Create index pattern: `aiplayground-*`
4. Set time field: `@timestamp`
5. Go to Discover to view logs

#### Example Queries

**Find all errors:**
```
level: "ERROR"
```

**Find slow requests:**
```
response_time_ms: >1000
```

**Find failed logins:**
```
event_type: "login" AND success: false
```

### Loki + Grafana

#### Start Loki Stack

```bash
docker-compose -f docker-compose.logging.yml up -d loki promtail grafana
```

#### Access Grafana

1. Open http://localhost:3000
2. Login: `admin` / `admin`
3. Go to Explore
4. Select Loki datasource
5. Query logs

#### Example LogQL Queries

**All logs from backend:**
```logql
{app="backend"}
```

**Error logs:**
```logql
{app="backend", level="ERROR"}
```

**Slow requests:**
```logql
{type="access"} | json | response_time_ms > 1000
```

**Failed authentication:**
```logql
{type="security", event_type="login"} | json | success="false"
```

## Viewing Logs

### Docker Logs

```bash
# View all services
docker-compose logs

# Follow logs
docker-compose logs -f

# Specific service
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend

# Timestamps
docker-compose logs -t backend
```

### File Logs

```bash
# View application log
tail -f /var/log/aiplayground/application.log

# View errors
tail -f /var/log/aiplayground/error.log

# View access log
tail -f /var/log/aiplayground/access.log

# Search for errors
grep "ERROR" /var/log/aiplayground/application.log

# Count errors
grep -c "ERROR" /var/log/aiplayground/application.log

# Parse JSON logs with jq
tail -f /var/log/aiplayground/application.log | jq .
```

### Search Logs

#### Using grep

```bash
# Find user ID in logs
grep "user-uuid" /var/log/aiplayground/*.log

# Find requests to specific endpoint
grep "/api/v1/datasets" /var/log/aiplayground/access.log

# Find errors in date range
grep "2026-01-01" /var/log/aiplayground/error.log | grep "ERROR"
```

#### Using jq (JSON logs)

```bash
# Find requests slower than 1 second
cat /var/log/aiplayground/access.log | \
  jq 'select(.response_time_ms > 1000)'

# Find requests by user
cat /var/log/aiplayground/access.log | \
  jq 'select(.user_id == "user-uuid")'

# Count requests by status code
cat /var/log/aiplayground/access.log | \
  jq -r '.status_code' | sort | uniq -c
```

## Log Rotation

### Logrotate (Linux)

#### Install Configuration

```bash
sudo cp docker/logging/logrotate.conf /etc/logrotate.d/aiplayground
sudo chmod 644 /etc/logrotate.d/aiplayground
```

#### Test Configuration

```bash
sudo logrotate -d /etc/logrotate.d/aiplayground
```

#### Force Rotation

```bash
sudo logrotate -f /etc/logrotate.d/aiplayground
```

#### Manual Configuration

See [docker/logging/logrotate.conf](docker/logging/logrotate.conf) for details.

### Built-in Python Rotation

Logs automatically rotate when they reach:
- **50 MB** per file (configurable)
- **30 backup files** kept (configurable)
- **Compressed** with gzip

### Retention Policy

| Log Type | Rotation | Retention |
|----------|----------|-----------|
| Application | Daily | 30 days |
| Error | Daily | 90 days |
| Access | Daily | 14 days |
| Security | Daily | 180 days (6 months) |
| Performance | Daily | 30 days |
| Celery | Daily | 14 days |
| Preprocessing | Daily | 30 days |

## Monitoring & Alerts

### Metrics to Monitor

1. **Error Rate**
   - Threshold: > 1% of requests
   - Alert: Email, Slack

2. **Slow Requests**
   - Threshold: > 1000ms
   - Alert: Performance log

3. **Failed Authentications**
   - Threshold: > 10 failures/minute from same IP
   - Alert: Security log, Email

4. **Disk Space**
   - Threshold: > 80% full
   - Alert: Email, Critical

5. **Log Volume**
   - Unexpected spike in logs
   - May indicate attack or issue

### Set Up Alerts (Grafana)

1. Go to Alerting → Alert Rules
2. Create New Alert Rule
3. Set conditions (e.g., error rate > 1%)
4. Add notification channel (Email, Slack)

### Example Alert Query (LogQL)

```logql
sum(count_over_time({app="backend", level="ERROR"}[5m]))
> 100
```

## Troubleshooting

### Common Issues

#### Logs Not Appearing

**Check log directory:**
```bash
ls -la /var/log/aiplayground/
```

**Check permissions:**
```bash
sudo chown -R www-data:www-data /var/log/aiplayground/
sudo chmod -R 755 /var/log/aiplayground/
```

**Check Docker volumes:**
```bash
docker volume ls | grep logs
docker volume inspect ai-playground_backend_logs
```

#### Logs Too Large

**Enable compression:**
```env
ENABLE_COMPRESSION=true
```

**Reduce log level:**
```env
LOG_LEVEL=WARNING
```

**Increase rotation:**
```bash
# Edit logrotate.conf
rotate 7  # Keep only 7 days
```

#### Elasticsearch Not Working

**Check Elasticsearch:**
```bash
curl http://localhost:9200/_cluster/health
```

**Check indices:**
```bash
curl http://localhost:9200/_cat/indices?v
```

**Clear old indices:**
```bash
curl -X DELETE "localhost:9200/aiplayground-2025.12.*"
```

#### Kibana Can't Connect

**Check network:**
```bash
docker network ls
docker network inspect aiplayground-network
```

**Restart Kibana:**
```bash
docker-compose -f docker-compose.logging.yml restart kibana
```

### Debug Logging

Enable debug logging temporarily:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Or specific logger
logging.getLogger('app.api.datasets').setLevel(logging.DEBUG)
```

### Log Analysis

#### Find Most Common Errors

```bash
cat /var/log/aiplayground/error.log | \
  jq -r '.message' | \
  sort | uniq -c | sort -rn | head -10
```

#### Request Latency Analysis

```bash
cat /var/log/aiplayground/access.log | \
  jq -r '.response_time_ms' | \
  awk '{sum+=$1; count++} END {print "Avg:", sum/count, "ms"}'
```

#### Requests per Hour

```bash
cat /var/log/aiplayground/access.log | \
  jq -r '.timestamp[:13]' | \
  uniq -c
```

## Best Practices

### DO:

✅ Use structured JSON logging in production
✅ Include request IDs for tracing
✅ Log security events (auth, access)
✅ Monitor error rates and set alerts
✅ Rotate logs regularly
✅ Use appropriate log levels
✅ Include context in log messages
✅ Test log aggregation before production

### DON'T:

❌ Log sensitive data (passwords, tokens, PII)
❌ Log at DEBUG level in production
❌ Ignore disk space for logs
❌ Skip log rotation
❌ Log excessive information
❌ Use print() instead of logging
❌ Ignore log errors

### Example Good Log Message

```python
logger.info(
    "Dataset uploaded successfully",
    extra={
        'dataset_id': dataset.id,
        'user_id': user.id,
        'file_size_mb': file_size / (1024 * 1024),
        'row_count': row_count,
        'processing_time_ms': processing_time
    }
)
```

## Additional Resources

- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)
- [Elasticsearch Guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [Grafana Loki Documentation](https://grafana.com/docs/loki/latest/)
- [Logrotate Manual](https://linux.die.net/man/8/logrotate)
- [FastAPI Logging](https://fastapi.tiangolo.com/tutorial/logging/)

## Support

For logging issues:
1. Check this documentation
2. View logs: `docker-compose logs -f`
3. Check log files in `/var/log/aiplayground/`
4. Verify configuration in `.env` files
5. Test log rotation: `sudo logrotate -d /etc/logrotate.d/aiplayground`
6. Open an issue with log samples (remove sensitive data!)
