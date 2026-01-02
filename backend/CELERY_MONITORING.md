# Celery Task Monitoring & Metrics Guide

## Overview

This document describes the comprehensive monitoring and metrics system for Celery async tasks in the AI-Playground application. The monitoring infrastructure provides visibility into task execution, worker health, queue status, and performance metrics.

## Architecture

### Components

1. **Prometheus Metrics** - Time-series metrics collection
2. **Celery Signal Handlers** - Real-time event tracking
3. **REST API Endpoints** - Programmatic access to monitoring data
4. **Flower Dashboard** - Web-based visual monitoring (optional)
5. **Resource Monitoring** - System resource tracking

## Features

### 1. Task Execution Metrics

#### Task Duration Tracking
- Histogram of task execution times
- Labeled by task name and status (success/failure)
- Percentile tracking (P50, P95, P99)

```python
# Metric: celery_task_duration_seconds
# Labels: task_name, status
# Buckets: 1s, 5s, 10s, 30s, 60s, 120s, 300s, 600s, 1800s, 3600s, 7200s
```

#### Task Counter
- Total count of executed tasks
- Labeled by task name and status

```python
# Metric: celery_task_total
# Labels: task_name, status
# Values: success, failure, retry, revoked
```

#### Active Tasks Gauge
- Current number of running tasks
- Labeled by task name

```python
# Metric: celery_active_tasks
# Labels: task_name
```

### 2. Worker Pool Metrics

#### Worker Pool Size
```python
# Metric: celery_worker_pool_size
# Labels: worker_name
# Description: Number of worker processes in the pool
```

#### Busy Workers
```python
# Metric: celery_worker_pool_busy
# Labels: worker_name
# Description: Number of currently executing tasks
```

#### Prefetch Count
```python
# Metric: celery_worker_prefetch_count
# Labels: worker_name
# Description: Number of tasks prefetched by worker
```

### 3. Queue Metrics

#### Queue Length
```python
# Metric: celery_queue_length
# Labels: queue_name
# Description: Number of tasks waiting in queue
```

### 4. Failure Tracking

#### Failure Rate
```python
# Metric: celery_task_failure_rate
# Labels: task_name
# Description: Task failures per minute (rolling 1-minute window)
```

#### Retry Tracking
```python
# Metric: celery_task_retries_total
# Labels: task_name, reason
# Description: Total number of task retries with reasons
```

#### Timeout Tracking
```python
# Metric: celery_task_timeouts_total
# Labels: task_name
# Description: Tasks that exceeded time limits
```

### 5. Task-Specific Metrics

#### Training Metrics
- Training duration by model type
- Number of training samples
- Number of features
- Training success/failure counts
- Models trained counter

#### Tuning Metrics
- Tuning duration by method
- Number of iterations
- Best score achieved
- Tuning success/failure counts

### 6. Resource Metrics

- Memory usage (bytes)
- CPU usage (percentage)
- Disk I/O (bytes)
- Process counts

## API Endpoints

### Task Monitoring

#### Get Task Status
```http
GET /api/v1/celery/tasks/status
```

Returns current status of all tasks including active, reserved, and scheduled tasks.

**Response:**
```json
{
  "timestamp": "2026-01-02T10:30:00Z",
  "summary": {
    "active": 5,
    "reserved": 2,
    "scheduled": 0,
    "total_pending": 7
  },
  "active_tasks": [
    {
      "worker": "celery@worker1",
      "task_id": "abc123",
      "task_name": "app.tasks.training_tasks.train_model",
      "time_start": 1641120600.0
    }
  ],
  "workers": ["celery@worker1"]
}
```

#### Get Task Status by ID
```http
GET /api/v1/celery/tasks/{task_id}/status
```

Returns detailed status of a specific task.

**Response:**
```json
{
  "task_id": "abc123",
  "state": "SUCCESS",
  "ready": true,
  "successful": true,
  "result": {
    "model_run_id": "...",
    "metrics": {...}
  }
}
```

#### Revoke Task
```http
POST /api/v1/celery/tasks/{task_id}/revoke?terminate=false
```

Revoke a running or pending task.

**Query Parameters:**
- `terminate` (bool): Whether to forcefully terminate (default: false)

### Worker Monitoring

#### Get Worker Status
```http
GET /api/v1/celery/workers/status
```

Returns status of all Celery workers.

**Response:**
```json
{
  "timestamp": "2026-01-02T10:30:00Z",
  "total_workers": 1,
  "workers": [
    {
      "name": "celery@worker1",
      "status": "online",
      "pool": {
        "max_concurrency": 4,
        "busy_workers": 2
      },
      "prefetch_count": 4
    }
  ]
}
```

### Queue Monitoring

#### Get Queue Status
```http
GET /api/v1/celery/queues/status
```

Returns current queue lengths and pending tasks.

### Metrics Summary

#### Get Metrics Summary
```http
GET /api/v1/celery/metrics/summary
```

Returns comprehensive metrics summary.

### Health Check

#### Check Celery Health
```http
GET /api/v1/celery/health
```

Returns health status of Celery workers and broker.

**Response:**
```json
{
  "status": "healthy",
  "message": "All systems operational",
  "workers": 1,
  "broker_connected": true,
  "timestamp": "2026-01-02T10:30:00Z"
}
```

## Prometheus Integration

### Metrics Endpoint

All Prometheus metrics are exposed at:
```
http://localhost:8000/metrics
```

### Example Prometheus Queries

#### Task Success Rate
```promql
rate(celery_task_total{status="success"}[5m]) /
rate(celery_task_total[5m])
```

#### Average Task Duration
```promql
rate(celery_task_duration_seconds_sum[5m]) /
rate(celery_task_duration_seconds_count[5m])
```

#### P95 Task Duration
```promql
histogram_quantile(0.95, rate(celery_task_duration_seconds_bucket[5m]))
```

#### Worker Utilization
```promql
celery_worker_pool_busy / celery_worker_pool_size * 100
```

#### Task Failure Rate
```promql
celery_task_failure_rate
```

## Flower Web Dashboard

### Installation

```bash
pip install flower
```

### Starting Flower

```bash
# Using the provided script
python start_flower.py

# Or manually
celery -A app.celery_app flower --port=5555
```

### Accessing Flower

Open your browser to:
```
http://localhost:5555/flower
```

### Flower Features

- Real-time task monitoring
- Worker management
- Task history and details
- Broker monitoring
- Task rate graphs
- Task execution timeline

### Flower Configuration

Edit `start_flower.py` to customize:
- Port (default: 5555)
- Database persistence
- Max tasks to display
- URL prefix
- Authentication (optional)

## Usage Examples

### Python Client

```python
import requests

# Get task status
response = requests.get('http://localhost:8000/api/v1/celery/tasks/status')
data = response.json()
print(f"Active tasks: {data['summary']['active']}")

# Check specific task
task_id = "abc123"
response = requests.get(f'http://localhost:8000/api/v1/celery/tasks/{task_id}/status')
task_data = response.json()
print(f"Task state: {task_data['state']}")

# Revoke a task
requests.post(f'http://localhost:8000/api/v1/celery/tasks/{task_id}/revoke')
```

### Monitoring in Code

```python
from app.monitoring.celery_metrics import get_celery_metrics_summary
from app.celery_app import celery_app

# Get metrics summary
summary = get_celery_metrics_summary(celery_app)
print(f"Total workers: {summary['workers']['total']}")
print(f"Active tasks: {summary['tasks']['active']}")
```

## Alerting Rules

### Prometheus Alerting Rules

Example `prometheus-rules.yml`:

```yaml
groups:
  - name: celery_alerts
    rules:
      - alert: HighTaskFailureRate
        expr: celery_task_failure_rate > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High task failure rate"
          description: "Task {{ $labels.task_name }} has {{ $value }} failures per minute"

      - alert: WorkerDown
        expr: up{job="celery_workers"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Celery worker is down"

      - alert: QueueBacklog
        expr: celery_queue_length > 100
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Large queue backlog"
          description: "Queue {{ $labels.queue_name }} has {{ $value }} pending tasks"

      - alert: TaskTimeout
        expr: rate(celery_task_timeouts_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Tasks are timing out"
          description: "Task {{ $labels.task_name }} is experiencing timeouts"
```

## Best Practices

### 1. Resource Monitoring

- Monitor worker memory usage to prevent OOM issues
- Track CPU usage to optimize worker pool size
- Set appropriate soft and hard time limits

### 2. Queue Management

- Monitor queue lengths regularly
- Set up alerts for queue backlogs
- Use priority queues for critical tasks

### 3. Task Design

- Keep tasks idempotent
- Use task timeouts appropriately
- Implement retry logic with exponential backoff
- Break large tasks into smaller chunks

### 4. Performance Optimization

- Use task routing to distribute load
- Adjust prefetch multiplier based on task duration
- Enable task result compression for large results
- Use result expiration to prevent backend bloat

### 5. Monitoring Strategy

- Set up dashboards for key metrics
- Configure alerts for critical issues
- Review metrics regularly
- Analyze task duration trends

## Troubleshooting

### High Failure Rate

1. Check task logs for errors
2. Review retry configuration
3. Verify resource availability
4. Check for timeouts

### Queue Backlog

1. Check worker availability
2. Review task duration
3. Consider adding more workers
4. Optimize task code

### Worker Issues

1. Check worker logs
2. Verify broker connectivity
3. Review resource usage
4. Restart workers if needed

### Monitoring Issues

1. Verify Celery signals are connected
2. Check Prometheus scrape configuration
3. Review metric exposition
4. Validate API endpoint responses

## Configuration

### Celery Configuration

In `app/celery_app.py`:

```python
celery_app.conf.update(
    task_track_started=True,  # Required for progress tracking
    task_time_limit=3600,     # Hard time limit (1 hour)
    task_soft_time_limit=3000, # Soft time limit (50 minutes)
    worker_prefetch_multiplier=4,  # Tasks to prefetch per worker
    task_acks_late=True,      # Acknowledge tasks after completion
    worker_max_tasks_per_child=1000,  # Restart worker after N tasks
)
```

### Monitoring Configuration

In `backend/celery_worker.py`:

```python
# Setup Celery metrics
setup_celery_metrics(celery_app)

# Start resource monitoring
start_global_monitor(interval=30.0, process_type='celery_worker')
```

## Metrics Export

### Exporting to Grafana

1. Configure Prometheus as data source in Grafana
2. Import Celery dashboard templates
3. Create custom dashboards for specific metrics

### Example Grafana Dashboard Panels

- Task execution rate (graph)
- Active vs. total workers (gauge)
- Queue length over time (graph)
- Task duration heatmap
- Failure rate by task (table)
- Resource usage (CPU, memory) (graph)

## Security Considerations

1. **API Authentication**: Implement authentication for monitoring endpoints in production
2. **Flower Authentication**: Enable basic auth or OAuth for Flower dashboard
3. **Metrics Exposure**: Limit /metrics endpoint to internal network
4. **Task Revocation**: Restrict revoke endpoint to authorized users

## Performance Impact

The monitoring system is designed to have minimal performance impact:

- Metrics updates use non-blocking operations
- Signal handlers execute quickly
- Resource monitoring runs in background thread
- Queue inspection is rate-limited

Typical overhead: < 1% CPU, < 10MB memory per worker

## References

- [Celery Monitoring Documentation](https://docs.celeryproject.org/en/stable/userguide/monitoring.html)
- [Prometheus Python Client](https://github.com/prometheus/client_python)
- [Flower Documentation](https://flower.readthedocs.io/)
- [Grafana Dashboards](https://grafana.com/grafana/dashboards/)
