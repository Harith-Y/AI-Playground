# Monitoring & Metrics System

## Overview

The AI-Playground application includes a comprehensive monitoring and metrics system for tracking:
- **Celery async tasks** (training, tuning, preprocessing)
- **API performance** (request duration, throughput, errors)
- **System resources** (CPU, memory, disk, network)
- **Database queries** (duration, counts, connection pool)
- **ML operations** (model training, evaluation, inference)

## Quick Start

```bash
# 1. Install monitoring dependencies
pip install -r requirements.monitoring.txt

# 2. Start the application with monitoring enabled
python -m uvicorn app.main:app --reload

# 3. Start Celery worker with monitoring
celery -A celery_worker.celery_app worker --loglevel=info

# 4. (Optional) Start Flower dashboard
python start_flower.py

# 5. Access monitoring interfaces
# - Prometheus metrics: http://localhost:8000/metrics
# - Celery API: http://localhost:8000/api/v1/celery/
# - Flower UI: http://localhost:5555/flower
```

## Documentation

### Main Guides

1. **[CELERY_MONITORING.md](CELERY_MONITORING.md)** - Complete Celery monitoring guide
   - Metrics definitions
   - API endpoints
   - Prometheus queries
   - Alerting rules
   - Best practices

2. **[CELERY_MONITORING_QUICKSTART.md](CELERY_MONITORING_QUICKSTART.md)** - Quick start guide
   - Common tasks
   - Key metrics to watch
   - Troubleshooting
   - Code examples

## Architecture

```
┌─────────────────┐
│   Application   │
│   (FastAPI)     │
└────────┬────────┘
         │
         ├─── Prometheus Metrics ───> /metrics endpoint
         │
         ├─── Celery Tasks ─────────> Celery Workers
         │                                    │
         │                                    ├─── Signal Handlers
         │                                    │    (prerun, postrun, failure)
         │                                    │
         │                                    └─── Metrics Collection
         │
         ├─── REST API ─────────────> /api/v1/celery/*
         │                             (status, health, stats)
         │
         └─── Resource Monitor ─────> Background Thread
                                      (CPU, memory, disk)

┌─────────────────┐
│   Flower UI     │  <──── Celery Events Bus
│ (port 5555)     │
└─────────────────┘

┌─────────────────┐
│  Prometheus     │  <──── Scrapes /metrics
│   (optional)    │
└─────────────────┘
         │
         v
┌─────────────────┐
│    Grafana      │
│  (optional)     │
└─────────────────┘
```

## Key Features

### 1. Celery Task Monitoring

**Automatic tracking of:**
- Task execution duration
- Success/failure counts
- Retry attempts and reasons
- Active task counts
- Queue lengths
- Worker pool utilization
- Failure rates (per minute)

**Available via:**
- REST API endpoints
- Prometheus metrics
- Flower dashboard

### 2. API Performance Monitoring

**Tracks:**
- Request duration histograms
- Request counts by endpoint/status
- Active request counts
- Request/response sizes

**Middleware:** `PerformanceMonitoringMiddleware` automatically instruments all API endpoints

### 3. Resource Monitoring

**Monitors:**
- CPU usage (%)
- Memory usage (bytes)
- Disk I/O (bytes read/written)
- Network I/O (bytes sent/received)
- Process counts

**Background monitoring:** Runs every 30 seconds in worker processes

### 4. ML-Specific Metrics

**Training metrics:**
- Training duration by model type
- Sample counts
- Feature counts
- Success/failure rates

**Tuning metrics:**
- Tuning duration by method
- Iteration counts
- Best scores achieved
- Success/failure rates

**Inference metrics:**
- Prediction duration
- Throughput (samples/sec)
- Batch sizes

## Available Endpoints

### Celery Monitoring API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/celery/tasks/status` | GET | Current task status |
| `/api/v1/celery/tasks/{task_id}/status` | GET | Specific task status |
| `/api/v1/celery/tasks/{task_id}/revoke` | POST | Cancel/revoke task |
| `/api/v1/celery/workers/status` | GET | Worker pool status |
| `/api/v1/celery/queues/status` | GET | Queue lengths |
| `/api/v1/celery/metrics/summary` | GET | Metrics summary |
| `/api/v1/celery/health` | GET | Health check |
| `/api/v1/celery/stats/detailed` | GET | Detailed statistics |

### Prometheus Metrics

- **Endpoint:** `http://localhost:8000/metrics`
- **Format:** Prometheus exposition format
- **Update:** Real-time on request

## Metrics Reference

### Celery Metrics

```
# Task execution
celery_task_duration_seconds{task_name, status}
celery_task_total{task_name, status}
celery_active_tasks{task_name}

# Worker pool
celery_worker_pool_size{worker_name}
celery_worker_pool_busy{worker_name}
celery_worker_prefetch_count{worker_name}

# Queue
celery_queue_length{queue_name}

# Failures
celery_task_failure_rate{task_name}
celery_task_retries_total{task_name, reason}
celery_task_timeouts_total{task_name}

# ML-specific
model_training_duration_seconds{model_type, task_type}
model_training_samples{model_type}
hyperparameter_tuning_duration_seconds{model_type, tuning_method}
```

### API Metrics

```
api_request_duration_seconds{method, endpoint, status_code}
api_requests_total{method, endpoint, status_code}
api_active_requests{method, endpoint}
api_request_size_bytes{method, endpoint}
api_response_size_bytes{method, endpoint}
```

### Resource Metrics

```
process_memory_usage_bytes{process_type}
process_cpu_usage_percent{process_type}
disk_io_bytes_total{operation, process_type}
```

## Integration Examples

### Python

```python
import requests

# Get task status
response = requests.get('http://localhost:8000/api/v1/celery/tasks/status')
data = response.json()
print(f"Active tasks: {data['summary']['active']}")

# Monitor specific task
from celery.result import AsyncResult
from app.celery_app import celery_app

result = AsyncResult(task_id, app=celery_app)
print(f"State: {result.state}")
if result.ready():
    print(f"Result: {result.result}")
```

### JavaScript/TypeScript

```typescript
// Fetch task status
const response = await fetch('/api/v1/celery/tasks/status');
const data = await response.json();

// Poll for updates
const interval = setInterval(async () => {
  const status = await fetch(`/api/v1/celery/tasks/${taskId}/status`);
  const taskData = await status.json();

  if (taskData.ready) {
    clearInterval(interval);
    console.log('Task completed:', taskData.result);
  }
}, 2000);
```

### Prometheus Queries

```promql
# Task success rate (last 5 minutes)
rate(celery_task_total{status="success"}[5m]) /
rate(celery_task_total[5m]) * 100

# Average task duration by task name
avg(rate(celery_task_duration_seconds_sum[5m]) /
    rate(celery_task_duration_seconds_count[5m]))
by (task_name)

# Worker utilization
celery_worker_pool_busy / celery_worker_pool_size * 100

# Queue backlog
sum(celery_queue_length)
```

## Alerting

### Recommended Alerts

**Critical:**
- No Celery workers available
- Broker connection lost
- Task failure rate > 50%

**Warning:**
- Queue length > 100 for > 10 minutes
- Worker utilization > 90%
- Task timeout rate > 10%
- Memory usage > 90%

**Info:**
- New worker started
- Task retry triggered
- Queue cleared after backlog

### Prometheus Alert Rules

See [CELERY_MONITORING.md](CELERY_MONITORING.md#alerting-rules) for complete alert rule examples.

## Dashboards

### Flower Dashboard

**Access:** http://localhost:5555/flower

**Features:**
- Task history and timeline
- Worker management (shutdown, pool grow/shrink)
- Real-time graphs
- Broker monitoring
- Task details and tracebacks

**Start with:**
```bash
python start_flower.py
```

### Grafana Dashboards (Optional)

**Setup:**
1. Install Grafana
2. Add Prometheus as data source
3. Import Celery monitoring dashboard
4. Create custom panels

**Example panels:**
- Task execution rate (line graph)
- Active workers (gauge)
- Queue length (area chart)
- Task duration heatmap
- Failure rate (stat panel)

## Configuration

### Enable Monitoring

Monitoring is enabled by default when you start the Celery worker:

```python
# backend/celery_worker.py
from app.monitoring.celery_metrics import setup_celery_metrics
from app.monitoring.resource_monitor import start_global_monitor

setup_celery_metrics(celery_app)
start_global_monitor(interval=30.0, process_type='celery_worker')
```

### Customize Monitoring

```python
# Adjust resource monitoring interval
start_global_monitor(interval=60.0, process_type='celery_worker')

# Add custom metrics
from prometheus_client import Counter

custom_metric = Counter(
    'my_custom_metric_total',
    'Description of my metric',
    ['label1', 'label2']
)

custom_metric.labels(label1='value1', label2='value2').inc()
```

### Configure Celery for Monitoring

```python
# app/celery_app.py
celery_app.conf.update(
    task_track_started=True,  # Required for started event
    task_send_sent_event=True,  # Send sent event
    worker_send_task_events=True,  # Enable task events
    task_time_limit=3600,
    task_soft_time_limit=3000,
)
```

## Best Practices

1. **Monitor Queue Length**
   - Set alerts for sustained queue growth
   - Scale workers based on queue depth

2. **Track Failure Rates**
   - Investigate when failure rate spikes
   - Review task logs and error patterns

3. **Optimize Worker Pool**
   - Monitor CPU and memory usage
   - Adjust concurrency based on workload
   - Use appropriate pool type (prefork vs gevent)

4. **Set Appropriate Timeouts**
   - Soft timeout for graceful cleanup
   - Hard timeout to prevent runaway tasks

5. **Use Result Expiration**
   - Clean up old task results
   - Prevent backend bloat

6. **Regular Health Checks**
   - Monitor worker and broker connectivity
   - Set up automated health checks

## Troubleshooting

### High Queue Length

1. Check worker availability
2. Review task duration trends
3. Consider scaling workers
4. Optimize task code

### Task Failures

1. Check failure rate metrics
2. Review task logs
3. Verify resource availability
4. Check for timeouts

### Performance Issues

1. Monitor task duration trends
2. Check resource usage (CPU, memory)
3. Review database query performance
4. Optimize task implementation

### Monitoring Not Working

1. Verify Celery worker started with monitoring enabled
2. Check `/metrics` endpoint is accessible
3. Verify Prometheus client installed
4. Check signal handlers are connected

## Security

### Production Recommendations

1. **Secure Monitoring Endpoints**
   ```python
   # Add authentication to monitoring endpoints
   from fastapi import Depends
   from app.api.deps import get_current_active_superuser

   @router.get("/celery/tasks/status", dependencies=[Depends(get_current_active_superuser)])
   ```

2. **Protect Flower Dashboard**
   ```bash
   # Enable basic auth
   celery -A celery_worker.celery_app flower --basic_auth=user:password
   ```

3. **Limit Metrics Exposure**
   - Expose `/metrics` only to internal network
   - Use firewall rules to restrict access
   - Consider reverse proxy with auth

4. **Sanitize Sensitive Data**
   - Avoid logging sensitive task arguments
   - Redact credentials in metrics
   - Use encrypted connections

## Performance Impact

The monitoring system has minimal overhead:
- **CPU:** < 1% additional usage
- **Memory:** ~10-20MB per worker process
- **Network:** Negligible (metrics pulled, not pushed)
- **Disk:** Minimal (Flower DB ~1-10MB)

## Dependencies

### Required
- `celery>=5.6.0`
- `prometheus-client>=0.19.0`
- `psutil>=5.9.0`

### Optional
- `flower>=2.0.1` - Web dashboard
- `redis>=5.0.0` - Result backend (if using Redis)
- `prometheus` - Metrics storage (external)
- `grafana` - Visualization (external)

## Support

For issues or questions:
1. Check [CELERY_MONITORING.md](CELERY_MONITORING.md) for detailed docs
2. Review [CELERY_MONITORING_QUICKSTART.md](CELERY_MONITORING_QUICKSTART.md) for examples
3. Check Celery logs for errors
4. Verify monitoring configuration

## Roadmap

Future enhancements:
- [ ] Pre-built Grafana dashboards
- [ ] Advanced alerting templates
- [ ] Task performance profiling
- [ ] Distributed tracing integration
- [ ] Custom metric collectors
- [ ] Automated performance reports
