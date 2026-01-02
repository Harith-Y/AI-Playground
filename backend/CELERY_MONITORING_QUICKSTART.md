# Celery Monitoring - Quick Start Guide

## TL;DR

```bash
# 1. Install monitoring dependencies
pip install flower

# 2. Start Celery worker with monitoring
celery -A celery_worker.celery_app worker --loglevel=info

# 3. Start Flower dashboard (optional)
python start_flower.py

# 4. Access monitoring
# - API: http://localhost:8000/api/v1/celery/tasks/status
# - Prometheus: http://localhost:8000/metrics
# - Flower: http://localhost:5555/flower
```

## What's Available

### 1. REST API Endpoints

```bash
# Get all tasks status
curl http://localhost:8000/api/v1/celery/tasks/status

# Get workers status
curl http://localhost:8000/api/v1/celery/workers/status

# Get queue status
curl http://localhost:8000/api/v1/celery/queues/status

# Get specific task
curl http://localhost:8000/api/v1/celery/tasks/{task_id}/status

# Health check
curl http://localhost:8000/api/v1/celery/health
```

### 2. Prometheus Metrics

Available at `http://localhost:8000/metrics`:

- `celery_task_duration_seconds` - Task execution time
- `celery_task_total` - Total task count by status
- `celery_active_tasks` - Currently running tasks
- `celery_task_failure_rate` - Failures per minute
- `celery_task_retries_total` - Retry counts
- `celery_worker_pool_size` - Worker pool size
- `celery_queue_length` - Queue depth

### 3. Flower Dashboard

Web-based monitoring at `http://localhost:5555/flower`:

- Task history and details
- Worker status and management
- Real-time graphs
- Broker monitoring

## Key Metrics to Watch

### Task Performance
```promql
# Average task duration
rate(celery_task_duration_seconds_sum[5m]) / rate(celery_task_duration_seconds_count[5m])

# Task success rate
rate(celery_task_total{status="success"}[5m]) / rate(celery_task_total[5m]) * 100
```

### Worker Health
```promql
# Worker utilization
celery_worker_pool_busy / celery_worker_pool_size * 100

# Active workers
count(celery_worker_pool_size)
```

### Queue Depth
```promql
# Current queue length
celery_queue_length

# Queue growth rate
rate(celery_queue_length[5m])
```

## Common Tasks

### Monitor Training Tasks

```python
import requests

# Get all active training tasks
response = requests.get('http://localhost:8000/api/v1/celery/tasks/status')
data = response.json()

training_tasks = [
    t for t in data['active_tasks']
    if 'train_model' in t['task_name']
]

print(f"Active training tasks: {len(training_tasks)}")
```

### Check Worker Health

```python
response = requests.get('http://localhost:8000/api/v1/celery/workers/status')
workers = response.json()

for worker in workers['workers']:
    utilization = (worker['pool']['busy_workers'] /
                  worker['pool']['max_concurrency'] * 100)
    print(f"{worker['name']}: {utilization:.1f}% utilized")
```

### Cancel Long-Running Task

```python
# Get task ID from task status endpoint
task_id = "abc123"

# Revoke the task
response = requests.post(
    f'http://localhost:8000/api/v1/celery/tasks/{task_id}/revoke',
    params={'terminate': True}
)
print(response.json())
```

## Alerts to Set Up

### Critical
- No workers available
- Broker connection lost
- Queue length > 1000 for > 15 minutes

### Warning
- Task failure rate > 10/min
- Worker utilization > 90%
- Queue length > 100 for > 5 minutes
- Task timeouts occurring

### Info
- New worker started
- Task retry triggered
- Queue backlog cleared

## Troubleshooting

### "No workers responding"

```bash
# Check if worker is running
celery -A celery_worker.celery_app inspect active

# If not, start worker
celery -A celery_worker.celery_app worker --loglevel=info
```

### High queue length

```bash
# Check worker status
curl http://localhost:8000/api/v1/celery/workers/status

# Add more workers
celery -A celery_worker.celery_app worker --concurrency=8
```

### Tasks failing

```bash
# Check failure rate
curl http://localhost:8000/api/v1/celery/tasks/status | jq '.summary'

# View task details in Flower
# Navigate to http://localhost:5555/flower/tasks
```

## Configuration Tips

### Optimize Worker Pool

```python
# For CPU-bound tasks
celery worker --concurrency=4 --pool=prefork

# For I/O-bound tasks
celery worker --concurrency=100 --pool=gevent
```

### Adjust Prefetch

```python
# Low prefetch for long tasks
celery worker --prefetch-multiplier=1

# Higher prefetch for quick tasks
celery worker --prefetch-multiplier=4
```

### Set Time Limits

```python
# In celery_app.py
celery_app.conf.update(
    task_time_limit=3600,      # 1 hour hard limit
    task_soft_time_limit=3000,  # 50 minute soft limit
)
```

## Integration with Frontend

### React Hook Example

```typescript
const useTaskMonitoring = () => {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      const response = await fetch('/api/v1/celery/metrics/summary');
      const data = await response.json();
      setMetrics(data);
    };

    const interval = setInterval(fetchMetrics, 5000); // Poll every 5s
    return () => clearInterval(interval);
  }, []);

  return metrics;
};
```

### Display Active Tasks

```typescript
const ActiveTasksWidget = () => {
  const [tasks, setTasks] = useState([]);

  useEffect(() => {
    fetch('/api/v1/celery/tasks/status')
      .then(res => res.json())
      .then(data => setTasks(data.active_tasks));
  }, []);

  return (
    <div>
      <h3>Active Tasks: {tasks.length}</h3>
      {tasks.map(task => (
        <div key={task.task_id}>
          {task.task_name} - {task.worker}
        </div>
      ))}
    </div>
  );
};
```

## Next Steps

1. **Set up Grafana dashboards** - Visualize metrics over time
2. **Configure alerting** - Get notified of issues
3. **Enable authentication** - Secure monitoring endpoints
4. **Optimize worker settings** - Tune for your workload
5. **Monitor production** - Set up proper observability

## Resources

- Full documentation: [CELERY_MONITORING.md](CELERY_MONITORING.md)
- Celery docs: https://docs.celeryproject.org/
- Flower docs: https://flower.readthedocs.io/
- Prometheus docs: https://prometheus.io/docs/
