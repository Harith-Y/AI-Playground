# Tuning Status Endpoint - Quick Reference

## Endpoint
```
GET /api/v1/tuning/tune/{tuning_run_id}/status
```

## Quick Example

```bash
curl http://localhost:8000/api/v1/tuning/tune/abc-123/status
```

## Response Structure

```json
{
  "tuning_run_id": "abc-123",
  "task_id": "xyz-789",
  "status": "PROGRESS",
  "progress": {
    "current": 15,
    "total": 36,
    "status": "Testing parameter combination 15/36...",
    "percentage": 41.67
  },
  "result": null,
  "error": null
}
```

## Status Values

| Status | Meaning | Next Action |
|--------|---------|-------------|
| `PENDING` | Queued, not started | Keep polling |
| `PROGRESS` | Running | Keep polling, show progress |
| `SUCCESS` | Completed | Get results |
| `FAILURE` | Failed | Show error |
| `REVOKED` | Cancelled | Inform user |

## Polling Pattern

### Simple Polling (JavaScript)
```javascript
const pollStatus = async (tuningRunId) => {
  const response = await fetch(
    `/api/v1/tuning/tune/${tuningRunId}/status`
  );
  const data = await response.json();
  
  if (data.status === 'SUCCESS') {
    console.log('Done!', data.result);
  } else if (data.status === 'FAILURE') {
    console.error('Failed:', data.error);
  } else {
    // Still running, poll again in 5 seconds
    setTimeout(() => pollStatus(tuningRunId), 5000);
  }
};
```

### With Progress Bar (React)
```typescript
const [progress, setProgress] = useState(0);
const [status, setStatus] = useState('PENDING');

useEffect(() => {
  const interval = setInterval(async () => {
    const res = await fetch(`/api/v1/tuning/tune/${id}/status`);
    const data = await res.json();
    
    setStatus(data.status);
    if (data.progress) {
      setProgress(data.progress.percentage);
    }
    
    if (data.status === 'SUCCESS' || data.status === 'FAILURE') {
      clearInterval(interval);
    }
  }, 5000);
  
  return () => clearInterval(interval);
}, [id]);
```

## Response Examples

### Pending
```json
{
  "tuning_run_id": "abc-123",
  "task_id": "xyz-789",
  "status": "PENDING",
  "progress": null,
  "result": null,
  "error": null
}
```

### In Progress
```json
{
  "tuning_run_id": "abc-123",
  "task_id": "xyz-789",
  "status": "PROGRESS",
  "progress": {
    "current": 15,
    "total": 36,
    "status": "Testing parameter combination 15/36...",
    "percentage": 41.67
  },
  "result": null,
  "error": null
}
```

### Completed
```json
{
  "tuning_run_id": "abc-123",
  "task_id": "xyz-789",
  "status": "SUCCESS",
  "progress": null,
  "result": {
    "best_params": {
      "n_estimators": 100,
      "max_depth": 10
    },
    "best_score": 0.9567,
    "total_combinations": 36,
    "tuning_time": 120.5
  },
  "error": null
}
```

### Failed
```json
{
  "tuning_run_id": "abc-123",
  "task_id": "xyz-789",
  "status": "FAILURE",
  "progress": null,
  "result": null,
  "error": {
    "type": "ValueError",
    "message": "Invalid parameter grid"
  }
}
```

## Common Errors

| Code | Reason | Solution |
|------|--------|----------|
| 400 | Invalid UUID | Check tuning_run_id format |
| 404 | Not found | Verify tuning_run_id exists |
| 403 | Unauthorized | Check user permissions |

## Polling Best Practices

### Recommended Intervals
- **First minute:** Every 2-3 seconds
- **Active tuning:** Every 5 seconds
- **Long-running:** Every 10 seconds

### Exponential Backoff
```javascript
let delay = 2000; // Start with 2 seconds
const maxDelay = 30000; // Max 30 seconds

const poll = async () => {
  const status = await getStatus();
  if (status === 'PROGRESS') {
    delay = Math.min(delay * 1.5, maxDelay);
    setTimeout(poll, delay);
  }
};
```

### Stop Conditions
Stop polling when:
- Status is `SUCCESS`
- Status is `FAILURE`
- Status is `REVOKED`
- Max attempts reached (e.g., 120 attempts = 10 minutes)

## Progress Information

When `status === 'PROGRESS'`, use the progress object:

```typescript
if (data.progress) {
  const { current, total, status, percentage } = data.progress;
  
  // Update progress bar
  progressBar.value = percentage;
  
  // Show status message
  statusText.innerText = status;
  
  // Calculate ETA (rough estimate)
  const elapsed = Date.now() - startTime;
  const eta = (elapsed / current) * (total - current);
}
```

## Complete Example

```typescript
interface TuningStatus {
  tuning_run_id: string;
  task_id: string | null;
  status: string;
  progress: {
    current: number;
    total: number;
    status: string;
    percentage: number;
  } | null;
  result: any | null;
  error: any | null;
}

async function monitorTuning(tuningRunId: string): Promise<void> {
  const maxAttempts = 120; // 10 minutes
  let attempts = 0;
  
  while (attempts < maxAttempts) {
    try {
      const response = await fetch(
        `/api/v1/tuning/tune/${tuningRunId}/status`
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const data: TuningStatus = await response.json();
      
      console.log(`Status: ${data.status}`);
      
      if (data.progress) {
        console.log(`Progress: ${data.progress.percentage}%`);
        console.log(`Message: ${data.progress.status}`);
      }
      
      if (data.status === 'SUCCESS') {
        console.log('Tuning completed!');
        console.log('Best params:', data.result.best_params);
        console.log('Best score:', data.result.best_score);
        return;
      }
      
      if (data.status === 'FAILURE') {
        console.error('Tuning failed:', data.error);
        throw new Error(data.error.message);
      }
      
      // Wait before next poll
      await new Promise(resolve => setTimeout(resolve, 5000));
      attempts++;
      
    } catch (error) {
      console.error('Error checking status:', error);
      await new Promise(resolve => setTimeout(resolve, 5000));
      attempts++;
    }
  }
  
  console.warn('Polling timeout reached');
}

// Usage
monitorTuning('abc-123-def-456');
```

## Related Endpoints

- `POST /api/v1/tuning/tune` - Start tuning
- `GET /api/v1/tuning/tune/{id}/results` - Get full results
- `GET /api/v1/models/train/{id}/status` - Similar for training

## Notes

- Status updates are real-time from Celery
- Falls back to database if Celery unavailable
- Progress percentage calculated automatically
- Task ID stored for tracking across requests
