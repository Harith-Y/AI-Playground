# Celery Monitoring - Caching Layer

## Overview

The Celery monitoring endpoints include an intelligent caching layer to reduce load on the Celery infrastructure while maintaining data freshness. The caching system uses TTL-based expiration with automatic cleanup and cache hit/miss tracking.

## Architecture

```
┌──────────────┐
│   Client     │
└──────┬───────┘
       │
       v
┌──────────────────────┐       Cache Hit
│  API Endpoint        │ ───────────────> Return cached data
│  (with caching)      │
└──────────┬───────────┘
           │ Cache Miss
           v
┌──────────────────────┐
│  Celery Inspect      │
│  (expensive call)    │
└──────────┬───────────┘
           │
           v
┌──────────────────────┐
│  Cache & Return      │
└──────────────────────┘
```

## Cache Configuration

### Default TTLs

Each endpoint type has an optimized TTL based on data volatility:

| Endpoint Type | TTL | Rationale |
|--------------|-----|-----------|
| Task Status | 2s | Tasks change frequently |
| Worker Status | 5s | Worker stats are relatively stable |
| Queue Status | 3s | Queue depth varies moderately |
| Metrics Summary | 5s | Aggregated data, can be slightly stale |
| Health Check | 10s | Health status changes infrequently |

### Cache Sizes

| Cache | Max Size | Description |
|-------|----------|-------------|
| Task Status | 500 entries | Handles many concurrent task queries |
| Worker Status | 100 entries | Limited by worker count |
| Queue Status | 100 entries | Limited by queue count |
| Metrics Summary | 100 entries | Single aggregate entry typically |
| Health Check | 50 entries | Minimal storage needed |

## Cache Headers

All cached responses include HTTP headers indicating cache status:

```http
X-Cache: HIT          # Response served from cache
X-Cache: MISS         # Response computed fresh
X-Cache-Age: 1        # Age of cached data in seconds (only on HIT)
```

### Example

```bash
$ curl -I http://localhost:8000/api/v1/celery/tasks/status
HTTP/1.1 200 OK
X-Cache: HIT
X-Cache-Age: 1
Content-Type: application/json
```

## API Usage

### Bypassing Cache

All caching-enabled endpoints support a `use_cache` query parameter:

```bash
# Use cache (default)
GET /api/v1/celery/tasks/status

# Bypass cache, force fresh data
GET /api/v1/celery/tasks/status?use_cache=false
```

### Cache Statistics

Get performance metrics for all caches:

```http
GET /api/v1/celery/cache/stats
```

**Response:**
```json
{
  "timestamp": "2026-01-02T12:00:00Z",
  "caches": {
    "task_status": {
      "size": 50,
      "max_size": 500,
      "utilization_percent": 10.0,
      "hits": 1500,
      "misses": 300,
      "hit_rate_percent": 83.33,
      "sets": 300,
      "evictions": 0,
      "expirations": 250,
      "invalidations": 5,
      "total_requests": 1800
    },
    "worker_status": { ... },
    "queue_status": { ... },
    "metrics_summary": { ... },
    "health_check": { ... }
  },
  "summary": {
    "total_hits": 3000,
    "total_misses": 500,
    "total_requests": 3500,
    "overall_hit_rate_percent": 85.71
  }
}
```

### Cache Invalidation

#### Invalidate All Caches

```http
POST /api/v1/celery/cache/invalidate
```

**Response:**
```json
{
  "timestamp": "2026-01-02T12:00:00Z",
  "message": "All monitoring caches invalidated",
  "cache_name": "all"
}
```

#### Invalidate Specific Cache

```http
POST /api/v1/celery/cache/invalidate?cache_name=task_status
```

**Valid cache names:**
- `task_status`
- `worker_status`
- `queue_status`
- `metrics_summary`
- `health_check`

**Response:**
```json
{
  "timestamp": "2026-01-02T12:00:00Z",
  "message": "Cache 'task_status' invalidated",
  "cache_name": "task_status"
}
```

## Cache Metrics

### Key Metrics

**Hit Rate:** Percentage of requests served from cache
- **Good:** > 80%
- **Fair:** 60-80%
- **Poor:** < 60%

**Cache Utilization:** Percentage of max cache size used
- **Ideal:** 40-70%
- **Underutilized:** < 20%
- **Overutilized:** > 90% (may cause frequent evictions)

**Evictions:** Number of entries removed due to size limits
- **Healthy:** 0-5 per hour
- **Warning:** > 10 per hour (consider increasing max_size)

**Expirations:** Number of entries removed due to TTL
- **Expected:** Varies based on request patterns
- **Issue:** If expirations >> hits, TTL may be too short

## Performance Impact

### Cache Hit Benefits

Typical performance improvements with cache:

| Metric | Without Cache | With Cache (Hit) | Improvement |
|--------|--------------|------------------|-------------|
| Response Time | 50-200ms | 1-5ms | 95% faster |
| Celery Load | High | Minimal | 80% reduction |
| CPU Usage | 5-10% | < 1% | 90% reduction |
| Network Calls | Every request | Every N seconds | 80-95% reduction |

### Memory Overhead

Approximate memory usage per cache:

| Cache Type | Per Entry | 100 Entries | 500 Entries |
|-----------|-----------|-------------|-------------|
| Task Status | 2-5 KB | 200-500 KB | 1-2.5 MB |
| Worker Status | 1-3 KB | 100-300 KB | N/A |
| Queue Status | 1-2 KB | 100-200 KB | N/A |
| Metrics Summary | 1-3 KB | 100-300 KB | N/A |

**Total maximum:** ~5-10 MB for all caches

## Implementation Details

### Cache Class

The `MetricsCache` class provides:

- **Thread-safe operations** using RLock
- **TTL-based expiration** per entry
- **LRU eviction** when max size reached
- **Automatic cleanup** of expired entries
- **Statistics tracking** for monitoring
- **Decorator support** for easy integration

### Example Usage

```python
from app.monitoring.metrics_cache import MetricsCache

# Create cache
cache = MetricsCache(default_ttl=5.0, max_size=100)

# Store value
cache.set('key', {'data': 'value'}, ttl=10.0)

# Retrieve value
value = cache.get('key')  # Returns value or None

# Using decorator
@cache.cached(ttl=5.0, key_prefix="metrics")
def expensive_function():
    return compute_metrics()
```

### Cache Cleanup

Expired entries are automatically cleaned up every 60 seconds (configurable). This prevents memory bloat while minimizing overhead.

## Best Practices

### 1. Use Cache by Default

Cache is enabled by default for all endpoints. Only disable (`use_cache=false`) when:
- Debugging issues
- Requiring real-time data
- Testing cache behavior

### 2. Monitor Cache Performance

Regularly check cache statistics:

```bash
curl http://localhost:8000/api/v1/celery/cache/stats
```

Look for:
- High hit rates (> 80%)
- Low eviction rates
- Appropriate utilization

### 3. Invalidate After Changes

Invalidate caches after operations that change state:

```python
# After worker restart
requests.post('http://localhost:8000/api/v1/celery/cache/invalidate?cache_name=worker_status')

# After bulk task submission
requests.post('http://localhost:8000/api/v1/celery/cache/invalidate?cache_name=task_status')
```

### 4. Tune TTLs if Needed

Default TTLs work well for most cases, but adjust if:
- **Data changes very frequently:** Reduce TTL
- **Staleness is acceptable:** Increase TTL
- **High load with stable data:** Increase TTL

### 5. Monitor for Issues

Watch for:
- **High miss rates:** May indicate TTL too short or cache too small
- **Frequent evictions:** Increase max_size
- **Stale data concerns:** Reduce TTL or use `use_cache=false`

## Troubleshooting

### Issue: Low Hit Rate

**Symptoms:** Hit rate < 60%

**Causes:**
1. TTL too short for access patterns
2. Cache keys changing frequently
3. Low request volume

**Solutions:**
1. Increase TTL for less critical data
2. Review cache key generation
3. Monitor request patterns

### Issue: Frequent Evictions

**Symptoms:** High eviction count in stats

**Causes:**
1. max_size too small for access patterns
2. Many unique cache keys

**Solutions:**
1. Increase max_size
2. Implement key pattern filtering
3. Reduce cache key variations

### Issue: Stale Data

**Symptoms:** Cached data doesn't reflect recent changes

**Causes:**
1. TTL too long
2. Cache not invalidated after changes

**Solutions:**
1. Reduce TTL
2. Implement automatic invalidation
3. Use `use_cache=false` for critical queries

### Issue: High Memory Usage

**Symptoms:** Cache using too much memory

**Causes:**
1. max_size too large
2. Large cached objects

**Solutions:**
1. Reduce max_size
2. Implement data compression
3. Reduce cached data size

## Advanced Configuration

### Custom Cache Instance

```python
from app.monitoring.metrics_cache import MetricsCache

# Create custom cache
my_cache = MetricsCache(
    default_ttl=10.0,      # 10 second TTL
    max_size=1000,         # 1000 entries max
    cleanup_interval=30.0  # Cleanup every 30s
)
```

### Programmatic Cache Control

```python
from app.monitoring.metrics_cache import (
    get_task_status_cache,
    invalidate_all_metrics_caches
)

# Get cache instance
cache = get_task_status_cache()

# Get statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate_percent']:.2f}%")

# Invalidate specific key
cache.invalidate('specific_key')

# Invalidate pattern
cache.invalidate_pattern('task_*')

# Clear entire cache
cache.clear()

# Invalidate all caches
invalidate_all_metrics_caches()
```

## Prometheus Metrics

Cache performance can be exposed via Prometheus (future enhancement):

```promql
# Cache hit rate by cache type
cache_hit_rate{cache_type="task_status"}

# Total cache requests
rate(cache_requests_total[5m])

# Cache evictions
rate(cache_evictions_total[5m])
```

## Security Considerations

1. **Cache Invalidation:** Restrict invalidation endpoint to admin users
2. **Cache Poisoning:** Validate data before caching
3. **Memory Limits:** Set appropriate max_size to prevent DoS
4. **Sensitive Data:** Don't cache sensitive information

## Future Enhancements

Potential improvements:

- [ ] Distributed cache (Redis) for multi-instance deployments
- [ ] Conditional HTTP caching (ETags, Last-Modified)
- [ ] Cache warming on startup
- [ ] Predictive cache population
- [ ] Per-user cache isolation
- [ ] Cache compression for large objects
- [ ] Prometheus metrics integration
- [ ] Automatic TTL adjustment based on access patterns

## References

- [Caching Implementation](app/monitoring/metrics_cache.py)
- [Cached Endpoints](app/api/v1/endpoints/celery_monitoring.py)
- [HTTP Caching Best Practices](https://developer.mozilla.org/en-US/docs/Web/HTTP/Caching)
