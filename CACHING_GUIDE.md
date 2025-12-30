# Caching Guide

## Overview

The AI-Playground backend implements a comprehensive Redis-based caching system to improve API response times and reduce database load. This guide covers the caching architecture, usage patterns, and best practices.

## Architecture

### Components

1. **CacheService** (`app/utils/cache.py`)

   - Core caching service with Redis operations
   - Singleton pattern for connection reuse
   - Graceful degradation when Redis unavailable

2. **Cache Keys** (`CacheKeys` class)

   - Standardized key generation
   - Consistent naming patterns
   - Support for parameterized keys

3. **TTL Constants** (`CacheTTL` class)

   - Predefined expiration times
   - Consistent across the application

4. **Decorators**

   - `@cache_response`: Function-level caching
   - `@invalidate_cache`: Post-execution invalidation

5. **Invalidation Logic**
   - Automatic cache invalidation on model updates
   - Pattern-based deletion
   - Manual clearing via API endpoints

## Cache Configuration

### Redis Setup

Redis is configured in `docker-compose.yml`:

```yaml
redis:
  image: redis:7-alpine
  container_name: ai_playground_redis
  ports:
    - "6379:6379"
  volumes:
    - redis_data:/data
    - ./docker/redis/redis.conf:/usr/local/etc/redis/redis.conf
  command: redis-server /usr/local/etc/redis/redis.conf
```

### Configuration Settings

```python
# app/core/config.py
REDIS_URL = "redis://localhost:6379/0"
```

### TTL Hierarchy

```python
CacheTTL.VERY_SHORT = 300    # 5 minutes
CacheTTL.SHORT = 900          # 15 minutes
CacheTTL.MEDIUM = 1800        # 30 minutes
CacheTTL.LONG = 3600          # 1 hour
CacheTTL.VERY_LONG = 7200     # 2 hours
CacheTTL.DAY = 86400          # 24 hours
CacheTTL.WEEK = 604800        # 7 days
```

## Cached Endpoints

### 1. Model Metrics

**Endpoint:** `GET /api/v1/models/train/{model_run_id}/metrics`  
**TTL:** 1 hour (`CacheTTL.LONG`)  
**Cache Key:** `model:metrics:{model_run_id}`

```python
# Example request
GET /api/v1/models/train/abc-123/metrics

# Bypass cache
GET /api/v1/models/train/abc-123/metrics?use_cache=false
```

**When Invalidated:**

- Model training completes
- Model status changes
- Model is deleted

### 2. Model Comparison

**Endpoint:** `POST /api/v1/models/compare`  
**TTL:** 30 minutes (`CacheTTL.MEDIUM`)  
**Cache Key:** `model:comparison:{hash(sorted_model_ids)}`

```python
# Example request
POST /api/v1/models/compare
{
    "model_run_ids": ["abc-123", "def-456", "ghi-789"],
    "metric": "accuracy"
}

# Bypass cache
POST /api/v1/models/compare?use_cache=false
{...}
```

**When Invalidated:**

- Any model in the comparison is updated
- Any model is deleted (clears all comparisons)

### 3. Feature Importance

**Endpoint:** `GET /api/v1/models/train/{model_run_id}/feature-importance`  
**TTL:** 30 minutes (`CacheTTL.MEDIUM`)  
**Cache Key:** `model:feature_importance:{model_run_id}` or `model:feature_importance:{model_run_id}:top_{n}`

```python
# Example request
GET /api/v1/models/train/abc-123/feature-importance

# With top_n parameter
GET /api/v1/models/train/abc-123/feature-importance?top_n=10

# Bypass cache
GET /api/v1/models/train/abc-123/feature-importance?use_cache=false
```

**When Invalidated:**

- Model training completes
- Model is deleted

## Usage Patterns

### Basic Cache Operations

```python
from app.utils.cache import cache_service, CacheKeys, CacheTTL

# Store value
await cache_service.set(
    key=CacheKeys.model_metrics("model-123"),
    value={"accuracy": 0.95, "f1_score": 0.93},
    ttl=CacheTTL.LONG
)

# Retrieve value
metrics = await cache_service.get(
    key=CacheKeys.model_metrics("model-123")
)

# Delete value
await cache_service.delete(
    key=CacheKeys.model_metrics("model-123")
)

# Delete by pattern
deleted_count = await cache_service.delete_pattern("model:metrics:*")

# Clear all cache
deleted_count = await cache_service.clear_all()
```

### Using Cache in Endpoints

```python
from app.utils.cache import cache_service, CacheKeys, CacheTTL

@router.get("/data/{id}")
async def get_data(
    id: str,
    use_cache: bool = True,
    db: Session = Depends(get_db)
):
    # Check cache first
    if use_cache:
        cache_key = CacheKeys.model_metrics(id)
        cached_result = await cache_service.get(cache_key)
        if cached_result:
            return cached_result

    # Query database
    result = db.query(Model).filter(Model.id == id).first()

    # Store in cache
    if use_cache and result:
        await cache_service.set(cache_key, result, ttl=CacheTTL.LONG)

    return result
```

### Using Decorators

```python
from app.utils.cache import cache_response, invalidate_cache, CacheTTL

# Cache function results
@cache_response(key_prefix="expensive:operation", ttl=CacheTTL.MEDIUM)
async def expensive_computation(x: int, y: int):
    # Expensive operation
    result = complex_calculation(x, y)
    return result

# Invalidate cache after operation
@invalidate_cache(patterns=["model:metrics:*", "model:comparison:*"])
async def update_model(model_id: str):
    # Update model
    update_operation(model_id)
```

### Custom Cache Keys

```python
# Simple key
CacheKeys.model_metrics("abc-123")
# Returns: "model:metrics:abc-123"

# Parameterized key
CacheKeys.feature_importance("abc-123", top_n=10)
# Returns: "model:feature_importance:abc-123:top_10"

# Hash-based key (for complex parameters)
CacheKeys.model_comparison(["abc-123", "def-456"])
# Returns: "model:comparison:{hash}"
```

## Cache Invalidation

### Automatic Invalidation

Cache is automatically invalidated when:

1. **Model Training Completes**

   - Invalidates: metrics, feature importance, all comparisons
   - Location: `app/tasks/training_tasks.py`

2. **Model Training Fails**

   - Invalidates: metrics, feature importance, all comparisons
   - Location: `app/services/training_error_handler.py`

3. **Model Deletion**
   - Invalidates: all model-specific caches, all comparisons
   - Location: `app/api/v1/endpoints/models.py`

### Manual Invalidation

```python
from app.utils.cache import invalidate_model_cache, invalidate_comparison_cache

# Invalidate all caches for a specific model
await invalidate_model_cache("model-123")

# Invalidate all comparison caches
await invalidate_comparison_cache()
```

### Invalidation via API

```bash
# Clear specific pattern
DELETE /api/v1/models/cache/clear?pattern=model:metrics:*

# Clear all cache (use with caution)
DELETE /api/v1/models/cache/clear
```

## Cache Monitoring

### View Cache Statistics

```bash
GET /api/v1/models/cache/stats
```

Response:

```json
{
  "available": true,
  "stats": {
    "used_memory": "1.5M",
    "total_keys": 42,
    "hits": 1245,
    "misses": 156,
    "hit_rate": "88.9%"
  },
  "config": {
    "ttl_very_short": 300,
    "ttl_short": 900,
    "ttl_medium": 1800,
    "ttl_long": 3600,
    "ttl_very_long": 7200,
    "ttl_day": 86400,
    "ttl_week": 604800
  },
  "message": "Cache is healthy"
}
```

### Redis CLI Monitoring

```bash
# Connect to Redis
docker exec -it ai_playground_redis redis-cli

# View all keys
KEYS *

# View keys by pattern
KEYS model:metrics:*

# Get key value
GET model:metrics:abc-123

# Check TTL
TTL model:metrics:abc-123

# View memory usage
INFO memory

# View stats
INFO stats

# Monitor commands in real-time
MONITOR
```

## Performance Benefits

### Benchmarks

Without cache:

```
Model Metrics:        ~500-800ms
Feature Importance:   ~300-600ms
Model Comparison:     ~1000-2000ms
```

With cache:

```
Model Metrics:        ~10-20ms (40-80x faster)
Feature Importance:   ~10-20ms (30-60x faster)
Model Comparison:     ~15-25ms (66-133x faster)
```

### Hit Rate Targets

- **Development:** 50-70%
- **Production:** 80-95%

## Best Practices

### 1. Always Provide `use_cache` Parameter

```python
@router.get("/data/{id}")
async def get_data(
    id: str,
    use_cache: bool = True,  # ✅ Default True, but allow bypass
    db: Session = Depends(get_db)
):
    ...
```

### 2. Choose Appropriate TTL

```python
# Frequently changing data
ttl = CacheTTL.VERY_SHORT  # 5 minutes

# Moderately stable data
ttl = CacheTTL.MEDIUM  # 30 minutes

# Stable data
ttl = CacheTTL.LONG  # 1 hour

# Nearly static data
ttl = CacheTTL.DAY  # 24 hours
```

### 3. Use Descriptive Cache Keys

```python
# ✅ Good
CacheKeys.model_metrics("abc-123")

# ❌ Bad
cache_key = "metrics_abc_123"
```

### 4. Handle Cache Unavailability

```python
# ✅ Good - graceful degradation
if cache_service.is_available():
    cached_result = await cache_service.get(key)
    if cached_result:
        return cached_result

# Continue with database query if cache unavailable or miss
result = query_database()
return result
```

### 5. Invalidate Related Caches

```python
# ✅ Good - invalidate all related caches
await invalidate_model_cache(model_id)
await invalidate_comparison_cache()

# ❌ Bad - only invalidate one cache type
await cache_service.delete(f"model:metrics:{model_id}")
```

### 6. Log Cache Operations

```python
# ✅ Good
if cached_result:
    logger.info(f"Cache hit for {cache_key}")
    return cached_result
else:
    logger.info(f"Cache miss for {cache_key}")
```

## Troubleshooting

### Cache Not Working

1. **Check Redis Connection**

   ```bash
   docker ps | grep redis
   docker logs ai_playground_redis
   ```

2. **Test Connection in Code**

   ```python
   if cache_service.is_available():
       print("Redis is available")
   else:
       print("Redis is NOT available")
   ```

3. **Check Redis Configuration**
   ```bash
   docker exec -it ai_playground_redis redis-cli CONFIG GET maxmemory
   docker exec -it ai_playground_redis redis-cli CONFIG GET maxmemory-policy
   ```

### High Cache Miss Rate

1. **Check TTL Settings**

   - TTL may be too short
   - Increase TTL for stable data

2. **Check Invalidation Logic**

   - Too aggressive invalidation
   - Review invalidation triggers

3. **Monitor Cache Keys**
   ```bash
   docker exec -it ai_playground_redis redis-cli KEYS *
   ```

### Memory Issues

1. **Check Redis Memory Usage**

   ```bash
   docker exec -it ai_playground_redis redis-cli INFO memory
   ```

2. **Adjust maxmemory in `redis.conf`**

   ```
   maxmemory 256mb
   maxmemory-policy allkeys-lru
   ```

3. **Clear Unnecessary Caches**
   ```bash
   DELETE /api/v1/models/cache/clear?pattern=old:pattern:*
   ```

## Security Considerations

### 1. Authentication Required

All cache management endpoints require authentication:

```python
user_id: str = Depends(get_current_user_id)
```

### 2. No Sensitive Data in Cache Keys

```python
# ✅ Good
CacheKeys.model_metrics(model_id)

# ❌ Bad
f"user:{email}:password:{password}"
```

### 3. TTL for All Entries

Always set TTL to prevent indefinite storage:

```python
await cache_service.set(key, value, ttl=CacheTTL.LONG)
```

## Testing

### Run Cache Tests

```bash
# All cache tests
pytest backend/tests/test_cache.py -v

# Specific test class
pytest backend/tests/test_cache.py::TestCacheService -v

# Specific test
pytest backend/tests/test_cache.py::TestCacheService::test_cache_set_and_get -v
```

### Test Coverage

```bash
pytest backend/tests/test_cache.py --cov=app/utils/cache --cov-report=html
```

## Future Enhancements

1. **Cache Warming**

   - Pre-populate cache on startup
   - Periodic refresh of popular entries

2. **Distributed Caching**

   - Redis Cluster for scalability
   - Sentinel for high availability

3. **Advanced Metrics**

   - Hit/miss rates per endpoint
   - Cache efficiency tracking
   - Cost savings calculations

4. **Cache Strategies**

   - Cache-aside (current)
   - Write-through
   - Write-behind

5. **Smart Invalidation**
   - Dependency tracking
   - Cascade invalidation
   - Lazy invalidation

## References

- [Redis Documentation](https://redis.io/documentation)
- [Python Redis Client](https://github.com/redis/redis-py)
- [Caching Best Practices](https://aws.amazon.com/caching/best-practices/)
- [Redis Persistence](https://redis.io/topics/persistence)
- [Redis Security](https://redis.io/topics/security)

## Support

For questions or issues:

1. Check logs: `docker logs ai_playground_redis`
2. Review cache stats: `GET /api/v1/models/cache/stats`
3. Test Redis connection: `docker exec -it ai_playground_redis redis-cli PING`
4. Clear cache if stuck: `DELETE /api/v1/models/cache/clear`
