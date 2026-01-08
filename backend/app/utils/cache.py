"""
Redis Caching Utility

Provides caching decorators and utilities for frequently accessed data
to improve API response times and reduce database load.
"""

import json
import hashlib
from typing import Any, Optional, Callable, Union
from functools import wraps
import redis
from datetime import timedelta

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CacheService:
    """Redis cache service for storing frequently accessed data."""
    
    def __init__(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def is_available(self) -> bool:
        """Check if Redis is available."""
        if self.redis_client is None:
            return False
        try:
            return self.redis_client.ping()
        except:
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found
        """
        if not self.is_available():
            return None
        
        try:
            value = self.redis_client.get(key)
            if value:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(value)
            else:
                logger.debug(f"Cache MISS: {key}")
                return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time to live in seconds or timedelta object
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            serialized_value = json.dumps(value)
            
            if ttl:
                if isinstance(ttl, timedelta):
                    ttl = int(ttl.total_seconds())
                self.redis_client.setex(key, ttl, serialized_value)
            else:
                self.redis_client.set(key, serialized_value)
            
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            self.redis_client.delete(key)
            logger.debug(f"Cache DELETE: {key}")
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.
        
        Args:
            pattern: Pattern to match (e.g., "model:*")
        
        Returns:
            Number of keys deleted
        """
        if not self.is_available():
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cache DELETE pattern {pattern}: {deleted} keys")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cache delete pattern error for {pattern}: {e}")
            return 0
    
    def clear_all(self) -> bool:
        """
        Clear all cache (use with caution).
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            self.redis_client.flushdb()
            logger.warning("Cache FLUSH: All keys cleared")
            return True
        except Exception as e:
            logger.error(f"Cache flush error: {e}")
            return False
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        if not self.is_available():
            return {"available": False}
        
        try:
            info = self.redis_client.info("stats")
            memory = self.redis_client.info("memory")
            
            return {
                "available": True,
                "keys": self.redis_client.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                ),
                "used_memory_human": memory.get("used_memory_human"),
                "used_memory_peak_human": memory.get("used_memory_peak_human"),
                "evicted_keys": info.get("evicted_keys", 0),
                "expired_keys": info.get("expired_keys", 0)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"available": False, "error": str(e)}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage."""
        total = hits + misses
        if total == 0:
            return 0.0
        return round((hits / total) * 100, 2)
    
    def generate_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Hashed cache key
        """
        # Create a string representation of arguments
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
        key_string = "|".join(key_parts)
        
        # Hash the key to ensure it's a valid Redis key
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return key_hash


# Global cache service instance
cache_service = CacheService()


def cache_response(
    ttl: Union[int, timedelta] = 3600,
    key_prefix: str = "",
    skip_cache: Optional[Callable] = None
):
    """
    Decorator to cache function responses.
    
    Args:
        ttl: Time to live in seconds or timedelta (default: 1 hour)
        key_prefix: Prefix for cache key
        skip_cache: Optional function to determine if caching should be skipped
    
    Example:
        @cache_response(ttl=1800, key_prefix="model:metrics")
        def get_model_metrics(model_id: str):
            # Expensive computation
            return metrics
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if caching should be skipped
            if skip_cache and skip_cache(*args, **kwargs):
                logger.debug(f"Skipping cache for {func.__name__}")
                return func(*args, **kwargs)
            
            # Generate cache key
            cache_key = f"{key_prefix}:{cache_service.generate_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_value = cache_service.get(cache_key)
            if cached_value is not None:
                logger.info(f"Returning cached response for {func.__name__}")
                return cached_value
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache the result
            if result is not None:
                cache_service.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def invalidate_cache(key_pattern: str):
    """
    Decorator to invalidate cache after function execution.
    
    Args:
        key_pattern: Pattern of keys to invalidate (e.g., "model:123:*")
    
    Example:
        @invalidate_cache("model:*:metrics")
        def update_model(model_id: str):
            # Update model
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Invalidate cache after successful execution
            cache_service.delete_pattern(key_pattern)
            logger.info(f"Invalidated cache pattern: {key_pattern}")
            
            return result
        
        return wrapper
    return decorator


# Cache key generators for specific use cases
class CacheKeys:
    """Predefined cache key patterns for consistency."""
    
    @staticmethod
    def model_metrics(model_run_id: str) -> str:
        """Cache key for model metrics."""
        return f"model:metrics:{model_run_id}"
    
    @staticmethod
    def model_comparison(model_ids: list) -> str:
        """Cache key for model comparison."""
        sorted_ids = sorted(model_ids)
        id_hash = hashlib.md5("|".join(sorted_ids).encode()).hexdigest()
        return f"model:comparison:{id_hash}"
    
    @staticmethod
    def feature_importance(model_run_id: str, top_n: Optional[int] = None) -> str:
        """Cache key for feature importance."""
        if top_n:
            return f"model:feature_importance:{model_run_id}:top_{top_n}"
        return f"model:feature_importance:{model_run_id}"
    
    @staticmethod
    def dataset_stats(dataset_id: str) -> str:
        """Cache key for dataset statistics."""
        return f"dataset:stats:{dataset_id}"
    
    @staticmethod
    def tuning_results(tuning_run_id: str) -> str:
        """Cache key for tuning results."""
        return f"tuning:results:{tuning_run_id}"
    
    @staticmethod
    def model_ranking(ranking_id: str) -> str:
        """Cache key for model ranking."""
        return f"model:ranking:{ranking_id}"


# TTL constants (in seconds)
class CacheTTL:
    """Standard TTL values for different data types."""
    
    VERY_SHORT = 300  # 5 minutes
    SHORT = 900  # 15 minutes
    MEDIUM = 1800  # 30 minutes
    LONG = 3600  # 1 hour
    VERY_LONG = 7200  # 2 hours
    DAY = 86400  # 24 hours
    WEEK = 604800  # 7 days


def warm_cache(func: Callable, *args, **kwargs) -> bool:
    """
    Warm cache by executing function and storing result.
    
    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments
    
    Returns:
        True if successful, False otherwise
    """
    try:
        result = func(*args, **kwargs)
        return result is not None
    except Exception as e:
        logger.error(f"Cache warming failed: {e}")
        return False


def invalidate_model_cache(model_run_id: str) -> None:
    """
    Invalidate all cache entries related to a specific model run.
    
    This should be called when:
    - Model training completes
    - Model is deleted
    - Model status changes
    - Model metadata is updated
    
    Args:
        model_run_id: UUID of the model run
    """
    try:
        patterns_to_delete = [
            f"model:metrics:{model_run_id}",
            f"model:feature_importance:{model_run_id}*",
            f"model:comparison:*{model_run_id}*"
        ]
        
        for pattern in patterns_to_delete:
            deleted_count = cache_service.delete_pattern(pattern)
            if deleted_count > 0:
                logger.info(
                    f"Invalidated {deleted_count} cache entries for pattern: {pattern}",
                    extra={'event': 'cache_invalidation', 'pattern': pattern}
                )
    except Exception as e:
        logger.error(
            f"Failed to invalidate cache for model {model_run_id}: {e}",
            extra={'event': 'cache_invalidation_failed', 'model_run_id': model_run_id}
        )


def invalidate_comparison_cache() -> None:
    """
    Invalidate all model comparison cache entries.
    
    This should be called when:
    - Any model is updated (as comparisons may include that model)
    - Bulk operations are performed
    """
    try:
        deleted_count = cache_service.delete_pattern("model:comparison:*")
        if deleted_count > 0:
            logger.info(
                f"Invalidated {deleted_count} comparison cache entries",
                extra={'event': 'comparison_cache_invalidation'}
            )
    except Exception as e:
        logger.error(
            f"Failed to invalidate comparison cache: {e}",
            extra={'event': 'comparison_cache_invalidation_failed'}
        )
