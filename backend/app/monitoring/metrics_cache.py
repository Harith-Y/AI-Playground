"""
Metrics Caching System

Provides intelligent caching for Celery monitoring metrics to reduce
infrastructure load while maintaining data freshness.

Features:
- TTL-based caching with configurable expiration
- Automatic cache invalidation
- Cache hit/miss tracking
- Separate TTLs for different metric types
- Memory-efficient storage
"""

import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import json
from dataclasses import dataclass, asdict
from threading import RLock

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached value with metadata."""
    value: Any
    cached_at: float
    expires_at: float
    key: str
    hit_count: int = 0

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'key': self.key,
            'cached_at': datetime.fromtimestamp(self.cached_at).isoformat(),
            'expires_at': datetime.fromtimestamp(self.expires_at).isoformat(),
            'hit_count': self.hit_count,
            'age_seconds': time.time() - self.cached_at,
            'ttl_remaining': max(0, self.expires_at - time.time())
        }


class MetricsCache:
    """
    Thread-safe cache for Celery monitoring metrics.

    Features:
    - TTL-based expiration
    - LRU eviction when max size reached
    - Cache statistics
    - Automatic cleanup of expired entries

    Example:
        cache = MetricsCache(default_ttl=5.0, max_size=100)

        # Store value
        cache.set('key', {'data': 'value'}, ttl=10.0)

        # Retrieve value
        value = cache.get('key')

        # Use decorator
        @cache.cached(ttl=5.0)
        def expensive_function():
            return compute_metrics()
    """

    def __init__(
        self,
        default_ttl: float = 5.0,
        max_size: int = 1000,
        cleanup_interval: float = 60.0
    ):
        """
        Initialize metrics cache.

        Args:
            default_ttl: Default TTL in seconds (default: 5)
            max_size: Maximum number of cache entries (default: 1000)
            cleanup_interval: Interval for cleanup of expired entries (default: 60)
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval

        self._cache: Dict[str, CacheEntry] = {}
        self._lock = RLock()
        self._last_cleanup = time.time()

        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0,
            'expirations': 0,
            'invalidations': 0
        }

        logger.info(
            f"MetricsCache initialized: ttl={default_ttl}s, max_size={max_size}"
        )

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            # Periodic cleanup
            self._maybe_cleanup()

            entry = self._cache.get(key)

            if entry is None:
                self._stats['misses'] += 1
                logger.debug(f"Cache miss: {key}")
                return None

            if entry.is_expired():
                del self._cache[key]
                self._stats['expirations'] += 1
                self._stats['misses'] += 1
                logger.debug(f"Cache expired: {key}")
                return None

            # Cache hit
            entry.hit_count += 1
            self._stats['hits'] += 1
            logger.debug(
                f"Cache hit: {key} (age={time.time() - entry.cached_at:.1f}s)"
            )

            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        with self._lock:
            if ttl is None:
                ttl = self.default_ttl

            now = time.time()
            entry = CacheEntry(
                value=value,
                cached_at=now,
                expires_at=now + ttl,
                key=key,
                hit_count=0
            )

            # Evict if cache is full
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()

            self._cache[key] = entry
            self._stats['sets'] += 1

            logger.debug(f"Cache set: {key} (ttl={ttl}s)")

    def invalidate(self, key: str) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry was found and removed
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats['invalidations'] += 1
                logger.debug(f"Cache invalidated: {key}")
                return True
            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all cache entries matching a pattern.

        Args:
            pattern: Pattern to match (substring match)

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys_to_remove = [
                key for key in self._cache.keys()
                if pattern in key
            ]

            for key in keys_to_remove:
                del self._cache[key]
                self._stats['invalidations'] += 1

            if keys_to_remove:
                logger.info(
                    f"Invalidated {len(keys_to_remove)} entries matching '{pattern}'"
                )

            return len(keys_to_remove)

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats['invalidations'] += count
            logger.info(f"Cache cleared: {count} entries removed")

    def _evict_lru(self):
        """Evict least recently used (by hit count) entry."""
        if not self._cache:
            return

        # Find entry with lowest hit count and oldest cache time
        lru_key = min(
            self._cache.keys(),
            key=lambda k: (self._cache[k].hit_count, -self._cache[k].cached_at)
        )

        del self._cache[lru_key]
        self._stats['evictions'] += 1
        logger.debug(f"Evicted LRU entry: {lru_key}")

    def _maybe_cleanup(self):
        """Perform periodic cleanup of expired entries."""
        now = time.time()

        if now - self._last_cleanup < self.cleanup_interval:
            return

        # Find expired entries
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]

        # Remove expired entries
        for key in expired_keys:
            del self._cache[key]
            self._stats['expirations'] += 1

        self._last_cleanup = now

        if expired_keys:
            logger.debug(f"Cleanup: removed {len(expired_keys)} expired entries")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (
                self._stats['hits'] / total_requests * 100
                if total_requests > 0 else 0
            )

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'utilization_percent': len(self._cache) / self.max_size * 100,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate_percent': hit_rate,
                'sets': self._stats['sets'],
                'evictions': self._stats['evictions'],
                'expirations': self._stats['expirations'],
                'invalidations': self._stats['invalidations'],
                'total_requests': total_requests
            }

    def get_entries(self) -> list:
        """
        Get information about cached entries.

        Returns:
            List of cache entry metadata
        """
        with self._lock:
            return [
                entry.to_dict()
                for entry in self._cache.values()
            ]

    def cached(
        self,
        ttl: Optional[float] = None,
        key_prefix: str = "",
        key_builder: Optional[Callable] = None
    ):
        """
        Decorator for caching function results.

        Args:
            ttl: Time to live in seconds
            key_prefix: Prefix for cache key
            key_builder: Custom function to build cache key from args

        Example:
            @cache.cached(ttl=10.0, key_prefix="metrics")
            def get_metrics(task_id):
                return expensive_operation(task_id)
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Build cache key
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    # Default key builder
                    key_parts = [key_prefix or func.__name__]
                    if args:
                        key_parts.append(str(args))
                    if kwargs:
                        key_parts.append(str(sorted(kwargs.items())))

                    # Hash to keep key size reasonable
                    key_str = ":".join(key_parts)
                    cache_key = hashlib.md5(key_str.encode()).hexdigest()

                # Try to get from cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value

                # Compute value
                result = func(*args, **kwargs)

                # Store in cache
                self.set(cache_key, result, ttl=ttl)

                return result

            return wrapper
        return decorator


# Global cache instances with different TTLs
_task_status_cache = MetricsCache(default_ttl=2.0, max_size=500)  # 2s for task status
_worker_status_cache = MetricsCache(default_ttl=5.0, max_size=100)  # 5s for worker status
_queue_status_cache = MetricsCache(default_ttl=3.0, max_size=100)  # 3s for queue status
_metrics_summary_cache = MetricsCache(default_ttl=5.0, max_size=100)  # 5s for summaries
_health_check_cache = MetricsCache(default_ttl=10.0, max_size=50)  # 10s for health checks


def get_task_status_cache() -> MetricsCache:
    """Get cache for task status endpoints."""
    return _task_status_cache


def get_worker_status_cache() -> MetricsCache:
    """Get cache for worker status endpoints."""
    return _worker_status_cache


def get_queue_status_cache() -> MetricsCache:
    """Get cache for queue status endpoints."""
    return _queue_status_cache


def get_metrics_summary_cache() -> MetricsCache:
    """Get cache for metrics summary endpoints."""
    return _metrics_summary_cache


def get_health_check_cache() -> MetricsCache:
    """Get cache for health check endpoints."""
    return _health_check_cache


def invalidate_all_metrics_caches():
    """Invalidate all metrics caches."""
    _task_status_cache.clear()
    _worker_status_cache.clear()
    _queue_status_cache.clear()
    _metrics_summary_cache.clear()
    _health_check_cache.clear()
    logger.info("All metrics caches invalidated")


def get_all_cache_stats() -> Dict[str, Any]:
    """
    Get statistics for all metrics caches.

    Returns:
        Dictionary with stats for each cache
    """
    return {
        'task_status': _task_status_cache.get_stats(),
        'worker_status': _worker_status_cache.get_stats(),
        'queue_status': _queue_status_cache.get_stats(),
        'metrics_summary': _metrics_summary_cache.get_stats(),
        'health_check': _health_check_cache.get_stats(),
    }
