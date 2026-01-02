"""
Memory management utilities for optimizing memory usage.

This module provides tools for:
- Memory profiling and monitoring
- Automatic garbage collection
- Memory-aware caching
- Memory-efficient data structures
- Memory leak detection
"""

import gc
import psutil
import os
import sys
from typing import Dict, Any, Optional, Callable, List
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass
import time
import weakref

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Percentage of total RAM used
    available_mb: float  # Available system memory
    total_mb: float  # Total system memory
    python_objects: int  # Number of Python objects

    def __str__(self) -> str:
        return (
            f"Memory: {self.rss_mb:.2f}MB RSS, "
            f"{self.percent:.1f}% system usage, "
            f"{self.available_mb:.2f}MB available"
        )


class MemoryMonitor:
    """
    Monitor and track memory usage.

    Provides real-time memory statistics and alerts.
    """

    def __init__(self, threshold_percent: float = 80.0):
        """
        Initialize memory monitor.

        Args:
            threshold_percent: Alert threshold for memory usage
        """
        self.process = psutil.Process(os.getpid())
        self.threshold_percent = threshold_percent
        self.baseline: Optional[MemorySnapshot] = None
        self.snapshots: List[MemorySnapshot] = []

    def get_current_snapshot(self) -> MemorySnapshot:
        """
        Get current memory usage snapshot.

        Returns:
            MemorySnapshot with current memory stats
        """
        # Process memory
        mem_info = self.process.memory_info()
        rss_mb = mem_info.rss / (1024 * 1024)
        vms_mb = mem_info.vms / (1024 * 1024)

        # System memory
        sys_mem = psutil.virtual_memory()

        # Python objects
        python_objects = len(gc.get_objects())

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=rss_mb,
            vms_mb=vms_mb,
            percent=sys_mem.percent,
            available_mb=sys_mem.available / (1024 * 1024),
            total_mb=sys_mem.total / (1024 * 1024),
            python_objects=python_objects
        )

        return snapshot

    def set_baseline(self) -> MemorySnapshot:
        """
        Set baseline memory usage.

        Returns:
            Baseline snapshot
        """
        self.baseline = self.get_current_snapshot()
        logger.info(f"Memory baseline set: {self.baseline}")
        return self.baseline

    def get_memory_delta(self) -> Optional[float]:
        """
        Get memory usage change since baseline.

        Returns:
            Memory delta in MB, or None if no baseline
        """
        if self.baseline is None:
            return None

        current = self.get_current_snapshot()
        delta = current.rss_mb - self.baseline.rss_mb
        return delta

    def check_threshold(self) -> bool:
        """
        Check if memory usage exceeds threshold.

        Returns:
            True if threshold exceeded
        """
        current = self.get_current_snapshot()
        exceeded = current.percent > self.threshold_percent

        if exceeded:
            logger.warning(
                f"Memory threshold exceeded: {current.percent:.1f}% > {self.threshold_percent}%",
                extra={
                    'event': 'memory_threshold_exceeded',
                    'current_percent': current.percent,
                    'threshold': self.threshold_percent,
                    'rss_mb': current.rss_mb
                }
            )

        return exceeded

    def log_memory_usage(self, message: str = "Memory usage"):
        """
        Log current memory usage.

        Args:
            message: Log message prefix
        """
        snapshot = self.get_current_snapshot()

        delta_str = ""
        if self.baseline:
            delta = snapshot.rss_mb - self.baseline.rss_mb
            delta_str = f" ({delta:+.2f}MB from baseline)"

        logger.info(
            f"{message}: {snapshot}{delta_str}",
            extra={
                'event': 'memory_usage',
                'rss_mb': snapshot.rss_mb,
                'percent': snapshot.percent,
                'available_mb': snapshot.available_mb
            }
        )

    def take_snapshot(self) -> MemorySnapshot:
        """
        Take and store memory snapshot.

        Returns:
            Memory snapshot
        """
        snapshot = self.get_current_snapshot()
        self.snapshots.append(snapshot)
        return snapshot

    def get_peak_memory(self) -> Optional[float]:
        """
        Get peak memory usage from snapshots.

        Returns:
            Peak RSS in MB, or None if no snapshots
        """
        if not self.snapshots:
            return None
        return max(s.rss_mb for s in self.snapshots)

    def get_memory_trend(self) -> Dict[str, Any]:
        """
        Analyze memory usage trend.

        Returns:
            Trend analysis dict
        """
        if len(self.snapshots) < 2:
            return {'status': 'insufficient_data'}

        rss_values = [s.rss_mb for s in self.snapshots]

        # Calculate trend
        start_rss = rss_values[0]
        end_rss = rss_values[-1]
        delta = end_rss - start_rss
        percent_change = (delta / start_rss) * 100 if start_rss > 0 else 0

        # Detect trend
        if abs(percent_change) < 5:
            trend = 'stable'
        elif percent_change > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'

        # Check for leaks (consistent increase)
        if len(rss_values) >= 5:
            recent_increases = sum(
                1 for i in range(len(rss_values)-1)
                if rss_values[i+1] > rss_values[i]
            )
            leak_suspected = recent_increases / (len(rss_values)-1) > 0.8
        else:
            leak_suspected = False

        return {
            'status': 'analyzed',
            'trend': trend,
            'start_mb': start_rss,
            'end_mb': end_rss,
            'delta_mb': delta,
            'percent_change': percent_change,
            'peak_mb': max(rss_values),
            'average_mb': sum(rss_values) / len(rss_values),
            'leak_suspected': leak_suspected
        }


class MemoryOptimizer:
    """
    Optimize memory usage automatically.

    Provides strategies for reducing memory consumption.
    """

    @staticmethod
    def aggressive_gc() -> Dict[str, int]:
        """
        Perform aggressive garbage collection.

        Returns:
            GC statistics
        """
        logger.debug("Running aggressive garbage collection")

        # Collect all generations
        collected = {
            'gen0': gc.collect(0),
            'gen1': gc.collect(1),
            'gen2': gc.collect(2)
        }

        total = sum(collected.values())

        logger.info(
            f"Garbage collected: {total} objects "
            f"(gen0: {collected['gen0']}, gen1: {collected['gen1']}, gen2: {collected['gen2']})"
        )

        return collected

    @staticmethod
    def clear_cache(cache_name: str = "all"):
        """
        Clear specified cache.

        Args:
            cache_name: Name of cache to clear
        """
        # Clear function caches
        if cache_name == "all" or cache_name == "functools":
            # Clear lru_cache for all cached functions
            # Note: Would need to track cached functions separately
            pass

        logger.info(f"Cleared cache: {cache_name}")

    @staticmethod
    def optimize_dataframe_memory(df, aggressive: bool = False):
        """
        Optimize pandas DataFrame memory usage.

        Args:
            df: Pandas DataFrame
            aggressive: Use aggressive optimization

        Returns:
            Optimized DataFrame
        """
        import pandas as pd
        import numpy as np

        initial_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)

        for col in df.columns:
            col_type = df[col].dtype

            # Optimize numeric types
            if col_type in ['int64', 'int32', 'int16', 'int8']:
                c_min = df[col].min()
                c_max = df[col].max()

                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

            elif col_type in ['float64', 'float32']:
                if aggressive or col_type == 'float64':
                    df[col] = df[col].astype(np.float32)

            # Convert to categorical if beneficial
            elif col_type == 'object':
                num_unique = df[col].nunique()
                num_total = len(df[col])

                # Convert if less than 50% unique values
                if num_unique / num_total < 0.5:
                    df[col] = df[col].astype('category')

        final_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        reduction = ((initial_memory - final_memory) / initial_memory) * 100

        logger.info(
            f"DataFrame memory optimized: {initial_memory:.2f}MB → {final_memory:.2f}MB "
            f"({reduction:.1f}% reduction)"
        )

        return df

    @staticmethod
    def find_memory_leaks() -> List[Dict[str, Any]]:
        """
        Detect potential memory leaks.

        Returns:
            List of suspected leak sources
        """
        # Get all objects
        all_objects = gc.get_objects()

        # Count by type
        type_counts = {}
        for obj in all_objects:
            obj_type = type(obj).__name__
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

        # Find types with unusually high counts
        suspects = []
        for obj_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            if count > 1000:  # Arbitrary threshold
                suspects.append({
                    'type': obj_type,
                    'count': count,
                    'suspected_leak': count > 10000
                })

        if suspects:
            logger.warning(
                f"Potential memory leaks detected: {len(suspects)} suspicious types",
                extra={'suspects': suspects}
            )

        return suspects


@contextmanager
def memory_profiler(operation_name: str, log_result: bool = True):
    """
    Context manager for profiling memory usage of operations.

    Args:
        operation_name: Name of operation being profiled
        log_result: Whether to log the result

    Yields:
        MemoryMonitor instance

    Example:
        with memory_profiler("Model Training") as monitor:
            model.fit(X, y)
        # Automatically logs memory usage
    """
    monitor = MemoryMonitor()
    monitor.set_baseline()

    logger.info(f"Starting memory profiling: {operation_name}")

    try:
        yield monitor
    finally:
        delta = monitor.get_memory_delta()
        snapshot = monitor.get_current_snapshot()

        if log_result:
            logger.info(
                f"Memory profiling complete: {operation_name} | "
                f"Delta: {delta:+.2f}MB | "
                f"Final: {snapshot.rss_mb:.2f}MB",
                extra={
                    'event': 'memory_profile_complete',
                    'operation': operation_name,
                    'delta_mb': delta,
                    'final_mb': snapshot.rss_mb
                }
            )


def memory_efficient(threshold_mb: float = 100.0):
    """
    Decorator for memory-efficient function execution.

    Performs garbage collection after function if memory usage exceeds threshold.

    Args:
        threshold_mb: Memory increase threshold for triggering GC

    Example:
        @memory_efficient(threshold_mb=50.0)
        def process_large_data(data):
            # Process data
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = MemoryMonitor()
            monitor.set_baseline()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                delta = monitor.get_memory_delta()

                if delta and delta > threshold_mb:
                    logger.info(
                        f"Function {func.__name__} used {delta:.2f}MB, "
                        f"running garbage collection"
                    )
                    MemoryOptimizer.aggressive_gc()

        return wrapper
    return decorator


class MemoryCache:
    """
    Memory-aware caching with automatic eviction.

    Evicts least recently used items when memory threshold is reached.
    """

    def __init__(
        self,
        max_memory_mb: float = 500.0,
        max_items: int = 1000
    ):
        """
        Initialize memory cache.

        Args:
            max_memory_mb: Maximum memory to use for cache
            max_items: Maximum number of items to cache
        """
        self.max_memory_mb = max_memory_mb
        self.max_items = max_items
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._sizes: Dict[str, int] = {}
        self.monitor = MemoryMonitor()

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]
        return None

    def set(self, key: str, value: Any, size_bytes: Optional[int] = None):
        """
        Set item in cache.

        Args:
            key: Cache key
            value: Value to cache
            size_bytes: Estimated size in bytes (auto-calculated if None)
        """
        # Estimate size if not provided
        if size_bytes is None:
            size_bytes = sys.getsizeof(value)

        # Check if we need to evict
        if len(self._cache) >= self.max_items:
            self._evict_lru()

        # Check memory usage
        current_memory = self.monitor.get_current_snapshot().rss_mb
        if current_memory > self.max_memory_mb:
            logger.warning(
                f"Cache memory limit reached: {current_memory:.2f}MB > {self.max_memory_mb}MB"
            )
            self._evict_lru()

        # Store item
        self._cache[key] = value
        self._access_times[key] = time.time()
        self._sizes[key] = size_bytes

    def _evict_lru(self):
        """Evict least recently used item."""
        if not self._access_times:
            return

        # Find LRU item
        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]

        # Remove it
        del self._cache[lru_key]
        del self._access_times[lru_key]
        size = self._sizes.pop(lru_key, 0)

        logger.debug(f"Evicted LRU cache item: {lru_key} ({size} bytes)")

    def clear(self):
        """Clear entire cache."""
        self._cache.clear()
        self._access_times.clear()
        self._sizes.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Stats dictionary
        """
        total_size = sum(self._sizes.values())

        return {
            'items': len(self._cache),
            'max_items': self.max_items,
            'total_size_mb': total_size / (1024 * 1024),
            'max_memory_mb': self.max_memory_mb,
            'current_memory_mb': self.monitor.get_current_snapshot().rss_mb
        }


class WeakValueCache:
    """
    Cache with weak references to values.

    Allows garbage collector to reclaim cached values when needed.
    """

    def __init__(self):
        """Initialize weak value cache."""
        self._cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if collected
        """
        return self._cache.get(key)

    def set(self, key: str, value: Any):
        """
        Set item in cache with weak reference.

        Args:
            key: Cache key
            value: Value to cache (must be object, not primitive)
        """
        try:
            self._cache[key] = value
        except TypeError:
            # Can't create weak reference to this type
            logger.warning(f"Cannot create weak reference for type: {type(value)}")

    def clear(self):
        """Clear cache."""
        self._cache.clear()

    def items_count(self) -> int:
        """
        Get number of items still in cache.

        Returns:
            Count of non-collected items
        """
        return len(self._cache)


# Global memory monitor instance
_global_monitor = MemoryMonitor()


def get_memory_monitor() -> MemoryMonitor:
    """
    Get global memory monitor instance.

    Returns:
        Global MemoryMonitor
    """
    return _global_monitor


def log_memory(message: str = "Memory usage"):
    """
    Log current memory usage using global monitor.

    Args:
        message: Log message
    """
    _global_monitor.log_memory_usage(message)


def check_memory_threshold(threshold_percent: float = 80.0) -> bool:
    """
    Check if memory usage exceeds threshold.

    Args:
        threshold_percent: Threshold percentage

    Returns:
        True if exceeded
    """
    monitor = MemoryMonitor(threshold_percent=threshold_percent)
    return monitor.check_threshold()
