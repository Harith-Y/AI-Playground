"""
Celery Task Monitoring API Endpoints

Provides endpoints for monitoring Celery task execution, worker status,
and queue metrics.

Features intelligent caching to reduce load on Celery infrastructure.
"""

from fastapi import APIRouter, HTTPException, Depends, Response
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import time

from app.celery_app import celery_app
from app.monitoring.celery_metrics import (
    get_celery_metrics_summary,
    update_queue_length,
    update_worker_stats,
)
from app.monitoring.metrics_cache import (
    get_task_status_cache,
    get_worker_status_cache,
    get_queue_status_cache,
    get_metrics_summary_cache,
    get_health_check_cache,
    get_all_cache_stats,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Cache instances
task_cache = get_task_status_cache()
worker_cache = get_worker_status_cache()
queue_cache = get_queue_status_cache()
summary_cache = get_metrics_summary_cache()
health_cache = get_health_check_cache()


@router.get("/tasks/status", response_model=Dict[str, Any])
async def get_task_status(response: Response, use_cache: bool = True):
    """
    Get current status of Celery tasks.

    Args:
        use_cache: Whether to use cached data (default: True)

    Returns:
        Dictionary with task status information including:
        - Active tasks
        - Reserved tasks
        - Scheduled tasks
        - Total pending tasks

    Cache TTL: 2 seconds
    """
    cache_key = "task_status"

    # Try cache first
    if use_cache:
        cached_data = task_cache.get(cache_key)
        if cached_data is not None:
            response.headers["X-Cache"] = "HIT"
            response.headers["X-Cache-Age"] = str(int(time.time() - cached_data.get('_cache_time', 0)))
            return cached_data

    response.headers["X-Cache"] = "MISS"

    try:
        from celery import current_app
        inspect = current_app.control.inspect()

        active = inspect.active() or {}
        reserved = inspect.reserved() or {}
        scheduled = inspect.scheduled() or {}

        # Aggregate task counts
        total_active = sum(len(tasks) for tasks in active.values())
        total_reserved = sum(len(tasks) for tasks in reserved.values())
        total_scheduled = sum(len(tasks) for tasks in scheduled.values())

        # Get task details
        active_tasks = []
        for worker_name, tasks in active.items():
            for task in tasks:
                active_tasks.append({
                    'worker': worker_name,
                    'task_id': task.get('id'),
                    'task_name': task.get('name'),
                    'time_start': task.get('time_start'),
                    'args': task.get('args'),
                    'kwargs': task.get('kwargs')
                })

        result = {
            'timestamp': datetime.utcnow().isoformat(),
            'summary': {
                'active': total_active,
                'reserved': total_reserved,
                'scheduled': total_scheduled,
                'total_pending': total_active + total_reserved + total_scheduled
            },
            'active_tasks': active_tasks[:50],  # Limit to 50 for performance
            'workers': list(active.keys()),
            '_cache_time': time.time()  # Internal field for cache age
        }

        # Cache the result
        if use_cache:
            task_cache.set(cache_key, result)

        return result

    except Exception as e:
        logger.error(f"Failed to get task status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")


@router.get("/workers/status", response_model=Dict[str, Any])
async def get_worker_status(response: Response, use_cache: bool = True):
    """
    Get current status of Celery workers.

    Args:
        use_cache: Whether to use cached data (default: True)

    Returns:
        Dictionary with worker status information including:
        - Worker names
        - Pool size
        - Active tasks
        - Resource usage

    Cache TTL: 5 seconds
    """
    cache_key = "worker_status"

    # Try cache first
    if use_cache:
        cached_data = worker_cache.get(cache_key)
        if cached_data is not None:
            response.headers["X-Cache"] = "HIT"
            response.headers["X-Cache-Age"] = str(int(time.time() - cached_data.get('_cache_time', 0)))
            return cached_data

    response.headers["X-Cache"] = "MISS"

    try:
        from celery import current_app
        inspect = current_app.control.inspect()

        stats = inspect.stats() or {}
        active = inspect.active() or {}

        workers = []
        for worker_name, worker_stats in stats.items():
            pool_info = worker_stats.get('pool', {})
            active_tasks = len(active.get(worker_name, []))

            workers.append({
                'name': worker_name,
                'status': 'online',
                'pool': {
                    'max_concurrency': pool_info.get('max-concurrency', 0),
                    'processes': pool_info.get('processes', []),
                    'busy_workers': active_tasks
                },
                'prefetch_count': worker_stats.get('prefetch_count', 0),
                'broker': worker_stats.get('broker', {}),
                'clock': worker_stats.get('clock', 0)
            })

        # Update worker stats metrics
        update_worker_stats(celery_app)

        result = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_workers': len(workers),
            'workers': workers,
            '_cache_time': time.time()
        }

        # Cache the result
        if use_cache:
            worker_cache.set(cache_key, result)

        return result

    except Exception as e:
        logger.error(f"Failed to get worker status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get worker status: {str(e)}")


@router.get("/queues/status", response_model=Dict[str, Any])
async def get_queue_status(response: Response, use_cache: bool = True):
    """
    Get current status of Celery queues.

    Args:
        use_cache: Whether to use cached data (default: True)

    Returns:
        Dictionary with queue status information

    Cache TTL: 3 seconds
    """
    cache_key = "queue_status"

    if use_cache:
        cached_data = queue_cache.get(cache_key)
        if cached_data is not None:
            response.headers["X-Cache"] = "HIT"
            return cached_data

    response.headers["X-Cache"] = "MISS"

    try:
        # Update queue metrics
        update_queue_length(celery_app, queue_name='celery')

        # Get metrics summary
        summary = get_celery_metrics_summary(celery_app)

        result = {
            'timestamp': datetime.utcnow().isoformat(),
            'queues': summary.get('queues', {}),
            'total_pending': summary.get('tasks', {}).get('total_pending', 0),
            '_cache_time': time.time()
        }

        if use_cache:
            queue_cache.set(cache_key, result)

        return result

    except Exception as e:
        logger.error(f"Failed to get queue status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get queue status: {str(e)}")


@router.get("/metrics/summary", response_model=Dict[str, Any])
async def get_metrics_summary(response: Response, use_cache: bool = True):
    """
    Get comprehensive metrics summary for Celery tasks.

    Args:
        use_cache: Whether to use cached data (default: True)

    Returns:
        Dictionary with:
        - Worker statistics
        - Task statistics
        - Queue statistics

    Cache TTL: 5 seconds
    """
    cache_key = "metrics_summary"

    if use_cache:
        cached_data = summary_cache.get(cache_key)
        if cached_data is not None:
            response.headers["X-Cache"] = "HIT"
            return cached_data

    response.headers["X-Cache"] = "MISS"

    try:
        summary = get_celery_metrics_summary(celery_app)

        result = {
            'timestamp': datetime.utcnow().isoformat(),
            **summary,
            '_cache_time': time.time()
        }

        if use_cache:
            summary_cache.set(cache_key, result)

        return result

    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get metrics summary: {str(e)}")


@router.get("/tasks/{task_id}/status")
async def get_task_status_by_id(task_id: str):
    """
    Get status of a specific task by ID.

    Args:
        task_id: Celery task ID

    Returns:
        Task status information
    """
    try:
        from celery.result import AsyncResult

        result = AsyncResult(task_id, app=celery_app)

        response = {
            'task_id': task_id,
            'state': result.state,
            'ready': result.ready(),
            'successful': result.successful() if result.ready() else None,
            'failed': result.failed() if result.ready() else None,
        }

        # Add result if task is ready
        if result.ready():
            if result.successful():
                response['result'] = result.result
            elif result.failed():
                response['error'] = str(result.info)
        else:
            # Add progress info if available
            if result.info:
                response['info'] = result.info

        return response

    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")


@router.post("/tasks/{task_id}/revoke")
async def revoke_task(task_id: str, terminate: bool = False):
    """
    Revoke a running task.

    Args:
        task_id: Celery task ID
        terminate: Whether to terminate the task (default: False)

    Returns:
        Confirmation message
    """
    try:
        from celery.result import AsyncResult

        result = AsyncResult(task_id, app=celery_app)
        result.revoke(terminate=terminate)

        logger.info(
            f"Task revoked: {task_id}",
            extra={
                'event': 'task_revoked',
                'task_id': task_id,
                'terminate': terminate
            }
        )

        return {
            'task_id': task_id,
            'revoked': True,
            'terminated': terminate,
            'message': f"Task {'terminated' if terminate else 'revoked'} successfully"
        }

    except Exception as e:
        logger.error(f"Failed to revoke task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to revoke task: {str(e)}")


@router.get("/health")
async def check_health():
    """
    Check health of Celery workers and message broker.

    Returns:
        Health status information
    """
    try:
        from celery import current_app
        inspect = current_app.control.inspect()

        # Check if workers are responding
        stats = inspect.stats()

        if not stats:
            return {
                'status': 'unhealthy',
                'message': 'No workers responding',
                'workers': 0,
                'timestamp': datetime.utcnow().isoformat()
            }

        # Check broker connection
        try:
            celery_app.connection().ensure_connection(max_retries=3)
            broker_healthy = True
        except Exception:
            broker_healthy = False

        return {
            'status': 'healthy' if broker_healthy else 'degraded',
            'message': 'All systems operational' if broker_healthy else 'Broker connection issues',
            'workers': len(stats),
            'broker_connected': broker_healthy,
            'timestamp': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            'status': 'unhealthy',
            'message': str(e),
            'workers': 0,
            'timestamp': datetime.utcnow().isoformat()
        }


@router.get("/stats/detailed")
async def get_detailed_stats():
    """
    Get detailed statistics about task execution.

    Returns:
        Detailed execution statistics
    """
    try:
        from celery import current_app
        inspect = current_app.control.inspect()

        stats = inspect.stats() or {}
        registered = inspect.registered() or {}
        active = inspect.active() or {}
        reserved = inspect.reserved() or {}

        detailed_stats = {}

        for worker_name, worker_stats in stats.items():
            detailed_stats[worker_name] = {
                'broker': worker_stats.get('broker', {}),
                'clock': worker_stats.get('clock', 0),
                'pool': worker_stats.get('pool', {}),
                'prefetch_count': worker_stats.get('prefetch_count', 0),
                'registered_tasks': len(registered.get(worker_name, [])),
                'active_tasks': len(active.get(worker_name, [])),
                'reserved_tasks': len(reserved.get(worker_name, [])),
                'total': worker_stats.get('total', {})
            }

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'workers': detailed_stats
        }

    except Exception as e:
        logger.error(f"Failed to get detailed stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get detailed stats: {str(e)}")


@router.get("/cache/stats")
async def get_cache_stats():
    """
    Get statistics about cache performance.

    Returns:
        Cache statistics for all endpoint caches
    """
    try:
        stats = get_all_cache_stats()

        # Add overall summary
        total_hits = sum(cache['hits'] for cache in stats.values())
        total_misses = sum(cache['misses'] for cache in stats.values())
        total_requests = total_hits + total_misses
        overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'caches': stats,
            'summary': {
                'total_hits': total_hits,
                'total_misses': total_misses,
                'total_requests': total_requests,
                'overall_hit_rate_percent': overall_hit_rate
            }
        }

    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@router.post("/cache/invalidate")
async def invalidate_cache(cache_name: Optional[str] = None):
    """
    Invalidate monitoring caches.

    Args:
        cache_name: Specific cache to invalidate (optional)
                   Options: task_status, worker_status, queue_status,
                           metrics_summary, health_check
                   If not provided, all caches are invalidated

    Returns:
        Confirmation message
    """
    try:
        if cache_name:
            # Invalidate specific cache
            cache_map = {
                'task_status': task_cache,
                'worker_status': worker_cache,
                'queue_status': queue_cache,
                'metrics_summary': summary_cache,
                'health_check': health_cache
            }

            if cache_name not in cache_map:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid cache name. Choose from: {', '.join(cache_map.keys())}"
                )

            cache_map[cache_name].clear()
            message = f"Cache '{cache_name}' invalidated"

        else:
            # Invalidate all caches
            from app.monitoring.metrics_cache import invalidate_all_metrics_caches
            invalidate_all_metrics_caches()
            message = "All monitoring caches invalidated"

        logger.info(message)

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'message': message,
            'cache_name': cache_name or 'all'
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to invalidate cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to invalidate cache: {str(e)}")
