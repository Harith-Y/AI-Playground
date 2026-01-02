"""
Celery Task Performance Monitoring

Integrates Prometheus metrics with Celery tasks for monitoring
training, tuning, and evaluation operations.
"""

import time
import psutil
from typing import Any, Dict
from celery import signals
from celery.app.task import Task
from prometheus_client import Gauge, Counter, Histogram

from .metrics import (
    task_duration_histogram,
    task_counter,
    active_tasks_gauge,
    memory_usage_gauge,
    cpu_usage_gauge,
    training_duration_histogram,
    training_samples_histogram,
    training_features_histogram,
    training_result_counter,
    tuning_duration_histogram,
    tuning_iterations_histogram,
    tuning_result_counter,
    tuning_best_score_histogram,
    models_trained_counter,
    task_queue_length_gauge,  # Import queue length metric from metrics.py
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Store task start times
_task_start_times: Dict[str, float] = {}

# ============================================================================
# Additional Celery-Specific Metrics
# ============================================================================

# Worker pool metrics
celery_worker_pool_size = Gauge(
    'celery_worker_pool_size',
    'Number of worker processes in the pool',
    ['worker_name']
)

celery_worker_pool_busy = Gauge(
    'celery_worker_pool_busy',
    'Number of busy worker processes',
    ['worker_name']
)

# Queue length tracking - use the one from metrics.py
celery_queue_length = task_queue_length_gauge

# Task failure rate
celery_task_failure_rate = Gauge(
    'celery_task_failure_rate',
    'Task failure rate (failures per minute)',
    ['task_name']
)

# Task retry tracking
celery_task_retry_counter = Counter(
    'celery_task_retries_total',
    'Total number of task retries',
    ['task_name', 'reason']
)

# Task timeout tracking
celery_task_timeout_counter = Counter(
    'celery_task_timeouts_total',
    'Total number of task timeouts',
    ['task_name']
)

# Prefetch multiplier tracking
celery_worker_prefetch_count = Gauge(
    'celery_worker_prefetch_count',
    'Number of tasks prefetched by worker',
    ['worker_name']
)

# Task result backend latency
celery_result_backend_latency = Histogram(
    'celery_result_backend_latency_seconds',
    'Latency of result backend operations',
    ['operation'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0)
)

# Task ETA (estimated time of arrival) tracking
celery_task_eta_delta = Histogram(
    'celery_task_eta_delta_seconds',
    'Difference between scheduled ETA and actual execution time',
    ['task_name'],
    buckets=(0, 1, 5, 10, 30, 60, 300, 600, 1800, 3600)
)

# Percentile tracking for task duration
celery_task_duration_percentiles = Gauge(
    'celery_task_duration_percentile_seconds',
    'Task duration percentiles',
    ['task_name', 'percentile']
)

# Store failure counts for rate calculation
_task_failure_counts: Dict[str, list] = {}
_failure_rate_window = 60.0  # 1 minute window


def setup_celery_metrics(celery_app):
    """
    Setup Celery signal handlers for metrics collection.
    
    Args:
        celery_app: Celery application instance
    """
    
    @signals.task_prerun.connect
    def task_prerun_handler(sender=None, task_id=None, task=None, **kwargs):
        """Called before task execution starts."""
        task_name = sender.name if sender else 'unknown'
        
        # Track active tasks
        active_tasks_gauge.labels(task_name=task_name).inc()
        
        # Store start time
        _task_start_times[task_id] = time.time()
        
        # Track resource usage
        process = psutil.Process()
        memory_usage_gauge.labels(process_type='celery_worker').set(
            process.memory_info().rss
        )
        cpu_usage_gauge.labels(process_type='celery_worker').set(
            process.cpu_percent(interval=0.1)
        )
        
        logger.debug(
            f"Task started: {task_name}",
            extra={
                'event': 'task_prerun',
                'task_id': task_id,
                'task_name': task_name
            }
        )
    
    @signals.task_postrun.connect
    def task_postrun_handler(sender=None, task_id=None, task=None, 
                            retval=None, state=None, **kwargs):
        """Called after task execution completes."""
        task_name = sender.name if sender else 'unknown'
        
        # Calculate duration
        start_time = _task_start_times.pop(task_id, None)
        if start_time:
            duration = time.time() - start_time
            
            # Track duration
            task_duration_histogram.labels(
                task_name=task_name,
                status='success'
            ).observe(duration)
            
            # Track specific metrics based on task type
            if 'train_model' in task_name and retval:
                _track_training_metrics(retval, duration)
            elif 'tune_hyperparameters' in task_name and retval:
                _track_tuning_metrics(retval, duration)
        
        # Track task completion
        task_counter.labels(
            task_name=task_name,
            status='success'
        ).inc()
        
        # Decrement active tasks
        active_tasks_gauge.labels(task_name=task_name).dec()
        
        # Track resource usage
        process = psutil.Process()
        memory_usage_gauge.labels(process_type='celery_worker').set(
            process.memory_info().rss
        )
        cpu_usage_gauge.labels(process_type='celery_worker').set(
            process.cpu_percent(interval=0.1)
        )
        
        logger.info(
            f"Task completed: {task_name}",
            extra={
                'event': 'task_postrun',
                'task_id': task_id,
                'task_name': task_name,
                'duration': duration if start_time else None
            }
        )
    
    @signals.task_failure.connect
    def task_failure_handler(sender=None, task_id=None, exception=None,
                            traceback=None, **kwargs):
        """Called when task fails."""
        task_name = sender.name if sender else 'unknown'

        # Calculate duration if available
        start_time = _task_start_times.pop(task_id, None)
        if start_time:
            duration = time.time() - start_time
            task_duration_histogram.labels(
                task_name=task_name,
                status='failure'
            ).observe(duration)

        # Track failure
        task_counter.labels(
            task_name=task_name,
            status='failure'
        ).inc()

        # Track failure rate
        _update_failure_rate(task_name)

        # Check for timeout
        if 'TimeLimitExceeded' in str(type(exception).__name__):
            celery_task_timeout_counter.labels(task_name=task_name).inc()

        # Decrement active tasks
        active_tasks_gauge.labels(task_name=task_name).dec()

        # Track specific failures
        if 'train_model' in task_name:
            training_result_counter.labels(
                model_type='unknown',
                result='failure'
            ).inc()
        elif 'tune_hyperparameters' in task_name:
            tuning_result_counter.labels(
                model_type='unknown',
                tuning_method='unknown',
                result='failure'
            ).inc()

        logger.error(
            f"Task failed: {task_name}",
            extra={
                'event': 'task_failure',
                'task_id': task_id,
                'task_name': task_name,
                'exception': str(exception)
            },
            exc_info=True
        )
    
    @signals.task_retry.connect
    def task_retry_handler(sender=None, task_id=None, reason=None, **kwargs):
        """Called when task is retried."""
        task_name = sender.name if sender else 'unknown'

        task_counter.labels(
            task_name=task_name,
            status='retry'
        ).inc()

        # Track retry with reason
        celery_task_retry_counter.labels(
            task_name=task_name,
            reason=str(reason) if reason else 'unknown'
        ).inc()

        logger.warning(
            f"Task retried: {task_name}",
            extra={
                'event': 'task_retry',
                'task_id': task_id,
                'task_name': task_name,
                'reason': str(reason)
            }
        )
    
    @signals.task_revoked.connect
    def task_revoked_handler(sender=None, request=None, terminated=None, 
                            signum=None, expired=None, **kwargs):
        """Called when task is revoked."""
        task_name = sender.name if sender else 'unknown'
        task_id = request.id if request else 'unknown'
        
        # Clean up start time
        _task_start_times.pop(task_id, None)
        
        # Track revocation
        task_counter.labels(
            task_name=task_name,
            status='revoked'
        ).inc()
        
        # Decrement active tasks
        active_tasks_gauge.labels(task_name=task_name).dec()
        
        logger.warning(
            f"Task revoked: {task_name}",
            extra={
                'event': 'task_revoked',
                'task_id': task_id,
                'task_name': task_name,
                'terminated': terminated,
                'expired': expired
            }
        )
    
    logger.info("Celery metrics monitoring configured")


def _track_training_metrics(result: Dict[str, Any], duration: float):
    """
    Track training-specific metrics.
    
    Args:
        result: Training task result dictionary
        duration: Task duration in seconds
    """
    try:
        model_type = result.get('model_type', 'unknown')
        task_type = result.get('task_type', 'unknown')
        
        # Track training duration
        training_duration_histogram.labels(
            model_type=model_type,
            task_type=task_type
        ).observe(result.get('training_time', duration))
        
        # Track training samples
        train_samples = result.get('train_samples', 0)
        if train_samples > 0:
            training_samples_histogram.labels(
                model_type=model_type
            ).observe(train_samples)
        
        # Track features
        n_features = result.get('n_features', 0)
        if n_features > 0:
            training_features_histogram.labels(
                model_type=model_type
            ).observe(n_features)
        
        # Track result
        status = result.get('status', 'unknown')
        training_result_counter.labels(
            model_type=model_type,
            result=status
        ).inc()
        
        # Track models trained
        if status == 'completed':
            models_trained_counter.labels(
                model_type=model_type,
                task_type=task_type
            ).inc()
        
    except Exception as e:
        logger.error(f"Failed to track training metrics: {e}")


def _track_tuning_metrics(result: Dict[str, Any], duration: float):
    """
    Track tuning-specific metrics.
    
    Args:
        result: Tuning task result dictionary
        duration: Task duration in seconds
    """
    try:
        model_type = result.get('model_type', 'unknown')
        tuning_method = result.get('tuning_method', 'unknown')
        
        # Track tuning duration
        tuning_duration_histogram.labels(
            model_type=model_type,
            tuning_method=tuning_method
        ).observe(result.get('tuning_time', duration))
        
        # Track iterations
        total_combinations = result.get('total_combinations', 0)
        if total_combinations > 0:
            tuning_iterations_histogram.labels(
                tuning_method=tuning_method
            ).observe(total_combinations)
        
        # Track best score
        best_score = result.get('best_score')
        if best_score is not None:
            tuning_best_score_histogram.labels(
                model_type=model_type,
                tuning_method=tuning_method
            ).observe(best_score)
        
        # Track result
        status = result.get('status', 'unknown')
        tuning_result_counter.labels(
            model_type=model_type,
            tuning_method=tuning_method,
            result=status
        ).inc()
        
    except Exception as e:
        logger.error(f"Failed to track tuning metrics: {e}")


def _update_failure_rate(task_name: str):
    """
    Update failure rate for a task.

    Args:
        task_name: Name of the task
    """
    current_time = time.time()

    if task_name not in _task_failure_counts:
        _task_failure_counts[task_name] = []

    # Add current failure
    _task_failure_counts[task_name].append(current_time)

    # Remove failures outside the time window
    cutoff_time = current_time - _failure_rate_window
    _task_failure_counts[task_name] = [
        t for t in _task_failure_counts[task_name] if t > cutoff_time
    ]

    # Calculate failures per minute
    failure_count = len(_task_failure_counts[task_name])
    failure_rate = failure_count / (_failure_rate_window / 60.0)

    celery_task_failure_rate.labels(task_name=task_name).set(failure_rate)


def update_queue_length(celery_app, queue_name: str = 'celery'):
    """
    Update queue length metrics.

    Args:
        celery_app: Celery application instance
        queue_name: Name of the queue to monitor
    """
    try:
        from celery import current_app
        inspect = current_app.control.inspect()

        # Get active queues
        active = inspect.active()
        reserved = inspect.reserved()
        scheduled = inspect.scheduled()

        total_tasks = 0
        if active:
            total_tasks += sum(len(tasks) for tasks in active.values())
        if reserved:
            total_tasks += sum(len(tasks) for tasks in reserved.values())
        if scheduled:
            total_tasks += sum(len(tasks) for tasks in scheduled.values())

        celery_queue_length.labels(queue_name=queue_name).set(total_tasks)

        logger.debug(f"Queue length for {queue_name}: {total_tasks}")

    except Exception as e:
        logger.error(f"Failed to update queue length: {e}")


def update_worker_stats(celery_app):
    """
    Update worker pool statistics.

    Args:
        celery_app: Celery application instance
    """
    try:
        from celery import current_app
        inspect = current_app.control.inspect()

        # Get worker stats
        stats = inspect.stats()

        if stats:
            for worker_name, worker_stats in stats.items():
                # Pool size
                pool_size = worker_stats.get('pool', {}).get('max-concurrency', 0)
                celery_worker_pool_size.labels(worker_name=worker_name).set(pool_size)

                # Active tasks (proxy for busy workers)
                active_tasks = inspect.active()
                if active_tasks and worker_name in active_tasks:
                    busy_count = len(active_tasks[worker_name])
                    celery_worker_pool_busy.labels(worker_name=worker_name).set(busy_count)

                # Prefetch count
                prefetch_count = worker_stats.get('prefetch_count', 0)
                celery_worker_prefetch_count.labels(worker_name=worker_name).set(prefetch_count)

                logger.debug(
                    f"Worker {worker_name}: pool_size={pool_size}, "
                    f"prefetch={prefetch_count}"
                )

    except Exception as e:
        logger.error(f"Failed to update worker stats: {e}")


def get_celery_metrics_summary(celery_app) -> Dict[str, Any]:
    """
    Get a summary of Celery metrics.

    Args:
        celery_app: Celery application instance

    Returns:
        Dictionary with metrics summary
    """
    try:
        from celery import current_app
        inspect = current_app.control.inspect()

        active = inspect.active() or {}
        reserved = inspect.reserved() or {}
        scheduled = inspect.scheduled() or {}
        stats = inspect.stats() or {}

        total_active = sum(len(tasks) for tasks in active.values())
        total_reserved = sum(len(tasks) for tasks in reserved.values())
        total_scheduled = sum(len(tasks) for tasks in scheduled.values())

        return {
            'workers': {
                'total': len(stats),
                'names': list(stats.keys())
            },
            'tasks': {
                'active': total_active,
                'reserved': total_reserved,
                'scheduled': total_scheduled,
                'total_pending': total_active + total_reserved + total_scheduled
            },
            'queues': {
                'celery': total_active + total_reserved + total_scheduled
            }
        }

    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        return {
            'error': str(e),
            'workers': {'total': 0, 'names': []},
            'tasks': {'active': 0, 'reserved': 0, 'scheduled': 0, 'total_pending': 0},
            'queues': {}
        }


def track_resource_usage():
    """
    Track current resource usage.

    Should be called periodically (e.g., every 30 seconds).
    """
    try:
        process = psutil.Process()

        # Memory usage
        memory_info = process.memory_info()
        memory_usage_gauge.labels(process_type='celery_worker').set(
            memory_info.rss
        )

        # CPU usage
        cpu_percent = process.cpu_percent(interval=1.0)
        cpu_usage_gauge.labels(process_type='celery_worker').set(
            cpu_percent
        )

    except Exception as e:
        logger.error(f"Failed to track resource usage: {e}")
