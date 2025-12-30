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
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Store task start times
_task_start_times: Dict[str, float] = {}


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
