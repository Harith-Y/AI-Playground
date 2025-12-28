"""
Logging configuration for the application

Provides structured logging with different levels and formatters
for preprocessing jobs, API requests, and general application logs.
"""

import logging
import logging.handlers
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
from pythonjsonlogger import jsonlogger


class StructuredFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter for structured logging

    Adds additional context fields for better log analysis and monitoring
    """

    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]):
        super().add_fields(log_record, record, message_dict)

        # Add timestamp in ISO format
        log_record['timestamp'] = datetime.utcnow().isoformat()

        # Add log level
        log_record['level'] = record.levelname

        # Add logger name
        log_record['logger'] = record.name

        # Add module and function info
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno

        # Add process and thread info for debugging
        log_record['process_id'] = record.process
        log_record['thread_id'] = record.thread


class PreprocessingJobFormatter(logging.Formatter):
    """
    Custom formatter for preprocessing job logs

    Formats logs with preprocessing-specific context
    """

    def format(self, record: logging.LogRecord) -> str:
        # Add preprocessing context if available
        if hasattr(record, 'dataset_id'):
            record.msg = f"[Dataset: {record.dataset_id}] {record.msg}"
        if hasattr(record, 'step_type'):
            record.msg = f"[Step: {record.step_type}] {record.msg}"
        if hasattr(record, 'task_id'):
            record.msg = f"[Task: {record.task_id}] {record.msg}"

        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_json_logging: bool = True,
    enable_file_logging: bool = True
):
    """
    Setup application logging configuration

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_json_logging: Whether to enable JSON formatted logging
        enable_file_logging: Whether to enable file-based logging
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    if enable_json_logging:
        console_formatter = StructuredFormatter(
            '%(timestamp)s %(level)s %(logger)s %(message)s'
        )
    else:
        console_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    if enable_file_logging:
        # General application log file
        app_file_handler = logging.handlers.RotatingFileHandler(
            log_path / 'app.log',
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        app_file_handler.setLevel(logging.INFO)
        app_file_handler.setFormatter(StructuredFormatter(
            '%(timestamp)s %(level)s %(logger)s %(message)s'
        ) if enable_json_logging else console_formatter)
        root_logger.addHandler(app_file_handler)

        # Error log file
        error_file_handler = logging.handlers.RotatingFileHandler(
            log_path / 'error.log',
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(StructuredFormatter(
            '%(timestamp)s %(level)s %(logger)s %(message)s'
        ) if enable_json_logging else console_formatter)
        root_logger.addHandler(error_file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('celery').setLevel(logging.INFO)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)


def setup_preprocessing_logger(
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_json_logging: bool = True
) -> logging.Logger:
    """
    Setup dedicated logger for preprocessing jobs

    Args:
        log_level: Logging level
        log_dir: Directory for log files
        enable_json_logging: Whether to use JSON format

    Returns:
        Configured preprocessing logger
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Create preprocessing logger
    logger = logging.getLogger('preprocessing')
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.propagate = False  # Don't propagate to root logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    if enable_json_logging:
        console_formatter = StructuredFormatter(
            '%(timestamp)s %(level)s %(message)s'
        )
    else:
        console_formatter = PreprocessingJobFormatter(
            '[%(asctime)s] %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler for preprocessing logs
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / 'preprocessing.log',
        maxBytes=20 * 1024 * 1024,  # 20 MB
        backupCount=10
    )
    file_handler.setLevel(logging.DEBUG)

    if enable_json_logging:
        file_formatter = StructuredFormatter(
            '%(timestamp)s %(level)s %(logger)s %(message)s'
        )
    else:
        file_formatter = PreprocessingJobFormatter(
            '[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Separate file for preprocessing job metrics
    metrics_handler = logging.handlers.RotatingFileHandler(
        log_path / 'preprocessing_metrics.log',
        maxBytes=20 * 1024 * 1024,  # 20 MB
        backupCount=10
    )
    metrics_handler.setLevel(logging.INFO)
    metrics_handler.addFilter(lambda record: hasattr(record, 'metrics'))
    metrics_handler.setFormatter(StructuredFormatter(
        '%(timestamp)s %(message)s'
    ))
    logger.addHandler(metrics_handler)

    return logger


def get_preprocessing_logger(
    dataset_id: str = None,
    task_id: str = None,
    user_id: str = None
) -> logging.LoggerAdapter:
    """
    Get a preprocessing logger with context

    Args:
        dataset_id: Dataset UUID
        task_id: Task UUID
        user_id: User UUID

    Returns:
        Logger adapter with context
    """
    logger = logging.getLogger('preprocessing')

    # Create context dictionary
    extra = {}
    if dataset_id:
        extra['dataset_id'] = dataset_id
    if task_id:
        extra['task_id'] = task_id
    if user_id:
        extra['user_id'] = user_id

    return logging.LoggerAdapter(logger, extra)


def log_preprocessing_metrics(
    logger: logging.Logger,
    dataset_id: str,
    task_id: str,
    step_type: str,
    execution_time: float,
    rows_before: int,
    rows_after: int,
    cols_before: int,
    cols_after: int,
    **kwargs
):
    """
    Log preprocessing job metrics in structured format

    Args:
        logger: Logger instance
        dataset_id: Dataset UUID
        task_id: Task UUID
        step_type: Type of preprocessing step
        execution_time: Time taken in seconds
        rows_before: Row count before processing
        rows_after: Row count after processing
        cols_before: Column count before processing
        cols_after: Column count after processing
        **kwargs: Additional metrics
    """
    metrics = {
        'event': 'preprocessing_step_completed',
        'dataset_id': dataset_id,
        'task_id': task_id,
        'step_type': step_type,
        'execution_time_seconds': round(execution_time, 3),
        'rows_before': rows_before,
        'rows_after': rows_after,
        'rows_changed': rows_after - rows_before,
        'cols_before': cols_before,
        'cols_after': cols_after,
        'cols_changed': cols_after - cols_before,
        **kwargs
    }

    # Create a log record with metrics attribute
    logger.info(
        f"Step completed: {step_type}",
        extra={'metrics': True, **metrics}
    )


# Initialize logging on module import
setup_logging()
preprocessing_logger = setup_preprocessing_logger()
