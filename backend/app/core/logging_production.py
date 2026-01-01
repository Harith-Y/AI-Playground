"""
Production logging configuration with advanced features

Features:
- Multiple log levels and handlers
- Log rotation with compression
- Separate logs by severity and component
- Request/response logging
- Performance monitoring
- Security event logging
- Integration with external systems (Sentry, ELK, Loki)
"""

import logging
import logging.handlers
import sys
import json
import gzip
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from pythonjsonlogger import jsonlogger

from app.core.config import settings


class ProductionFormatter(jsonlogger.JsonFormatter):
    """
    Production-grade JSON formatter with enhanced metadata
    """

    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]):
        super().add_fields(log_record, record, message_dict)

        # Core fields
        log_record['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['message'] = record.getMessage()

        # Source location
        log_record['source'] = {
            'file': record.pathname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Process/thread info
        log_record['process'] = {
            'id': record.process,
            'name': record.processName
        }
        log_record['thread'] = {
            'id': record.thread,
            'name': record.threadName
        }

        # Environment info
        log_record['environment'] = settings.ENVIRONMENT

        # Request context (if available)
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id
        if hasattr(record, 'ip_address'):
            log_record['ip_address'] = record.ip_address

        # Custom fields
        if hasattr(record, 'extra_fields'):
            log_record.update(record.extra_fields)


class GZipRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Rotating file handler that compresses old log files with gzip
    """

    def rotation_filename(self, default_name):
        """
        Compress rotated log file
        """
        return default_name + '.gz'

    def rotate(self, source, dest):
        """
        Compress the log file during rotation
        """
        with open(source, 'rb') as f_in:
            with gzip.open(dest, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        Path(source).unlink()


class TimedGZipRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """
    Time-based rotating file handler with compression
    """

    def rotation_filename(self, default_name):
        """
        Compress rotated log file
        """
        return default_name + '.gz'

    def rotate(self, source, dest):
        """
        Compress the log file during rotation
        """
        if not Path(dest).exists():
            with open(source, 'rb') as f_in:
                with gzip.open(dest, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            Path(source).unlink()


def setup_production_logging(
    log_level: str = None,
    log_dir: str = None,
    enable_json: bool = True,
    enable_compression: bool = True,
    max_bytes: int = 50 * 1024 * 1024,  # 50 MB
    backup_count: int = 30,
    enable_sentry: bool = False,
    sentry_dsn: str = None
):
    """
    Setup production logging with comprehensive handlers and formatters

    Args:
        log_level: Logging level (default from settings)
        log_dir: Log directory (default from settings)
        enable_json: Use JSON formatting
        enable_compression: Compress rotated logs
        max_bytes: Max log file size before rotation
        backup_count: Number of backup files to keep
        enable_sentry: Enable Sentry error tracking
        sentry_dsn: Sentry DSN
    """
    # Use settings as defaults
    log_level = log_level or settings.LOG_LEVEL
    log_dir = log_dir or settings.LOG_DIR

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.handlers = []

    # Choose handler class based on compression setting
    RotatingHandler = GZipRotatingFileHandler if enable_compression else logging.handlers.RotatingFileHandler

    # Formatter
    if enable_json:
        formatter = ProductionFormatter(
            '%(timestamp)s %(level)s %(logger)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)-8s [%(name)s:%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # --- Console Handler (stdout) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # --- Application Log (all levels) ---
    app_handler = RotatingHandler(
        log_path / 'application.log',
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(formatter)
    root_logger.addHandler(app_handler)

    # --- Error Log (errors and above) ---
    error_handler = RotatingHandler(
        log_path / 'error.log',
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # --- Access Log (API requests) ---
    access_logger = logging.getLogger('api.access')
    access_logger.setLevel(logging.INFO)
    access_logger.propagate = False

    access_handler = RotatingHandler(
        log_path / 'access.log',
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    access_handler.setLevel(logging.INFO)
    access_handler.setFormatter(formatter)
    access_logger.addHandler(access_handler)

    # --- Security Log (authentication, authorization) ---
    security_logger = logging.getLogger('security')
    security_logger.setLevel(logging.INFO)
    security_logger.propagate = False

    security_handler = RotatingHandler(
        log_path / 'security.log',
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    security_handler.setLevel(logging.INFO)
    security_handler.setFormatter(formatter)
    security_logger.addHandler(security_handler)

    # --- Performance Log (slow queries, bottlenecks) ---
    perf_logger = logging.getLogger('performance')
    perf_logger.setLevel(logging.INFO)
    perf_logger.propagate = False

    perf_handler = RotatingHandler(
        log_path / 'performance.log',
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(formatter)
    perf_logger.addHandler(perf_handler)

    # --- Celery Log (async tasks) ---
    celery_logger = logging.getLogger('celery')
    celery_logger.setLevel(logging.INFO)
    celery_logger.propagate = False

    celery_handler = RotatingHandler(
        log_path / 'celery.log',
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    celery_handler.setLevel(logging.INFO)
    celery_handler.setFormatter(formatter)
    celery_logger.addHandler(celery_handler)

    # --- Database Log (SQL queries) ---
    if settings.DEBUG:
        db_logger = logging.getLogger('sqlalchemy.engine')
        db_logger.setLevel(logging.INFO)
    else:
        db_logger = logging.getLogger('sqlalchemy.engine')
        db_logger.setLevel(logging.WARNING)

    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.error').setLevel(logging.INFO)
    logging.getLogger('fastapi').setLevel(logging.INFO)

    # --- Sentry Integration (optional) ---
    if enable_sentry and sentry_dsn:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.logging import LoggingIntegration

            sentry_logging = LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR
            )

            sentry_sdk.init(
                dsn=sentry_dsn,
                environment=settings.ENVIRONMENT,
                integrations=[sentry_logging],
                traces_sample_rate=0.1,
                profiles_sample_rate=0.1,
            )

            logging.info("Sentry error tracking enabled")
        except ImportError:
            logging.warning("Sentry SDK not installed, skipping Sentry integration")

    logging.info(f"Production logging initialized: level={log_level}, dir={log_dir}")


def get_request_logger(request_id: str = None, user_id: str = None, ip_address: str = None) -> logging.LoggerAdapter:
    """
    Get logger with request context

    Args:
        request_id: Unique request ID
        user_id: Authenticated user ID
        ip_address: Client IP address

    Returns:
        Logger adapter with context
    """
    extra = {}
    if request_id:
        extra['request_id'] = request_id
    if user_id:
        extra['user_id'] = user_id
    if ip_address:
        extra['ip_address'] = ip_address

    return logging.LoggerAdapter(logging.getLogger('api.access'), extra)


def log_api_request(
    method: str,
    path: str,
    status_code: int,
    response_time_ms: float,
    request_id: str = None,
    user_id: str = None,
    ip_address: str = None,
    **kwargs
):
    """
    Log API request details

    Args:
        method: HTTP method
        path: Request path
        status_code: Response status code
        response_time_ms: Response time in milliseconds
        request_id: Request ID
        user_id: User ID
        ip_address: Client IP
        **kwargs: Additional fields
    """
    logger = logging.getLogger('api.access')

    log_data = {
        'event': 'api_request',
        'method': method,
        'path': path,
        'status_code': status_code,
        'response_time_ms': round(response_time_ms, 2),
        'request_id': request_id,
        'user_id': user_id,
        'ip_address': ip_address,
        **kwargs
    }

    logger.info(
        f"{method} {path} {status_code} {response_time_ms:.2f}ms",
        extra={'extra_fields': log_data}
    )


def log_security_event(
    event_type: str,
    user_id: str = None,
    ip_address: str = None,
    success: bool = True,
    details: Dict[str, Any] = None
):
    """
    Log security-related events

    Args:
        event_type: Type of security event (login, logout, auth_failure, etc.)
        user_id: User ID
        ip_address: Client IP
        success: Whether the event was successful
        details: Additional details
    """
    logger = logging.getLogger('security')

    log_data = {
        'event': 'security_event',
        'event_type': event_type,
        'user_id': user_id,
        'ip_address': ip_address,
        'success': success,
        'details': details or {}
    }

    level = logging.INFO if success else logging.WARNING
    logger.log(
        level,
        f"Security event: {event_type} - {'SUCCESS' if success else 'FAILURE'}",
        extra={'extra_fields': log_data}
    )


def log_performance_metric(
    operation: str,
    duration_ms: float,
    details: Dict[str, Any] = None
):
    """
    Log performance metrics

    Args:
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        details: Additional details
    """
    logger = logging.getLogger('performance')

    log_data = {
        'event': 'performance_metric',
        'operation': operation,
        'duration_ms': round(duration_ms, 2),
        'details': details or {}
    }

    # Log as warning if operation is slow
    level = logging.WARNING if duration_ms > 1000 else logging.INFO

    logger.log(
        level,
        f"Performance: {operation} took {duration_ms:.2f}ms",
        extra={'extra_fields': log_data}
    )
