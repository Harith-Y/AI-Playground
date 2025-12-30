"""
Performance Monitoring Middleware for FastAPI

Tracks API request metrics including duration, size, and status codes.
"""

import time
import sys
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .metrics import (
    api_request_duration_histogram,
    api_request_counter,
    api_active_requests_gauge,
    api_request_size_histogram,
    api_response_size_histogram,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware to monitor API performance metrics.
    
    Tracks:
    - Request duration
    - Request/response sizes
    - Status codes
    - Active requests
    - Error rates
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and track metrics.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain
            
        Returns:
            HTTP response
        """
        # Extract request info
        method = request.method
        path = request.url.path
        
        # Normalize path for metrics (remove IDs)
        endpoint = self._normalize_path(path)
        
        # Track active requests
        api_active_requests_gauge.labels(
            method=method,
            endpoint=endpoint
        ).inc()
        
        # Track request size
        request_size = int(request.headers.get('content-length', 0))
        if request_size > 0:
            api_request_size_histogram.labels(
                method=method,
                endpoint=endpoint
            ).observe(request_size)
        
        # Start timer
        start_time = time.time()
        
        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            
        except Exception as e:
            # Log error
            logger.error(
                f"Request failed: {method} {path}",
                extra={
                    'event': 'request_error',
                    'method': method,
                    'path': path,
                    'error': str(e)
                },
                exc_info=True
            )
            
            # Track error
            status_code = 500
            raise
            
        finally:
            # Calculate duration
            duration = time.time() - start_time
            
            # Track metrics
            api_request_duration_histogram.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).observe(duration)
            
            api_request_counter.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            # Decrement active requests
            api_active_requests_gauge.labels(
                method=method,
                endpoint=endpoint
            ).dec()
            
            # Log slow requests (> 5 seconds)
            if duration > 5.0:
                logger.warning(
                    f"Slow request detected: {method} {path} took {duration:.2f}s",
                    extra={
                        'event': 'slow_request',
                        'method': method,
                        'path': path,
                        'duration': duration,
                        'status_code': status_code
                    }
                )
        
        # Track response size
        response_size = int(response.headers.get('content-length', 0))
        if response_size > 0:
            api_response_size_histogram.labels(
                method=method,
                endpoint=endpoint
            ).observe(response_size)
        
        return response
    
    def _normalize_path(self, path: str) -> str:
        """
        Normalize path by replacing UUIDs and IDs with placeholders.
        
        This prevents metric explosion from unique IDs.
        
        Args:
            path: Original request path
            
        Returns:
            Normalized path
        """
        import re
        
        # Replace UUIDs
        path = re.sub(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '{id}',
            path,
            flags=re.IGNORECASE
        )
        
        # Replace numeric IDs
        path = re.sub(r'/\d+/', '/{id}/', path)
        path = re.sub(r'/\d+$', '/{id}', path)
        
        return path
