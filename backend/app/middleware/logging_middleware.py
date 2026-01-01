"""
Logging middleware for FastAPI applications

Provides:
- Request/response logging
- Request ID tracking
- Performance monitoring
- Error logging
"""

import time
import uuid
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.logging_production import log_api_request, log_performance_metric


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all HTTP requests and responses
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = logging.getLogger('api.access')

    async def dispatch(self, request: Request, call_next: Callable):
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Start timer
        start_time = time.time()

        # Get client IP
        client_ip = self.get_client_ip(request)

        # Get user ID if authenticated
        user_id = None
        if hasattr(request.state, 'user'):
            user_id = str(request.state.user.id)

        # Log request
        self.logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                'request_id': request_id,
                'method': request.method,
                'path': request.url.path,
                'ip_address': client_ip,
                'user_id': user_id
            }
        )

        # Process request
        try:
            response = await call_next(request)

            # Calculate response time
            process_time = (time.time() - start_time) * 1000  # Convert to ms

            # Add custom headers
            response.headers['X-Request-ID'] = request_id
            response.headers['X-Process-Time'] = f"{process_time:.2f}ms"

            # Log response
            log_api_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                response_time_ms=process_time,
                request_id=request_id,
                user_id=user_id,
                ip_address=client_ip
            )

            # Log slow requests as performance issues
            if process_time > 1000:  # > 1 second
                log_performance_metric(
                    operation=f"{request.method} {request.url.path}",
                    duration_ms=process_time,
                    details={
                        'request_id': request_id,
                        'user_id': user_id,
                        'status_code': response.status_code
                    }
                )

            return response

        except Exception as e:
            # Calculate response time even on error
            process_time = (time.time() - start_time) * 1000

            # Log error
            self.logger.error(
                f"Request failed: {request.method} {request.url.path} - {str(e)}",
                extra={
                    'request_id': request_id,
                    'method': request.method,
                    'path': request.url.path,
                    'ip_address': client_ip,
                    'user_id': user_id,
                    'response_time_ms': process_time,
                    'error': str(e)
                },
                exc_info=True
            )

            # Re-raise exception
            raise

    @staticmethod
    def get_client_ip(request: Request) -> str:
        """
        Extract client IP address from request

        Handles proxy headers (X-Forwarded-For, X-Real-IP)
        """
        # Check forwarded headers (for reverse proxies)
        forwarded = request.headers.get('X-Forwarded-For')
        if forwarded:
            # X-Forwarded-For can contain multiple IPs, get the first one
            return forwarded.split(',')[0].strip()

        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip

        # Fallback to direct client
        if request.client:
            return request.client.host

        return 'unknown'


class SecurityLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log security-related events
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = logging.getLogger('security')

    async def dispatch(self, request: Request, call_next: Callable):
        # Track authentication attempts
        if request.url.path.startswith('/api/v1/auth/'):
            client_ip = self.get_client_ip(request)

            self.logger.info(
                f"Auth request: {request.method} {request.url.path}",
                extra={
                    'path': request.url.path,
                    'ip_address': client_ip,
                    'event': 'auth_attempt'
                }
            )

        response = await call_next(request)

        # Log failed authentication
        if request.url.path.startswith('/api/v1/auth/') and response.status_code == 401:
            client_ip = self.get_client_ip(request)

            self.logger.warning(
                f"Authentication failed: {request.url.path}",
                extra={
                    'path': request.url.path,
                    'ip_address': client_ip,
                    'status_code': response.status_code,
                    'event': 'auth_failure'
                }
            )

        return response

    @staticmethod
    def get_client_ip(request: Request) -> str:
        """Extract client IP from request"""
        forwarded = request.headers.get('X-Forwarded-For')
        if forwarded:
            return forwarded.split(',')[0].strip()

        if request.client:
            return request.client.host

        return 'unknown'


class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to catch and log all unhandled exceptions
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = logging.getLogger('error')

    async def dispatch(self, request: Request, call_next: Callable):
        try:
            return await call_next(request)
        except Exception as e:
            # Log the error with full context
            self.logger.error(
                f"Unhandled exception: {str(e)}",
                extra={
                    'method': request.method,
                    'path': request.url.path,
                    'query_params': dict(request.query_params),
                    'client_ip': self.get_client_ip(request),
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                },
                exc_info=True
            )

            # Re-raise to let FastAPI handle the response
            raise

    @staticmethod
    def get_client_ip(request: Request) -> str:
        """Extract client IP from request"""
        forwarded = request.headers.get('X-Forwarded-For')
        if forwarded:
            return forwarded.split(',')[0].strip()

        if request.client:
            return request.client.host

        return 'unknown'
