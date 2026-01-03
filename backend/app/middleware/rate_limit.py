"""
Rate Limiting Middleware

Implements rate limiting to prevent abuse and manage server load.
Uses Redis for distributed rate limiting across multiple instances.
"""
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import time
from datetime import datetime, timedelta
import redis.asyncio as redis

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using Redis for distributed tracking.
    
    Implements sliding window rate limiting per IP address.
    """
    
    def __init__(self, app, redis_client: redis.Redis = None):
        super().__init__(app)
        self.redis_client = redis_client
        self.enabled = settings.ENABLE_RATE_LIMITING
        
        # Rate limits
        self.per_minute_limit = settings.RATE_LIMIT_PER_MINUTE
        self.per_hour_limit = settings.RATE_LIMIT_PER_HOUR
        
        # Exempt paths (health checks, docs)
        self.exempt_paths = {
            "/health",
            "/health/migrations",
            "/docs",
            "/redoc",
            "/openapi.json",
        }
        
        logger.info(
            f"Rate limiting {'enabled' if self.enabled else 'disabled'}: "
            f"{self.per_minute_limit}/min, {self.per_hour_limit}/hour"
        )
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Process the request and apply rate limiting"""
        
        # Skip if disabled
        if not self.enabled:
            return await call_next(request)
        
        # Skip exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)
        
        # Get client identifier (IP address)
        client_ip = self._get_client_ip(request)
        
        # Check rate limits
        try:
            is_allowed, retry_after = await self._check_rate_limit(client_ip)
            
            if not is_allowed:
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": "Rate limit exceeded. Please try again later.",
                        "retry_after": retry_after
                    },
                    headers={"Retry-After": str(retry_after)}
                )
        except Exception as e:
            # If rate limiting fails, allow request but log error
            logger.error(f"Rate limiting error: {e}", exc_info=True)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit-Minute"] = str(self.per_minute_limit)
        response.headers["X-RateLimit-Limit-Hour"] = str(self.per_hour_limit)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        # Try to get real IP from headers (behind proxy/load balancer)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    async def _check_rate_limit(self, client_ip: str) -> tuple[bool, int]:
        """
        Check if client has exceeded rate limits.
        
        Returns:
            tuple: (is_allowed, retry_after_seconds)
        """
        if not self.redis_client:
            # If no Redis, use simple in-memory fallback (not distributed)
            return await self._check_rate_limit_memory(client_ip)
        
        try:
            return await self._check_rate_limit_redis(client_ip)
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fallback to allowing request on Redis failure
            return True, 0
    
    async def _check_rate_limit_redis(self, client_ip: str) -> tuple[bool, int]:
        """Redis-based distributed rate limiting"""
        current_time = int(time.time())
        minute_key = f"ratelimit:{client_ip}:minute:{current_time // 60}"
        hour_key = f"ratelimit:{client_ip}:hour:{current_time // 3600}"
        
        # Use pipeline for atomic operations
        pipe = self.redis_client.pipeline()
        
        # Increment counters
        pipe.incr(minute_key)
        pipe.expire(minute_key, 60)  # Expire after 1 minute
        pipe.incr(hour_key)
        pipe.expire(hour_key, 3600)  # Expire after 1 hour
        
        results = await pipe.execute()
        minute_count = results[0]
        hour_count = results[2]
        
        # Check limits
        if minute_count > self.per_minute_limit:
            retry_after = 60 - (current_time % 60)
            return False, retry_after
        
        if hour_count > self.per_hour_limit:
            retry_after = 3600 - (current_time % 3600)
            return False, retry_after
        
        return True, 0
    
    async def _check_rate_limit_memory(self, client_ip: str) -> tuple[bool, int]:
        """
        Simple in-memory rate limiting (not suitable for production with multiple instances).
        This is a fallback when Redis is not available.
        """
        # Note: This won't work across multiple server instances
        # For production, always use Redis
        if not hasattr(self, '_memory_store'):
            self._memory_store = {}
        
        current_time = time.time()
        
        if client_ip not in self._memory_store:
            self._memory_store[client_ip] = {
                'minute': [],
                'hour': []
            }
        
        client_data = self._memory_store[client_ip]
        
        # Clean old entries
        client_data['minute'] = [t for t in client_data['minute'] if current_time - t < 60]
        client_data['hour'] = [t for t in client_data['hour'] if current_time - t < 3600]
        
        # Check limits
        if len(client_data['minute']) >= self.per_minute_limit:
            oldest = min(client_data['minute'])
            retry_after = int(60 - (current_time - oldest))
            return False, retry_after
        
        if len(client_data['hour']) >= self.per_hour_limit:
            oldest = min(client_data['hour'])
            retry_after = int(3600 - (current_time - oldest))
            return False, retry_after
        
        # Add current request
        client_data['minute'].append(current_time)
        client_data['hour'].append(current_time)
        
        return True, 0


def create_rate_limiter(redis_url: str = None) -> RateLimitMiddleware:
    """
    Create rate limiter with optional Redis connection.
    
    Args:
        redis_url: Redis connection URL. If None, uses settings.REDIS_URL
    
    Returns:
        RateLimitMiddleware instance
    """
    redis_client = None
    
    if settings.ENABLE_RATE_LIMITING:
        try:
            url = redis_url or settings.REDIS_URL
            redis_client = redis.from_url(
                url,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info("Rate limiter connected to Redis")
        except Exception as e:
            logger.warning(f"Failed to connect Redis for rate limiting: {e}")
            logger.warning("Rate limiting will use in-memory fallback (not suitable for production)")
    
    return lambda app: RateLimitMiddleware(app, redis_client)
