"""
Error Recovery Utilities

Provides robust error recovery patterns including:
- Retry logic with exponential backoff
- Circuit breaker pattern
- Fallback strategies
- Transaction management
- Timeout handling
"""

import time
import functools
from typing import Callable, Any, Optional, Type, Tuple, List, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio

from app.utils.logger import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"  # Linear backoff
    FIXED = "fixed"  # Fixed delay
    FIBONACCI = "fibonacci"  # Fibonacci backoff


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascade failures by failing fast when a service is down.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Service is down, fail fast
    - HALF_OPEN: Testing if service recovered
    
    Example:
        >>> circuit = CircuitBreaker(
        ...     failure_threshold=5,
        ...     recovery_timeout=60,
        ...     expected_exception=requests.ConnectionError
        ... )
        >>> 
        >>> @circuit
        ... def call_external_api():
        ...     return requests.get("https://api.example.com")
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception,
        name: Optional[str] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            expected_exception: Exception type to catch
            name: Optional name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name or "CircuitBreaker"
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"{self.name}: Circuit half-open, attempting reset")
            else:
                raise Exception(
                    f"{self.name}: Circuit breaker is OPEN. "
                    f"Service unavailable (failures: {self.failure_count})"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"{self.name}: Circuit reset to CLOSED")
        
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(
                f"{self.name}: Circuit breaker OPENED "
                f"(failures: {self.failure_count}/{self.failure_threshold})"
            )
        else:
            logger.warning(
                f"{self.name}: Failure {self.failure_count}/{self.failure_threshold}"
            )
    
    def reset(self):
        """Manually reset circuit breaker."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        logger.info(f"{self.name}: Circuit manually reset")


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    jitter: bool = True
):
    """
    Retry decorator with configurable backoff strategies.
    
    Args:
        max_attempts: Maximum number of attempts (including initial)
        delay: Initial delay in seconds
        backoff: Backoff multiplier for exponential strategy
        max_delay: Maximum delay between retries
        exceptions: Tuple of exception types to catch
        strategy: Retry strategy (exponential, linear, fixed, fibonacci)
        on_retry: Optional callback(exception, attempt) called before retry
        jitter: Add random jitter to delay (±25%)
    
    Example:
        >>> @retry(
        ...     max_attempts=5,
        ...     delay=1.0,
        ...     backoff=2.0,
        ...     exceptions=(ConnectionError, TimeoutError)
        ... )
        ... def call_api():
        ...     return requests.get("https://api.example.com")
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt >= max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    # Calculate delay
                    wait_time = _calculate_delay(
                        attempt=attempt,
                        base_delay=delay,
                        backoff=backoff,
                        max_delay=max_delay,
                        strategy=strategy,
                        jitter=jitter
                    )
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    
                    # Call retry callback
                    if on_retry:
                        try:
                            on_retry(e, attempt)
                        except Exception as callback_error:
                            logger.error(f"Retry callback failed: {callback_error}")
                    
                    time.sleep(wait_time)
            
            # Should never reach here
            raise last_exception
        
        return wrapper
    return decorator


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    jitter: bool = True
):
    """
    Async retry decorator with configurable backoff strategies.
    
    Same as retry() but for async functions.
    
    Example:
        >>> @async_retry(
        ...     max_attempts=5,
        ...     delay=1.0,
        ...     exceptions=(aiohttp.ClientError,)
        ... )
        ... async def call_api():
        ...     async with aiohttp.ClientSession() as session:
        ...         async with session.get("https://api.example.com") as resp:
        ...             return await resp.json()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt >= max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    # Calculate delay
                    wait_time = _calculate_delay(
                        attempt=attempt,
                        base_delay=delay,
                        backoff=backoff,
                        max_delay=max_delay,
                        strategy=strategy,
                        jitter=jitter
                    )
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    
                    # Call retry callback
                    if on_retry:
                        try:
                            on_retry(e, attempt)
                        except Exception as callback_error:
                            logger.error(f"Retry callback failed: {callback_error}")
                    
                    await asyncio.sleep(wait_time)
            
            # Should never reach here
            raise last_exception
        
        return wrapper
    return decorator


def _calculate_delay(
    attempt: int,
    base_delay: float,
    backoff: float,
    max_delay: float,
    strategy: RetryStrategy,
    jitter: bool
) -> float:
    """
    Calculate retry delay based on strategy.
    
    Args:
        attempt: Current attempt number (1-indexed)
        base_delay: Base delay in seconds
        backoff: Backoff multiplier
        max_delay: Maximum delay
        strategy: Retry strategy
        jitter: Add random jitter
    
    Returns:
        Delay in seconds
    """
    import random
    
    if strategy == RetryStrategy.FIXED:
        delay = base_delay
    elif strategy == RetryStrategy.LINEAR:
        delay = base_delay * attempt
    elif strategy == RetryStrategy.EXPONENTIAL:
        delay = base_delay * (backoff ** (attempt - 1))
    elif strategy == RetryStrategy.FIBONACCI:
        # Calculate Fibonacci number for attempt
        fib = _fibonacci(attempt)
        delay = base_delay * fib
    else:
        delay = base_delay
    
    # Cap at max_delay
    delay = min(delay, max_delay)
    
    # Add jitter (±25%)
    if jitter:
        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)
        delay = max(0, delay)  # Ensure non-negative
    
    return delay


def _fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def fallback(
    fallback_func: Callable,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    log_error: bool = True
):
    """
    Fallback decorator - use alternative function if primary fails.
    
    Args:
        fallback_func: Function to call if primary fails
        exceptions: Exceptions to catch
        log_error: Whether to log the error
    
    Example:
        >>> def get_cached_data():
        ...     return {"cached": True}
        >>> 
        >>> @fallback(get_cached_data)
        ... def get_live_data():
        ...     return requests.get("https://api.example.com").json()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_error:
                    logger.warning(
                        f"{func.__name__} failed: {e}. "
                        f"Using fallback: {fallback_func.__name__}"
                    )
                return fallback_func(*args, **kwargs)
        return wrapper
    return decorator


def timeout(seconds: float):
    """
    Timeout decorator for functions.
    
    Args:
        seconds: Timeout in seconds
    
    Example:
        >>> @timeout(30.0)
        ... def long_running_task():
        ...     # Task must complete within 30 seconds
        ...     pass
    
    Note: Uses signal on Unix, threading on Windows
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import platform
            
            if platform.system() == "Windows":
                return _timeout_threading(func, seconds, args, kwargs)
            else:
                return _timeout_signal(func, seconds, args, kwargs)
        
        return wrapper
    return decorator


def _timeout_signal(func: Callable, seconds: float, args: tuple, kwargs: dict) -> Any:
    """Timeout implementation using signals (Unix only)."""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")
    
    # Set alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))
    
    try:
        result = func(*args, **kwargs)
    finally:
        signal.alarm(0)  # Cancel alarm
        signal.signal(signal.SIGALRM, old_handler)
    
    return result


def _timeout_threading(func: Callable, seconds: float, args: tuple, kwargs: dict) -> Any:
    """Timeout implementation using threading (cross-platform)."""
    import threading
    
    result = []
    exception = []
    
    def target():
        try:
            result.append(func(*args, **kwargs))
        except Exception as e:
            exception.append(e)
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(seconds)
    
    if thread.is_alive():
        # Thread still running, timeout occurred
        logger.error(f"Function {func.__name__} timed out after {seconds}s")
        raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")
    
    if exception:
        raise exception[0]
    
    return result[0] if result else None


class TransactionManager:
    """
    Context manager for database transactions with automatic rollback.
    
    Example:
        >>> with TransactionManager(db) as tx:
        ...     # Perform database operations
        ...     user = User(name="John")
        ...     db.add(user)
        ...     
        ...     # If exception occurs, automatic rollback
        ...     # Otherwise, automatic commit on exit
    """
    
    def __init__(self, session, auto_commit: bool = True):
        """
        Initialize transaction manager.
        
        Args:
            session: SQLAlchemy session
            auto_commit: Whether to auto-commit on success
        """
        self.session = session
        self.auto_commit = auto_commit
        self._committed = False
    
    def __enter__(self):
        """Start transaction."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End transaction - commit or rollback."""
        try:
            if exc_type is not None:
                # Exception occurred, rollback
                self.session.rollback()
                logger.warning(
                    f"Transaction rolled back due to {exc_type.__name__}: {exc_val}"
                )
                return False  # Re-raise exception
            elif self.auto_commit and not self._committed:
                # Success, commit
                self.session.commit()
                self._committed = True
                logger.debug("Transaction committed successfully")
            return True
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error during transaction finalization: {e}")
            raise
    
    def commit(self):
        """Manually commit transaction."""
        self.session.commit()
        self._committed = True
        logger.debug("Transaction manually committed")
    
    def rollback(self):
        """Manually rollback transaction."""
        self.session.rollback()
        logger.debug("Transaction manually rolled back")


def safe_execute(
    func: Callable,
    *args,
    default: Any = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    log_error: bool = True,
    **kwargs
) -> Any:
    """
    Safely execute function, returning default value on exception.
    
    Args:
        func: Function to execute
        *args: Function arguments
        default: Default value to return on exception
        exceptions: Exceptions to catch
        log_error: Whether to log errors
        **kwargs: Function keyword arguments
    
    Returns:
        Function result or default value
    
    Example:
        >>> result = safe_execute(
        ...     risky_operation,
        ...     arg1, arg2,
        ...     default=None,
        ...     exceptions=(ValueError, KeyError)
        ... )
    """
    try:
        return func(*args, **kwargs)
    except exceptions as e:
        if log_error:
            logger.error(f"safe_execute: {func.__name__} failed: {e}")
        return default


def batch_with_retry(
    items: List[Any],
    process_func: Callable[[Any], Any],
    batch_size: int = 100,
    max_retries: int = 3,
    on_item_error: Optional[Callable[[Any, Exception], None]] = None
) -> Tuple[List[Any], List[Tuple[Any, Exception]]]:
    """
    Process items in batches with retry on failures.
    
    Args:
        items: Items to process
        process_func: Function to process each item
        batch_size: Number of items per batch
        max_retries: Max retries per item
        on_item_error: Optional callback for failed items
    
    Returns:
        Tuple of (successful_results, failed_items_with_errors)
    
    Example:
        >>> def process_item(item):
        ...     # Process item
        ...     return result
        >>> 
        >>> success, failures = batch_with_retry(
        ...     items=data,
        ...     process_func=process_item,
        ...     batch_size=50
        ... )
    """
    results = []
    failures = []
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} items)")
        
        for item in batch:
            # Retry each item
            success = False
            last_error = None
            
            for attempt in range(1, max_retries + 1):
                try:
                    result = process_func(item)
                    results.append(result)
                    success = True
                    break
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        wait_time = 2 ** (attempt - 1)  # Exponential backoff
                        logger.warning(
                            f"Item processing failed (attempt {attempt}/{max_retries}): {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
            
            if not success:
                failures.append((item, last_error))
                if on_item_error:
                    try:
                        on_item_error(item, last_error)
                    except Exception as callback_error:
                        logger.error(f"Error callback failed: {callback_error}")
    
    logger.info(
        f"Batch processing complete: {len(results)} succeeded, {len(failures)} failed"
    )
    return results, failures
