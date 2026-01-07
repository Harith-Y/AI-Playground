"""
Database Error Recovery Utilities

Provides database-specific error recovery patterns including:
- Transaction retry with deadlock handling
- Connection pool management
- Query retry with backoff
- Savepoint support
"""

from typing import Callable, Any, Optional, TypeVar, Type
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import (
    OperationalError,
    IntegrityError,
    DatabaseError,
    DBAPIError,
    DisconnectionError
)
import functools
import time

from app.utils.logger import get_logger
from app.utils.error_recovery import retry, RetryStrategy

logger = get_logger(__name__)

T = TypeVar('T')


# Database-specific retry decorator
def db_retry(
    max_attempts: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    on_retry: Optional[Callable[[Exception, int, Session], None]] = None
):
    """
    Retry decorator specifically for database operations.
    
    Handles common database errors:
    - Connection errors
    - Deadlocks
    - Lock timeouts
    - Transient failures
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay in seconds
        backoff: Backoff multiplier
        on_retry: Optional callback(exception, attempt, session)
    
    Example:
        >>> @db_retry(max_attempts=5, delay=1.0)
        ... def update_user(db: Session, user_id: int, data: dict):
        ...     user = db.query(User).filter(User.id == user_id).first()
        ...     for key, value in data.items():
        ...         setattr(user, key, value)
        ...     db.commit()
    """
    # Database errors that are retryable
    retryable_exceptions = (
        OperationalError,  # Connection errors, lock timeouts
        DisconnectionError,  # Connection lost
    )
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to find Session in args/kwargs
            session = None
            for arg in args:
                if isinstance(arg, Session):
                    session = arg
                    break
            if session is None:
                session = kwargs.get('db') or kwargs.get('session')
            
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt >= max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}",
                            extra={'error_type': type(e).__name__}
                        )
                        raise
                    
                    # Rollback on error
                    if session:
                        try:
                            session.rollback()
                            logger.debug(f"Rolled back session after error: {e}")
                        except Exception as rollback_error:
                            logger.error(f"Rollback failed: {rollback_error}")
                    
                    # Calculate delay
                    wait_time = delay * (backoff ** (attempt - 1))
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {wait_time:.2f}s...",
                        extra={
                            'attempt': attempt,
                            'max_attempts': max_attempts,
                            'error_type': type(e).__name__
                        }
                    )
                    
                    # Call retry callback
                    if on_retry and session:
                        try:
                            on_retry(e, attempt, session)
                        except Exception as callback_error:
                            logger.error(f"Retry callback failed: {callback_error}")
                    
                    time.sleep(wait_time)
            
            raise last_exception
        
        return wrapper
    return decorator


def with_savepoint(func: Callable) -> Callable:
    """
    Execute function within a database savepoint.
    
    Allows partial rollback without affecting outer transaction.
    
    Example:
        >>> @with_savepoint
        ... def risky_operation(db: Session):
        ...     # If this fails, only this operation rolls back
        ...     db.add(User(name="test"))
        ...     db.flush()
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Find session
        session = None
        for arg in args:
            if isinstance(arg, Session):
                session = arg
                break
        if session is None:
            session = kwargs.get('db') or kwargs.get('session')
        
        if session is None:
            logger.warning(f"{func.__name__}: No session found, executing without savepoint")
            return func(*args, **kwargs)
        
        # Create savepoint
        savepoint = session.begin_nested()
        
        try:
            result = func(*args, **kwargs)
            savepoint.commit()
            return result
        except Exception as e:
            logger.warning(f"{func.__name__}: Rolling back to savepoint due to {e}")
            savepoint.rollback()
            raise
    
    return wrapper


def atomic_update(
    session: Session,
    model_class: Type[T],
    record_id: Any,
    updates: dict,
    max_retries: int = 3
) -> T:
    """
    Perform atomic update with optimistic locking retry.
    
    Args:
        session: Database session
        model_class: SQLAlchemy model class
        record_id: Primary key value
        updates: Dictionary of updates to apply
        max_retries: Maximum retry attempts
    
    Returns:
        Updated record
    
    Raises:
        Exception: If update fails after all retries
    
    Example:
        >>> user = atomic_update(
        ...     db,
        ...     User,
        ...     user_id,
        ...     {'status': 'active', 'last_login': datetime.utcnow()}
        ... )
    """
    for attempt in range(1, max_retries + 1):
        try:
            # Get record with FOR UPDATE lock
            record = session.query(model_class).filter(
                model_class.id == record_id
            ).with_for_update().first()
            
            if not record:
                raise ValueError(f"{model_class.__name__} with id {record_id} not found")
            
            # Apply updates
            for key, value in updates.items():
                setattr(record, key, value)
            
            # Commit
            session.commit()
            session.refresh(record)
            
            logger.debug(
                f"Atomic update successful for {model_class.__name__}({record_id})"
            )
            return record
            
        except (OperationalError, IntegrityError) as e:
            session.rollback()
            
            if attempt >= max_retries:
                logger.error(
                    f"Atomic update failed after {max_retries} attempts: {e}"
                )
                raise
            
            wait_time = 0.1 * (2 ** (attempt - 1))
            logger.warning(
                f"Atomic update attempt {attempt}/{max_retries} failed: {e}. "
                f"Retrying in {wait_time:.2f}s..."
            )
            time.sleep(wait_time)


def safe_bulk_insert(
    session: Session,
    records: list,
    batch_size: int = 1000,
    on_error: str = 'skip'
) -> dict:
    """
    Safely insert records in batches with error handling.
    
    Args:
        session: Database session
        records: List of model instances to insert
        batch_size: Number of records per batch
        on_error: 'skip', 'rollback', or 'continue'
            - skip: Skip failed records, continue with rest
            - rollback: Rollback entire batch on error
            - continue: Try to insert failed records individually
    
    Returns:
        Dictionary with 'success' count, 'failed' count, and 'errors' list
    
    Example:
        >>> users = [User(name=f"user{i}") for i in range(1000)]
        >>> result = safe_bulk_insert(db, users, batch_size=100)
        >>> print(f"Inserted {result['success']} users")
    """
    success_count = 0
    failed_count = 0
    errors = []
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        
        try:
            session.bulk_save_objects(batch)
            session.commit()
            success_count += len(batch)
            
            logger.info(
                f"Batch {i//batch_size + 1}: Inserted {len(batch)} records"
            )
            
        except Exception as e:
            session.rollback()
            logger.error(f"Batch insert failed: {e}")
            
            if on_error == 'rollback':
                # Rollback entire batch
                failed_count += len(batch)
                errors.append({
                    'batch': i//batch_size + 1,
                    'error': str(e),
                    'count': len(batch)
                })
                
            elif on_error == 'continue':
                # Try inserting individually
                for record in batch:
                    try:
                        session.add(record)
                        session.commit()
                        success_count += 1
                    except Exception as record_error:
                        session.rollback()
                        failed_count += 1
                        errors.append({
                            'record': str(record),
                            'error': str(record_error)
                        })
            
            else:  # skip
                failed_count += len(batch)
                errors.append({
                    'batch': i//batch_size + 1,
                    'error': str(e),
                    'count': len(batch)
                })
                logger.warning(f"Skipped batch {i//batch_size + 1}")
    
    result = {
        'success': success_count,
        'failed': failed_count,
        'errors': errors
    }
    
    logger.info(
        f"Bulk insert complete: {success_count} succeeded, {failed_count} failed"
    )
    
    return result


class ConnectionPoolMonitor:
    """
    Monitor database connection pool health.
    
    Example:
        >>> monitor = ConnectionPoolMonitor(engine)
        >>> stats = monitor.get_stats()
        >>> if stats['utilization'] > 0.9:
        ...     logger.warning("Connection pool near capacity")
    """
    
    def __init__(self, engine):
        """
        Initialize monitor.
        
        Args:
            engine: SQLAlchemy engine
        """
        self.engine = engine
        self.pool = engine.pool
    
    def get_stats(self) -> dict:
        """Get connection pool statistics."""
        return {
            'size': self.pool.size(),
            'checked_in': self.pool.checkedin(),
            'checked_out': self.pool.checkedout(),
            'overflow': self.pool.overflow(),
            'utilization': self._calculate_utilization()
        }
    
    def _calculate_utilization(self) -> float:
        """Calculate pool utilization percentage."""
        total_capacity = self.pool.size() + self.pool.overflow()
        if total_capacity == 0:
            return 0.0
        used = self.pool.checkedout()
        return used / total_capacity
    
    def check_health(self) -> dict:
        """
        Check pool health and return status.
        
        Returns:
            Dictionary with 'healthy' boolean and 'issues' list
        """
        stats = self.get_stats()
        issues = []
        
        # Check high utilization
        if stats['utilization'] > 0.9:
            issues.append(f"High utilization: {stats['utilization']*100:.1f}%")
        
        # Check overflow
        if stats['overflow'] > 0:
            issues.append(f"Connection overflow: {stats['overflow']}")
        
        # Check if maxed out
        if stats['checked_out'] >= stats['size']:
            issues.append("Pool at maximum capacity")
        
        return {
            'healthy': len(issues) == 0,
            'issues': issues,
            'stats': stats
        }


def ensure_connection(func: Callable) -> Callable:
    """
    Ensure database connection is alive before executing function.
    
    Attempts to reconnect if connection is lost.
    
    Example:
        >>> @ensure_connection
        ... def query_users(db: Session):
        ...     return db.query(User).all()
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Find session
        session = None
        for arg in args:
            if isinstance(arg, Session):
                session = arg
                break
        if session is None:
            session = kwargs.get('db') or kwargs.get('session')
        
        if session is None:
            return func(*args, **kwargs)
        
        # Test connection
        try:
            session.execute(text("SELECT 1"))
        except (OperationalError, DisconnectionError) as e:
            logger.warning(f"Connection lost: {e}. Attempting to reconnect...")
            try:
                session.close()
                session.bind.connect()
                logger.info("Reconnected successfully")
            except Exception as reconnect_error:
                logger.error(f"Reconnection failed: {reconnect_error}")
                raise
        
        return func(*args, **kwargs)
    
    return wrapper
