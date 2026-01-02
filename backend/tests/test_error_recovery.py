"""
Tests for Error Recovery Utilities

Tests retry logic, circuit breaker, fallback strategies, and database recovery.
"""

import pytest
import time
from unittest.mock import Mock, patch
from sqlalchemy.exc import OperationalError, IntegrityError
from sqlalchemy.orm import Session

from app.utils.error_recovery import (
    retry,
    CircuitBreaker,
    CircuitState,
    fallback,
    safe_execute,
    batch_with_retry,
    TransactionManager,
    RetryStrategy
)
from app.utils.db_recovery import (
    db_retry,
    atomic_update,
    safe_bulk_insert,
    ensure_connection
)


class TestRetryDecorator:
    """Test retry decorator with various strategies."""
    
    def test_retry_success_on_first_attempt(self):
        """Test successful execution on first attempt."""
        call_count = [0]
        
        @retry(max_attempts=3, delay=0.1)
        def successful_func():
            call_count[0] += 1
            return "success"
        
        result = successful_func()
        assert result == "success"
        assert call_count[0] == 1
    
    def test_retry_success_after_failures(self):
        """Test successful execution after some failures."""
        call_count = [0]
        
        @retry(max_attempts=3, delay=0.1)
        def eventually_successful():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Not yet")
            return "success"
        
        result = eventually_successful()
        assert result == "success"
        assert call_count[0] == 3
    
    def test_retry_exhausted(self):
        """Test failure after all retries exhausted."""
        call_count = [0]
        
        @retry(max_attempts=3, delay=0.1)
        def always_fails():
            call_count[0] += 1
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            always_fails()
        
        assert call_count[0] == 3
    
    def test_retry_exponential_backoff(self):
        """Test exponential backoff timing."""
        timestamps = []
        
        @retry(max_attempts=3, delay=0.1, backoff=2.0, jitter=False)
        def fails_twice():
            timestamps.append(time.time())
            if len(timestamps) < 3:
                raise ValueError("Fail")
            return "success"
        
        result = fails_twice()
        assert result == "success"
        assert len(timestamps) == 3
        
        # Check delays (0.1s, 0.2s)
        delay1 = timestamps[1] - timestamps[0]
        delay2 = timestamps[2] - timestamps[1]
        
        assert 0.08 < delay1 < 0.15  # ~0.1s
        assert 0.18 < delay2 < 0.25  # ~0.2s
    
    def test_retry_specific_exceptions(self):
        """Test retry only catches specified exceptions."""
        call_count = [0]
        
        @retry(max_attempts=3, delay=0.1, exceptions=(ValueError,))
        def raises_different_error():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Retry this")
            raise TypeError("Don't retry this")
        
        with pytest.raises(TypeError, match="Don't retry this"):
            raises_different_error()
        
        assert call_count[0] == 2  # First ValueError, then TypeError
    
    def test_retry_callback(self):
        """Test on_retry callback."""
        retry_info = []
        
        def on_retry_callback(exc, attempt):
            retry_info.append((attempt, str(exc)))
        
        @retry(max_attempts=3, delay=0.1, on_retry=on_retry_callback)
        def fails_twice():
            if len(retry_info) < 2:
                raise ValueError(f"Attempt {len(retry_info) + 1}")
            return "success"
        
        result = fails_twice()
        assert result == "success"
        assert len(retry_info) == 2
        assert retry_info[0] == (1, "Attempt 1")
        assert retry_info[1] == (2, "Attempt 2")


class TestCircuitBreaker:
    """Test circuit breaker pattern."""
    
    def test_circuit_closed_normal_operation(self):
        """Test circuit stays closed during normal operation."""
        circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        
        @circuit
        def successful_operation():
            return "success"
        
        # Should work fine
        for _ in range(5):
            result = successful_operation()
            assert result == "success"
        
        assert circuit.state == CircuitState.CLOSED
        assert circuit.failure_count == 0
    
    def test_circuit_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        circuit = CircuitBreaker(
            failure_threshold=3,
            expected_exception=ValueError
        )
        
        call_count = [0]
        
        @circuit
        def failing_operation():
            call_count[0] += 1
            raise ValueError("Fail")
        
        # First 3 failures should open circuit
        for i in range(3):
            with pytest.raises(ValueError):
                failing_operation()
        
        assert circuit.state == CircuitState.OPEN
        assert call_count[0] == 3
        
        # Next call should fail fast
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            failing_operation()
        
        # Should not have called function
        assert call_count[0] == 3
    
    def test_circuit_half_open_recovery(self):
        """Test circuit transitions to half-open and recovers."""
        circuit = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.5,
            expected_exception=ValueError
        )
        
        call_count = [0]
        
        @circuit
        def sometimes_fails():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ValueError("Fail")
            return "success"
        
        # Open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                sometimes_fails()
        
        assert circuit.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.6)
        
        # Next call should try (half-open)
        result = sometimes_fails()
        assert result == "success"
        assert circuit.state == CircuitState.CLOSED
        assert circuit.failure_count == 0


class TestFallback:
    """Test fallback decorator."""
    
    def test_fallback_on_failure(self):
        """Test fallback function called on failure."""
        def fallback_func():
            return "fallback"
        
        @fallback(fallback_func)
        def primary_func():
            raise ValueError("Primary failed")
        
        result = primary_func()
        assert result == "fallback"
    
    def test_fallback_with_args(self):
        """Test fallback receives same arguments."""
        def fallback_func(x, y):
            return x + y + 10
        
        @fallback(fallback_func)
        def primary_func(x, y):
            raise ValueError("Fail")
        
        result = primary_func(1, 2)
        assert result == 13
    
    def test_no_fallback_on_success(self):
        """Test primary function used when successful."""
        fallback_called = [False]
        
        def fallback_func():
            fallback_called[0] = True
            return "fallback"
        
        @fallback(fallback_func)
        def primary_func():
            return "primary"
        
        result = primary_func()
        assert result == "primary"
        assert not fallback_called[0]


class TestSafeExecute:
    """Test safe_execute utility."""
    
    def test_safe_execute_success(self):
        """Test successful execution."""
        def successful_func():
            return "success"
        
        result = safe_execute(successful_func, default="default")
        assert result == "success"
    
    def test_safe_execute_failure(self):
        """Test returns default on failure."""
        def failing_func():
            raise ValueError("Fail")
        
        result = safe_execute(failing_func, default="default")
        assert result == "default"
    
    def test_safe_execute_with_args(self):
        """Test with function arguments."""
        def add(a, b):
            if b == 0:
                raise ValueError("Cannot add zero")
            return a + b
        
        result = safe_execute(add, 5, 0, default=-1)
        assert result == -1


class TestBatchWithRetry:
    """Test batch processing with retry."""
    
    def test_batch_all_success(self):
        """Test batch processing with all successes."""
        def process_item(item):
            return item * 2
        
        items = list(range(10))
        success, failures = batch_with_retry(
            items,
            process_item,
            batch_size=5,
            max_retries=2
        )
        
        assert len(success) == 10
        assert len(failures) == 0
        assert success == [i * 2 for i in range(10)]
    
    def test_batch_with_failures(self):
        """Test batch processing with some failures."""
        def process_item(item):
            if item == 5:
                raise ValueError("Item 5 always fails")
            return item * 2
        
        items = list(range(10))
        success, failures = batch_with_retry(
            items,
            process_item,
            batch_size=5,
            max_retries=2
        )
        
        assert len(success) == 9
        assert len(failures) == 1
        assert failures[0][0] == 5


class TestTransactionManager:
    """Test database transaction manager."""
    
    def test_transaction_commit_on_success(self):
        """Test transaction commits on success."""
        mock_session = Mock(spec=Session)
        
        with TransactionManager(mock_session) as tx:
            # Simulate some operations
            pass
        
        mock_session.commit.assert_called_once()
        mock_session.rollback.assert_not_called()
    
    def test_transaction_rollback_on_exception(self):
        """Test transaction rolls back on exception."""
        mock_session = Mock(spec=Session)
        
        with pytest.raises(ValueError):
            with TransactionManager(mock_session) as tx:
                raise ValueError("Intentional error")
        
        mock_session.rollback.assert_called_once()
        mock_session.commit.assert_not_called()
    
    def test_transaction_manual_commit(self):
        """Test manual commit."""
        mock_session = Mock(spec=Session)
        
        with TransactionManager(mock_session, auto_commit=False) as tx:
            tx.commit()
        
        # Should be called twice: manual + auto on exit
        assert mock_session.commit.call_count == 2


class TestDatabaseRetry:
    """Test database-specific retry."""
    
    def test_db_retry_on_operational_error(self):
        """Test retry on OperationalError."""
        call_count = [0]
        mock_session = Mock(spec=Session)
        
        @db_retry(max_attempts=3, delay=0.1)
        def flaky_db_operation(db: Session):
            call_count[0] += 1
            if call_count[0] < 3:
                raise OperationalError("stmt", "params", "orig")
            return "success"
        
        result = flaky_db_operation(mock_session)
        assert result == "success"
        assert call_count[0] == 3
        # Should rollback on each failure
        assert mock_session.rollback.call_count == 2
    
    def test_db_retry_exhausted(self):
        """Test db_retry raises after max attempts."""
        call_count = [0]
        mock_session = Mock(spec=Session)
        
        @db_retry(max_attempts=3, delay=0.1)
        def always_fails_db(db: Session):
            call_count[0] += 1
            raise OperationalError("stmt", "params", "orig")
        
        with pytest.raises(OperationalError):
            always_fails_db(mock_session)
        
        assert call_count[0] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
