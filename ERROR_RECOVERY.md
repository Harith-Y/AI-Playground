# Error Recovery Patterns

Comprehensive guide to error recovery utilities and resilience patterns in the AI Playground ML system.

## Overview

The error recovery system provides robust handling of transient failures, cascading failures, and graceful degradation through multiple patterns:

- **Retry Logic**: Automatic retry with exponential backoff
- **Circuit Breaker**: Prevent cascading failures
- **Fallback Strategies**: Alternative execution paths
- **Transaction Management**: Automatic rollback on errors
- **Database Recovery**: Specialized database error handling
- **Checkpoint/Resume**: Resume training from failures

## Architecture

```
app/utils/
├── error_recovery.py     # Generic error recovery patterns
├── db_recovery.py        # Database-specific recovery
└── logger.py             # Centralized logging

app/tasks/
└── training_tasks.py     # Enhanced with error recovery

app/services/
└── training_error_handler.py  # Training-specific errors
```

## Core Patterns

### 1. Retry Logic

Automatically retry failed operations with configurable backoff strategies.

#### Basic Usage

```python
from app.utils.error_recovery import retry

@retry(max_attempts=3, delay=1.0, backoff=2.0)
def unreliable_operation():
    # May fail transiently
    result = external_api_call()
    return result
```

#### Retry Strategies

**Exponential Backoff** (Recommended)

```python
@retry(
    max_attempts=5,
    delay=1.0,        # Initial delay
    backoff=2.0,      # Multiply delay by 2 each attempt
    jitter=True       # Add randomness to prevent thundering herd
)
def api_call():
    return requests.get("https://api.example.com/data")
```

**Linear Backoff**

```python
@retry(
    max_attempts=5,
    delay=2.0,
    strategy=RetryStrategy.LINEAR  # 2s, 4s, 6s, 8s, 10s
)
def database_query():
    return db.execute(query)
```

**Fixed Delay**

```python
@retry(
    max_attempts=3,
    delay=5.0,
    strategy=RetryStrategy.FIXED  # 5s, 5s, 5s
)
def file_operation():
    return read_file("/path/to/file")
```

**Fibonacci Backoff**

```python
@retry(
    max_attempts=7,
    delay=1.0,
    strategy=RetryStrategy.FIBONACCI  # 1s, 1s, 2s, 3s, 5s, 8s, 13s
)
def network_operation():
    return download_file(url)
```

#### Exception Filtering

```python
# Only retry on specific exceptions
@retry(
    max_attempts=3,
    exceptions=(TimeoutError, ConnectionError)
)
def network_call():
    return requests.get(url, timeout=5)

# Retry on any exception except specified
@retry(
    max_attempts=3,
    exceptions=(Exception,),  # Base exception
)
def risky_operation():
    # Will retry on any error
    pass
```

#### Retry Callbacks

```python
def log_retry(exception, attempt):
    logger.warning(f"Attempt {attempt} failed: {exception}")

@retry(max_attempts=5, on_retry=log_retry)
def monitored_operation():
    return perform_task()
```

#### Async Retry

```python
from app.utils.error_recovery import async_retry

@async_retry(max_attempts=3, delay=1.0)
async def async_operation():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

### 2. Circuit Breaker

Prevent cascading failures by "opening" the circuit after repeated failures.

#### States

- **CLOSED**: Normal operation, all requests pass through
- **OPEN**: Circuit tripped, requests fail fast without attempting operation
- **HALF_OPEN**: Testing if service recovered, allow one request through

#### Usage

```python
from app.utils.error_recovery import CircuitBreaker, CircuitState

# Create circuit breaker
external_service_circuit = CircuitBreaker(
    failure_threshold=5,      # Open after 5 failures
    recovery_timeout=60,      # Try recovery after 60 seconds
    expected_exception=RequestException
)

@external_service_circuit
def call_external_service():
    response = requests.get("https://external-api.com/data")
    return response.json()

# Check circuit state
if external_service_circuit.state == CircuitState.OPEN:
    logger.warning("Circuit is open, using cached data")
    return get_cached_data()
```

#### Advanced Configuration

```python
# Custom circuit breaker for database
db_circuit = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=30,
    expected_exception=(OperationalError, DisconnectionError),
    on_open=lambda: logger.critical("Database circuit opened!"),
    on_close=lambda: logger.info("Database circuit closed")
)

@db_circuit
def database_query(db: Session):
    return db.query(Model).all()
```

#### Manual Circuit Control

```python
# Manually open circuit
circuit.open()

# Manually reset circuit
circuit.reset()

# Force half-open state for testing
circuit.state = CircuitState.HALF_OPEN
```

### 3. Fallback Strategies

Provide alternative execution paths when primary operation fails.

#### Basic Fallback

```python
from app.utils.error_recovery import fallback

def use_cached_data():
    return cache.get("user_data", [])

@fallback(use_cached_data)
def get_live_data():
    # Try to get fresh data
    response = requests.get("/api/data")
    return response.json()
```

#### Fallback with Arguments

```python
def get_default_config(env: str):
    defaults = {
        "dev": {"timeout": 30},
        "prod": {"timeout": 10}
    }
    return defaults.get(env, {"timeout": 30})

@fallback(get_default_config)
def load_config(env: str):
    # Try to load from file
    with open(f"config.{env}.json") as f:
        return json.load(f)

config = load_config("prod")  # Falls back to defaults if file missing
```

#### Fallback Chain

```python
def tertiary_fallback():
    return "emergency_value"

def secondary_fallback():
    # This might also fail
    return load_from_backup()

@fallback(secondary_fallback)
def primary_operation():
    return fetch_from_primary()

# Can nest fallbacks
@fallback(tertiary_fallback)
@fallback(secondary_fallback)
def highly_reliable_operation():
    return primary_operation()
```

### 4. Transaction Management

Automatic database transaction handling with rollback on errors.

#### Basic Transaction

```python
from app.utils.error_recovery import TransactionManager

def update_user_data(db: Session, user_id: int, data: dict):
    with TransactionManager(db) as tx:
        user = db.query(User).filter_by(id=user_id).first()
        user.name = data["name"]
        user.email = data["email"]

        # If any exception occurs, transaction rolls back
        # Otherwise commits automatically
```

#### Manual Commit

```python
def complex_transaction(db: Session):
    with TransactionManager(db, auto_commit=False) as tx:
        # Do some operations
        update_records(db)

        # Check if we should commit
        if validation_passes():
            tx.commit()
        else:
            # Explicitly rollback
            tx.rollback()
```

#### Nested Transactions with Savepoints

```python
from app.utils.db_recovery import with_savepoint

@with_savepoint
def inner_transaction(db: Session):
    # This uses a savepoint
    db.execute(insert_statement)

    if error_condition:
        # Rolls back to savepoint, not entire transaction
        raise ValueError("Partial rollback")

def outer_transaction(db: Session):
    with TransactionManager(db) as tx:
        db.execute(other_statement)

        try:
            inner_transaction(db)
        except ValueError:
            # Inner rolled back, outer continues
            pass

        db.execute(final_statement)
        # Commits all successfully executed statements
```

### 5. Database Recovery

Specialized patterns for database error handling.

#### Database Retry

```python
from app.utils.db_recovery import db_retry

@db_retry(max_attempts=3, delay=1.0)
def query_with_retry(db: Session):
    return db.query(Model).filter_by(status="active").all()
```

#### Connection Validation

```python
from app.utils.db_recovery import ensure_connection

@ensure_connection
@db_retry(max_attempts=3)
def critical_db_operation(db: Session):
    # Connection tested before operation
    # Retries if connection lost
    return db.execute(complex_query)
```

#### Optimistic Locking

```python
from app.utils.db_recovery import atomic_update

result = atomic_update(
    db=db,
    model_class=User,
    record_id=user_id,
    updates={"credits": user.credits + 100},
    max_retries=5
)

if result:
    logger.info("Update successful")
else:
    logger.error("Update failed after retries")
```

#### Safe Bulk Insert

```python
from app.utils.db_recovery import safe_bulk_insert

records = [
    {"name": "Alice", "email": "alice@example.com"},
    {"name": "Bob", "email": "bob@example.com"},
    # ... 1000 more records
]

success, failed = safe_bulk_insert(
    db=db,
    model_class=User,
    records=records,
    batch_size=100
)

logger.info(f"Inserted {len(success)} records")
if failed:
    logger.error(f"Failed to insert {len(failed)} records")
```

### 6. Timeout Protection

Prevent operations from hanging indefinitely.

#### Basic Timeout

```python
from app.utils.error_recovery import timeout

@timeout(seconds=30)
def long_running_operation():
    # Will raise TimeoutError if takes > 30 seconds
    result = expensive_computation()
    return result
```

#### Timeout with Fallback

```python
@timeout(seconds=10)
@fallback(lambda: "default_value")
def api_call_with_timeout():
    return requests.get(url).json()
```

### 7. Safe Execution

Execute functions safely with default return values.

#### Safe Execute

```python
from app.utils.error_recovery import safe_execute

# Returns default on any error
config = safe_execute(
    load_config,
    default={"debug": False},
    "production"
)

# With multiple arguments
result = safe_execute(
    calculate_score,
    default=0.0,
    features,
    weights,
    bias
)
```

#### Safe Batch Processing

```python
from app.utils.error_recovery import batch_with_retry

def process_item(item):
    # May fail for some items
    return transform(item)

successful, failed = batch_with_retry(
    items=data_items,
    process_func=process_item,
    batch_size=100,
    max_retries=3
)

logger.info(f"Processed {len(successful)}/{len(data_items)} items")
```

## Integration Examples

### Training Tasks with Full Error Recovery

```python
from app.utils.error_recovery import retry, TransactionManager, safe_execute
from app.utils.db_recovery import db_retry, ensure_connection

@db_retry(max_attempts=3)
@ensure_connection
def initialize_progress_tracking(db: Session, model_run_id: int) -> dict:
    """Initialize progress tracking with retry and connection validation."""
    with TransactionManager(db) as tx:
        progress = TrainingProgress(
            model_run_id=model_run_id,
            status="initializing",
            progress_percentage=0.0
        )
        db.add(progress)

    return {"progress_id": progress.id}

@db_retry(max_attempts=3)
@ensure_connection
def update_training_progress(
    db: Session,
    model_run_id: int,
    progress: float,
    stage: str
):
    """Update training progress with error recovery."""
    with TransactionManager(db) as tx:
        prog = db.query(TrainingProgress).filter_by(
            model_run_id=model_run_id
        ).first()

        if prog:
            prog.progress_percentage = progress
            prog.stage = stage
            prog.updated_at = datetime.utcnow()

@celery_app.task(bind=True, max_retries=3)
def train_model(self, model_run_id: int, config: dict):
    """Training task with checkpoint/resume capability."""
    try:
        # Check for existing checkpoint
        checkpoint = safe_execute(
            load_checkpoint,
            default=None,
            model_run_id
        )

        if checkpoint:
            logger.info(f"Resuming from checkpoint: {checkpoint['epoch']}")
            start_epoch = checkpoint['epoch']
            model_state = checkpoint['model_state']
        else:
            start_epoch = 0
            model_state = None

        # Initialize with retry
        progress_data = initialize_progress_tracking(db, model_run_id)

        # Train with periodic checkpoints
        for epoch in range(start_epoch, config['num_epochs']):
            try:
                # Training step
                metrics = train_epoch(model, data, epoch)

                # Update progress with retry
                update_training_progress(
                    db,
                    model_run_id,
                    progress=(epoch + 1) / config['num_epochs'] * 100,
                    stage=f"epoch_{epoch + 1}"
                )

                # Save checkpoint every 10 epochs
                if epoch % 10 == 0:
                    save_checkpoint(model_run_id, {
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'metrics': metrics
                    })

            except Exception as e:
                logger.error(f"Error in epoch {epoch}: {e}")
                # Save checkpoint before failing
                save_checkpoint(model_run_id, {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'error': str(e)
                })
                raise

        return {"status": "success", "metrics": metrics}

    except Exception as e:
        # Retry task if retries available
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60)

        # Final failure
        logger.error(f"Training failed after retries: {e}")
        raise
```

### API Endpoints with Circuit Breaker

```python
from fastapi import APIRouter, HTTPException
from app.utils.error_recovery import CircuitBreaker, CircuitState

router = APIRouter()

# Circuit breaker for external ML model API
model_api_circuit = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=120,
    expected_exception=RequestException
)

@router.post("/predict")
async def predict(data: PredictRequest):
    """Prediction endpoint with circuit breaker."""

    # Check circuit state
    if model_api_circuit.state == CircuitState.OPEN:
        # Use fallback model
        logger.warning("Using local fallback model")
        return local_model_predict(data)

    try:
        # Try external model with circuit breaker
        @model_api_circuit
        def call_external_model():
            response = requests.post(
                "https://ml-api.example.com/predict",
                json=data.dict(),
                timeout=10
            )
            return response.json()

        result = call_external_model()
        return result

    except Exception as e:
        # Circuit opened or other error
        logger.error(f"Prediction failed: {e}")

        # Use fallback
        return local_model_predict(data)
```

## Best Practices

### 1. Retry Guidelines

✅ **DO:**

- Use exponential backoff for external services
- Add jitter to prevent thundering herd
- Set reasonable max_attempts (3-5 typically)
- Log retry attempts for monitoring
- Use specific exception types

❌ **DON'T:**

- Retry non-idempotent operations without checks
- Retry user input validation errors
- Use infinite retries
- Retry without delays
- Ignore retry metrics

### 2. Circuit Breaker Guidelines

✅ **DO:**

- Use for external service dependencies
- Set failure_threshold based on service SLA
- Monitor circuit state changes
- Provide fallback mechanisms
- Test recovery behavior

❌ **DON'T:**

- Use for fast-failing operations
- Set threshold too low (causes false positives)
- Use without fallback strategy
- Ignore circuit open events

### 3. Transaction Guidelines

✅ **DO:**

- Keep transactions short
- Use savepoints for partial rollback
- Handle concurrent updates with optimistic locking
- Log transaction failures
- Test rollback scenarios

❌ **DON'T:**

- Hold transactions during long operations
- Nest transactions without savepoints
- Ignore deadlock scenarios
- Mix transaction patterns

### 4. Error Recovery Composition

Combine patterns effectively:

```python
# Good: Layered defense
@timeout(seconds=30)              # Outer: prevent hanging
@retry(max_attempts=3, delay=1.0) # Middle: retry transient failures
@circuit_breaker                   # Inner: prevent cascade failures
def robust_external_call():
    return external_service.call()

# Good: Database operation with full protection
@ensure_connection                 # Test connection first
@db_retry(max_attempts=3)         # Retry database errors
def safe_database_operation(db: Session):
    with TransactionManager(db):   # Auto rollback on error
        return db.query(Model).all()
```

## Monitoring and Observability

### Metrics to Track

1. **Retry Metrics**

   - Retry count per operation
   - Success rate after retries
   - Average retry delay
   - Operations exceeding max retries

2. **Circuit Breaker Metrics**

   - Circuit state changes
   - Time in OPEN state
   - Failure rate when CLOSED
   - Recovery success rate

3. **Transaction Metrics**
   - Rollback frequency
   - Average transaction duration
   - Deadlock occurrences
   - Concurrent update conflicts

### Logging

```python
# Configure structured logging
import logging
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Log retry attempts
@retry(
    max_attempts=3,
    on_retry=lambda exc, attempt: logger.warning(
        f"Retry attempt {attempt}: {exc}",
        extra={
            "attempt": attempt,
            "exception_type": type(exc).__name__,
            "operation": "external_api_call"
        }
    )
)
def monitored_operation():
    pass

# Log circuit state changes
circuit = CircuitBreaker(
    on_open=lambda: logger.critical(
        "Circuit opened",
        extra={"circuit": "external_service", "failures": circuit.failure_count}
    ),
    on_close=lambda: logger.info(
        "Circuit closed",
        extra={"circuit": "external_service"}
    )
)
```

## Testing Error Recovery

### Unit Tests

```python
import pytest
from unittest.mock import Mock, patch

def test_retry_success_after_failure():
    """Test retry succeeds after transient failure."""
    call_count = [0]

    @retry(max_attempts=3, delay=0.1)
    def flaky_function():
        call_count[0] += 1
        if call_count[0] < 2:
            raise ConnectionError("Transient failure")
        return "success"

    result = flaky_function()
    assert result == "success"
    assert call_count[0] == 2

def test_circuit_opens_after_threshold():
    """Test circuit breaker opens after failures."""
    circuit = CircuitBreaker(failure_threshold=3)

    @circuit
    def always_fails():
        raise ValueError("Fail")

    # Trigger failures
    for _ in range(3):
        with pytest.raises(ValueError):
            always_fails()

    assert circuit.state == CircuitState.OPEN

    # Next call fails fast
    with pytest.raises(Exception, match="Circuit breaker is OPEN"):
        always_fails()
```

### Integration Tests

```python
@pytest.mark.integration
def test_database_retry_integration(test_db):
    """Test database retry in real scenario."""

    @db_retry(max_attempts=3, delay=0.5)
    def flaky_db_operation(db: Session):
        # Simulate intermittent connection issues
        if random.random() < 0.5:
            raise OperationalError("stmt", "params", "orig")

        user = User(name="Test", email="test@example.com")
        db.add(user)
        db.commit()
        return user.id

    user_id = flaky_db_operation(test_db)
    assert user_id is not None

    # Verify record was created
    user = test_db.query(User).filter_by(id=user_id).first()
    assert user is not None
    assert user.name == "Test"
```

## Performance Considerations

### Retry Impact

```python
# High-frequency operations: Use lower retry counts
@retry(max_attempts=2, delay=0.1)  # Fast fail
def high_frequency_operation():
    pass

# Critical operations: Use more retries
@retry(max_attempts=5, delay=1.0, backoff=2.0)
def critical_operation():
    pass

# Batch operations: Retry per-item instead of entire batch
successful, failed = batch_with_retry(
    items=large_dataset,
    process_func=process_item,
    max_retries=2  # Lower per-item retries
)
```

### Circuit Breaker Tuning

```python
# High-traffic service: Lower threshold, faster recovery
high_traffic_circuit = CircuitBreaker(
    failure_threshold=3,   # Trip quickly
    recovery_timeout=30    # Test recovery sooner
)

# Critical service: Higher threshold, longer recovery
critical_circuit = CircuitBreaker(
    failure_threshold=10,  # More tolerance
    recovery_timeout=120   # Give more time to recover
)
```

## Troubleshooting

### Common Issues

**1. Retry Loops**

```python
# Problem: Retrying non-transient errors
@retry(max_attempts=5)
def parse_data(data):
    return json.loads(data)  # Always fails on invalid JSON

# Solution: Don't retry validation errors
@retry(max_attempts=3, exceptions=(IOError, TimeoutError))
def fetch_and_parse(url):
    data = fetch(url)  # Retry network errors
    return json.loads(data)  # Don't retry parse errors
```

**2. Circuit Breaker False Positives**

```python
# Problem: Circuit opens due to legitimate load spikes
circuit = CircuitBreaker(failure_threshold=3)  # Too sensitive

# Solution: Increase threshold and add exception filtering
circuit = CircuitBreaker(
    failure_threshold=10,
    expected_exception=RequestException  # Only count network errors
)
```

**3. Transaction Deadlocks**

```python
# Problem: Long-held transactions causing deadlocks
def slow_operation(db: Session):
    with TransactionManager(db):
        records = db.query(Model).all()
        for record in records:
            time.sleep(1)  # Don't do this!
            process(record)

# Solution: Release transaction between operations
def fast_operation(db: Session):
    records = db.query(Model).all()

    for record in records:
        process(record)  # Process outside transaction

        # Quick transaction per record
        with TransactionManager(db):
            record.status = "processed"
```

## Migration Guide

### Adding Error Recovery to Existing Code

1. **Identify Critical Paths**

   - Database operations
   - External API calls
   - File I/O operations
   - Long-running computations

2. **Add Appropriate Patterns**

   ```python
   # Before
   def fetch_data():
       return requests.get(url).json()

   # After
   @retry(max_attempts=3, delay=1.0)
   @timeout(seconds=30)
   def fetch_data():
       return requests.get(url).json()
   ```

3. **Test Thoroughly**

   - Unit tests for retry logic
   - Integration tests with real failures
   - Load tests with circuit breakers

4. **Monitor and Tune**
   - Track retry rates
   - Adjust thresholds based on observed behavior
   - Fine-tune timeouts and delays

## Additional Resources

- [Martin Fowler - Circuit Breaker](https://martinfowler.com/bliki/CircuitBreaker.html)
- [AWS - Exponential Backoff](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)
- [Google SRE Book - Handling Overload](https://sre.google/sre-book/handling-overload/)
- [Resilience4j Documentation](https://resilience4j.readme.io/)

## Summary

Error recovery patterns provide:

- ✅ Resilience to transient failures
- ✅ Protection against cascading failures
- ✅ Graceful degradation
- ✅ Automatic recovery mechanisms
- ✅ Improved system reliability

Choose patterns based on:

- Operation criticality
- Failure characteristics
- Performance requirements
- User experience impact
