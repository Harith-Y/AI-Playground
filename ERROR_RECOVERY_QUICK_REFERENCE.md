# Error Recovery Quick Reference

Quick lookup guide for common error recovery patterns.

## Quick Pattern Selection

| Scenario | Pattern | Code |
|----------|---------|------|
| External API call | Retry + Timeout | `@timeout(30) @retry(max_attempts=3)` |
| Database query | DB Retry | `@db_retry(max_attempts=3)` |
| File I/O | Retry + Fallback | `@fallback(use_cache) @retry(max_attempts=2)` |
| External service | Circuit Breaker | `@circuit_breaker` |
| Batch processing | Batch with Retry | `batch_with_retry(items, func)` |
| Transaction | Transaction Manager | `with TransactionManager(db):` |

## Common Patterns

### Retry with Exponential Backoff
```python
from app.utils.error_recovery import retry

@retry(
    max_attempts=3,
    delay=1.0,
    backoff=2.0,    # 1s, 2s, 4s
    jitter=True     # Add randomness
)
def api_call():
    return requests.get(url).json()
```

### Database Operation with Retry
```python
from app.utils.db_recovery import db_retry, ensure_connection

@ensure_connection  # Test connection first
@db_retry(max_attempts=3, delay=1.0)
def query_database(db: Session):
    return db.query(Model).filter_by(status="active").all()
```

### Circuit Breaker for External Service
```python
from app.utils.error_recovery import CircuitBreaker

external_api_circuit = CircuitBreaker(
    failure_threshold=5,      # Open after 5 failures
    recovery_timeout=60,      # Try recovery after 60s
    expected_exception=RequestException
)

@external_api_circuit
def call_external_api():
    return requests.post(api_url, json=data).json()
```

### Fallback Strategy
```python
from app.utils.error_recovery import fallback

def use_cached_data():
    return cache.get("data", default_value)

@fallback(use_cached_data)
def fetch_live_data():
    return api.get_data()
```

### Safe Transaction
```python
from app.utils.error_recovery import TransactionManager

def update_records(db: Session, updates: list):
    with TransactionManager(db) as tx:
        for update in updates:
            record = db.query(Model).get(update['id'])
            record.value = update['value']
        # Auto-commits on success, rolls back on error
```

### Timeout Protection
```python
from app.utils.error_recovery import timeout

@timeout(seconds=30)
def long_operation():
    # Raises TimeoutError if exceeds 30s
    return compute_result()
```

### Safe Execution with Default
```python
from app.utils.error_recovery import safe_execute

# Returns default if function fails
result = safe_execute(
    risky_function,
    arg1, arg2,
    default="fallback_value"
)
```

### Batch Processing with Retry
```python
from app.utils.error_recovery import batch_with_retry

successful, failed = batch_with_retry(
    items=data_items,
    process_func=transform_item,
    batch_size=100,
    max_retries=2
)
```

### Optimistic Locking
```python
from app.utils.db_recovery import atomic_update

success = atomic_update(
    db=db,
    model_class=User,
    record_id=user_id,
    updates={"credits": user.credits + 100},
    max_retries=5
)
```

### Safe Bulk Insert
```python
from app.utils.db_recovery import safe_bulk_insert

successful, failed = safe_bulk_insert(
    db=db,
    model_class=User,
    records=user_records,
    batch_size=1000
)
```

## Composition Patterns

### Full Protection Stack
```python
@timeout(seconds=30)                    # Prevent hanging
@retry(max_attempts=3, delay=1.0)      # Retry failures
@fallback(use_cache)                    # Fallback if all retries fail
def robust_operation():
    return external_service.call()
```

### Database Operation with Full Recovery
```python
@ensure_connection                      # Test connection
@db_retry(max_attempts=3)              # Retry DB errors
def safe_db_operation(db: Session):
    with TransactionManager(db):        # Auto-rollback
        return db.query(Model).all()
```

### Celery Task with Retry
```python
from celery import Task

@celery_app.task(bind=True, max_retries=3)
@retry(max_attempts=2, delay=5.0)      # Inner retry
def resilient_task(self, data):
    try:
        return process(data)
    except Exception as exc:
        # Celery-level retry with backoff
        raise self.retry(exc=exc, countdown=60)
```

## Retry Strategies

| Strategy | Delay Pattern | Use Case |
|----------|---------------|----------|
| Exponential | 1s, 2s, 4s, 8s | External APIs (recommended) |
| Linear | 1s, 2s, 3s, 4s | Moderate load services |
| Fixed | 2s, 2s, 2s, 2s | Predictable recovery time |
| Fibonacci | 1s, 1s, 2s, 3s, 5s | Gradual backoff needed |

```python
from app.utils.error_recovery import RetryStrategy

# Exponential (default)
@retry(max_attempts=5, delay=1.0, backoff=2.0)

# Linear
@retry(max_attempts=5, delay=1.0, strategy=RetryStrategy.LINEAR)

# Fixed
@retry(max_attempts=5, delay=2.0, strategy=RetryStrategy.FIXED)

# Fibonacci
@retry(max_attempts=7, delay=1.0, strategy=RetryStrategy.FIBONACCI)
```

## Circuit Breaker States

```
CLOSED (Normal)
    ↓ (failures ≥ threshold)
OPEN (Fail Fast)
    ↓ (after recovery_timeout)
HALF_OPEN (Testing)
    ↓ (success)           ↓ (failure)
CLOSED               OPEN
```

### Check Circuit State
```python
if circuit.state == CircuitState.OPEN:
    return use_fallback()
```

### Manual Control
```python
circuit.reset()      # Force CLOSED
circuit.open()       # Force OPEN
circuit.half_open()  # Force HALF_OPEN
```

## Exception Handling

### Retry Specific Exceptions Only
```python
@retry(
    max_attempts=3,
    exceptions=(TimeoutError, ConnectionError, RequestException)
)
def network_call():
    return requests.get(url)
```

### Database Exceptions (Auto-handled by db_retry)
- `OperationalError`: Connection lost, server restart
- `DisconnectionError`: Connection pool exhausted
- `TimeoutError`: Query timeout
- `IntegrityError`: NOT retried (data issue, not transient)

### Circuit Breaker Exception Filtering
```python
circuit = CircuitBreaker(
    failure_threshold=5,
    expected_exception=(RequestException, TimeoutError)  # Only count these
)
```

## Logging and Monitoring

### Retry Callback
```python
def log_retry(exc, attempt):
    logger.warning(f"Retry {attempt}: {exc}")

@retry(max_attempts=3, on_retry=log_retry)
def monitored_func():
    pass
```

### Circuit State Callbacks
```python
circuit = CircuitBreaker(
    on_open=lambda: logger.critical("Circuit OPENED!"),
    on_close=lambda: logger.info("Circuit CLOSED"),
    on_half_open=lambda: logger.info("Circuit HALF_OPEN")
)
```

## Common Mistakes

### ❌ Don't Retry Non-Idempotent Operations
```python
# BAD: May create duplicate charges
@retry(max_attempts=3)
def charge_customer(amount):
    payment_gateway.charge(amount)

# GOOD: Check for duplicates
@retry(max_attempts=3)
def charge_customer(idempotency_key, amount):
    if not already_charged(idempotency_key):
        payment_gateway.charge(amount, key=idempotency_key)
```

### ❌ Don't Retry User Input Errors
```python
# BAD: Retries validation errors
@retry(max_attempts=3)
def parse_input(data):
    return json.loads(data)

# GOOD: Only retry transient errors
@retry(max_attempts=3, exceptions=(IOError, TimeoutError))
def fetch_and_parse(url):
    data = fetch_url(url)  # Retry this
    return json.loads(data)  # Don't retry this
```

### ❌ Don't Hold Transactions During Long Operations
```python
# BAD: Holds transaction for minutes
with TransactionManager(db):
    records = db.query(Model).all()
    for record in records:
        expensive_computation(record)  # Don't do this!
        record.status = "processed"

# GOOD: Short transactions
records = db.query(Model).all()
for record in records:
    result = expensive_computation(record)
    
    # Quick transaction per record
    with TransactionManager(db):
        record.status = "processed"
        record.result = result
```

### ❌ Don't Ignore Circuit State
```python
# BAD: Ignores circuit protection
@circuit_breaker
def call_service():
    return external_api.call()

result = call_service()  # May raise CircuitBreakerError

# GOOD: Check state and provide fallback
if circuit.state == CircuitState.OPEN:
    result = get_cached_result()
else:
    result = call_service()
```

## Performance Tips

1. **Use Lower Retry Counts for High-Frequency Operations**
   ```python
   @retry(max_attempts=2, delay=0.1)  # Fast fail
   def frequent_operation():
       pass
   ```

2. **Batch Operations: Per-Item Retry**
   ```python
   # Don't retry entire batch, retry per item
   successful, failed = batch_with_retry(items, process_item, max_retries=2)
   ```

3. **Circuit Breaker for External Dependencies**
   ```python
   # Prevents hammering failing service
   @external_service_circuit
   def call_external():
       pass
   ```

4. **Timeout for Unbounded Operations**
   ```python
   @timeout(seconds=30)  # Prevent hanging
   def potentially_slow():
       pass
   ```

## Testing

### Mock Failures
```python
def test_retry_success():
    call_count = [0]
    
    @retry(max_attempts=3, delay=0.1)
    def flaky():
        call_count[0] += 1
        if call_count[0] < 2:
            raise ConnectionError("Fail")
        return "success"
    
    assert flaky() == "success"
    assert call_count[0] == 2
```

### Test Circuit Breaker
```python
def test_circuit_opens():
    circuit = CircuitBreaker(failure_threshold=3)
    
    @circuit
    def fails():
        raise ValueError("Fail")
    
    for _ in range(3):
        with pytest.raises(ValueError):
            fails()
    
    assert circuit.state == CircuitState.OPEN
```

## When to Use Each Pattern

| Pattern | Use When | Don't Use When |
|---------|----------|----------------|
| **Retry** | Transient failures (network, DB) | User input errors, validation |
| **Circuit Breaker** | External dependencies | Internal functions, fast operations |
| **Fallback** | Acceptable degraded service | Data consistency critical |
| **Timeout** | Unbounded operations | Fast operations (<1s) |
| **Transaction** | Multiple DB operations | Long-running processes |
| **Batch Retry** | Large datasets, some may fail | All-or-nothing operations |

## Configuration Recommendations

### External API
```python
@timeout(seconds=30)
@retry(max_attempts=3, delay=1.0, backoff=2.0, jitter=True)
@circuit_breaker
def api_call():
    pass
```

### Database Query
```python
@ensure_connection
@db_retry(max_attempts=3, delay=0.5)
def db_query(db: Session):
    with TransactionManager(db):
        pass
```

### File Operation
```python
@retry(max_attempts=2, delay=0.5)
@fallback(use_default_file)
def read_file(path):
    pass
```

### Celery Task
```python
@celery_app.task(bind=True, max_retries=3)
@retry(max_attempts=2, delay=5.0)
def async_task(self, data):
    try:
        return process(data)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Too many retries | Lower `max_attempts` or add exception filtering |
| Retry loops | Only retry transient errors, not validation |
| Circuit opens too often | Increase `failure_threshold` |
| Deadlocks | Keep transactions short, use smaller batch sizes |
| Timeout too strict | Increase timeout or optimize operation |
| No fallback when circuit open | Always provide fallback for circuit-protected calls |

## See Also

- [ERROR_RECOVERY.md](ERROR_RECOVERY.md) - Full documentation
- [backend/app/utils/error_recovery.py](backend/app/utils/error_recovery.py) - Implementation
- [backend/app/utils/db_recovery.py](backend/app/utils/db_recovery.py) - Database utilities
- [backend/tests/test_error_recovery.py](backend/tests/test_error_recovery.py) - Test examples
