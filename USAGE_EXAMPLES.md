# Usage Examples - Error Recovery Patterns

Ready-to-use code snippets for implementing error recovery in your application.

## Table of Contents

1. [API Endpoints](#api-endpoints)
2. [Database Operations](#database-operations)
3. [Celery Tasks](#celery-tasks)
4. [File Operations](#file-operations)
5. [External Services](#external-services)
6. [ML Pipeline Operations](#ml-pipeline-operations)
7. [Batch Processing](#batch-processing)

---

## API Endpoints

### Basic API Call with Retry and Timeout

```python
from fastapi import APIRouter, HTTPException
from app.utils.error_recovery import retry, timeout
import requests

router = APIRouter()

@router.get("/external-data")
@timeout(seconds=30)
@retry(max_attempts=3, delay=1.0, backoff=2.0)
async def fetch_external_data():
    """Fetch data from external API with retry."""
    try:
        response = requests.get(
            "https://api.example.com/data",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"External service unavailable: {e}")
```

### Prediction Endpoint with Circuit Breaker

```python
from fastapi import APIRouter, BackgroundTasks
from app.utils.error_recovery import CircuitBreaker, CircuitState, fallback
from app.schemas.prediction import PredictRequest, PredictResponse
import requests

router = APIRouter()

# Create circuit breaker for prediction service
prediction_circuit = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=requests.RequestException
)

def local_model_fallback(features: list) -> dict:
    """Fallback to local model when external service fails."""
    from app.ml_engine.inference import local_predictor
    prediction = local_predictor.predict(features)
    return {
        "prediction": prediction,
        "model": "local_fallback",
        "confidence": 0.85
    }

@router.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    """Prediction with circuit breaker and fallback."""

    # Check circuit state
    if prediction_circuit.state == CircuitState.OPEN:
        result = local_model_fallback(request.features)
        result["circuit_status"] = "open"
        return result

    try:
        @prediction_circuit
        def call_external_predictor():
            response = requests.post(
                "https://ml-api.example.com/predict",
                json={"features": request.features},
                timeout=10
            )
            response.raise_for_status()
            return response.json()

        result = call_external_predictor()
        result["circuit_status"] = "closed"
        return result

    except Exception as e:
        # Circuit opened or prediction failed
        result = local_model_fallback(request.features)
        result["circuit_status"] = prediction_circuit.state.name.lower()
        result["fallback_reason"] = str(e)
        return result
```

### Database Query Endpoint with Error Recovery

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.utils.db_recovery import db_retry, ensure_connection
from app.models.model_run import ModelRun

router = APIRouter()

@router.get("/models/{model_id}")
@ensure_connection
@db_retry(max_attempts=3, delay=0.5)
async def get_model(model_id: int, db: Session = Depends(get_db)):
    """Get model with database retry."""
    model = db.query(ModelRun).filter_by(id=model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model
```

---

## Database Operations

### Create Record with Transaction

```python
from sqlalchemy.orm import Session
from app.utils.error_recovery import TransactionManager
from app.utils.db_recovery import db_retry, ensure_connection
from app.models.model_run import ModelRun
from datetime import datetime

@ensure_connection
@db_retry(max_attempts=3, delay=0.5)
def create_model_run(
    db: Session,
    name: str,
    model_type: str,
    config: dict
) -> ModelRun:
    """Create model run with automatic rollback on error."""
    with TransactionManager(db) as tx:
        model_run = ModelRun(
            name=name,
            model_type=model_type,
            config=config,
            status="created",
            created_at=datetime.utcnow()
        )
        db.add(model_run)
        db.flush()  # Get ID without committing

        # Create related records
        from app.models.training_progress import TrainingProgress
        progress = TrainingProgress(
            model_run_id=model_run.id,
            status="initialized",
            progress_percentage=0.0
        )
        db.add(progress)

        # Transaction commits automatically on exit
        return model_run
```

### Update with Optimistic Locking

```python
from sqlalchemy.orm import Session
from app.utils.db_recovery import atomic_update
from app.models.model_run import ModelRun

def increment_model_usage(db: Session, model_id: int) -> bool:
    """Increment usage count with concurrent update protection."""

    # Get current model
    model = db.query(ModelRun).filter_by(id=model_id).first()
    if not model:
        return False

    # Atomic update with retry on conflict
    success = atomic_update(
        db=db,
        model_class=ModelRun,
        record_id=model_id,
        updates={"usage_count": model.usage_count + 1},
        max_retries=5
    )

    return success
```

### Bulk Insert with Error Handling

```python
from sqlalchemy.orm import Session
from app.utils.db_recovery import safe_bulk_insert
from app.models.prediction import Prediction
from typing import List, Dict

def save_predictions_batch(
    db: Session,
    predictions: List[Dict]
) -> tuple[int, List[tuple]]:
    """Save predictions in batch with error handling."""

    # Convert to model instances
    records = [
        {
            "model_run_id": pred["model_run_id"],
            "input_data": pred["input_data"],
            "prediction": pred["prediction"],
            "confidence": pred.get("confidence", 0.0)
        }
        for pred in predictions
    ]

    # Bulk insert with retry
    successful, failed = safe_bulk_insert(
        db=db,
        model_class=Prediction,
        records=records,
        batch_size=100
    )

    if failed:
        from app.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.error(f"Failed to insert {len(failed)} predictions")
        for record, error in failed:
            logger.error(f"Record: {record}, Error: {error}")

    return len(successful), failed
```

### Complex Transaction with Savepoints

```python
from sqlalchemy.orm import Session
from app.utils.error_recovery import TransactionManager
from app.utils.db_recovery import with_savepoint
from app.models.model_run import ModelRun
from app.models.dataset import Dataset

def create_model_with_dataset(
    db: Session,
    model_data: dict,
    dataset_data: dict
) -> tuple[ModelRun, Dataset]:
    """Create model and dataset with partial rollback capability."""

    with TransactionManager(db) as tx:
        # Create model
        model = ModelRun(**model_data)
        db.add(model)
        db.flush()

        # Try to create dataset (may fail)
        @with_savepoint
        def create_dataset_with_savepoint(db: Session):
            dataset = Dataset(**dataset_data)
            db.add(dataset)
            db.flush()
            return dataset

        try:
            dataset = create_dataset_with_savepoint(db)
        except Exception as e:
            # Dataset creation failed, but model still saved
            from app.utils.logger import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Dataset creation failed: {e}, continuing with model only")
            dataset = None

        # Transaction commits model (and dataset if successful)
        return model, dataset
```

---

## Celery Tasks

### Basic Task with Retry

```python
from app.celery_app import celery_app
from app.utils.error_recovery import retry
from app.utils.logger import get_logger

logger = get_logger(__name__)

@celery_app.task(bind=True, max_retries=3)
@retry(max_attempts=2, delay=5.0, backoff=2.0)
def process_data_task(self, data_id: int):
    """Process data with multi-level retry."""
    try:
        # Load data
        data = load_data(data_id)

        # Process
        result = process(data)

        # Save result
        save_result(data_id, result)

        return {"status": "success", "data_id": data_id}

    except Exception as exc:
        # Inner retry failed, try Celery-level retry
        if self.request.retries < self.max_retries:
            logger.warning(f"Task failed, retrying in 60s. Attempt {self.request.retries + 1}")
            raise self.retry(exc=exc, countdown=60)

        # All retries exhausted
        logger.error(f"Task failed permanently: {exc}")
        raise
```

### Training Task with Checkpoint/Resume

```python
from app.celery_app import celery_app
from app.utils.error_recovery import retry, safe_execute, TransactionManager
from app.utils.db_recovery import db_retry, ensure_connection
from app.db.session import get_db
import pickle
import os

@celery_app.task(bind=True, max_retries=3)
def train_model_task(self, model_run_id: int, config: dict):
    """Training with checkpoint/resume capability."""
    from app.utils.logger import get_logger
    logger = get_logger(__name__)

    checkpoint_path = f"/tmp/checkpoint_{model_run_id}.pkl"

    try:
        # Try to load checkpoint
        checkpoint = safe_execute(
            lambda: pickle.load(open(checkpoint_path, 'rb')),
            default=None
        )

        if checkpoint:
            logger.info(f"Resuming from epoch {checkpoint['epoch']}")
            start_epoch = checkpoint['epoch']
            model_state = checkpoint['model_state']
            best_score = checkpoint['best_score']
        else:
            start_epoch = 0
            model_state = None
            best_score = 0.0

        # Initialize model
        from app.ml_engine.training import ModelTrainer
        trainer = ModelTrainer(config)
        if model_state:
            trainer.load_state(model_state)

        # Training loop with checkpointing
        for epoch in range(start_epoch, config['num_epochs']):
            try:
                # Train epoch
                metrics = trainer.train_epoch(epoch)

                # Update progress in database with retry
                update_progress(model_run_id, epoch, metrics)

                # Save checkpoint every 10 epochs
                if epoch % 10 == 0 or metrics['score'] > best_score:
                    checkpoint_data = {
                        'epoch': epoch,
                        'model_state': trainer.get_state(),
                        'best_score': max(best_score, metrics['score']),
                        'metrics': metrics
                    }
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(checkpoint_data, f)

                    if metrics['score'] > best_score:
                        best_score = metrics['score']
                        logger.info(f"New best score: {best_score}")

            except Exception as e:
                logger.error(f"Error in epoch {epoch}: {e}")
                # Save checkpoint before raising
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state': trainer.get_state(),
                    'best_score': best_score,
                    'error': str(e)
                }
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                raise

        # Training complete
        final_model = trainer.get_state()

        # Save final model with retry
        save_final_model(model_run_id, final_model, best_score)

        # Cleanup checkpoint
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        return {
            "status": "success",
            "model_run_id": model_run_id,
            "best_score": best_score
        }

    except Exception as exc:
        if self.request.retries < self.max_retries:
            logger.warning(f"Training failed, will retry from checkpoint")
            raise self.retry(exc=exc, countdown=300)  # 5 min delay

        logger.error(f"Training failed permanently: {exc}")
        raise

@ensure_connection
@db_retry(max_attempts=3)
def update_progress(model_run_id: int, epoch: int, metrics: dict):
    """Update training progress with retry."""
    db = next(get_db())
    try:
        with TransactionManager(db):
            from app.models.training_progress import TrainingProgress
            progress = db.query(TrainingProgress).filter_by(
                model_run_id=model_run_id
            ).first()

            if progress:
                progress.current_epoch = epoch
                progress.metrics = metrics
                progress.updated_at = datetime.utcnow()
    finally:
        db.close()

@ensure_connection
@db_retry(max_attempts=5)
def save_final_model(model_run_id: int, model_state: dict, score: float):
    """Save final model with retry."""
    db = next(get_db())
    try:
        with TransactionManager(db):
            from app.models.model_run import ModelRun
            model_run = db.query(ModelRun).filter_by(id=model_run_id).first()

            if model_run:
                model_run.status = "completed"
                model_run.final_score = score
                model_run.model_state = model_state
                model_run.completed_at = datetime.utcnow()
    finally:
        db.close()
```

### Batch Processing Task

```python
from app.celery_app import celery_app
from app.utils.error_recovery import batch_with_retry
from typing import List

@celery_app.task
def process_batch_task(item_ids: List[int]):
    """Process batch of items with per-item retry."""

    def process_single_item(item_id: int):
        # Load item
        item = load_item(item_id)

        # Process (may fail for some items)
        result = complex_processing(item)

        # Save result
        save_result(item_id, result)

        return result

    # Process with per-item retry
    successful, failed = batch_with_retry(
        items=item_ids,
        process_func=process_single_item,
        batch_size=50,
        max_retries=2
    )

    return {
        "total": len(item_ids),
        "successful": len(successful),
        "failed": len(failed),
        "failed_ids": [item_id for item_id, _ in failed]
    }
```

---

## File Operations

### Read File with Retry and Fallback

```python
from app.utils.error_recovery import retry, fallback
import json

def get_default_config():
    """Return default configuration."""
    return {
        "debug": False,
        "timeout": 30,
        "max_retries": 3
    }

@fallback(get_default_config)
@retry(max_attempts=3, delay=0.5, exceptions=(IOError, OSError))
def load_config(config_path: str) -> dict:
    """Load config with retry and fallback to defaults."""
    with open(config_path, 'r') as f:
        return json.load(f)

# Usage
config = load_config("/path/to/config.json")
```

### Save File with Retry

```python
from app.utils.error_recovery import retry
import json
import os
from pathlib import Path

@retry(max_attempts=3, delay=1.0, exceptions=(IOError, OSError))
def save_results(results: dict, output_path: str):
    """Save results with retry."""
    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file first
    temp_path = f"{output_path}.tmp"
    with open(temp_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Atomic rename
    os.replace(temp_path, output_path)
```

### Load Model with Timeout

```python
from app.utils.error_recovery import timeout, retry
import pickle

@timeout(seconds=60)
@retry(max_attempts=2, delay=5.0)
def load_model_file(model_path: str):
    """Load model with timeout and retry."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)
```

---

## External Services

### API Client with Circuit Breaker

```python
from app.utils.error_recovery import CircuitBreaker, retry, timeout
import requests
from typing import Optional

class ExternalAPIClient:
    """API client with built-in error recovery."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.circuit = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=requests.RequestException
        )

    @timeout(seconds=30)
    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def _make_request(self, endpoint: str, method: str = "GET", **kwargs):
        """Internal request method with retry."""
        url = f"{self.base_url}{endpoint}"
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

    def get(self, endpoint: str, use_circuit: bool = True) -> Optional[dict]:
        """GET request with optional circuit breaker."""
        if use_circuit:
            @self.circuit
            def protected_get():
                return self._make_request(endpoint, "GET")
            return protected_get()
        else:
            return self._make_request(endpoint, "GET")

    def post(self, endpoint: str, data: dict, use_circuit: bool = True) -> Optional[dict]:
        """POST request with optional circuit breaker."""
        if use_circuit:
            @self.circuit
            def protected_post():
                return self._make_request(endpoint, "POST", json=data)
            return protected_post()
        else:
            return self._make_request(endpoint, "POST", json=data)

# Usage
client = ExternalAPIClient("https://api.example.com")

# This uses circuit breaker
data = client.get("/users")

# This bypasses circuit breaker
critical_data = client.get("/critical", use_circuit=False)
```

### Service with Health Check

```python
from app.utils.error_recovery import CircuitBreaker, CircuitState, retry
import requests

class ExternalMLService:
    """ML service with health checking."""

    def __init__(self, service_url: str):
        self.service_url = service_url
        self.circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

    @retry(max_attempts=2, delay=1.0)
    def health_check(self) -> bool:
        """Check if service is healthy."""
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def predict(self, features: list) -> dict:
        """Make prediction with circuit protection."""
        # Check circuit state
        if self.circuit.state == CircuitState.OPEN:
            if self.health_check():
                self.circuit.reset()  # Manually close circuit
            else:
                raise Exception("Service unavailable (circuit open)")

        @self.circuit
        @retry(max_attempts=2, delay=0.5)
        def _predict():
            response = requests.post(
                f"{self.service_url}/predict",
                json={"features": features},
                timeout=10
            )
            response.raise_for_status()
            return response.json()

        return _predict()
```

---

## ML Pipeline Operations

### Data Loading with Retry

```python
from app.utils.error_recovery import retry, safe_execute
import pandas as pd
from typing import Optional

@retry(max_attempts=3, delay=2.0, exceptions=(IOError, OSError, pd.errors.ParserError))
def load_dataset(file_path: str) -> pd.DataFrame:
    """Load dataset with retry."""
    return pd.read_csv(file_path)

def load_dataset_with_fallback(
    primary_path: str,
    backup_path: Optional[str] = None
) -> pd.DataFrame:
    """Load dataset with fallback to backup."""

    # Try primary
    df = safe_execute(
        load_dataset,
        primary_path,
        default=None
    )

    if df is not None:
        return df

    # Try backup
    if backup_path:
        df = safe_execute(
            load_dataset,
            backup_path,
            default=None
        )

        if df is not None:
            return df

    # Raise if both failed
    raise FileNotFoundError(f"Could not load dataset from {primary_path} or {backup_path}")
```

### Feature Engineering with Error Recovery

```python
from app.utils.error_recovery import safe_execute, batch_with_retry
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features with safe execution of optional steps."""
    df = df.copy()

    # Required transformations (will raise on error)
    df['age_squared'] = df['age'] ** 2

    # Optional transformations (won't break pipeline)
    # Add interaction feature (may fail on missing columns)
    interaction = safe_execute(
        lambda: df['feature1'] * df['feature2'],
        default=pd.Series([0] * len(df))
    )
    df['interaction'] = interaction

    # Add log transform (may fail on negative values)
    log_feature = safe_execute(
        lambda: np.log1p(df['amount']),
        default=pd.Series([0] * len(df))
    )
    df['log_amount'] = log_feature

    return df
```

### Model Training with Error Handling

```python
from app.utils.error_recovery import retry, TransactionManager
from app.utils.db_recovery import db_retry
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

@retry(max_attempts=2, delay=5.0)
def train_model_with_retry(X_train, y_train, params: dict):
    """Train model with retry on failure."""
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

def train_and_save_model(
    db,
    model_run_id: int,
    X_train,
    y_train,
    params: dict
):
    """Train and save model with full error recovery."""
    try:
        # Train model
        model = train_model_with_retry(X_train, y_train, params)

        # Calculate metrics
        from sklearn.metrics import accuracy_score
        train_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, train_pred)

        # Save to database with retry
        save_model_results(db, model_run_id, model, accuracy)

        return model, accuracy

    except Exception as e:
        # Update status to failed with retry
        update_model_status(db, model_run_id, "failed", str(e))
        raise

@db_retry(max_attempts=3)
def save_model_results(db, model_run_id: int, model, accuracy: float):
    """Save model results with retry."""
    import pickle

    with TransactionManager(db):
        from app.models.model_run import ModelRun
        model_run = db.query(ModelRun).filter_by(id=model_run_id).first()

        if model_run:
            model_run.model_object = pickle.dumps(model)
            model_run.accuracy = accuracy
            model_run.status = "completed"

@db_retry(max_attempts=3)
def update_model_status(db, model_run_id: int, status: str, error_msg: str = None):
    """Update model status with retry."""
    with TransactionManager(db):
        from app.models.model_run import ModelRun
        model_run = db.query(ModelRun).filter_by(id=model_run_id).first()

        if model_run:
            model_run.status = status
            if error_msg:
                model_run.error_message = error_msg
```

---

## Batch Processing

### Process Large Dataset in Batches

```python
from app.utils.error_recovery import batch_with_retry
import pandas as pd
from typing import List, Callable

def process_dataframe_in_batches(
    df: pd.DataFrame,
    process_func: Callable,
    batch_size: int = 1000,
    max_retries: int = 2
) -> tuple[pd.DataFrame, List]:
    """Process DataFrame in batches with retry."""

    # Convert to list of row dicts
    rows = df.to_dict('records')

    # Process with retry
    successful, failed = batch_with_retry(
        items=rows,
        process_func=process_func,
        batch_size=batch_size,
        max_retries=max_retries
    )

    # Convert successful back to DataFrame
    result_df = pd.DataFrame(successful) if successful else pd.DataFrame()

    return result_df, failed

# Usage example
def transform_row(row: dict) -> dict:
    """Transform single row."""
    # May fail for some rows
    row['processed'] = row['value'] * 2
    row['validated'] = validate(row)
    return row

df = pd.read_csv("large_dataset.csv")
processed_df, failures = process_dataframe_in_batches(
    df,
    transform_row,
    batch_size=500
)

print(f"Processed {len(processed_df)} rows successfully")
if failures:
    print(f"{len(failures)} rows failed")
```

### Parallel Batch Processing

```python
from app.utils.error_recovery import batch_with_retry, retry
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any

def parallel_batch_process(
    items: List[Any],
    process_func: Callable,
    num_workers: int = 4,
    batch_size: int = 100
) -> tuple[List, List]:
    """Process items in parallel batches with retry."""

    # Split into chunks for workers
    chunk_size = len(items) // num_workers
    chunks = [
        items[i:i + chunk_size]
        for i in range(0, len(items), chunk_size)
    ]

    all_successful = []
    all_failed = []

    def process_chunk(chunk):
        return batch_with_retry(
            items=chunk,
            process_func=process_func,
            batch_size=batch_size,
            max_retries=2
        )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_chunk, chunks)

        for successful, failed in results:
            all_successful.extend(successful)
            all_failed.extend(failed)

    return all_successful, all_failed

# Usage
items = list(range(10000))

def expensive_operation(item):
    # Process item
    return item ** 2

successful, failed = parallel_batch_process(
    items,
    expensive_operation,
    num_workers=4,
    batch_size=250
)
```

---

## Complete Application Example

### FastAPI App with Full Error Recovery

```python
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.utils.error_recovery import (
    CircuitBreaker,
    CircuitState,
    retry,
    timeout,
    fallback
)
from app.utils.db_recovery import db_retry, ensure_connection
from app.celery_app import celery_app
import requests

app = FastAPI(title="ML API with Error Recovery")

# Circuit breakers for external services
model_service_circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
data_service_circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

# ===== API Endpoints =====

@app.post("/train")
@ensure_connection
@db_retry(max_attempts=3)
async def start_training(
    config: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start model training with error recovery."""
    from app.models.model_run import ModelRun
    from app.utils.error_recovery import TransactionManager

    # Create model run with transaction
    with TransactionManager(db) as tx:
        model_run = ModelRun(
            name=config.get("name", "Unnamed Model"),
            model_type=config["model_type"],
            config=config,
            status="queued"
        )
        db.add(model_run)
        db.flush()
        model_run_id = model_run.id

    # Start training task
    from app.tasks.training_tasks import train_model_task
    background_tasks.add_task(
        train_model_task.delay,
        model_run_id,
        config
    )

    return {
        "model_run_id": model_run_id,
        "status": "queued"
    }

@app.get("/models/{model_id}")
@ensure_connection
@db_retry(max_attempts=3)
async def get_model(model_id: int, db: Session = Depends(get_db)):
    """Get model with database retry."""
    from app.models.model_run import ModelRun

    model = db.query(ModelRun).filter_by(id=model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "id": model.id,
        "name": model.name,
        "status": model.status,
        "accuracy": model.accuracy
    }

@app.post("/predict")
async def predict(request: dict):
    """Predict with circuit breaker and fallback."""
    features = request["features"]

    # Check circuit state
    if model_service_circuit.state == CircuitState.OPEN:
        return use_local_model(features)

    try:
        @model_service_circuit
        @timeout(seconds=10)
        @retry(max_attempts=2, delay=1.0)
        def call_model_service():
            response = requests.post(
                "https://ml-service.example.com/predict",
                json={"features": features},
                timeout=5
            )
            response.raise_for_status()
            return response.json()

        result = call_model_service()
        result["source"] = "external"
        return result

    except Exception as e:
        # Fallback to local model
        result = use_local_model(features)
        result["fallback_reason"] = str(e)
        return result

def use_local_model(features):
    """Fallback local model prediction."""
    # Load and use local model
    prediction = sum(features) / len(features)  # Dummy logic
    return {
        "prediction": prediction,
        "source": "local_fallback",
        "confidence": 0.75
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "circuits": {
            "model_service": model_service_circuit.state.name,
            "data_service": data_service_circuit.state.name
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Testing Examples

### Unit Test with Mocked Failures

```python
import pytest
from unittest.mock import Mock, patch
from app.utils.error_recovery import retry, CircuitBreaker, CircuitState

def test_retry_succeeds_after_failures():
    """Test retry succeeds after transient failures."""
    call_count = [0]

    @retry(max_attempts=3, delay=0.1)
    def flaky_function():
        call_count[0] += 1
        if call_count[0] < 3:
            raise ConnectionError("Transient failure")
        return "success"

    result = flaky_function()
    assert result == "success"
    assert call_count[0] == 3

def test_circuit_breaker_opens():
    """Test circuit breaker opens after threshold."""
    circuit = CircuitBreaker(failure_threshold=3)

    @circuit
    def failing_operation():
        raise ValueError("Always fails")

    # Trigger failures
    for _ in range(3):
        with pytest.raises(ValueError):
            failing_operation()

    assert circuit.state == CircuitState.OPEN

    # Next call should fail fast
    with pytest.raises(Exception, match="Circuit breaker is OPEN"):
        failing_operation()

@patch('requests.get')
def test_api_call_with_retry(mock_get):
    """Test API call retries on failure."""
    # First two calls fail, third succeeds
    mock_get.side_effect = [
        ConnectionError("Network error"),
        ConnectionError("Network error"),
        Mock(status_code=200, json=lambda: {"data": "success"})
    ]

    @retry(max_attempts=3, delay=0.1)
    def api_call():
        response = requests.get("https://api.example.com/data")
        return response.json()

    result = api_call()
    assert result == {"data": "success"}
    assert mock_get.call_count == 3
```

---

## Configuration Templates

### Production Configuration

```python
# config/error_recovery.py

from app.utils.error_recovery import CircuitBreaker, RetryStrategy

# External API configuration
EXTERNAL_API_RETRY_CONFIG = {
    "max_attempts": 3,
    "delay": 1.0,
    "backoff": 2.0,
    "jitter": True,
    "strategy": RetryStrategy.EXPONENTIAL
}

EXTERNAL_API_TIMEOUT = 30  # seconds

EXTERNAL_API_CIRCUIT = {
    "failure_threshold": 5,
    "recovery_timeout": 60,
    "expected_exception": (ConnectionError, TimeoutError)
}

# Database configuration
DATABASE_RETRY_CONFIG = {
    "max_attempts": 3,
    "delay": 0.5,
    "backoff": 2.0
}

DATABASE_CONNECTION_TIMEOUT = 10  # seconds

# File I/O configuration
FILE_RETRY_CONFIG = {
    "max_attempts": 2,
    "delay": 0.5,
    "exceptions": (IOError, OSError)
}

# Batch processing configuration
BATCH_SIZE = 1000
BATCH_MAX_RETRIES = 2
BATCH_PARALLEL_WORKERS = 4

# Training configuration
TRAINING_CHECKPOINT_INTERVAL = 10  # epochs
TRAINING_MAX_RETRIES = 3
TRAINING_RETRY_DELAY = 300  # seconds

# Monitoring
LOG_RETRY_ATTEMPTS = True
LOG_CIRCUIT_STATE_CHANGES = True
ALERT_ON_CIRCUIT_OPEN = True
```

---

## Quick Reference

### Common Patterns

```python
# API call with full protection
@timeout(seconds=30)
@retry(max_attempts=3, delay=1.0, backoff=2.0)
@circuit_breaker
def api_call(): ...

# Database operation
@ensure_connection
@db_retry(max_attempts=3)
def db_operation(db: Session):
    with TransactionManager(db): ...

# File operation with fallback
@fallback(use_default)
@retry(max_attempts=2)
def load_file(path): ...

# Celery task
@celery_app.task(bind=True, max_retries=3)
@retry(max_attempts=2, delay=5.0)
def task(self): ...

# Batch processing
successful, failed = batch_with_retry(
    items, process_func, batch_size=100, max_retries=2
)
```

---

For more details, see:

- [ERROR_RECOVERY.md](ERROR_RECOVERY.md) - Complete documentation
- [ERROR_RECOVERY_QUICK_REFERENCE.md](ERROR_RECOVERY_QUICK_REFERENCE.md) - Quick lookup
- [backend/app/utils/error_recovery.py](backend/app/utils/error_recovery.py) - Implementation
