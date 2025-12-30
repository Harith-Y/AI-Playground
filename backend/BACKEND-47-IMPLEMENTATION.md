# BACKEND-47: Grid/Random Search Orchestration

**Implementation Date**: December 30, 2025  
**Status**: ✅ Complete  
**Version**: 1.0.0

## Overview

BACKEND-47 implements comprehensive orchestration capabilities for hyperparameter tuning workflows. This feature provides high-level abstractions for complex tuning strategies including progressive refinement and multi-model comparison.

### Key Features

1. **Progressive Search Workflow**: Automatic grid → random → bayesian progression
2. **Multi-Model Comparison**: Parallel tuning across multiple models
3. **Automatic Parameter Refinement**: Intelligent parameter space narrowing
4. **Workflow Management**: Status tracking and orchestration control
5. **Best Model Selection**: Automatic identification of optimal model

---

## Architecture

### Components

```
backend/
├── app/
│   ├── services/
│   │   └── tuning_orchestration_service.py    # Core orchestration logic
│   ├── api/v1/endpoints/
│   │   └── tuning_orchestration.py            # REST API endpoints
│   ├── schemas/
│   │   └── tuning_orchestration.py            # Pydantic models
│   └── tasks/
│       └── tuning_tasks.py                    # Celery tasks (reused)
└── tests/
    └── test_tuning_orchestration.py           # Comprehensive tests
```

### Class Structure

```python
TuningOrchestrationService
├── progressive_search()           # Grid → Random → Bayesian
├── trigger_next_stage()          # Auto-advance workflow stages
├── multi_model_comparison()      # Parallel model tuning
├── get_orchestration_status()    # Track workflow progress
└── get_best_model_from_comparison()  # Find optimal model

ProgressiveSearchConfig            # Progressive search settings
MultiModelConfig                   # Multi-model comparison settings
```

---

## Implementation Details

### 1. Progressive Search Workflow

**Purpose**: Automatically refine hyperparameter search through three stages

**Workflow**:

```
Stage 1: Grid Search
├─ Exhaustive exploration of initial parameter space
├─ Identifies promising parameter regions
└─ Generates baseline results

Stage 2: Random Search
├─ Explores refined space around best grid results
├─ Parameter space narrowed by refinement_factor
├─ Increases exploration within promising regions
└─ Balances exploration vs exploitation

Stage 3: Bayesian Optimization
├─ Fine-tunes in best random search region
├─ Efficient convergence to optimum
├─ Uses Gaussian processes for intelligent search
└─ Final optimization stage
```

**Key Methods**:

```python
def progressive_search(
    model_run_id: UUID,
    config: ProgressiveSearchConfig,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Initiate progressive search workflow.

    Creates 3 tuning runs (grid, random, bayesian) and triggers
    grid search. Subsequent stages triggered automatically upon
    completion via trigger_next_stage().
    """
```

**Parameter Refinement Algorithm**:

```python
def _refine_param_grid(
    best_params: Dict[str, Any],
    top_results: List[Dict[str, Any]],
    refinement_factor: float = 0.3
) -> Dict[str, List]:
    """
    Refine parameter grid based on top results.

    For numeric parameters:
    - Find min/max from top N results
    - Narrow range: new_range = best ± (range * refinement_factor)

    For categorical parameters:
    - Keep best value + values from top results
    - Reduces search space intelligently
    """
```

### 2. Multi-Model Comparison

**Purpose**: Compare multiple models with parallel or sequential tuning

**Features**:

- Parallel execution using Celery groups
- Model-specific parameter grids
- Automatic best model selection
- Fair comparison with consistent settings

**Key Methods**:

```python
def multi_model_comparison(
    config: MultiModelConfig,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute parallel tuning across multiple models.

    If parallel=True:
    - Uses Celery group for simultaneous execution
    - All models tuned concurrently
    - Returns group_task_id for tracking

    If parallel=False:
    - Sequential execution
    - One model at a time
    - Useful for resource-constrained environments
    """
```

**Best Model Selection**:

```python
def get_best_model_from_comparison(
    orchestration_id: str
) -> Dict[str, Any]:
    """
    Identify best performing model from comparison.

    Returns:
    - Best model details (ID, type, score, params)
    - All models ranked by score
    - Tuning method used
    """
```

### 3. Workflow Status Tracking

**Purpose**: Monitor orchestration progress and stage completion

**Status Values**:

- `PENDING`: Not yet started
- `RUNNING`: Currently executing
- `COMPLETED`: Successfully finished
- `FAILED`: Error occurred

**Tracking Method**:

```python
def get_orchestration_status(
    orchestration_id: str
) -> Dict[str, Any]:
    """
    Get comprehensive workflow status.

    Returns:
    - Overall status
    - Progress (completed/total stages)
    - Per-stage status and scores
    - Workflow type (progressive/multi-model)
    """
```

---

## API Endpoints

### 1. Start Progressive Search

**Endpoint**: `POST /api/v1/tuning-orchestration/progressive-search`

**Request**:

```json
{
  "model_run_id": "123e4567-e89b-12d3-a456-426614174000",
  "initial_param_grid": {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 5, 10]
  },
  "refinement_factor": 0.3,
  "cv_folds": 5,
  "scoring_metric": "accuracy",
  "n_iter_random": 50,
  "n_iter_bayesian": 30
}
```

**Response (202)**:

```json
{
  "orchestration_id": "abc-123-def-456",
  "model_run_id": "123e4567-e89b-12d3-a456-426614174000",
  "workflow": "progressive_search",
  "stages": [
    {
      "stage": "grid_search",
      "tuning_run_id": "111-222-333",
      "method": "grid_search"
    },
    {
      "stage": "random_search",
      "tuning_run_id": "444-555-666",
      "method": "random_search"
    },
    {
      "stage": "bayesian_optimization",
      "tuning_run_id": "777-888-999",
      "method": "bayesian"
    }
  ],
  "grid_search": {
    "tuning_run_id": "111-222-333",
    "task_id": "task-abc-123",
    "status": "RUNNING"
  },
  "random_search": {
    "tuning_run_id": "444-555-666",
    "status": "PENDING"
  },
  "bayesian_optimization": {
    "tuning_run_id": "777-888-999",
    "status": "PENDING"
  },
  "message": "Progressive search workflow initiated. Grid search is running."
}
```

### 2. Start Multi-Model Comparison

**Endpoint**: `POST /api/v1/tuning-orchestration/multi-model-comparison`

**Request**:

```json
{
  "model_run_ids": [
    "123e4567-e89b-12d3-a456-426614174000",
    "123e4567-e89b-12d3-a456-426614174001",
    "123e4567-e89b-12d3-a456-426614174002"
  ],
  "tuning_method": "bayesian",
  "param_grids": {
    "123e4567-e89b-12d3-a456-426614174000": {
      "n_estimators": [50, 100, 200],
      "max_depth": [5, 10, 20]
    }
  },
  "cv_folds": 5,
  "scoring_metric": "accuracy",
  "n_iter": 30,
  "parallel": true
}
```

**Response (202)**:

```json
{
  "orchestration_id": "abc-123-def-456",
  "workflow": "multi_model_comparison",
  "n_models": 3,
  "tuning_method": "bayesian",
  "parallel": true,
  "group_task_id": "group-task-xyz-789",
  "tuning_runs": [
    {
      "model_run_id": "123e4567-e89b-12d3-a456-426614174000",
      "tuning_run_id": "111-222-333"
    },
    {
      "model_run_id": "123e4567-e89b-12d3-a456-426614174001",
      "tuning_run_id": "444-555-666"
    },
    {
      "model_run_id": "123e4567-e89b-12d3-a456-426614174002",
      "tuning_run_id": "777-888-999"
    }
  ],
  "message": "Multi-model comparison initiated for 3 models"
}
```

### 3. Get Orchestration Status

**Endpoint**: `GET /api/v1/tuning-orchestration/orchestration/{orchestration_id}/status`

**Response**:

```json
{
  "orchestration_id": "abc-123-def-456",
  "workflow_type": "progressive_search",
  "overall_status": "RUNNING",
  "progress": {
    "completed": 1,
    "total": 3,
    "percentage": 33.33
  },
  "statuses": {
    "completed": 1,
    "running": 1,
    "failed": 0,
    "pending": 1
  },
  "stages": [
    {
      "tuning_run_id": "111-222-333",
      "model_run_id": "123e4567-e89b-12d3-a456-426614174000",
      "method": "grid_search",
      "status": "COMPLETED",
      "stage": "grid_search",
      "task_id": "task-abc-123",
      "best_score": 0.95
    },
    {
      "tuning_run_id": "444-555-666",
      "model_run_id": "123e4567-e89b-12d3-a456-426614174000",
      "method": "random_search",
      "status": "RUNNING",
      "stage": "random_search",
      "task_id": "task-def-456"
    },
    {
      "tuning_run_id": "777-888-999",
      "model_run_id": "123e4567-e89b-12d3-a456-426614174000",
      "method": "bayesian",
      "status": "PENDING",
      "stage": "bayesian_optimization"
    }
  ]
}
```

### 4. Get Best Model

**Endpoint**: `GET /api/v1/tuning-orchestration/orchestration/{orchestration_id}/best-model`

**Response**:

```json
{
  "orchestration_id": "abc-123-def-456",
  "best_model": {
    "model_run_id": "123e4567-e89b-12d3-a456-426614174001",
    "tuning_run_id": "444-555-666",
    "model_type": "random_forest_classifier",
    "best_score": 0.96,
    "best_params": {
      "n_estimators": 150,
      "max_depth": 15,
      "min_samples_split": 5
    },
    "tuning_method": "bayesian"
  },
  "all_models": [
    {
      "model_run_id": "123e4567-e89b-12d3-a456-426614174000",
      "tuning_run_id": "111-222-333",
      "score": 0.94,
      "method": "bayesian"
    },
    {
      "model_run_id": "123e4567-e89b-12d3-a456-426614174001",
      "tuning_run_id": "444-555-666",
      "score": 0.96,
      "method": "bayesian"
    },
    {
      "model_run_id": "123e4567-e89b-12d3-a456-426614174002",
      "tuning_run_id": "777-888-999",
      "score": 0.92,
      "method": "bayesian"
    }
  ]
}
```

### 5. Trigger Next Stage (Manual)

**Endpoint**: `POST /api/v1/tuning-orchestration/trigger-next-stage`

**Request**:

```json
{
  "orchestration_id": "abc-123-def-456",
  "completed_tuning_run_id": "111-222-333"
}
```

**Response (202)**:

```json
{
  "stage": "random_search",
  "tuning_run_id": "444-555-666",
  "task_id": "task-def-456",
  "status": "RUNNING",
  "message": "Next stage (random_search) triggered successfully"
}
```

---

## Usage Examples

### Example 1: Progressive Search

```python
import httpx

# 1. Start progressive search
response = httpx.post(
    "http://localhost:8000/api/v1/tuning-orchestration/progressive-search",
    json={
        "model_run_id": "abc-123",
        "initial_param_grid": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10]
        },
        "refinement_factor": 0.3,
        "cv_folds": 5,
        "scoring_metric": "f1",
        "n_iter_random": 50,
        "n_iter_bayesian": 30
    }
)
orchestration_id = response.json()["orchestration_id"]

# 2. Monitor progress
import time
while True:
    status = httpx.get(
        f"http://localhost:8000/api/v1/tuning-orchestration/orchestration/{orchestration_id}/status"
    ).json()

    print(f"Status: {status['overall_status']}")
    print(f"Progress: {status['progress']['completed']}/{status['progress']['total']}")

    if status['overall_status'] in ['COMPLETED', 'FAILED']:
        break

    time.sleep(30)

# 3. Get final results
if status['overall_status'] == 'COMPLETED':
    # Get the last (bayesian) stage results
    final_stage = status['stages'][-1]
    tuning_run_id = final_stage['tuning_run_id']

    results = httpx.get(
        f"http://localhost:8000/api/v1/tuning/tune/{tuning_run_id}/results"
    ).json()

    print(f"Best score: {results['best_score']:.4f}")
    print(f"Best params: {results['best_params']}")
```

### Example 2: Multi-Model Comparison

```python
import httpx

# 1. Train multiple models
model_ids = []
for model_type in ['random_forest_classifier', 'gradient_boosting_classifier', 'svm_classifier']:
    response = httpx.post(
        "http://localhost:8000/api/v1/models/train",
        json={
            "experiment_id": "exp-123",
            "dataset_id": "dataset-456",
            "model_type": model_type
        }
    )
    model_ids.append(response.json()["model_run_id"])

# 2. Start multi-model comparison
response = httpx.post(
    "http://localhost:8000/api/v1/tuning-orchestration/multi-model-comparison",
    json={
        "model_run_ids": model_ids,
        "tuning_method": "bayesian",
        "cv_folds": 5,
        "scoring_metric": "roc_auc",
        "n_iter": 30,
        "parallel": True
    }
)
orchestration_id = response.json()["orchestration_id"]

# 3. Wait for completion
import time
while True:
    status = httpx.get(
        f"http://localhost:8000/api/v1/tuning-orchestration/orchestration/{orchestration_id}/status"
    ).json()

    if status['overall_status'] == 'COMPLETED':
        break

    time.sleep(30)

# 4. Get best model
best_model = httpx.get(
    f"http://localhost:8000/api/v1/tuning-orchestration/orchestration/{orchestration_id}/best-model"
).json()

print(f"Best model: {best_model['best_model']['model_type']}")
print(f"Best score: {best_model['best_model']['best_score']:.4f}")
print(f"Best params: {best_model['best_model']['best_params']}")

# 5. Compare all models
print("\nModel Rankings:")
for i, model in enumerate(best_model['all_models'], 1):
    print(f"{i}. Score: {model['score']:.4f}")
```

### Example 3: Custom Param Grids per Model

```python
import httpx

# Different search spaces for different model types
param_grids = {
    "rf-model-id": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20]
    },
    "svm-model-id": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"]
    },
    "gb-model-id": {
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7]
    }
}

response = httpx.post(
    "http://localhost:8000/api/v1/tuning-orchestration/multi-model-comparison",
    json={
        "model_run_ids": list(param_grids.keys()),
        "tuning_method": "random_search",
        "param_grids": param_grids,
        "cv_folds": 5,
        "n_iter": 50,
        "parallel": True
    }
)
```

---

## Testing

### Test Coverage

**File**: `backend/tests/test_tuning_orchestration.py`

**Test Classes**:

1. `TestProgressiveSearch` - 30+ tests for progressive workflow
2. `TestMultiModelComparison` - 25+ tests for multi-model tuning
3. `TestOrchestrationStatus` - 15+ tests for status tracking
4. `TestOrchestrationAPI` - 20+ tests for API endpoints

**Total**: 90+ comprehensive tests

### Running Tests

```bash
# Run all orchestration tests
pytest backend/tests/test_tuning_orchestration.py -v

# Run specific test class
pytest backend/tests/test_tuning_orchestration.py::TestProgressiveSearch -v

# Run with coverage
pytest backend/tests/test_tuning_orchestration.py --cov=app.services.tuning_orchestration_service --cov-report=html
```

---

## Benefits

### 1. Progressive Search

✅ **Comprehensive Coverage**: Grid search ensures no region is missed  
✅ **Efficient Refinement**: Random search explores promising regions faster  
✅ **Optimal Fine-Tuning**: Bayesian optimization converges efficiently  
✅ **Automatic Workflow**: No manual intervention between stages  
✅ **Parameter Narrowing**: Intelligent space reduction saves time

### 2. Multi-Model Comparison

✅ **Parallel Execution**: All models tuned simultaneously  
✅ **Fair Comparison**: Consistent settings across models  
✅ **Automatic Selection**: Best model identified automatically  
✅ **Resource Efficient**: Optional sequential mode  
✅ **Custom Grids**: Model-specific parameter spaces

### 3. Orchestration Management

✅ **Progress Tracking**: Real-time workflow status  
✅ **Error Handling**: Robust failure detection  
✅ **State Persistence**: Results stored in database  
✅ **Workflow Control**: Manual stage triggering available  
✅ **Comprehensive Logging**: Full audit trail

---

## Performance Considerations

### Time Estimates

**Progressive Search** (typical):

- Grid Search: 5-30 minutes (depends on param space size)
- Random Search: 10-20 minutes (50 iterations)
- Bayesian: 5-15 minutes (30 iterations)
- **Total**: ~30-65 minutes

**Multi-Model Comparison** (3 models, parallel):

- Single model tuning: 10-20 minutes
- Parallel execution: 10-20 minutes (same as single)
- Sequential execution: 30-60 minutes (3x single)

### Resource Usage

**Parallel Multi-Model**:

- CPU: High (all cores utilized)
- Memory: Medium-High (multiple models in memory)
- Recommended: 8+ cores, 16+ GB RAM

**Sequential Workflows**:

- CPU: Medium (single model at a time)
- Memory: Low-Medium
- Recommended: 4+ cores, 8+ GB RAM

### Optimization Tips

1. **Reduce CV Folds**: Use 3-5 instead of 10 for faster results
2. **Limit Iterations**: Start with n_iter=20 for Bayesian
3. **Narrow Initial Grid**: Fewer initial parameters = faster grid search
4. **Sequential Mode**: Use when resources limited
5. **Increase Refinement Factor**: Higher factor = narrower search = faster

---

## Integration Points

### Database Models

- ✅ `TuningRun`: Stores tuning configuration and results
- ✅ `ModelRun`: Links to trained models
- ✅ `Experiment`: Groups related work

### Services

- ✅ `TuningOrchestrationService`: Core orchestration logic
- ✅ Reuses `tune_hyperparameters` Celery task from BACKEND-46

### ML Engine

- ✅ `run_grid_search()`: Grid search execution
- ✅ `run_random_search()`: Random search execution
- ✅ `run_bayesian_search()`: Bayesian optimization

---

## Future Enhancements

### Planned Features

1. **Early Stopping**: Stop stages if no improvement
2. **Adaptive Refinement**: Dynamic refinement_factor based on results
3. **Multi-Objective**: Optimize for multiple metrics simultaneously
4. **Ensemble Creation**: Auto-create ensemble from top models
5. **Warm Start**: Resume from previous orchestration
6. **Cost Estimation**: Predict time/resources before execution
7. **Experiment Comparison**: Compare orchestrations across experiments

### Potential Improvements

- Webhook notifications on completion
- Real-time progress updates via WebSocket
- Visualization of parameter space exploration
- A/B testing framework integration
- Auto-scaling for parallel execution

---

## Troubleshooting

### Issue: Progressive search stuck

**Solution**:

```python
# Check orchestration status
status = httpx.get(f"/orchestration/{orch_id}/status").json()

# If stage completed but next didn't start, manually trigger
httpx.post("/trigger-next-stage", json={
    "orchestration_id": orch_id,
    "completed_tuning_run_id": completed_run_id
})
```

### Issue: Multi-model comparison slow

**Solution**:

- Reduce `cv_folds` to 3
- Decrease `n_iter` to 20
- Use sequential mode if memory limited

### Issue: Best model selection fails

**Cause**: Orchestration not fully completed

**Solution**:

- Check `overall_status == 'COMPLETED'`
- Ensure all tuning runs have `status == COMPLETED`

---

## Migration Guide

### From Manual Tuning to Orchestration

**Before** (BACKEND-46):

```python
# Manual 3-stage tuning
grid_response = httpx.post("/tuning/tune", json={...})
# Wait and check status
random_response = httpx.post("/tuning/tune", json={...})
# Wait and check status
bayesian_response = httpx.post("/tuning/tune", json={...})
```

**After** (BACKEND-47):

```python
# Automatic 3-stage workflow
response = httpx.post("/tuning-orchestration/progressive-search", json={...})
# All stages handled automatically
```

---

## Summary

BACKEND-47 provides production-ready orchestration for complex hyperparameter tuning workflows. The implementation offers:

- 🚀 **Progressive Search**: Automatic grid → random → bayesian refinement
- 🔄 **Multi-Model Comparison**: Parallel tuning across models
- 📊 **Workflow Management**: Comprehensive status tracking
- ⚡ **Automatic Optimization**: Intelligent parameter space refinement
- 🎯 **Best Model Selection**: Automatic identification of optimal model

**Files Created**:

1. `backend/app/services/tuning_orchestration_service.py` (820 lines)
2. `backend/app/schemas/tuning_orchestration.py` (380 lines)
3. `backend/app/api/v1/endpoints/tuning_orchestration.py` (500 lines)
4. `backend/tests/test_tuning_orchestration.py` (900 lines)
5. `backend/BACKEND-47-IMPLEMENTATION.md` (this file)

**Total**: 2,600+ lines of production code + tests + documentation

---

**Ready for deployment!** 🎉
