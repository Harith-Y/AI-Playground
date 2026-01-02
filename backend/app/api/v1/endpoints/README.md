# Model Training API Endpoints

## Overview

This document provides comprehensive documentation for the Model Training API endpoints. These endpoints enable users to train machine learning models, retrieve training results, analyze model performance, and manage model runs.

## Table of Contents

- [Model Metrics Endpoint](#model-metrics-endpoint)
- [Feature Importance Endpoint](#feature-importance-endpoint)
- [Delete Model Run Endpoint](#delete-model-run-endpoint)

---

# Model Metrics Endpoint

## Overview

The `/api/v1/models/train/{model_run_id}/metrics` endpoint provides detailed evaluation metrics for completed model training runs. This endpoint is designed to give comprehensive performance insights including metrics, training metadata, and feature importance.

## Endpoint

```
GET /api/v1/models/train/{model_run_id}/metrics
```

## Purpose

This endpoint serves as a dedicated metrics retrieval interface that:
- Returns comprehensive evaluation metrics
- Provides training metadata (time, samples, features)
- Includes feature importance when available
- Supports all task types (classification, regression, clustering)
- Enables detailed performance analysis

## Request

### Path Parameters

- `model_run_id` (string, required): UUID of the model run

### Headers

- Authentication headers (when implemented)

### Example Request

```bash
curl -X GET "http://localhost:8000/api/v1/models/train/123e4567-e89b-12d3-a456-426614174002/metrics" \
  -H "Authorization: Bearer <token>"
```

## Response

### Success Response (200 OK)

#### Classification Model

```json
{
  "model_run_id": "123e4567-e89b-12d3-a456-426614174002",
  "model_type": "random_forest_classifier",
  "task_type": "classification",
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.94,
    "recall": 0.93,
    "f1_score": 0.935,
    "auc_roc": 0.96
  },
  "training_metadata": {
    "training_time": 45.5,
    "created_at": "2025-12-29T10:00:00Z",
    "hyperparameters": {
      "n_estimators": 100,
      "max_depth": 10,
      "min_samples_split": 2
    },
    "train_samples": 120,
    "test_samples": 30,
    "n_features": 4
  },
  "feature_importance": {
    "sepal_length": 0.35,
    "sepal_width": 0.25,
    "petal_length": 0.30,
    "petal_width": 0.10
  }
}
```

#### Regression Model

```json
{
  "model_run_id": "456e7890-e89b-12d3-a456-426614174003",
  "model_type": "random_forest_regressor",
  "task_type": "regression",
  "metrics": {
    "mae": 2.5,
    "mse": 8.3,
    "rmse": 2.88,
    "r2": 0.85
  },
  "training_metadata": {
    "training_time": 25.0,
    "created_at": "2025-12-29T11:00:00Z",
    "hyperparameters": {
      "n_estimators": 50,
      "max_depth": 8
    },
    "train_samples": 200,
    "test_samples": 50,
    "n_features": 10
  },
  "feature_importance": {
    "feature1": 0.25,
    "feature2": 0.20,
    "feature3": 0.15
  }
}
```

#### Clustering Model

```json
{
  "model_run_id": "789e0123-e89b-12d3-a456-426614174004",
  "model_type": "kmeans",
  "task_type": "clustering",
  "metrics": {
    "silhouette_score": 0.65,
    "davies_bouldin_score": 0.8,
    "calinski_harabasz_score": 150.5
  },
  "training_metadata": {
    "training_time": 15.0,
    "created_at": "2025-12-29T12:00:00Z",
    "hyperparameters": {
      "n_clusters": 3,
      "init": "k-means++",
      "max_iter": 300
    },
    "train_samples": 150,
    "n_features": 5
  }
}
```

### Response Fields

#### Root Level

- `model_run_id` (string): UUID of the model run
- `model_type` (string): Type of model trained
- `task_type` (string): Task type (classification, regression, clustering)
- `metrics` (object): Performance metrics
- `training_metadata` (object): Training information
- `feature_importance` (object, optional): Feature importance scores

#### Metrics Object

**Classification:**
- `accuracy`: Overall accuracy (0-1)
- `precision`: Precision score (0-1)
- `recall`: Recall score (0-1)
- `f1_score`: F1 score (0-1)
- `auc_roc`: Area under ROC curve (0-1)

**Regression:**
- `mae`: Mean Absolute Error
- `mse`: Mean Squared Error
- `rmse`: Root Mean Squared Error
- `r2`: R-squared score (-∞ to 1)

**Clustering:**
- `silhouette_score`: Silhouette coefficient (-1 to 1)
- `davies_bouldin_score`: Davies-Bouldin index (lower is better)
- `calinski_harabasz_score`: Calinski-Harabasz index (higher is better)

#### Training Metadata Object

- `training_time` (float): Training duration in seconds
- `created_at` (string): ISO 8601 timestamp
- `hyperparameters` (object): Model hyperparameters used
- `train_samples` (int, optional): Number of training samples
- `test_samples` (int, optional): Number of test samples
- `n_features` (int, optional): Number of features used

#### Feature Importance Object

- Key-value pairs where:
  - Key: Feature name (string)
  - Value: Importance score (float, 0-1)

### Error Responses

#### 400 Bad Request - Invalid UUID

```json
{
  "detail": "Invalid model_run_id format: invalid-uuid"
}
```

#### 400 Bad Request - Not Completed

```json
{
  "detail": "Model run is not completed yet. Current status: running"
}
```

#### 403 Forbidden - No Permission

```json
{
  "detail": "You don't have permission to access this model run"
}
```

#### 404 Not Found - Model Run Not Found

```json
{
  "detail": "Model run with id 123e4567-e89b-12d3-a456-426614174002 not found"
}
```

#### 404 Not Found - No Metrics Available

```json
{
  "detail": "No metrics available for this model run"
}
```

## Use Cases

### 1. Display Model Performance Dashboard

Frontend displays comprehensive metrics in a dashboard:

```javascript
const response = await fetch(`/api/v1/models/train/${runId}/metrics`);
const data = await response.json();

// Display metrics
console.log(`Accuracy: ${data.metrics.accuracy}`);
console.log(`Training Time: ${data.training_metadata.training_time}s`);

// Render feature importance chart
renderFeatureImportanceChart(data.feature_importance);
```

### 2. Compare Multiple Models

Fetch metrics for multiple runs to compare:

```javascript
const runs = ['run-1', 'run-2', 'run-3'];
const metricsPromises = runs.map(runId => 
  fetch(`/api/v1/models/train/${runId}/metrics`).then(r => r.json())
);

const allMetrics = await Promise.all(metricsPromises);
const comparison = allMetrics.map(m => ({
  model: m.model_type,
  accuracy: m.metrics.accuracy,
  time: m.training_metadata.training_time
}));
```

### 3. Export Metrics Report

Generate a detailed report:

```javascript
const response = await fetch(`/api/v1/models/train/${runId}/metrics`);
const data = await response.json();

const report = {
  model: data.model_type,
  performance: data.metrics,
  configuration: data.training_metadata.hyperparameters,
  topFeatures: Object.entries(data.feature_importance || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
};

downloadReport(report);
```

## Differences from /result Endpoint

| Feature | /metrics | /result |
|---------|----------|---------|
| **Purpose** | Detailed metrics analysis | Complete training results |
| **Metrics** | ✅ Comprehensive | ✅ Basic |
| **Feature Importance** | ✅ Included | ✅ Included |
| **Training Metadata** | ✅ Detailed | ✅ Basic |
| **Model Artifact Path** | ❌ Not included | ✅ Included |
| **Experiment ID** | ❌ Not included | ✅ Included |
| **Task Type** | ✅ Included | ❌ Not included |
| **Use Case** | Performance analysis | General results |

## Task Type Detection

The endpoint automatically detects task type from model name:

**Classification Keywords:**
- classifier, classification, logistic, naive_bayes, svc
- decision_tree_class, random_forest_class, gradient_boosting_class
- ada_boost_class, extra_trees_class, xgb_class, lgbm_class, catboost_class

**Regression Keywords:**
- regressor, regression, linear_reg, ridge, lasso, elastic, svr
- decision_tree_reg, random_forest_reg, gradient_boosting_reg
- ada_boost_reg, extra_trees_reg, xgb_reg, lgbm_reg, catboost_reg

**Clustering Keywords:**
- kmeans, dbscan, hierarchical, agglomerative, spectral
- mean_shift, birch, optics, gaussian_mixture

## Security

- ✅ User ownership verification through experiment
- ✅ UUID validation
- ✅ Status validation (must be completed)
- ✅ Metrics existence check

## Performance Considerations

- Fast response time (< 100ms typical)
- No heavy computation (reads from database)
- Efficient JSONB field access
- Indexed queries on model_run_id

## Integration with Frontend

### React Example

```typescript
interface ModelMetrics {
  model_run_id: string;
  model_type: string;
  task_type: string;
  metrics: Record<string, number>;
  training_metadata: {
    training_time: number;
    created_at: string;
    hyperparameters: Record<string, any>;
    train_samples?: number;
    test_samples?: number;
    n_features?: number;
  };
  feature_importance?: Record<string, number>;
}

async function fetchModelMetrics(runId: string): Promise<ModelMetrics> {
  const response = await fetch(`/api/v1/models/train/${runId}/metrics`);
  if (!response.ok) {
    throw new Error('Failed to fetch metrics');
  }
  return response.json();
}

// Usage
const metrics = await fetchModelMetrics('abc-123');
console.log(`Model: ${metrics.model_type}`);
console.log(`Accuracy: ${metrics.metrics.accuracy}`);
```

## Testing

Comprehensive unit tests in `backend/tests/test_model_training_api.py`:

- ✅ Get metrics for completed run
- ✅ Get metrics with feature importance
- ✅ Get metrics for different task types
- ✅ Handle non-existent model run
- ✅ Handle non-completed model run
- ✅ Handle missing metrics
- ✅ Handle invalid UUID format
- ✅ Verify permission checks

Run tests:
```bash
pytest backend/tests/test_model_training_api.py::TestGetModelMetrics -v
```

## Future Enhancements

Potential improvements:
- Add confusion matrix data for classification
- Add residual plots data for regression
- Add cluster visualization data
- Add cross-validation metrics
- Add metric history over time
- Add comparison with baseline models
- Add statistical significance tests
- Cache frequently accessed metrics

---

# Feature Importance Endpoint

## Overview

The Feature Importance endpoint provides insights into which features (input variables) had the most significant impact on a trained machine learning model's predictions. This is crucial for:

- **Model Interpretability**: Understanding what drives model predictions
- **Feature Selection**: Identifying which features to keep or remove
- **Domain Insights**: Discovering relationships in your data
- **Model Debugging**: Detecting if the model is using unexpected features

## Endpoint

```
GET /api/v1/models/train/{model_run_id}/feature-importance
```

## Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `top_n` | integer | No | 10 | Number of top features to return in `top_features` field |

## Response Schema

### Success Response (200 OK)

```json
{
  "model_run_id": "string (UUID)",
  "model_type": "string",
  "task_type": "string (classification|regression|clustering)",
  "has_feature_importance": "boolean",
  "feature_importance": [
    {
      "feature": "string",
      "importance": "float",
      "rank": "integer"
    }
  ],
  "feature_importance_dict": {
    "feature_name": "float"
  },
  "total_features": "integer",
  "top_features": [
    {
      "feature": "string",
      "importance": "float",
      "rank": "integer"
    }
  ],
  "importance_method": "string",
  "message": "string|null"
}
```

### Field Descriptions

- **model_run_id**: UUID of the model run
- **model_type**: Type of model (e.g., "random_forest_classifier")
- **task_type**: Machine learning task type
- **has_feature_importance**: Whether feature importance is available
- **feature_importance**: Complete list of all features with importance scores, sorted by importance (descending)
- **feature_importance_dict**: Dictionary mapping feature names to importance scores
- **total_features**: Total number of features
- **top_features**: Top N most important features (default: 10)
- **importance_method**: Method used to calculate importance (e.g., "feature_importances_", "coef_")
- **message**: Informational message (e.g., why feature importance is not available)

## Supported Models

### Tree-Based Models (feature_importances_)
- ✅ Random Forest (Classifier/Regressor)
- ✅ Decision Tree (Classifier/Regressor)
- ✅ Gradient Boosting (Classifier/Regressor)
- ✅ AdaBoost (Classifier/Regressor)
- ✅ Extra Trees (Classifier/Regressor)
- ✅ XGBoost (Classifier/Regressor)
- ✅ LightGBM (Classifier/Regressor)
- ✅ CatBoost (Classifier/Regressor)

### Linear Models (coef_)
- ✅ Linear Regression
- ✅ Ridge Regression
- ✅ Lasso Regression
- ✅ Elastic Net
- ✅ Logistic Regression
- ✅ SGD Classifier/Regressor

### Support Vector Machines (coef_ for linear kernel)
- ✅ SVC (Support Vector Classifier)
- ✅ SVR (Support Vector Regressor)

### Not Supported
- ❌ K-Nearest Neighbors (KNN)
- ❌ Naive Bayes
- ❌ Clustering models (K-Means, DBSCAN, etc.)

## Examples

### Example 1: Get All Feature Importance

**Request:**
```bash
GET /api/v1/models/train/123e4567-e89b-12d3-a456-426614174002/feature-importance
```

**Response:**
```json
{
  "model_run_id": "123e4567-e89b-12d3-a456-426614174002",
  "model_type": "random_forest_classifier",
  "task_type": "classification",
  "has_feature_importance": true,
  "feature_importance": [
    {
      "feature": "sepal_length",
      "importance": 0.35,
      "rank": 1
    },
    {
      "feature": "petal_length",
      "importance": 0.30,
      "rank": 2
    },
    {
      "feature": "sepal_width",
      "importance": 0.25,
      "rank": 3
    },
    {
      "feature": "petal_width",
      "importance": 0.10,
      "rank": 4
    }
  ],
  "feature_importance_dict": {
    "sepal_length": 0.35,
    "petal_length": 0.30,
    "sepal_width": 0.25,
    "petal_width": 0.10
  },
  "total_features": 4,
  "top_features": [
    {
      "feature": "sepal_length",
      "importance": 0.35,
      "rank": 1
    },
    {
      "feature": "petal_length",
      "importance": 0.30,
      "rank": 2
    },
    {
      "feature": "sepal_width",
      "importance": 0.25,
      "rank": 3
    }
  ],
  "importance_method": "feature_importances_",
  "message": null
}
```

### Example 2: Get Top 5 Features

**Request:**
```bash
GET /api/v1/models/train/123e4567-e89b-12d3-a456-426614174002/feature-importance?top_n=5
```

**Response:**
```json
{
  "model_run_id": "123e4567-e89b-12d3-a456-426614174002",
  "model_type": "xgboost_classifier",
  "task_type": "classification",
  "has_feature_importance": true,
  "feature_importance": [...],
  "feature_importance_dict": {...},
  "total_features": 20,
  "top_features": [
    {"feature": "feature_1", "importance": 0.45, "rank": 1},
    {"feature": "feature_5", "importance": 0.30, "rank": 2},
    {"feature": "feature_3", "importance": 0.15, "rank": 3},
    {"feature": "feature_7", "importance": 0.05, "rank": 4},
    {"feature": "feature_2", "importance": 0.03, "rank": 5}
  ],
  "importance_method": "feature_importances_",
  "message": null
}
```

### Example 3: Model Without Feature Importance

**Request:**
```bash
GET /api/v1/models/train/abc-123-def-456/feature-importance
```

**Response:**
```json
{
  "model_run_id": "abc-123-def-456",
  "model_type": "knn_classifier",
  "task_type": "classification",
  "has_feature_importance": false,
  "feature_importance": null,
  "feature_importance_dict": null,
  "total_features": 0,
  "top_features": null,
  "importance_method": null,
  "message": "Model type 'knn_classifier' does not support feature importance"
}
```

## Error Responses

### 400 Bad Request - Invalid UUID
```json
{
  "detail": "Invalid model_run_id format: invalid-uuid"
}
```

### 400 Bad Request - Training Not Completed
```json
{
  "detail": "Model run is not completed yet. Current status: running"
}
```

### 403 Forbidden - Unauthorized Access
```json
{
  "detail": "You don't have permission to access this model run"
}
```

### 404 Not Found - Model Run Not Found
```json
{
  "detail": "Model run with id 123e4567-e89b-12d3-a456-426614174002 not found"
}
```

## Feature Importance Calculation Methods

### 1. Tree-Based Models (feature_importances_)

Tree-based models calculate feature importance based on how much each feature decreases impurity (Gini impurity for classification, variance for regression) across all trees.

**Interpretation:**
- Higher values = more important features
- Values sum to 1.0
- Based on how often and how effectively a feature is used for splitting

**Example Models:** Random Forest, XGBoost, LightGBM

### 2. Linear Models (coef_)

Linear models use the absolute value of coefficients as feature importance. For multi-class classification, the mean of absolute coefficients across all classes is used.

**Interpretation:**
- Higher absolute values = more important features
- Positive coefficients = positive correlation with target
- Negative coefficients = negative correlation with target

**Example Models:** Linear Regression, Logistic Regression, Ridge, Lasso

### 3. Support Vector Machines (coef_)

For linear kernel SVMs, the coefficients of the hyperplane are used as feature importance.

**Interpretation:**
- Similar to linear models
- Only available for linear kernel
- RBF and polynomial kernels don't provide direct feature importance

## Use Cases

### 1. Feature Selection
Identify and remove low-importance features to:
- Reduce model complexity
- Improve training speed
- Reduce overfitting
- Lower data collection costs

### 2. Model Interpretation
Understand which features drive predictions:
- Validate domain knowledge
- Discover unexpected patterns
- Build trust in model decisions
- Communicate results to stakeholders

### 3. Feature Engineering
Guide feature engineering efforts:
- Focus on improving important features
- Create interactions between important features
- Remove or transform unimportant features

### 4. Debugging
Detect potential issues:
- Data leakage (unexpected important features)
- Missing important features
- Incorrect feature encoding
- Model relying on noise

## Best Practices

### 1. Compare Across Models
Feature importance can vary between model types. Compare importance across different models to get robust insights.

### 2. Consider Feature Correlation
Highly correlated features may split importance between them. Use correlation analysis alongside feature importance.

### 3. Validate with Domain Knowledge
Feature importance should align with domain expertise. Unexpected results may indicate data issues or new insights.

### 4. Use with Caution for Correlated Features
When features are highly correlated, importance may be distributed arbitrarily between them.

### 5. Combine with Other Interpretability Methods
Use alongside:
- SHAP values (Shapley Additive Explanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Partial Dependence Plots
- Permutation Importance

## Integration with Frontend

### Visualization Recommendations

1. **Bar Chart**: Most common visualization
   - X-axis: Feature names
   - Y-axis: Importance scores
   - Sort by importance (descending)

2. **Horizontal Bar Chart**: Better for many features
   - Y-axis: Feature names
   - X-axis: Importance scores
   - Show top N features

3. **Pie Chart**: Show relative importance
   - Each slice = feature
   - Size = importance score

4. **Table**: Detailed view
   - Columns: Rank, Feature, Importance, Percentage
   - Sortable and searchable

### Example Frontend Code (React)

```typescript
import { useEffect, useState } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from 'recharts';

interface FeatureImportanceData {
  model_run_id: string;
  has_feature_importance: boolean;
  feature_importance: Array<{
    feature: string;
    importance: number;
    rank: number;
  }>;
  top_features: Array<{
    feature: string;
    importance: number;
    rank: number;
  }>;
}

function FeatureImportanceChart({ modelRunId }: { modelRunId: string }) {
  const [data, setData] = useState<FeatureImportanceData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get(`/api/v1/models/train/${modelRunId}/feature-importance?top_n=10`)
      .then(response => {
        setData(response.data);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error fetching feature importance:', error);
        setLoading(false);
      });
  }, [modelRunId]);

  if (loading) return <div>Loading...</div>;
  if (!data?.has_feature_importance) return <div>Feature importance not available</div>;

  return (
    <div>
      <h3>Top 10 Most Important Features</h3>
      <BarChart width={600} height={400} data={data.top_features}>
        <XAxis dataKey="feature" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="importance" fill="#8884d8" />
      </BarChart>
    </div>
  );
}
```

## Notes

- Feature importance is calculated during model training and stored in `ModelRun.run_metadata`
- The endpoint returns cached importance scores; it does not recalculate them
- For models without native feature importance, consider using permutation importance (future enhancement)
- Feature importance is relative within a model; absolute values are not comparable across different models

## Future Enhancements

- [ ] Permutation importance for all model types
- [ ] SHAP values integration
- [ ] Feature importance visualization generation
- [ ] Feature importance comparison across multiple runs
- [ ] Export feature importance to CSV/JSON
- [ ] Feature importance trends over time

---

# Delete Model Run Endpoint

## Overview

The delete model run endpoint (`DELETE /api/v1/models/train/{model_run_id}`) provides a comprehensive way to remove model runs and their associated resources from the system.

## Endpoint

```
DELETE /api/v1/models/train/{model_run_id}
```

## Features

### 1. Permission Verification
- Verifies the model run exists
- Confirms user owns the model run through experiment ownership
- Returns 403 Forbidden if user doesn't have permission

### 2. Task Revocation
- Checks if a Celery task is still running
- Revokes the task with SIGTERM signal if in PENDING, STARTED, RETRY, or PROGRESS state
- Continues with deletion even if task revocation fails (logs warning)

### 3. Artifact Deletion
- Deletes the model artifact file (.joblib)
- Deletes associated metadata files (_config.json, _metadata.json)
- Uses ModelSerializationService for safe file deletion
- Continues with database deletion even if file deletion fails (logs error)

### 4. Database Record Deletion
- Removes the ModelRun record from the database
- Commits the transaction
- Rolls back and returns 500 error if database deletion fails

### 5. Detailed Response
- Returns comprehensive deletion summary
- Indicates which operations succeeded
- Includes timestamp for audit purposes

## Request

### Path Parameters
- `model_run_id` (string, required): UUID of the model run to delete

### Headers
- Authentication headers (when implemented)

### Example Request
```bash
curl -X DELETE "http://localhost:8000/api/v1/models/train/123e4567-e89b-12d3-a456-426614174002" \
  -H "Authorization: Bearer <token>"
```

## Response

### Success Response (200 OK)

```json
{
  "message": "Model run deleted successfully",
  "deletion_summary": {
    "model_run_id": "123e4567-e89b-12d3-a456-426614174002",
    "model_type": "random_forest_classifier",
    "status": "completed",
    "task_revoked": false,
    "artifact_deleted": true,
    "database_record_deleted": true
  },
  "timestamp": "2025-12-29T10:00:00Z"
}
```

### Deletion Summary Fields

- `model_run_id`: UUID of the deleted model run
- `model_type`: Type of model that was trained
- `status`: Status of the model run at deletion time
- `task_revoked`: Whether a running Celery task was revoked
- `artifact_deleted`: Whether the model artifact file was deleted
- `database_record_deleted`: Whether the database record was deleted

### Error Responses

#### 400 Bad Request
Invalid model_run_id format
```json
{
  "detail": "Invalid model_run_id format: invalid-uuid"
}
```

#### 404 Not Found
Model run doesn't exist
```json
{
  "detail": "Model run with id 123e4567-e89b-12d3-a456-426614174002 not found"
}
```

#### 403 Forbidden
User doesn't own the model run
```json
{
  "detail": "You don't have permission to delete this model run"
}
```

#### 500 Internal Server Error
Database deletion failed
```json
{
  "detail": "Failed to delete model run: <error message>"
}
```

## Use Cases

### 1. Delete Completed Model Run
User wants to clean up old experiments and free storage space.

```bash
DELETE /api/v1/models/train/abc-123
```

Response indicates artifact and database record deleted, task not revoked (already completed).

### 2. Cancel Running Training
User realizes they configured training incorrectly and wants to stop it.

```bash
DELETE /api/v1/models/train/def-456
```

Response indicates task revoked, artifact may not exist yet, database record deleted.

### 3. Delete Failed Model Run
User wants to remove a failed training attempt.

```bash
DELETE /api/v1/models/train/ghi-789
```

Response indicates database record deleted, artifact may or may not exist depending on when failure occurred.

## Behavior by Model Run Status

| Status | Task Revoked | Artifact Deleted | Notes |
|--------|--------------|------------------|-------|
| pending | Yes (if task exists) | No (not created yet) | Stops training before it starts |
| running | Yes | Maybe (depends on progress) | Terminates active training |
| completed | No (already done) | Yes | Clean deletion of finished run |
| failed | No (already done) | Maybe (depends on failure point) | Cleanup of failed attempt |
| cancelled | No (already done) | Maybe | Cleanup of previously cancelled run |

## Error Handling

The endpoint implements graceful degradation:

1. **Task Revocation Failure**: Logs warning, continues with deletion
2. **Artifact Deletion Failure**: Logs error, continues with database deletion
3. **Database Deletion Failure**: Rolls back, returns 500 error (critical failure)

This ensures that even if some cleanup operations fail, the system attempts to complete as much as possible.

## Logging

All operations are logged with structured logging:

- `model_run_delete_start`: Deletion request received
- `task_revoked`: Celery task successfully revoked
- `task_revoke_failed`: Failed to revoke task (warning)
- `artifact_deleted`: Model artifact successfully deleted
- `artifact_not_found`: Artifact already deleted or never created (warning)
- `artifact_delete_failed`: Failed to delete artifact (error)
- `model_run_deleted`: Database record successfully deleted
- `database_delete_failed`: Failed to delete from database (error)

## Security Considerations

1. **Ownership Verification**: Always verifies user owns the experiment before allowing deletion
2. **UUID Validation**: Validates model_run_id format before querying database
3. **Audit Trail**: Logs all deletion operations with user_id and timestamps
4. **Graceful Failure**: Doesn't expose internal errors to users

## Integration with Other Services

### ModelSerializationService
Uses `delete_model()` method to remove artifacts and metadata files.

### Celery
Uses `AsyncResult.revoke()` to terminate running tasks.

### Database
Uses SQLAlchemy ORM for transactional deletion with rollback support.

## Future Enhancements

Potential improvements:
- Soft delete with retention period
- Bulk deletion of multiple model runs
- Cascade deletion of related resources
- Async deletion for large artifacts
- Deletion confirmation requirement for completed runs
- Restore deleted model runs (if soft delete implemented)
- Webhook notifications on deletion

---

# Related Endpoints

For complete API documentation, see also:

- `POST /api/v1/models/train` - Start a new model training run
- `GET /api/v1/models/train/{model_run_id}/status` - Get training status
- `GET /api/v1/models/train/{model_run_id}/result` - Get complete training results
- `GET /api/v1/experiments/{experiment_id}/runs` - List all runs for an experiment
