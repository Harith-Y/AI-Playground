# Feature Importance Endpoint - Quick Reference

## Endpoint
```
GET /api/v1/models/train/{model_run_id}/feature-importance
```

## Quick Examples

### Get all feature importance
```bash
curl http://localhost:8000/api/v1/models/train/abc-123/feature-importance
```

### Get top 5 features
```bash
curl http://localhost:8000/api/v1/models/train/abc-123/feature-importance?top_n=5
```

## Response Structure

```json
{
  "has_feature_importance": true,
  "feature_importance": [
    {"feature": "age", "importance": 0.35, "rank": 1},
    {"feature": "income", "importance": 0.30, "rank": 2}
  ],
  "top_features": [...],
  "total_features": 10,
  "importance_method": "feature_importances_"
}
```

## Supported Models

✅ **Tree-based:** Random Forest, XGBoost, LightGBM, CatBoost, Decision Tree, Gradient Boosting  
✅ **Linear:** Linear/Ridge/Lasso Regression, Logistic Regression  
✅ **SVM:** Support Vector Machines (linear kernel)  
❌ **Not supported:** KNN, Naive Bayes, Clustering

## Common Errors

| Code | Reason | Solution |
|------|--------|----------|
| 400 | Model not completed | Wait for training to finish |
| 403 | Unauthorized | Check user permissions |
| 404 | Model run not found | Verify model_run_id |

## Frontend Integration

```typescript
// React/TypeScript example
const response = await fetch(
  `/api/v1/models/train/${modelRunId}/feature-importance?top_n=10`
);
const data = await response.json();

if (data.has_feature_importance) {
  // Display feature importance chart
  renderChart(data.top_features);
} else {
  // Show message
  showMessage(data.message);
}
```

## Visualization Ideas

1. **Horizontal Bar Chart** - Best for many features
2. **Vertical Bar Chart** - Good for few features
3. **Pie Chart** - Show relative importance
4. **Table** - Detailed view with sorting

## Related Endpoints

- `GET /api/v1/models/train/{model_run_id}/metrics` - Model metrics
- `GET /api/v1/models/train/{model_run_id}/status` - Training status
- `GET /api/v1/models/train/{model_run_id}/result` - Full results

## Notes

- Feature importance is calculated during training
- Scores are relative within a model
- Higher score = more important feature
- Scores may not sum to 1.0 for all models
