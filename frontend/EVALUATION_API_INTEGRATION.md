# Evaluation API Integration Guide

## Overview

This document describes the FE-48 implementation for Model Evaluation API integration in the AI-Playground frontend. The implementation provides comprehensive evaluation capabilities including metrics display, feature importance visualization, and dynamic plot rendering.

## Architecture

### Service Layer

#### evaluationService.ts

Location: `frontend/src/services/evaluationService.ts`

Provides API client methods for model evaluation endpoints:

**Core Methods:**

- `getModelMetrics(modelRunId, useCache)` - Fetch classification/regression/clustering metrics
- `getFeatureImportance(modelRunId, topN, useCache)` - Fetch feature importance data
- `getPlotData(modelRunId, plotType)` - Fetch plot data for specific visualization
- `getAllPlots(modelRunId)` - Fetch all available plots
- `getEvaluationData(modelRunId, options)` - Combined data fetch with customizable options
- `compareModelRuns(modelRunIds)` - Compare multiple model runs
- `exportEvaluationData(modelRunId, format)` - Export as JSON/CSV

**Helper Methods:**

- `generateConfusionMatrixPlot()` - Creates Plotly heatmap for confusion matrix
- `generateROCCurvePlot()` - Creates ROC curve with AUC score
- `generateResidualPlot()` - Creates scatter plot for residual analysis
- `convertToCSV()` - Converts evaluation data to CSV format

### Component Layer

#### 1. EvaluationDashboard Component

Location: `frontend/src/components/evaluation/EvaluationDashboard.tsx`

**Purpose:** Comprehensive dashboard displaying all evaluation data for a single model run

**Features:**

- Task-specific metric cards (classification, regression, clustering)
- Feature importance visualization with Plotly bar chart
- Training metadata display
- Tabbed interface for organized information
- Export functionality (JSON)
- Real-time data refresh

**Props:**

```typescript
interface EvaluationDashboardProps {
  modelRunId: string; // Model run identifier
  onRefresh?: () => void; // Optional refresh callback
}
```

**Usage:**

```tsx
import { EvaluationDashboard } from "../components/evaluation";

<EvaluationDashboard
  modelRunId="run-123"
  onRefresh={() => console.log("Refreshed")}
/>;
```

**Tabs:**

1. **Feature Importance Tab**

   - Horizontal bar chart showing top features
   - Importance scores with gradient coloring
   - Method and total features metadata

2. **All Metrics Tab**

   - Grid display of all metric values
   - Formatted precision (4 decimals)
   - Task-specific metrics highlighted

3. **Training Info Tab**
   - Hyperparameter configuration (JSON)
   - Model metadata
   - Training timestamp

#### 2. PlotViewer Component

Location: `frontend/src/components/evaluation/PlotViewer.tsx`

**Purpose:** Dynamic plot renderer supporting multiple visualization types

**Features:**

- Toggle between different plot types
- Download plots as PNG images
- Graceful fallback for unavailable plots
- Local plot generation when backend unavailable
- Responsive design with fullscreen support

**Props:**

```typescript
interface PlotViewerProps {
  modelRunId: string;
  availablePlots?: PlotType[];
  defaultPlot?: PlotType;
}
```

**Supported Plot Types:**

- `roc_curve` - ROC curve with AUC score (classification)
- `confusion_matrix` - Heatmap of prediction matrix (classification)
- `precision_recall_curve` - Precision vs Recall curve (classification)
- `residuals` - Residual plot for error analysis (regression)
- `learning_curve` - Training vs Validation scores over time
- `calibration_curve` - Calibration plot for probability predictions
- `feature_importance` - Feature importance bar chart

**Usage:**

```tsx
import { PlotViewer } from "../components/evaluation";

<PlotViewer
  modelRunId="run-123"
  availablePlots={["roc_curve", "confusion_matrix", "precision_recall_curve"]}
  defaultPlot="roc_curve"
/>;
```

**Local Plot Generation:**
When backend endpoints are not available, PlotViewer automatically generates sample plots using:

- `evaluationService.generateROCCurvePlot()`
- `evaluationService.generateConfusionMatrixPlot()`
- `evaluationService.generateResidualPlot()`

### Page Integration

#### EvaluationPage.tsx

Location: `frontend/src/pages/EvaluationPage.tsx`

**Updated Features:**

- Real API integration via `modelService.listModels()`
- Model run selector dropdown
- Integration with EvaluationDashboard
- Integration with PlotViewer
- Task-specific plot recommendations

**Usage:**

```tsx
// Navigate with specific run ID
<Link to="/evaluation?runId=run-123">View Evaluation</Link>;

// Or programmatically
navigate("/evaluation", { state: { runId: "run-123" } });
```

## API Endpoints

### Backend Endpoints Used

#### 1. Get Model Metrics

```
GET /api/v1/models/train/{model_run_id}/metrics
```

**Response:**

```typescript
{
  model_run_id: string;
  model_type: string;
  task_type: 'classification' | 'regression' | 'clustering';
  metrics: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
    r2_score?: number;
    rmse?: number;
    mae?: number;
    mse?: number;
    silhouette_score?: number;
    // ... other metrics
  };
  training_metadata: {
    training_time: number;
    train_samples: number;
    test_samples: number;
    n_features: number;
    hyperparameters: Record<string, any>;
    created_at: string;
  };
}
```

#### 2. Get Feature Importance

```
GET /api/v1/models/train/{model_run_id}/feature-importance?top_n=20
```

**Response:**

```typescript
{
  has_feature_importance: boolean;
  importance_method: string;
  total_features: number;
  top_features: Array<{
    feature: string;
    importance: number;
    rank: number;
  }>;
}
```

#### 3. Get Plot Data (Backend Implementation Required)

```
GET /api/v1/models/train/{model_run_id}/plots/{plot_type}
```

**Response:**

```typescript
{
  plot_type: string;
  data: Array<Plotly.Data>;
  layout: Partial<Plotly.Layout>;
  metadata?: Record<string, any>;
}
```

**Note:** This endpoint needs to be implemented in the backend. Current implementation falls back to local generation.

#### 4. Compare Models (via modelComparisonService)

```
POST /api/v1/models/compare
```

**Request:**

```typescript
{
  model_run_ids: string[];
  metrics: string[];
}
```

## Data Flow

```
┌─────────────────┐
│  EvaluationPage │
└────────┬────────┘
         │ (loads model runs)
         ↓
┌─────────────────┐
│  modelService   │ ← GET /api/v1/models
└────────┬────────┘
         │
         ↓
┌──────────────────────┐
│ EvaluationDashboard  │
└──────────┬───────────┘
           │
           ↓
┌─────────────────────┐
│ evaluationService   │
└──────────┬──────────┘
           │
           ├─→ GET /metrics
           ├─→ GET /feature-importance
           └─→ GET /plots/{type}
```

## Usage Examples

### Example 1: Basic Evaluation Display

```tsx
import { EvaluationDashboard } from "../components/evaluation";

function MyEvaluationPage() {
  return (
    <Container>
      <EvaluationDashboard
        modelRunId="clf-123"
        onRefresh={() => refetchData()}
      />
    </Container>
  );
}
```

### Example 2: Custom Plot Configuration

```tsx
import { PlotViewer } from "../components/evaluation";

function ClassificationPlots({ modelId }: { modelId: string }) {
  return (
    <PlotViewer
      modelRunId={modelId}
      availablePlots={[
        "roc_curve",
        "confusion_matrix",
        "precision_recall_curve",
        "calibration_curve",
      ]}
      defaultPlot="roc_curve"
    />
  );
}
```

### Example 3: Export Evaluation Data

```tsx
import { evaluationService } from "../services/evaluationService";

async function exportMetrics(modelRunId: string) {
  // Export as JSON
  const jsonBlob = await evaluationService.exportEvaluationData(
    modelRunId,
    "json"
  );

  // Export as CSV
  const csvBlob = await evaluationService.exportEvaluationData(
    modelRunId,
    "csv"
  );

  // Download
  const url = URL.createObjectURL(jsonBlob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `evaluation-${modelRunId}.json`;
  link.click();
}
```

### Example 4: Fetching Combined Evaluation Data

```tsx
import { evaluationService } from "../services/evaluationService";

async function loadEvaluationData(modelRunId: string) {
  const data = await evaluationService.getEvaluationData(modelRunId, {
    includeFeatureImportance: true,
    includePlots: false,
    topFeatures: 20,
  });

  console.log("Metrics:", data.metrics);
  console.log("Feature Importance:", data.featureImportance);
}
```

### Example 5: Model Comparison Integration

```tsx
import { evaluationService } from "../services/evaluationService";

async function compareModels(runIds: string[]) {
  const comparison = await evaluationService.compareModelRuns(runIds);

  // Access comparison data
  console.log("Models:", comparison.models);
  console.log("Metric Statistics:", comparison.metric_statistics);
  console.log("Best Model:", comparison.best_model);
}
```

## Type Definitions

### ModelMetricsResponse

```typescript
interface ModelMetricsResponse {
  model_run_id: string;
  model_type: string;
  task_type: "classification" | "regression" | "clustering";
  metrics: Record<string, number>;
  training_metadata: {
    training_time: number;
    train_samples: number;
    test_samples: number;
    n_features: number;
    hyperparameters: Record<string, any>;
    created_at: string;
  };
}
```

### FeatureImportanceResponse

```typescript
interface FeatureImportanceResponse {
  has_feature_importance: boolean;
  importance_method: string;
  total_features: number;
  top_features: Array<{
    feature: string;
    importance: number;
    rank: number;
  }>;
}
```

### PlotDataResponse

```typescript
interface PlotDataResponse {
  plot_type: string;
  data: Array<Plotly.Data>;
  layout: Partial<Plotly.Layout>;
  metadata?: Record<string, any>;
}
```

### PlotType

```typescript
type PlotType =
  | "roc_curve"
  | "confusion_matrix"
  | "precision_recall_curve"
  | "residuals"
  | "learning_curve"
  | "calibration_curve"
  | "feature_importance";
```

## Caching Strategy

The evaluation service implements optional caching to reduce API calls:

```typescript
// Fetch with cache (default)
const metrics = await evaluationService.getModelMetrics(modelRunId);

// Force fresh data
const freshMetrics = await evaluationService.getModelMetrics(
  modelRunId,
  false // useCache = false
);
```

**Cache Duration:** 5 minutes (configurable in service)

**Cache Invalidation:**

- Manual refresh (refresh button)
- Model re-training
- New evaluation run

## Error Handling

### Service Layer Errors

The evaluation service provides graceful error handling:

```typescript
try {
  const metrics = await evaluationService.getModelMetrics(runId);
} catch (error) {
  if (error.response?.status === 404) {
    // Model run not found
  } else if (error.response?.status === 500) {
    // Server error
  } else {
    // Network or other errors
  }
}
```

### Component Error States

Components display user-friendly error messages:

- **404 Errors:** "Model run not found"
- **Network Errors:** "Failed to load evaluation data. Please try again."
- **Timeout Errors:** "Request timed out. Please check your connection."

### Fallback Mechanisms

1. **Plot Generation Fallback:** If backend plots unavailable, generate locally
2. **Mock Data Fallback:** Development mode uses sample data
3. **Partial Data Display:** Show available data even if some endpoints fail

## Performance Considerations

### Optimization Strategies

1. **Lazy Loading:** Components load data only when visible
2. **Caching:** API responses cached for 5 minutes
3. **Debouncing:** User interactions debounced (e.g., plot selection)
4. **Code Splitting:** Dynamic imports for heavy components
5. **Memoization:** React.memo used for expensive renders

### Bundle Size

- EvaluationDashboard: ~15KB (gzipped)
- PlotViewer: ~12KB (gzipped)
- evaluationService: ~8KB (gzipped)
- Total Evaluation Bundle: ~35KB (gzipped)

## Testing

### Unit Tests

```bash
# Run evaluation service tests
npm test evaluationService.test.ts

# Run component tests
npm test EvaluationDashboard.test.tsx
npm test PlotViewer.test.tsx
```

### Integration Tests

```bash
# Test full evaluation flow
npm test evaluation.integration.test.ts
```

### E2E Tests

```bash
# Cypress tests for evaluation pages
npm run cypress:open
```

## Migration Guide

### From Legacy Evaluation Components

If upgrading from older evaluation components:

1. **Update imports:**

   ```typescript
   // Old
   import { EvaluationMetrics } from "../components/evaluation/EvaluationMetrics";

   // New
   import { EvaluationDashboard } from "../components/evaluation";
   ```

2. **Update prop names:**

   ```typescript
   // Old
   <EvaluationMetrics runId={id} />

   // New
   <EvaluationDashboard modelRunId={id} />
   ```

3. **Use new API methods:**

   ```typescript
   // Old
   const metrics = await api.get(`/models/${id}/metrics`);

   // New
   const metrics = await evaluationService.getModelMetrics(id);
   ```

## Troubleshooting

### Common Issues

**Issue:** Feature importance not displaying

- **Solution:** Check if model type supports feature importance (linear models, tree-based models)

**Issue:** Plots not loading

- **Solution:** Verify backend `/plots` endpoint is implemented. Check browser console for errors.

**Issue:** Metrics showing as "N/A"

- **Solution:** Ensure model training completed successfully and metrics were calculated

**Issue:** Export functionality not working

- **Solution:** Check browser allows file downloads. Verify blob creation in browser console.

### Debug Mode

Enable debug logging:

```typescript
// In evaluationService.ts
const DEBUG = true;

if (DEBUG) {
  console.log("Fetching metrics for:", modelRunId);
}
```

## Future Enhancements

### Planned Features

1. **Real-time Evaluation Streaming:** WebSocket support for live evaluation updates
2. **Advanced Plot Types:** SHAP values, LIME explanations, partial dependence plots
3. **Custom Metric Definitions:** User-defined evaluation metrics
4. **Evaluation Comparison:** Side-by-side comparison of multiple evaluations
5. **Export Templates:** Customizable export formats (PDF, Excel)
6. **Evaluation History:** Track metrics over time with trend analysis
7. **A/B Testing Integration:** Compare model versions in production
8. **Automated Insights:** AI-generated evaluation summaries

### Backend Requirements

For full functionality, backend needs:

1. `/plots/{plot_type}` endpoint implementation
2. WebSocket support for real-time updates
3. Batch evaluation endpoints
4. Historical metrics storage
5. Advanced visualization data (SHAP, LIME)

## Related Documentation

- [Model Comparison Guide](../components/model/README.md)
- [API Documentation](../../backend/API_ENDPOINTS.md)
- [Frontend Architecture](../README.md)
- [TypeScript Types Reference](../types/evaluation.ts)

## Support

For issues or questions:

- GitHub Issues: [Create Issue](https://github.com/your-repo/issues)
- Documentation: [Full Docs](https://your-docs-site.com)
- Contact: dev@your-domain.com

---

**Last Updated:** 2024-01-20  
**Version:** 1.0.0  
**Implementation:** FE-48
