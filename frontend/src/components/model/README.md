# Model Comparison Feature

## Overview

The Model Comparison feature enables users to compare multiple ML/DL model runs side-by-side, analyzing their performance metrics, training characteristics, and receiving intelligent recommendations for model selection. This feature is essential for experiment tracking and making data-driven decisions about which model to deploy.

## 🎯 Key Features

### 1. **Comprehensive Comparison Analysis**

- Compare 2-10 models simultaneously
- Auto-detect task type (classification, regression, clustering)
- Auto-select appropriate metrics based on task
- Statistical summaries for each metric (mean, std, min, max)
- Best and worst model identification

### 2. **Visual Analytics**

- **Bar Charts**: Side-by-side metric comparison
- **Radar Charts**: Overall performance visualization
- **Scatter Plots**: Training time vs performance trade-offs
- **Box Plots**: Statistical distribution of metrics

### 3. **Intelligent Recommendations**

- Automated best model identification
- Performance vs efficiency insights
- Ensemble suggestions
- Statistical significance indicators

### 4. **Interactive Selection**

- Filter by model type, status, dataset
- Sort by any metric or training time
- Search by model name
- Multi-select with configurable limits (2-10 models)

## 📁 File Structure

```
frontend/src/
├── components/model/
│   ├── ModelComparisonView.tsx           # Legacy view (basic comparison)
│   ├── ModelComparisonViewEnhanced.tsx   # New enhanced view (with API integration)
│   ├── ModelListSelector.tsx             # Model selection interface
│   └── README.md                         # This file
├── pages/
│   └── ModelComparisonPage.tsx           # Main comparison page
├── services/
│   └── modelComparisonService.ts         # API service for comparison
└── types/
    └── modelComparison.ts                # TypeScript type definitions
```

## 🚀 Usage

### Basic Usage

```tsx
import ModelComparisonViewEnhanced from "./components/model/ModelComparisonViewEnhanced";

function MyComponent() {
  return (
    <ModelComparisonViewEnhanced
      modelRunIds={["model-id-1", "model-id-2", "model-id-3"]}
      comparisonMetrics={["accuracy", "f1_score", "precision"]}
      rankingCriteria="f1_score"
      onRefresh={() => console.log("Refreshing...")}
    />
  );
}
```

### Complete Page Implementation

```tsx
import { ModelComparisonPage } from "./pages/ModelComparisonPage";

// In your router
<Route path="/compare" element={<ModelComparisonPage />} />;

// Navigate with pre-selected models
navigate("/compare?models=model-1,model-2,model-3");
```

### Programmatic Comparison

```typescript
import { modelComparisonService } from "./services/modelComparisonService";

// Compare models
const result = await modelComparisonService.compareModels({
  model_run_ids: ["uuid-1", "uuid-2", "uuid-3"],
  comparison_metrics: ["accuracy", "f1_score"],
  ranking_criteria: "f1_score",
  include_statistical_tests: false,
});

console.log("Best model:", result.best_model);
console.log("Recommendations:", result.recommendations);
```

## 📊 Components

### ModelComparisonViewEnhanced

The main comparison component that displays comprehensive analysis.

**Props:**

```typescript
interface ModelComparisonViewEnhancedProps {
  modelRunIds: string[]; // 2-10 model run UUIDs
  comparisonMetrics?: string[]; // Optional metrics (auto-detected if not provided)
  rankingCriteria?: string; // Primary metric for ranking
  onRefresh?: () => void; // Callback when refresh is triggered
}
```

**Features:**

- Best Model Banner with key metrics
- Recommendations panel
- Statistical summary cards
- Interactive tabs:
  - Metrics Table with best/worst indicators
  - Bar Chart comparison
  - Radar Chart for overall performance
  - Scatter Plot for time vs performance

### ModelListSelector

Interactive model selection interface with filtering and sorting.

**Props:**

```typescript
interface ModelListSelectorProps {
  models: ModelComparisonData[]; // Available models
  selectedModels: string[]; // Currently selected model IDs
  onSelectionChange: (ids: string[]) => void;
  maxSelection?: number; // Default: 4
  loading?: boolean;
}
```

**Features:**

- Search by model name
- Filter by type, status, dataset, accuracy, training time
- Sort by any field
- Bulk select/deselect
- Visual selection state

### ModelComparisonPage

Complete page with selection, configuration, and comparison.

**Features:**

- Breadcrumb navigation
- Comparison options configuration
- Model selection interface
- Real-time comparison results
- Error handling with fallback to mock data

## 🔌 API Integration

### Endpoints Used

```typescript
// Compare models
POST /api/v1/models/compare
{
  "model_run_ids": ["uuid-1", "uuid-2"],
  "comparison_metrics": ["accuracy", "f1_score"],
  "ranking_criteria": "f1_score",
  "include_statistical_tests": false
}

// List model runs
GET /api/v1/models/runs?status=completed&limit=100

// Get model metrics
GET /api/v1/models/train/{model_run_id}/metrics
```

### Response Schema

```typescript
interface ModelComparisonResponse {
  comparison_id: string;
  task_type: string;
  total_models: number;
  compared_models: ModelComparisonItem[];
  best_model: ModelComparisonItem;
  metric_statistics: MetricStatistics[];
  ranking_criteria: string;
  recommendations: string[];
  timestamp: string;
}
```

## 🎨 Styling & Theming

The components use Material-UI (MUI) for consistent styling:

```tsx
// Custom theme integration
import { ThemeProvider } from "@mui/material";
import { myTheme } from "./theme";

<ThemeProvider theme={myTheme}>
  <ModelComparisonPage />
</ThemeProvider>;
```

**Color Coding:**

- 🟢 **Green** - Best performer
- 🔴 **Red** - Worst performer
- 🟣 **Purple** - Gradient for best model banner
- 🔵 **Blue** - Primary actions

## 📈 Charts & Visualizations

### Bar Chart

- **Purpose**: Compare individual metrics across models
- **Library**: Plotly.js
- **Interactive**: Yes (zoom, pan, hover)

### Radar Chart

- **Purpose**: Overall performance profile
- **Use Case**: Quick visual comparison of all metrics
- **Best For**: 3-6 models

### Scatter Plot

- **Purpose**: Efficiency analysis (time vs performance)
- **X-Axis**: Training Time
- **Y-Axis**: Ranking Score
- **Insight**: Identify fast and accurate models

### Box Plot

- **Purpose**: Statistical distribution
- **Shows**: Min, mean, max, std deviation
- **Use Case**: Understand metric variability

## ⚙️ Configuration Options

### Comparison Metrics

Specify which metrics to compare:

```typescript
comparisonMetrics={['accuracy', 'precision', 'recall', 'f1_score']}
```

Leave empty for auto-detection based on task type:

- **Classification**: accuracy, precision, recall, f1_score
- **Regression**: r2_score, rmse, mae, mse

### Ranking Criteria

Primary metric for ranking models:

```typescript
rankingCriteria = "f1_score";
```

Options: `accuracy`, `f1_score`, `precision`, `recall`, `r2_score`, `rmse`, etc.

### Max Selection

Limit how many models can be compared:

```typescript
<ModelListSelector maxSelection={10} />
```

Range: 2-10 models (backend limit)

## 🧪 Testing

### Unit Tests

```typescript
import { render, screen } from "@testing-library/react";
import ModelComparisonViewEnhanced from "./ModelComparisonViewEnhanced";

test("renders comparison view with models", () => {
  render(<ModelComparisonViewEnhanced modelRunIds={["id-1", "id-2"]} />);

  expect(screen.getByText(/Best Model/i)).toBeInTheDocument();
});
```

### Integration Tests

```typescript
import { modelComparisonService } from "./services/modelComparisonService";

test("fetches and compares models", async () => {
  const result = await modelComparisonService.compareModels({
    model_run_ids: ["uuid-1", "uuid-2"],
  });

  expect(result.total_models).toBe(2);
  expect(result.best_model).toBeDefined();
});
```

## 🐛 Error Handling

### Network Errors

- Automatic retry with visual feedback
- Fallback to cached results if available
- Mock data in development mode

### Invalid Inputs

- Client-side validation for 2-10 models
- Clear error messages for invalid requests
- Graceful degradation

### API Errors

```typescript
try {
  const result = await modelComparisonService.compareModels(request);
} catch (error) {
  if (error.response?.status === 400) {
    // Invalid request (e.g., incompatible models)
    showError("Models must have the same task type");
  } else if (error.response?.status === 404) {
    // Models not found
    showError("One or more models not found");
  } else {
    // Generic error
    showError("Failed to compare models");
  }
}
```

## 🚀 Performance Optimization

### Caching

- API responses cached for 30 minutes
- Automatic cache invalidation on updates
- Optional cache bypass: `use_cache=false`

### Lazy Loading

```tsx
import { lazy, Suspense } from "react";

const ModelComparisonViewEnhanced = lazy(
  () => import("./components/model/ModelComparisonViewEnhanced")
);

<Suspense fallback={<CircularProgress />}>
  <ModelComparisonViewEnhanced {...props} />
</Suspense>;
```

### Pagination

```typescript
// Load models in batches
const models = await modelComparisonService.listModelRuns(
  undefined,
  "completed",
  undefined,
  50, // limit
  0 // offset
);
```

## 📝 Best Practices

1. **Compare Similar Models**: Only compare models trained on the same dataset with the same task type
2. **Limit Selection**: Keep comparisons to 4-6 models for optimal visualization
3. **Custom Metrics**: Use custom metrics when you need specific evaluation criteria
4. **Ranking Criteria**: Choose ranking criteria that aligns with your business objective
5. **Statistical Significance**: Enable statistical tests for more confident decisions

## 🔮 Future Enhancements

- [ ] Export comparison results (PDF, CSV, PNG)
- [ ] Save comparison configurations
- [ ] Share comparisons via link
- [ ] Statistical significance tests
- [ ] Model versioning support
- [ ] Real-time collaborative comparison
- [ ] A/B testing integration
- [ ] Cost analysis (training cost vs performance)
- [ ] Model explainability comparison
- [ ] Production deployment comparison

## 📚 Related Documentation

- [Backend MODEL_COMPARISON_GUIDE.md](../../../backend/MODEL_COMPARISON_GUIDE.md)
- [API Endpoints Documentation](../../../backend/API_ENDPOINTS.md)
- [Type Definitions](../../types/modelComparison.ts)
- [Service Implementation](../../services/modelComparisonService.ts)

## 🤝 Contributing

When adding features to the comparison module:

1. Update type definitions in `types/modelComparison.ts`
2. Add service methods in `services/modelComparisonService.ts`
3. Update components with new features
4. Add tests for new functionality
5. Update this README with usage examples
6. Update backend documentation if API changes

## 📄 License

This feature is part of the AI-Playground project.

---

**Last Updated**: December 30, 2025  
**Version**: 1.0.0  
**Feature ID**: FE-47
