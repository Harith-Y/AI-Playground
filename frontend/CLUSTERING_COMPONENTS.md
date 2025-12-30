# Clustering Evaluation & Visualizations

## Overview

This document describes the clustering evaluation and visualization components implemented for FE-46. These components provide comprehensive metrics and visual analysis for clustering models.

## Components

### 1. ClusteringCharts (Container)

**Location:** `frontend/src/components/evaluation/charts/ClusteringCharts.tsx`

**Purpose:** Main container component that orchestrates all clustering visualizations.

**Props:**

```typescript
interface ClusteringChartsProps {
  metrics: ClusteringMetrics; // Required clustering metrics
  projectionData?: {
    // Optional 2D projection data
    x: number[];
    y: number[];
    labels: number[];
  };
  isLoading?: boolean; // Loading state
  error?: string; // Error message
  showSilhouette?: boolean; // Show silhouette chart (default: true)
  showInertia?: boolean; // Show inertia chart (default: true)
  showDistribution?: boolean; // Show distribution chart (default: true)
  showProjection?: boolean; // Show 2D projection (default: true)
}
```

**Features:**

- Displays 2x2 grid of clustering visualizations
- Handles loading and error states
- Validates metric data before rendering
- Shows additional metrics (Davies-Bouldin, Calinski-Harabasz) in info alert

### 2. SilhouetteScorePlot

**Location:** `frontend/src/components/evaluation/charts/SilhouetteScorePlot.tsx`

**Purpose:** Displays the silhouette coefficient metric with visual indicators.

**Props:**

```typescript
interface SilhouetteScorePlotProps {
  silhouetteScore: number; // Silhouette coefficient (-1 to 1)
  nClusters: number; // Number of clusters
}
```

**Features:**

- Large numeric display of silhouette score
- Color-coded quality indicator:
  - 🟢 **Excellent**: ≥ 0.7 (Green)
  - 🔵 **Good**: ≥ 0.5 (Blue)
  - 🟠 **Fair**: ≥ 0.3 (Orange)
  - 🔴 **Poor**: < 0.3 (Red)
- Linear progress bar showing score on -1 to 1 scale
- Quality label (Excellent/Good/Fair/Poor)
- Educational tooltip and description

**Interpretation:**

- **1.0**: Perfect clustering (samples well-matched to clusters)
- **0.7-1.0**: Strong, well-defined clusters
- **0.5-0.7**: Reasonable structure, acceptable separation
- **0.3-0.5**: Weak structure, clusters overlap
- **< 0.3**: Poor clustering, consider different k or algorithm
- **< 0**: Samples likely assigned to wrong clusters

### 3. InertiaPlot

**Location:** `frontend/src/components/evaluation/charts/InertiaPlot.tsx`

**Purpose:** Displays the Within-Cluster Sum of Squares (WCSS/Inertia) metric.

**Props:**

```typescript
interface InertiaPlotProps {
  inertia: number; // WCSS value
  nClusters: number; // Number of clusters
  previousInertia?: number; // Optional previous inertia for comparison
}
```

**Features:**

- Large numeric display with K/M suffixes for readability
- Improvement percentage badge (if previousInertia provided)
- Visual compactness indicator
- Raw inertia value display
- Educational tooltip

**Interpretation:**

- **Lower is better**: Indicates tighter, more compact clusters
- Use with **Elbow Method**: Plot inertia vs. k to find optimal cluster count
- Decreases as k increases (trade-off with complexity)
- Look for "elbow point" where improvement plateaus

### 4. ClusterDistributionPlot

**Location:** `frontend/src/components/evaluation/charts/ClusterDistributionPlot.tsx`

**Purpose:** Shows how samples are distributed across clusters as a bar chart.

**Props:**

```typescript
interface ClusterDistributionPlotProps {
  clusterSizes: number[]; // Array of sample counts per cluster
  nClusters: number; // Number of clusters
}
```

**Features:**

- Interactive Plotly bar chart
- Percentage labels on bars
- Hover tooltips with exact counts
- Imbalance detection warning:
  - Warns if any cluster has >50% or <5% of samples
  - Yellow alert for imbalanced distributions
- Summary statistics:
  - Total samples
  - Number of clusters
  - Average samples per cluster

**Interpretation:**

- **Balanced**: Similar sizes indicate natural groupings
- **Imbalanced**: May indicate outlier clusters or need for different k
- Very small clusters (<5%) might be noise or outliers
- Very large clusters (>50%) might need to be split

### 5. ClusterProjection2D

**Location:** `frontend/src/components/evaluation/charts/ClusterProjection2D.tsx`

**Purpose:** Visualizes clusters in 2D space using dimensionality reduction.

**Props:**

```typescript
interface ClusterProjection2DProps {
  data: {
    x: number[]; // X coordinates
    y: number[]; // Y coordinates
    labels: number[]; // Cluster assignments
  };
  nClusters: number; // Number of clusters
  method?: "PCA" | "t-SNE" | "UMAP" | "Other"; // Reduction method
}
```

**Features:**

- Interactive Plotly scatter plot
- Color-coded clusters (10 base colors, generates more if needed)
- Hover tooltips showing cluster and coordinates
- Method badge (PCA, t-SNE, UMAP)
- Zoom, pan, and selection tools
- Legend with cluster names

**Interpretation:**

- **Well-separated**: Good visual separation indicates quality clustering
- **Overlapping**: May indicate ambiguous cluster boundaries
- **Outliers**: Points far from any cluster
- Note: 2D projection may not capture all dimensions - use with other metrics

## Usage Examples

### Basic Usage (EvaluationPage)

```typescript
import { ClusteringCharts } from "../components/evaluation/charts";

// In your component
<ClusteringCharts
  metrics={clusteringMetrics}
  isLoading={false}
  showSilhouette={true}
  showInertia={true}
  showDistribution={true}
  showProjection={false}
/>;
```

### With 2D Projection Data

```typescript
const projectionData = {
  x: [1.2, 2.3, 1.5, ...],
  y: [3.4, 2.1, 3.8, ...],
  labels: [0, 1, 0, ...]
};

<ClusteringCharts
  metrics={clusteringMetrics}
  projectionData={projectionData}
  showProjection={true}
/>
```

### Individual Components

You can also use components individually:

```typescript
import {
  SilhouetteScorePlot,
  InertiaPlot,
  ClusterDistributionPlot
} from '../components/evaluation/charts';

<SilhouetteScorePlot
  silhouetteScore={0.65}
  nClusters={3}
/>

<InertiaPlot
  inertia={1234.56}
  nClusters={3}
/>

<ClusterDistributionPlot
  clusterSizes={[150, 200, 175]}
  nClusters={3}
/>
```

## Integration Points

### 1. EvaluationPage

**File:** `frontend/src/pages/EvaluationPage.tsx`

The clustering tab now displays:

1. MetricsDisplay with numeric metrics
2. ClusteringCharts with all visualizations

```typescript
<TabPanel value={EvaluationTabEnum.CLUSTERING} currentValue={currentTab}>
  <MetricsDisplay
    taskType={TaskTypeEnum.CLUSTERING}
    metrics={clusteringRuns[0]?.metrics}
  />

  <ClusteringCharts metrics={clusteringRuns[0].metrics} isLoading={isLoading} />
</TabPanel>
```

### 2. MetricsExamplePage

**File:** `frontend/src/pages/MetricsExamplePage.tsx`

Shows example clustering metrics and visualizations with mock data.

## Data Requirements

### ClusteringMetrics Type

```typescript
interface ClusteringMetrics {
  silhouetteScore: number; // Required: -1 to 1
  inertia?: number; // Optional: WCSS value
  daviesBouldinScore?: number; // Optional: DB index (lower is better)
  calinskiHarabaszScore?: number; // Optional: CH score (higher is better)
  nClusters: number; // Required: number of clusters
  clusterSizes?: number[]; // Optional: samples per cluster
  clusterCenters?: number[][]; // Optional: cluster centroids
}
```

### Minimum Required Data

To display charts, you need:

- `silhouetteScore` (required)
- `nClusters` (required)
- `clusterSizes` (required for distribution chart)
- `inertia` (required for inertia chart)

## Clustering Metrics Reference

### Silhouette Coefficient

- **Range:** -1 to 1
- **Optimal:** Close to 1
- **Interpretation:**
  - Measures how similar samples are to their own cluster vs. other clusters
  - Considers both cohesion and separation
  - Per-sample metric averaged across all samples

### Inertia (WCSS)

- **Range:** 0 to ∞
- **Optimal:** Lower is better
- **Interpretation:**
  - Sum of squared distances from samples to cluster centers
  - Always decreases with more clusters
  - Use Elbow Method to find optimal k

### Davies-Bouldin Index

- **Range:** 0 to ∞
- **Optimal:** Lower is better
- **Interpretation:**
  - Average similarity between each cluster and its most similar cluster
  - Considers both separation and compactness
  - Zero means perfectly separated clusters

### Calinski-Harabasz Score (Variance Ratio)

- **Range:** 0 to ∞
- **Optimal:** Higher is better
- **Interpretation:**
  - Ratio of between-cluster to within-cluster variance
  - Higher values indicate well-defined clusters
  - Also known as Variance Ratio Criterion

## Styling & Theme

All components follow the existing design system:

- MUI components and styling
- Consistent color palette
- Responsive grid layouts
- Loading and error states
- Tooltips and help text

### Color Scheme

**Chart Colors:**

- Primary: #3b82f6 (Blue)
- Success: #22c55e (Green)
- Warning: #f59e0b (Orange)
- Error: #ef4444 (Red)
- Purple: #8b5cf6
- Pink: #ec4899
- Teal: #14b8a6

**Quality Indicators:**

- Excellent: Green (#22c55e)
- Good: Blue (#3b82f6)
- Fair: Orange (#f59e0b)
- Poor: Red (#ef4444)

## Future Enhancements

Potential improvements:

1. **Elbow Plot**: Multi-k inertia comparison chart
2. **Dendrogram**: Hierarchical clustering visualization
3. **Silhouette Plot**: Per-sample silhouette scores
4. **3D Projection**: Interactive 3D scatter plot
5. **Cluster Profiling**: Feature importance per cluster
6. **Comparison Mode**: Compare different clustering runs
7. **Export**: Download charts as images
8. **Animation**: Cluster evolution over iterations

## Testing

To test the components:

1. **With Mock Data** (MetricsExamplePage):

   - Navigate to `/metrics-example`
   - Click "Clustering" tab
   - View pre-populated charts with mock data

2. **With Real Data** (EvaluationPage):
   - Train a clustering model
   - Navigate to `/evaluation`
   - Click "Clustering" tab
   - View charts with actual model metrics

## Dependencies

- **react**: ^18.3.1
- **@mui/material**: ^7.0.0
- **react-plotly.js**: Chart rendering
- **@mui/icons-material**: Icons

## Files Created/Modified

### New Files

1. `frontend/src/components/evaluation/charts/ClusteringCharts.tsx`
2. `frontend/src/components/evaluation/charts/SilhouetteScorePlot.tsx`
3. `frontend/src/components/evaluation/charts/InertiaPlot.tsx`
4. `frontend/src/components/evaluation/charts/ClusterDistributionPlot.tsx`
5. `frontend/src/components/evaluation/charts/ClusterProjection2D.tsx`

### Modified Files

1. `frontend/src/components/evaluation/charts/index.ts` - Added exports
2. `frontend/src/pages/EvaluationPage.tsx` - Integrated ClusteringCharts
3. `frontend/src/pages/MetricsExamplePage.tsx` - Added clustering examples

### Type Definitions

- Uses existing `ClusteringMetrics` from `frontend/src/types/evaluation.ts`
- No type changes required

## Support

For issues or questions:

1. Check this README for usage examples
2. Review component prop documentation
3. Test with MetricsExamplePage first
4. Verify ClusteringMetrics data format

---

**Implementation:** FE-46: Clustering evaluation & visualizations  
**Status:** ✅ Complete  
**Date:** 2024
