# Frontend - AI-Playground

React + TypeScript frontend for the AI-Playground ML platform. Modern, responsive UI for dataset management, visualization, preprocessing pipeline building, and ML model training.

## 🏗️ Architecture

```
frontend/
├── src/
│   ├── main.tsx              # React entry point
│   ├── App.tsx               # Main application component
│   ├── components/           # React components
│   │   ├── common/          # Shared components (Layout, Header, Sidebar)
│   │   ├── dataset/         # Dataset components (Upload, Preview, Stats, Viz)
│   │   ├── features/        # Feature engineering components
│   │   ├── evaluation/      # Model evaluation components
│   │   └── modeling/        # Model training components
│   ├── pages/               # Page components (React Router)
│   ├── store/               # Redux state management
│   │   ├── index.ts        # Store configuration
│   │   └── slices/         # Redux slices
│   ├── services/            # API services (Axios)
│   ├── types/               # TypeScript type definitions
│   ├── utils/               # Utility functions
│   └── hooks/               # Custom React hooks
├── public/                  # Static assets
├── package.json            # Dependencies
├── vite.config.ts          # Vite configuration
└── tsconfig.json           # TypeScript configuration
```

## ✨ Features

### ✅ Implemented Components

#### Dataset Management
- **DatasetUpload** (236 lines) - File upload with drag-and-drop, progress tracking
- **DatasetPreview** (336 lines) - Interactive data table with MUI DataGrid
- **DatasetStats** (343 lines) - Statistical summary cards with detailed metrics
- **VisualizationGallery** (483 lines) - Interactive Plotly charts with grid/list view, search, filter, fullscreen
- **MissingValuesHeatmap** (359 lines) - Color-coded heatmap for missing data
- **MissingValuesAnalysis** (96 lines) - Missing data statistics and insights

#### Common Components
- **Layout** (67 lines) - Main app layout with header and sidebar
- **Header** (114 lines) - Top navigation with breadcrumbs
- **Sidebar** (232 lines) - Side navigation with collapsible menu
- **ErrorBoundary** (175 lines) - Error handling with fallback UI
- **LoadingState** (73 lines) - Loading indicators
- **EmptyState** (86 lines) - Empty state placeholders
- **ErrorState** (106 lines) - Error display with retry
- **LoadingSpinner** (66 lines) - Custom loading animations

#### Feature Engineering
- **CorrelationHeatmap** (292 lines) - Interactive correlation matrix visualization
- **CorrelationMatrix** (90 lines) - Correlation matrix display
- **FeatureImportance** (45 lines) - Feature importance charts

#### Model Evaluation
- **EvaluationMetrics** (59 lines) - Model performance metrics display
- **TrainingProgress** (129 lines) - Real-time training progress tracking

### ✅ Redux State Management

#### Dataset Slice (204 lines)
Complete CRUD operations with 6 async thunks:
- `uploadDataset` - Upload dataset file
- `fetchDatasets` - Get all datasets
- `fetchDataset` - Get single dataset
- `fetchDatasetStats` - Get dataset statistics
- `fetchDatasetPreview` - Get preview rows
- `deleteDataset` - Delete dataset

4 synchronous actions:
- `setCurrentDataset` - Set active dataset
- `clearDatasetError` - Clear error state
- `setUploadProgress` - Update upload progress
- `resetDatasetState` - Reset to initial state

**State Shape:**
```typescript
interface DatasetState {
  currentDataset: Dataset | null;
  datasets: Dataset[];
  stats: DatasetStats | null;
  columns: ColumnInfo[];
  preview: any[];
  isLoading: boolean;
  error: string | null;
  uploadProgress: number;
}
```

### ✅ Visualization Features

#### Distribution Plots (visualizations.ts - 420+ lines)
- **createHistogram** - Single column histogram with statistics
- **createMultipleHistograms** - Multiple histograms in subplot
- **createGroupedHistogram** - Grouped histogram by category
- **createHistogramWithNormal** - Histogram with normal distribution overlay
- **createCustomBinHistogram** - Custom bin sizes
- **createAutoBinnedHistogram** - Auto-binning (Sturges' or Freedman-Diaconis)
- **createHistogramWithKDE** - Histogram with KDE curve
- **createCumulativeHistogram** - Cumulative distribution
- **createLogHistogram** - Logarithmic scale histogram

**Statistical Calculations:**
- Mean, median, mode
- Standard deviation, variance
- Quartiles (Q1, Q3), IQR
- Skewness, kurtosis
- Normality test (Shapiro-Wilk)

#### Visualization Gallery
- **Grid/List View** - Toggle between grid and list layouts
- **Search** - Find visualizations by name
- **Filter** - Filter by type (histogram, scatter, box, correlation, missing)
- **Fullscreen** - Expand charts for detailed analysis
- **Download** - Export visualization data as JSON
- **Interactive Charts** - Plotly.js with zoom, pan, hover

### 🚧 Coming Soon

- **Preprocessing UI** - Drag-and-drop step builder
- **Model Training UI** - Model selection and configuration
- **Hyperparameter Tuning UI** - Grid search parameter builder
- **Code Generation UI** - Code preview and download
- **Experiment Comparison** - Side-by-side model comparison

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- npm or yarn

### Installation

```powershell
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
# or
yarn install
```

### Configuration

Create a `.env` file in the frontend directory:

```env
# API Configuration
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws

# Optional: Analytics
VITE_ANALYTICS_ID=your-analytics-id
```

### Development

```powershell
# Start development server (port 5173)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run linter
npm run lint

# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# Type check
npm run type-check
```

### Access Application

- **Development**: http://localhost:5173
- **Preview**: http://localhost:4173

## 📦 Tech Stack

### Core
- **React** 19.2.0 - UI library
- **TypeScript** 5.9.3 - Type safety
- **Vite** 7.2.4 - Build tool

### UI & Styling
- **Material-UI** 7.3.6 - Component library
- **@mui/x-data-grid** 9.1.0 - Data tables
- **@emotion/react** 11.15.0 - CSS-in-JS
- **@emotion/styled** 11.15.0 - Styled components

### State Management
- **Redux Toolkit** 2.11.2 - State management
- **React-Redux** 9.2.0 - React bindings

### Routing & HTTP
- **React Router** 7.11.0 - Client-side routing
- **Axios** 1.13.2 - HTTP client

### Visualization
- **Plotly.js** 3.3.1 - Interactive charts
- **react-plotly.js** 2.6.0 - React wrapper for Plotly

### Forms & Validation
- **React Hook Form** 7.69.0 - Form management
- **Yup** 1.7.1 - Schema validation

### Testing
- **Vitest** 4.0.16 - Test framework
- **@testing-library/react** 16.3.0 - Testing utilities
- **@testing-library/user-event** 14.6.1 - User interaction simulation

## 🎨 Component Library

### Common Components

#### Layout
```tsx
import { Layout } from './components/common/Layout';

function App() {
  return (
    <Layout>
      <YourContent />
    </Layout>
  );
}
```

#### ErrorBoundary
```tsx
import { ErrorBoundary } from './components/common/ErrorBoundary';

<ErrorBoundary>
  <YourComponent />
</ErrorBoundary>
```

#### LoadingState
```tsx
import { LoadingState } from './components/common/LoadingState';

{isLoading && <LoadingState message="Loading data..." />}
```

### Dataset Components

#### DatasetUpload
```tsx
import { DatasetUpload } from './components/dataset/DatasetUpload';

<DatasetUpload
  onUploadSuccess={(dataset) => console.log('Uploaded:', dataset)}
  onUploadError={(error) => console.error('Error:', error)}
/>
```

#### DatasetPreview
```tsx
import { DatasetPreview } from './components/dataset/DatasetPreview';

<DatasetPreview
  datasetId="uuid"
  data={previewData}
  columns={columns}
  isLoading={false}
/>
```

#### VisualizationGallery
```tsx
import { VisualizationGallery } from './components/dataset/VisualizationGallery';

<VisualizationGallery
  visualizations={visualizations}
  isLoading={false}
  error={null}
  onVisualizationSelect={(viz) => console.log(viz)}
  maxColumns={3}
/>
```

## 🔧 Redux Store Usage

### Basic Usage

```tsx
import { useAppDispatch, useAppSelector } from './store/hooks';
import { uploadDataset, fetchDatasets } from './store/slices/datasetSlice';

function MyComponent() {
  const dispatch = useAppDispatch();
  const { datasets, isLoading, error } = useAppSelector((state) => state.dataset);

  const handleUpload = async (file: File) => {
    try {
      const result = await dispatch(uploadDataset(file)).unwrap();
      console.log('Success:', result);
    } catch (err) {
      console.error('Error:', err);
    }
  };

  useEffect(() => {
    dispatch(fetchDatasets());
  }, [dispatch]);

  return (
    // Your component JSX
  );
}
```

### Custom Hooks Pattern

```tsx
// hooks/useDataset.ts
export function useDataset(datasetId?: string) {
  const dispatch = useAppDispatch();
  const state = useAppSelector((state) => state.dataset);

  useEffect(() => {
    if (datasetId) {
      dispatch(fetchDataset(datasetId));
      dispatch(fetchDatasetStats(datasetId));
    }
  }, [dispatch, datasetId]);

  const upload = async (file: File) => {
    return dispatch(uploadDataset(file)).unwrap();
  };

  const remove = async (id: string) => {
    return dispatch(deleteDataset(id)).unwrap();
  };

  return { ...state, upload, remove };
}
```

## 📊 Visualization Utilities

### Creating Histograms

```typescript
import { createHistogram, createHistogramWithNormal } from './utils/visualizations';

// Basic histogram
const histogram = createHistogram(
  data,
  'age',
  { bins: 20, color: '#1976d2', showMean: true, showMedian: true }
);

// Histogram with normal distribution overlay
const histWithNormal = createHistogramWithNormal(
  data,
  'salary',
  { bins: 30, color: '#2196f3' }
);
```

### Auto-binning

```typescript
import { createAutoBinnedHistogram } from './utils/visualizations';

// Sturges' rule (default)
const hist1 = createAutoBinnedHistogram(data, 'age', { method: 'sturges' });

// Freedman-Diaconis rule
const hist2 = createAutoBinnedHistogram(data, 'salary', { method: 'fd' });
```

## 🧪 Testing

### Component Tests

```tsx
import { render, screen } from '@testing-library/react';
import { DatasetStats } from './components/dataset/DatasetStats';

test('renders dataset statistics', () => {
  const stats = {
    rowCount: 1000,
    columnCount: 10,
    missingValues: 15
  };

  render(<DatasetStats stats={stats} />);

  expect(screen.getByText('1000')).toBeInTheDocument();
  expect(screen.getByText('10')).toBeInTheDocument();
});
```

### Redux Tests

```tsx
import { configureStore } from '@reduxjs/toolkit';
import datasetReducer from './store/slices/datasetSlice';

test('should handle dataset upload', () => {
  const store = configureStore({ reducer: { dataset: datasetReducer } });

  // Test your Redux logic
});
```

## 🎨 Theming

### MUI Theme Customization

```typescript
// styles/theme.ts
import { createTheme } from '@mui/material/styles';

export const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
  typography: {
    fontFamily: 'Inter, system-ui, Avenir, Helvetica, Arial, sans-serif',
  },
});
```

## 🔌 API Integration

### Dataset Service

```typescript
// services/datasetService.ts
import api from './api';

class DatasetService {
  async uploadDataset(file: File): Promise<Dataset> {
    return api.upload<Dataset>('/api/v1/datasets/upload', file);
  }

  async getDatasets(): Promise<Dataset[]> {
    return api.get<Dataset[]>('/api/v1/datasets');
  }

  async getDatasetStats(id: string): Promise<DatasetStatsResponse> {
    return api.get(`/api/v1/datasets/${id}/stats`);
  }
}
```

## 📁 Project Structure

```
src/
├── components/         # React components
│   ├── common/        # Shared components
│   ├── dataset/       # Dataset-specific components
│   ├── features/      # Feature engineering components
│   ├── evaluation/    # Evaluation components
│   └── modeling/      # Model training components
├── pages/             # Page components
│   ├── HomePage.tsx
│   ├── DatasetUploadPage.tsx
│   └── ExplorationPage.tsx
├── store/             # Redux store
│   ├── index.ts
│   ├── hooks.ts
│   └── slices/
│       └── datasetSlice.ts
├── services/          # API services
│   ├── api.ts
│   └── datasetService.ts
├── types/             # TypeScript types
│   ├── dataset.ts
│   └── index.ts
├── utils/             # Utilities
│   ├── visualizations.ts
│   └── constants.ts
└── hooks/             # Custom hooks
    └── useDataset.ts
```

## 🐛 Troubleshooting

### Common Issues

**Module not found errors**
```powershell
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Type errors with Plotly**
- Custom type declarations in `src/types/react-plotly.d.ts`
- Restart TS server if needed

**API connection errors**
- Verify VITE_API_URL in .env
- Check backend is running on port 8000
- CORS must be enabled on backend

**Build failures**
```powershell
# Clear Vite cache
rm -rf node_modules/.vite
npm run build
```

## 📚 Resources

- **[Main README](../README.md)** - Project overview
- **[Backend README](../backend/README.md)** - Backend API documentation
- **[Redux Store Docs](src/store/README.md)** - State management guide
- **[Material-UI Docs](https://mui.com/)** - Component library
- **[Plotly.js Docs](https://plotly.com/javascript/)** - Visualization library

## 🤝 Contributing

1. Create feature branch
2. Write tests for new components
3. Follow ESLint rules
4. Use TypeScript types
5. Update documentation
6. Submit pull request

## 📄 License

This project is for educational purposes.
