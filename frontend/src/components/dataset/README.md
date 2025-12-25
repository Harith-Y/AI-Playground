# Dataset Components

This directory contains reusable components for dataset management.

## Components

### 1. DatasetUpload

A drag-and-drop file upload component with validation and progress tracking.

**Props:**
- `onFileSelect: (file: File) => void` - Callback when file is selected
- `onUpload: () => void` - Callback when upload button is clicked
- `selectedFile: File | null` - Currently selected file
- `isLoading?: boolean` - Loading state (default: false)
- `uploadProgress?: number` - Upload progress 0-100 (default: 0)
- `error?: string | null` - Error message (default: null)

**Features:**
- Drag and drop file upload
- Click to browse files
- File type validation (CSV, XLSX, XLS, JSON)
- File size validation (max 100MB)
- Upload progress indicator
- Error handling and display

**Example:**
```tsx
import DatasetUpload from '../components/dataset/DatasetUpload';

function MyPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleUpload = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    // Upload logic here
    setIsUploading(false);
  };

  return (
    <DatasetUpload
      onFileSelect={setSelectedFile}
      onUpload={handleUpload}
      selectedFile={selectedFile}
      isLoading={isUploading}
      uploadProgress={50}
    />
  );
}
```

### 2. DatasetPreview

A table component for previewing dataset contents with column information.

**Props:**
- `preview?: any[][]` - Array of rows (array of values)
- `columns?: Column[]` - Array of column metadata
- `isLoading?: boolean` - Loading state (default: false)
- `error?: string | null` - Error message (default: null)
- `maxRows?: number` - Maximum rows to display (default: 10)

**Column Interface:**
```tsx
interface Column {
  name: string;
  dataType: string;
  nullCount: number;
  uniqueCount: number;
  sampleValues: any[];
}
```

**Features:**
- Collapsible table view
- Sticky header with scrolling
- Data type color coding
- Row numbering
- Null value styling
- Column metadata chips
- Footer statistics
- Custom scrollbar styling

**Example:**
```tsx
import DatasetPreview from '../components/dataset/DatasetPreview';

function MyPage() {
  const columns = [
    { name: 'age', dataType: 'int64', nullCount: 0, uniqueCount: 50, sampleValues: [25, 30, 35] },
    { name: 'name', dataType: 'object', nullCount: 2, uniqueCount: 98, sampleValues: ['John', 'Jane'] },
  ];

  const preview = [
    [25, 'John'],
    [30, 'Jane'],
    [35, 'Bob'],
  ];

  return (
    <DatasetPreview
      preview={preview}
      columns={columns}
      maxRows={10}
    />
  );
}
```

### 3. DatasetStats

A comprehensive statistics component with visual stat cards and data quality indicators.

**Props:**
- `stats?: DatasetStats` - Dataset statistics object
- `columns?: ColumnInfo[]` - Array of column metadata
- `isLoading?: boolean` - Loading state (default: false)
- `error?: string | null` - Error message (default: null)

**DatasetStats Interface:**
```typescript
{
  rowCount: number;
  columnCount: number;
  numericColumns: number;
  categoricalColumns: number;
  missingValues: number;
  duplicateRows: number;
  memoryUsage: number; // in bytes
}
```

**Features:**
- 7 color-coded stat cards with icons
- Total rows and columns
- Numeric vs categorical column breakdown
- Missing values with percentage
- Duplicate rows detection
- Memory usage display
- Column types breakdown with chips
- Data quality score with progress bar
- Responsive grid layout
- Loading and error states

**Example:**
```tsx
import DatasetStats from '../components/dataset/DatasetStats';

function MyPage() {
  const stats = {
    rowCount: 1000,
    columnCount: 10,
    numericColumns: 5,
    categoricalColumns: 3,
    missingValues: 15,
    duplicateRows: 2,
    memoryUsage: 80000,
  };

  const columns = [
    { name: 'age', dataType: 'int64', nullCount: 0, uniqueCount: 50, sampleValues: [25, 30, 35] },
    { name: 'name', dataType: 'object', nullCount: 2, uniqueCount: 98, sampleValues: ['John', 'Jane'] },
  ];

  return (
    <DatasetStats
      stats={stats}
      columns={columns}
    />
  );
}
```

### 4. VisualizationGallery

A comprehensive visualization gallery component for displaying EDA charts with filtering and fullscreen viewing.

**Props:**
- `visualizations: VisualizationData[]` - Array of visualization objects
- `isLoading?: boolean` - Loading state (default: false)
- `error?: string | null` - Error message (default: null)
- `onVisualizationSelect?: (viz: VisualizationData) => void` - Callback when visualization is selected
- `maxColumns?: number` - Maximum columns in grid (default: 3)

**VisualizationData Interface:**
```typescript
{
  type: 'histogram' | 'scatter' | 'box' | 'correlation' | 'missing';
  data: any; // Plotly data array
  title: string;
  xLabel?: string;
  yLabel?: string;
}
```

**Features:**
- Responsive grid/list view toggle
- Search functionality (by title or type)
- Filter by visualization type (histogram, scatter, box, correlation, missing)
- Color-coded cards by visualization type
- Individual card actions:
  - Fullscreen/expand view with interactive Plotly controls
  - Download visualization data as JSON
- Empty states for no data and no search results
- Loading and error states
- Plotly.js integration for interactive charts

**Visualization Type Colors:**
- **Blue (#3B82F6)**: Histogram
- **Green (#10B981)**: Scatter
- **Purple (#8B5CF6)**: Box
- **Orange (#F59E0B)**: Correlation
- **Red (#EF4444)**: Missing data

**Example:**
```tsx
import VisualizationGallery from '../components/dataset/VisualizationGallery';

function MyPage() {
  const visualizations = [
    {
      type: 'histogram',
      title: 'Age Distribution',
      data: [{ x: [25, 30, 35, 40, 45], type: 'histogram' }],
      xLabel: 'Age',
      yLabel: 'Frequency',
    },
    {
      type: 'scatter',
      title: 'Income vs Age',
      data: [{
        x: [25, 30, 35, 40, 45],
        y: [50000, 60000, 70000, 80000, 90000],
        mode: 'markers',
        type: 'scatter'
      }],
      xLabel: 'Age',
      yLabel: 'Income',
    },
    {
      type: 'correlation',
      title: 'Feature Correlation Matrix',
      data: [{
        z: [[1, 0.8, 0.3], [0.8, 1, 0.5], [0.3, 0.5, 1]],
        type: 'heatmap',
        colorscale: 'RdBu',
      }],
    },
  ];

  const handleSelect = (viz) => {
    console.log('Selected visualization:', viz.title);
  };

  return (
    <VisualizationGallery
      visualizations={visualizations}
      onVisualizationSelect={handleSelect}
      maxColumns={3}
    />
  );
}
```

## Data Type Color Coding

All dataset components use consistent color coding for different data types:

- **Blue (#3B82F6)**: Numeric types (int, float, number)
- **Green (#10B981)**: Text types (object, str, string)
- **Orange (#F59E0B)**: Boolean types
- **Purple (#8B5CF6)**: DateTime types
- **Gray (#6B7280)**: Other types

## Complete Usage Example

```tsx
import React, { useState } from 'react';
import { Box } from '@mui/material';
import DatasetUpload from '../components/dataset/DatasetUpload';
import DatasetPreview from '../components/dataset/DatasetPreview';

function DatasetPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [preview, setPreview] = useState<any[][]>([]);
  const [columns, setColumns] = useState<Column[]>([]);

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    try {
      // Upload file to API
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch('/api/v1/datasets/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      // Fetch preview data
      const previewResponse = await fetch(`/api/v1/datasets/${data.id}/preview`);
      const previewData = await previewResponse.json();

      setPreview(previewData.preview);
      setColumns(previewData.columns);
      setSelectedFile(null);
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <Box sx={{ p: 4 }}>
      <DatasetUpload
        onFileSelect={setSelectedFile}
        onUpload={handleUpload}
        selectedFile={selectedFile}
        isLoading={isUploading}
      />

      <Box sx={{ mt: 3 }}>
        <DatasetPreview
          preview={preview}
          columns={columns}
        />
      </Box>
    </Box>
  );
}

export default DatasetPage;
```
