# Redux Store Documentation

Redux state management for the AI-Playground frontend application.

## Overview

The Redux store manages global application state using Redux Toolkit with TypeScript. State is organized into feature-based slices.

## Store Structure

```
store/
├── index.ts              # Store configuration
├── hooks.ts              # Typed hooks (useAppDispatch, useAppSelector)
└── slices/
    ├── datasetSlice.ts   # Dataset state management
    ├── preprocessingSlice.ts
    ├── featureSlice.ts
    ├── modelingSlice.ts
    ├── evaluationSlice.ts
    └── tuningSlice.ts
```

## Dataset Slice

Complete CRUD operations for dataset management.

### State Shape

```typescript
interface DatasetState {
  currentDataset: Dataset | null;  // Currently selected dataset
  datasets: Dataset[];              // List of all datasets
  stats: DatasetStats | null;       // Statistics for current dataset
  columns: ColumnInfo[];            // Column metadata
  preview: any[];                   // Preview rows
  isLoading: boolean;               // Loading state
  error: string | null;             // Error message
  uploadProgress: number;           // Upload progress (0-100)
}
```

### Actions

#### Async Thunks (API calls)

**`uploadDataset(file: File)`**
- Uploads a dataset file to the backend
- Returns: `Dataset` object
- Updates: `currentDataset`, `datasets`, `uploadProgress`

```typescript
const result = await dispatch(uploadDataset(file)).unwrap();
```

**`fetchDatasets()`**
- Fetches all datasets for the current user
- Returns: `Dataset[]`
- Updates: `datasets`

```typescript
await dispatch(fetchDatasets());
```

**`fetchDataset(datasetId: string)`**
- Fetches a specific dataset by ID
- Returns: `Dataset`
- Updates: `currentDataset`, `datasets` (upsert)

```typescript
await dispatch(fetchDataset('dataset-id'));
```

**`fetchDatasetStats(datasetId: string)`**
- Fetches statistics for a dataset
- Returns: `{ stats: DatasetStats, columns: ColumnInfo[] }`
- Updates: `stats`, `columns`

```typescript
await dispatch(fetchDatasetStats('dataset-id'));
```

**`fetchDatasetPreview(datasetId: string)`**
- Fetches preview rows for a dataset
- Returns: `any[]` (2D array of values)
- Updates: `preview`

```typescript
await dispatch(fetchDatasetPreview('dataset-id'));
```

**`deleteDataset(datasetId: string)`**
- Deletes a dataset
- Returns: `string` (deleted dataset ID)
- Updates: `datasets` (removes), `currentDataset` (clears if deleted)

```typescript
await dispatch(deleteDataset('dataset-id')).unwrap();
```

#### Synchronous Actions

**`setCurrentDataset(dataset: Dataset | null)`**
- Manually set the current dataset

```typescript
dispatch(setCurrentDataset(dataset));
```

**`clearDatasetError()`**
- Clear error message

```typescript
dispatch(clearDatasetError());
```

**`setUploadProgress(progress: number)`**
- Update upload progress (0-100)

```typescript
dispatch(setUploadProgress(50));
```

**`resetDatasetState()`**
- Reset entire dataset state to initial values

```typescript
dispatch(resetDatasetState());
```

### Usage Examples

#### Basic Upload

```typescript
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { uploadDataset } from '../store/slices/datasetSlice';

function UploadComponent() {
  const dispatch = useAppDispatch();
  const { isLoading, error, uploadProgress } = useAppSelector(
    (state) => state.dataset
  );

  const handleUpload = async (file: File) => {
    try {
      const dataset = await dispatch(uploadDataset(file)).unwrap();
      console.log('Uploaded:', dataset);
    } catch (err) {
      console.error('Upload failed:', err);
    }
  };

  return (
    <>
      <input type="file" onChange={(e) => handleUpload(e.target.files[0])} />
      {isLoading && <progress value={uploadProgress} max={100} />}
      {error && <div>{error}</div>}
    </>
  );
}
```

#### Fetch and Display List

```typescript
import { useEffect } from 'react';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { fetchDatasets } from '../store/slices/datasetSlice';

function DatasetList() {
  const dispatch = useAppDispatch();
  const { datasets, isLoading } = useAppSelector((state) => state.dataset);

  useEffect(() => {
    dispatch(fetchDatasets());
  }, [dispatch]);

  if (isLoading) return <div>Loading...</div>;

  return (
    <ul>
      {datasets.map((dataset) => (
        <li key={dataset.id}>{dataset.name}</li>
      ))}
    </ul>
  );
}
```

#### Load Dataset with Details

```typescript
import { useEffect } from 'react';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  fetchDataset,
  fetchDatasetStats,
  fetchDatasetPreview,
} from '../store/slices/datasetSlice';

function DatasetDetails({ datasetId }: { datasetId: string }) {
  const dispatch = useAppDispatch();
  const { currentDataset, stats, preview } = useAppSelector(
    (state) => state.dataset
  );

  useEffect(() => {
    const loadDataset = async () => {
      await dispatch(fetchDataset(datasetId));
      await Promise.all([
        dispatch(fetchDatasetStats(datasetId)),
        dispatch(fetchDatasetPreview(datasetId)),
      ]);
    };

    loadDataset();
  }, [dispatch, datasetId]);

  return (
    <div>
      <h2>{currentDataset?.name}</h2>
      <p>Rows: {stats?.rowCount}</p>
      <p>Columns: {stats?.columnCount}</p>
      {/* Preview table */}
    </div>
  );
}
```

#### Delete with Confirmation

```typescript
import { useAppDispatch } from '../store/hooks';
import { deleteDataset, fetchDatasets } from '../store/slices/datasetSlice';

function DeleteButton({ datasetId }: { datasetId: string }) {
  const dispatch = useAppDispatch();

  const handleDelete = async () => {
    if (!confirm('Delete this dataset?')) return;

    try {
      await dispatch(deleteDataset(datasetId)).unwrap();
      // Refresh list
      await dispatch(fetchDatasets());
    } catch (err) {
      console.error('Delete failed:', err);
    }
  };

  return <button onClick={handleDelete}>Delete</button>;
}
```

#### Custom Hook Pattern

```typescript
import { useEffect } from 'react';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  fetchDataset,
  fetchDatasetStats,
  deleteDataset,
} from '../store/slices/datasetSlice';

export function useDataset(datasetId?: string) {
  const dispatch = useAppDispatch();
  const state = useAppSelector((state) => state.dataset);

  useEffect(() => {
    if (datasetId) {
      dispatch(fetchDataset(datasetId));
      dispatch(fetchDatasetStats(datasetId));
    }
  }, [dispatch, datasetId]);

  const remove = async (id: string) => {
    return dispatch(deleteDataset(id)).unwrap();
  };

  return {
    ...state,
    remove,
  };
}

// Usage
function MyComponent({ datasetId }: { datasetId: string }) {
  const { currentDataset, stats, isLoading, remove } = useDataset(datasetId);

  return (
    <div>
      <h2>{currentDataset?.name}</h2>
      <button onClick={() => remove(datasetId)}>Delete</button>
    </div>
  );
}
```

## Typed Hooks

Use typed hooks for better TypeScript support:

```typescript
import { useAppDispatch, useAppSelector } from './store/hooks';

// Instead of:
import { useDispatch, useSelector } from 'react-redux';
```

**Benefits:**
- Type inference for `dispatch`
- Type-safe state selection
- Autocomplete for actions

## Error Handling

All async thunks follow this pattern:

```typescript
// Pending state
.addCase(action.pending, (state) => {
  state.isLoading = true;
  state.error = null;
})

// Success state
.addCase(action.fulfilled, (state, action) => {
  state.isLoading = false;
  // Update state with payload
})

// Error state
.addCase(action.rejected, (state, action) => {
  state.isLoading = false;
  state.error = action.payload as string;
})
```

**Usage in components:**

```typescript
try {
  await dispatch(someAction()).unwrap();
  // Success
} catch (err) {
  // Error is already in Redux state
  console.error(err);
}
```

## Best Practices

### 1. Always use typed hooks

```typescript
// ✅ Good
import { useAppDispatch, useAppSelector } from '../store/hooks';

// ❌ Bad
import { useDispatch, useSelector } from 'react-redux';
```

### 2. Clear errors when retrying

```typescript
const handleRetry = () => {
  dispatch(clearDatasetError());
  dispatch(uploadDataset(file));
};
```

### 3. Use `.unwrap()` for error handling

```typescript
// ✅ Good - catch errors
try {
  const result = await dispatch(uploadDataset(file)).unwrap();
} catch (err) {
  // Handle error
}

// ❌ Bad - errors silent
await dispatch(uploadDataset(file));
```

### 4. Cleanup on unmount

```typescript
useEffect(() => {
  return () => {
    dispatch(resetDatasetState());
  };
}, []);
```

### 5. Combine related actions

```typescript
// Load dataset with all details
const loadDatasetComplete = async (id: string) => {
  await dispatch(fetchDataset(id));
  await Promise.all([
    dispatch(fetchDatasetStats(id)),
    dispatch(fetchDatasetPreview(id)),
  ]);
};
```

## File Reference

- **Store config**: [store/index.ts](./index.ts)
- **Typed hooks**: [store/hooks.ts](./hooks.ts)
- **Dataset slice**: [store/slices/datasetSlice.ts](./slices/datasetSlice.ts)
- **Usage examples**: [store/slices/datasetSlice.example.tsx](./slices/datasetSlice.example.tsx)

## Summary

The Redux store provides complete CRUD operations for datasets with:
- ✅ 6 async thunks for API operations
- ✅ 4 synchronous actions for state updates
- ✅ Comprehensive error and loading state management
- ✅ TypeScript type safety throughout
- ✅ Optimistic updates support
- ✅ Custom hook patterns
