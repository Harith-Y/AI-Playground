/**
 * Example usage of dataset Redux actions
 *
 * This file demonstrates how to use the dataset slice actions in React components.
 */

import React, { useEffect } from 'react';
import { useAppDispatch, useAppSelector } from '../hooks';
import {
  uploadDataset,
  fetchDatasets,
  fetchDataset,
  fetchDatasetStats,
  fetchDatasetPreview,
  deleteDataset,
  setCurrentDataset,
  clearDatasetError,
  setUploadProgress,
  resetDatasetState,
} from './slices/datasetSlice';
import { Box, Button, CircularProgress, Alert } from '@mui/material';

// ============================================================================
// Example 1: Upload Dataset with Progress Tracking
// ============================================================================

export const DatasetUploadExample: React.FC = () => {
  const dispatch = useAppDispatch();
  const { isLoading, error, uploadProgress } = useAppSelector((state) => state.dataset);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Clear any previous errors
    dispatch(clearDatasetError());

    // Dispatch upload action
    try {
      const result = await dispatch(uploadDataset(file)).unwrap();
      console.log('Upload successful:', result);

      // Automatically fetch stats after upload
      await dispatch(fetchDatasetStats(result.id));
    } catch (err) {
      console.error('Upload failed:', err);
    }
  };

  return (
    <Box>
      <input type="file" onChange={handleFileUpload} disabled={isLoading} />
      {isLoading && <CircularProgress value={uploadProgress} variant="determinate" />}
      {error && <Alert severity="error">{error}</Alert>}
    </Box>
  );
};

// ============================================================================
// Example 2: Fetch and Display All Datasets
// ============================================================================

export const DatasetListExample: React.FC = () => {
  const dispatch = useAppDispatch();
  const { datasets, isLoading, error } = useAppSelector((state) => state.dataset);

  useEffect(() => {
    // Fetch datasets on component mount
    dispatch(fetchDatasets());
  }, [dispatch]);

  const handleRefresh = () => {
    dispatch(fetchDatasets());
  };

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box>
      <Button onClick={handleRefresh}>Refresh</Button>
      <ul>
        {datasets.map((dataset) => (
          <li key={dataset.id}>
            {dataset.name} - {dataset.rowCount} rows, {dataset.columnCount} cols
          </li>
        ))}
      </ul>
    </Box>
  );
};

// ============================================================================
// Example 3: Fetch Single Dataset and Display Details
// ============================================================================

export const DatasetDetailsExample: React.FC<{ datasetId: string }> = ({ datasetId }) => {
  const dispatch = useAppDispatch();
  const { currentDataset, isLoading, error } = useAppSelector((state) => state.dataset);

  useEffect(() => {
    // Fetch specific dataset
    dispatch(fetchDataset(datasetId));
  }, [dispatch, datasetId]);

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">{error}</Alert>;
  if (!currentDataset) return <div>No dataset loaded</div>;

  return (
    <Box>
      <h2>{currentDataset.name}</h2>
      <p>Filename: {currentDataset.filename}</p>
      <p>Size: {currentDataset.size} bytes</p>
      <p>Rows: {currentDataset.rowCount}</p>
      <p>Columns: {currentDataset.columnCount}</p>
      <p>Status: {currentDataset.status}</p>
      <p>Created: {new Date(currentDataset.createdAt).toLocaleString()}</p>
    </Box>
  );
};

// ============================================================================
// Example 4: Fetch and Display Dataset Statistics
// ============================================================================

export const DatasetStatsExample: React.FC<{ datasetId: string }> = ({ datasetId }) => {
  const dispatch = useAppDispatch();
  const { stats, columns, isLoading, error } = useAppSelector((state) => state.dataset);

  useEffect(() => {
    dispatch(fetchDatasetStats(datasetId));
  }, [dispatch, datasetId]);

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">{error}</Alert>;
  if (!stats) return <div>No stats available</div>;

  return (
    <Box>
      <h3>Dataset Statistics</h3>
      <p>Total Rows: {stats.rowCount}</p>
      <p>Total Columns: {stats.columnCount}</p>
      <p>Numeric Columns: {stats.numericColumns}</p>
      <p>Categorical Columns: {stats.categoricalColumns}</p>
      <p>Missing Values: {stats.missingValues}</p>
      <p>Duplicate Rows: {stats.duplicateRows}</p>
      <p>Memory Usage: {stats.memoryUsage} bytes</p>

      <h4>Columns:</h4>
      <ul>
        {columns.map((col) => (
          <li key={col.name}>
            {col.name} ({col.dataType}) - {col.uniqueCount} unique, {col.nullCount} nulls
          </li>
        ))}
      </ul>
    </Box>
  );
};

// ============================================================================
// Example 5: Fetch and Display Dataset Preview
// ============================================================================

export const DatasetPreviewExample: React.FC<{ datasetId: string }> = ({ datasetId }) => {
  const dispatch = useAppDispatch();
  const { preview, columns, isLoading, error } = useAppSelector((state) => state.dataset);

  useEffect(() => {
    // Fetch both preview and columns
    dispatch(fetchDatasetPreview(datasetId));
    dispatch(fetchDatasetStats(datasetId)); // For column info
  }, [dispatch, datasetId]);

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">{error}</Alert>;
  if (preview.length === 0) return <div>No preview data</div>;

  return (
    <Box>
      <h3>Dataset Preview</h3>
      <table>
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col.name}>{col.name}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {preview.map((row, idx) => (
            <tr key={idx}>
              {row.map((cell: any, cellIdx: number) => (
                <td key={cellIdx}>{cell ?? 'null'}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </Box>
  );
};

// ============================================================================
// Example 6: Delete Dataset with Confirmation
// ============================================================================

export const DeleteDatasetExample: React.FC<{ datasetId: string }> = ({ datasetId }) => {
  const dispatch = useAppDispatch();
  const { isLoading, error } = useAppSelector((state) => state.dataset);

  const handleDelete = async () => {
    if (!window.confirm('Are you sure you want to delete this dataset?')) {
      return;
    }

    try {
      await dispatch(deleteDataset(datasetId)).unwrap();
      console.log('Dataset deleted successfully');

      // Optionally refresh the dataset list
      dispatch(fetchDatasets());
    } catch (err) {
      console.error('Delete failed:', err);
    }
  };

  return (
    <Box>
      <Button
        variant="contained"
        color="error"
        onClick={handleDelete}
        disabled={isLoading}
      >
        {isLoading ? 'Deleting...' : 'Delete Dataset'}
      </Button>
      {error && <Alert severity="error">{error}</Alert>}
    </Box>
  );
};

// ============================================================================
// Example 7: Complete Dataset Management Component
// ============================================================================

export const CompleteDatasetManager: React.FC = () => {
  const dispatch = useAppDispatch();
  const {
    datasets,
    currentDataset,
    stats,
    preview,
    isLoading,
    error,
  } = useAppSelector((state) => state.dataset);

  // Fetch datasets on mount
  useEffect(() => {
    dispatch(fetchDatasets());
  }, [dispatch]);

  // Select a dataset
  const handleSelectDataset = async (datasetId: string) => {
    try {
      // Fetch dataset details
      await dispatch(fetchDataset(datasetId)).unwrap();

      // Fetch stats and preview
      await Promise.all([
        dispatch(fetchDatasetStats(datasetId)),
        dispatch(fetchDatasetPreview(datasetId)),
      ]);
    } catch (err) {
      console.error('Failed to load dataset:', err);
    }
  };

  // Upload new dataset
  const handleUpload = async (file: File) => {
    try {
      const result = await dispatch(uploadDataset(file)).unwrap();

      // Auto-select the uploaded dataset
      handleSelectDataset(result.id);
    } catch (err) {
      console.error('Upload failed:', err);
    }
  };

  // Delete dataset
  const handleDelete = async (datasetId: string) => {
    if (!window.confirm('Delete this dataset?')) return;

    try {
      await dispatch(deleteDataset(datasetId)).unwrap();
      console.log('Dataset deleted');
    } catch (err) {
      console.error('Delete failed:', err);
    }
  };

  // Clear error
  const handleClearError = () => {
    dispatch(clearDatasetError());
  };

  // Reset entire state
  const handleReset = () => {
    dispatch(resetDatasetState());
  };

  return (
    <Box>
      {/* Error Display */}
      {error && (
        <Alert severity="error" onClose={handleClearError}>
          {error}
        </Alert>
      )}

      {/* Dataset List */}
      <Box>
        <h3>Datasets ({datasets.length})</h3>
        {isLoading && <CircularProgress />}
        <ul>
          {datasets.map((dataset) => (
            <li key={dataset.id}>
              <Button onClick={() => handleSelectDataset(dataset.id)}>
                {dataset.name}
              </Button>
              <Button color="error" onClick={() => handleDelete(dataset.id)}>
                Delete
              </Button>
            </li>
          ))}
        </ul>
      </Box>

      {/* Current Dataset Details */}
      {currentDataset && (
        <Box>
          <h3>Current Dataset: {currentDataset.name}</h3>
          <p>Rows: {currentDataset.rowCount}</p>
          <p>Columns: {currentDataset.columnCount}</p>
        </Box>
      )}

      {/* Statistics */}
      {stats && (
        <Box>
          <h4>Statistics</h4>
          <p>Missing Values: {stats.missingValues}</p>
          <p>Duplicates: {stats.duplicateRows}</p>
        </Box>
      )}

      {/* Preview */}
      {preview.length > 0 && (
        <Box>
          <h4>Preview ({preview.length} rows)</h4>
          {/* Render preview table */}
        </Box>
      )}

      {/* Actions */}
      <Box>
        <Button onClick={() => dispatch(fetchDatasets())}>Refresh List</Button>
        <Button onClick={handleReset}>Reset State</Button>
      </Box>
    </Box>
  );
};

// ============================================================================
// Example 8: Using with Custom Hooks
// ============================================================================

// Custom hook for dataset operations
export const useDataset = (datasetId?: string) => {
  const dispatch = useAppDispatch();
  const state = useAppSelector((state) => state.dataset);

  useEffect(() => {
    if (datasetId) {
      dispatch(fetchDataset(datasetId));
      dispatch(fetchDatasetStats(datasetId));
      dispatch(fetchDatasetPreview(datasetId));
    }
  }, [dispatch, datasetId]);

  const upload = async (file: File) => {
    return dispatch(uploadDataset(file)).unwrap();
  };

  const remove = async (id: string) => {
    return dispatch(deleteDataset(id)).unwrap();
  };

  const refresh = async () => {
    if (datasetId) {
      await dispatch(fetchDataset(datasetId));
    }
  };

  return {
    ...state,
    upload,
    remove,
    refresh,
  };
};

// Usage of custom hook
export const DatasetWithHookExample: React.FC<{ datasetId: string }> = ({ datasetId }) => {
  const { currentDataset, stats, isLoading, error, refresh } = useDataset(datasetId);

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box>
      <h2>{currentDataset?.name}</h2>
      <p>Rows: {stats?.rowCount}</p>
      <Button onClick={refresh}>Refresh</Button>
    </Box>
  );
};

// ============================================================================
// Example 9: Error Handling Patterns
// ============================================================================

export const ErrorHandlingExample: React.FC = () => {
  const dispatch = useAppDispatch();
  const { error } = useAppSelector((state) => state.dataset);

  const handleUploadWithErrorHandling = async (file: File) => {
    // Clear previous errors
    dispatch(clearDatasetError());

    try {
      // Attempt upload
      const result = await dispatch(uploadDataset(file)).unwrap();
      console.log('Success:', result);
    } catch (err: any) {
      // Error is already in Redux state
      console.error('Upload failed:', err);

      // Could show toast notification here
      // toast.error(`Upload failed: ${err}`);
    }
  };

  return (
    <Box>
      {error && (
        <Alert
          severity="error"
          onClose={() => dispatch(clearDatasetError())}
        >
          {error}
        </Alert>
      )}
    </Box>
  );
};

// ============================================================================
// Example 10: Optimistic Updates
// ============================================================================

export const OptimisticDeleteExample: React.FC<{ datasetId: string }> = ({ datasetId }) => {
  const dispatch = useAppDispatch();
  const datasets = useAppSelector((state) => state.dataset.datasets);
  const [localDatasets, setLocalDatasets] = React.useState(datasets);

  const handleOptimisticDelete = async (id: string) => {
    // Optimistically remove from UI
    setLocalDatasets((prev) => prev.filter((d) => d.id !== id));

    try {
      // Attempt delete
      await dispatch(deleteDataset(id)).unwrap();
      console.log('Delete successful');
    } catch (err) {
      // Revert on error
      setLocalDatasets(datasets);
      console.error('Delete failed, reverting:', err);
    }
  };

  return (
    <ul>
      {localDatasets.map((dataset) => (
        <li key={dataset.id}>
          {dataset.name}
          <Button onClick={() => handleOptimisticDelete(dataset.id)}>Delete</Button>
        </li>
      ))}
    </ul>
  );
};
