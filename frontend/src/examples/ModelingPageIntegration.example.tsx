/**
 * Example: How to integrate FE-39 API wiring into ModelingPage
 * 
 * This file shows how to use the new hooks and components
 * for complete training management.
 */

import React, { useState } from 'react';
import { Box, Container, Snackbar, Alert } from '@mui/material';
import { useNavigate } from 'react-router-dom';

// Import the new hooks
import { useTrainingRuns } from '../hooks/useTrainingRuns';
import { useAvailableModels } from '../hooks/useAvailableModels';
import { useModelTraining } from '../hooks/useModelTraining';

// Import the new component
import TrainingRunsList from '../components/modeling/TrainingRunsList';

// Import types
import type { TrainingConfig } from '../types/training';

const ModelingPageExample: React.FC = () => {
  const navigate = useNavigate();
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({
    open: false,
    message: '',
    severity: 'success',
  });

  // 1. Fetch available models
  const { models } = useAvailableModels({
    category: 'classification', // or 'regression', 'clustering'
    autoFetch: true,
  });

  // 2. Manage training runs list
  const {
    runs,
    total,
    page,
    pageSize,
    loading: runsLoading,
    error: runsError,
    refresh,
    deleteRun,
    deleteRuns,
    cancelRun,
    retryRun,
    setPage,
    setPageSize,
  } = useTrainingRuns({
    autoRefresh: true, // Auto-refresh every 5 seconds
    refreshInterval: 5000,
    initialPageSize: 10,
    filters: {
      // status: 'running', // Optional: filter by status
    },
  });

  // 3. Start training
  const { startTraining } = useModelTraining({
    onSuccess: (runId) => {
      setSnackbar({
        open: true,
        message: 'Training started successfully!',
        severity: 'success',
      });
      // Navigate to training progress page
      navigate(`/training/${runId}`);
    },
    onError: (error) => {
      setSnackbar({
        open: true,
        message: `Failed to start training: ${error}`,
        severity: 'error',
      });
    },
    validateBeforeStart: true,
  });

  // Handle training start
  const handleStartTraining = async (config: TrainingConfig) => {
    await startTraining(config);
  };

  // Use models and handleStartTraining in your UI
  console.log('Available models:', models.length);
  console.log('Start training handler:', handleStartTraining);

  // Handle delete with confirmation
  const handleDelete = async (runId: string) => {
    if (window.confirm('Are you sure you want to delete this training run?')) {
      try {
        await deleteRun(runId);
        setSnackbar({
          open: true,
          message: 'Training run deleted successfully',
          severity: 'success',
        });
      } catch (err) {
        setSnackbar({
          open: true,
          message: 'Failed to delete training run',
          severity: 'error',
        });
      }
    }
  };

  // Handle batch delete
  const handleDeleteMultiple = async (runIds: string[]) => {
    if (window.confirm(`Are you sure you want to delete ${runIds.length} training runs?`)) {
      try {
        await deleteRuns(runIds);
        setSnackbar({
          open: true,
          message: `${runIds.length} training runs deleted successfully`,
          severity: 'success',
        });
      } catch (err) {
        setSnackbar({
          open: true,
          message: 'Failed to delete training runs',
          severity: 'error',
        });
      }
    }
  };

  // Handle cancel
  const handleCancel = async (runId: string) => {
    if (window.confirm('Are you sure you want to cancel this training?')) {
      try {
        await cancelRun(runId);
        setSnackbar({
          open: true,
          message: 'Training cancelled successfully',
          severity: 'success',
        });
      } catch (err) {
        setSnackbar({
          open: true,
          message: 'Failed to cancel training',
          severity: 'error',
        });
      }
    }
  };

  // Handle retry
  const handleRetry = async (runId: string) => {
    try {
      await retryRun(runId);
      setSnackbar({
        open: true,
        message: 'Training restarted successfully',
        severity: 'success',
      });
    } catch (err) {
      setSnackbar({
        open: true,
        message: 'Failed to retry training',
        severity: 'error',
      });
    }
  };

  // Handle export
  const handleExport = (runId: string) => {
    // Export model logic
    console.log('Exporting model:', runId);
  };

  return (
    <Container maxWidth="xl">
      <Box py={4}>
        {/* Your existing ModelingPage content here */}
        {/* Model selection, hyperparameter configuration, etc. */}

        {/* Training Runs List */}
        <Box mt={4}>
          <TrainingRunsList
            runs={runs}
            total={total}
            page={page}
            pageSize={pageSize}
            loading={runsLoading}
            error={runsError}
            onPageChange={setPage}
            onPageSizeChange={setPageSize}
            onRefresh={refresh}
            onDelete={handleDelete}
            onDeleteMultiple={handleDeleteMultiple}
            onCancel={handleCancel}
            onRetry={handleRetry}
            onExport={handleExport}
          />
        </Box>

        {/* Snackbar for notifications */}
        <Snackbar
          open={snackbar.open}
          autoHideDuration={6000}
          onClose={() => setSnackbar({ ...snackbar, open: false })}
        >
          <Alert severity={snackbar.severity} onClose={() => setSnackbar({ ...snackbar, open: false })}>
            {snackbar.message}
          </Alert>
        </Snackbar>
      </Box>
    </Container>
  );
};

export default ModelingPageExample;

/**
 * USAGE NOTES:
 * 
 * 1. Available Models:
 *    - Use `useAvailableModels()` to fetch models
 *    - Filter by category or task type
 *    - Display in model selector component
 * 
 * 2. Training Runs:
 *    - Use `useTrainingRuns()` for list management
 *    - Enable auto-refresh for real-time updates
 *    - Apply filters for status, model type, dataset
 * 
 * 3. Start Training:
 *    - Use `useModelTraining()` to start training
 *    - Validation happens automatically
 *    - Navigate to progress page on success
 * 
 * 4. Training Runs List:
 *    - Use `TrainingRunsList` component
 *    - Supports pagination, filtering, sorting
 *    - Batch operations (delete multiple)
 *    - Action buttons (view, cancel, retry, delete, export)
 * 
 * 5. Error Handling:
 *    - All hooks provide error states
 *    - Use snackbars for user feedback
 *    - Validation errors shown before training
 * 
 * 6. Real-Time Updates:
 *    - Enable auto-refresh in useTrainingRuns
 *    - Configurable refresh interval
 *    - Manual refresh button available
 */
