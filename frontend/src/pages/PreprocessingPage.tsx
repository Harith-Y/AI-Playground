import React, { useEffect, useState, useRef } from 'react';
import {
  Typography,
  Box,
  Container,
  Button,
  Alert,
  Snackbar,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  LinearProgress,
  Chip,
  Paper,
} from '@mui/material';
import { Transform, Add, CheckCircle } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import LoadingState from '../components/common/LoadingState';
import EmptyState from '../components/common/EmptyState';
import StepBuilder from '../components/preprocessing/StepBuilder';
import StepHistory from '../components/preprocessing/StepHistory';
import {
  fetchPreprocessingSteps,
  createPreprocessingStep,
  updatePreprocessingStep,
  deletePreprocessingStep,
  reorderPreprocessingSteps,
  applyPreprocessingPipelineAsync,
  getPreprocessingTaskStatus,
  clearError,
  clearPipelineResult,
} from '../store/slices/preprocessingSlice';
import { fetchDatasetStats } from '../store/slices/datasetSlice';
import type { PreprocessingStep, PreprocessingStepCreate } from '../types/preprocessing';

const PreprocessingPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const {
    steps,
    isLoading,
    isProcessing,
    isExecuting,
    error,
    pipelineResult,
    currentTaskId,
    taskStatus,
  } = useAppSelector((state) => state.preprocessing);
  const { currentDataset, columns } = useAppSelector((state) => state.dataset);

  const [isBuilderOpen, setIsBuilderOpen] = useState(false);
  const [editingStep, setEditingStep] = useState<PreprocessingStep | null>(null);
  const [showProgressDialog, setShowProgressDialog] = useState(false);
  const [showResultDialog, setShowResultDialog] = useState(false);
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({
    open: false,
    message: '',
    severity: 'success',
  });

  const pollingIntervalRef = useRef<number | null>(null);

  useEffect(() => {
    if (currentDataset?.id) {
      // Fetch preprocessing steps for the current dataset
      dispatch(fetchPreprocessingSteps(currentDataset.id));

      // Fetch dataset stats if not already loaded
      if (columns.length === 0) {
        dispatch(fetchDatasetStats(currentDataset.id));
      }
    }
  }, [currentDataset?.id, dispatch, columns.length]);

  const handleAddStep = () => {
    setEditingStep(null);
    setIsBuilderOpen(true);
  };

  const handleEditStep = (step: PreprocessingStep) => {
    setEditingStep(step);
    setIsBuilderOpen(true);
  };

  const handleSaveStep = async (stepData: PreprocessingStepCreate) => {
    try {
      if (editingStep) {
        // Update existing step
        await dispatch(updatePreprocessingStep({
          id: editingStep.id,
          data: {
            step_type: stepData.step_type,
            parameters: stepData.parameters,
            column_name: stepData.column_name,
          },
        })).unwrap();
        showSnackbar('Step updated successfully', 'success');
      } else {
        // Create new step
        await dispatch(createPreprocessingStep(stepData)).unwrap();
        showSnackbar('Step added successfully', 'success');
      }
    } catch (error: any) {
      showSnackbar(error || 'Failed to save step', 'error');
    }
  };

  const handleDeleteStep = async (stepId: string) => {
    if (confirm('Are you sure you want to delete this step?')) {
      try {
        await dispatch(deletePreprocessingStep(stepId)).unwrap();
        showSnackbar('Step deleted successfully', 'success');
      } catch (error: any) {
        showSnackbar(error || 'Failed to delete step', 'error');
      }
    }
  };

  const handleReorderSteps = async (reorderedSteps: PreprocessingStep[]) => {
    try {
      const stepIds = reorderedSteps.map(step => step.id);
      await dispatch(reorderPreprocessingSteps(stepIds)).unwrap();
      showSnackbar('Steps reordered successfully', 'success');
    } catch (error: any) {
      showSnackbar(error || 'Failed to reorder steps', 'error');
    }
  };

  const handleExecutePipeline = async () => {
    if (!currentDataset?.id || steps.length === 0) {
      showSnackbar('No steps to execute', 'error');
      return;
    }

    try {
      // Start async preprocessing task
      await dispatch(applyPreprocessingPipelineAsync({
        dataset_id: currentDataset.id,
        save_output: true,
        output_name: `${currentDataset.name}_preprocessed`,
      })).unwrap();

      setShowProgressDialog(true);
    } catch (error: any) {
      showSnackbar(error || 'Failed to start preprocessing', 'error');
    }
  };

  // Poll for task status
  useEffect(() => {
    if (currentTaskId && isExecuting) {
      // Start polling every 2 seconds
      pollingIntervalRef.current = setInterval(() => {
        dispatch(getPreprocessingTaskStatus(currentTaskId));
      }, 2000);

      return () => {
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
      };
    }
  }, [currentTaskId, isExecuting, dispatch]);

  // Handle task completion
  useEffect(() => {
    if (taskStatus) {
      if (taskStatus.state === 'SUCCESS' && pipelineResult) {
        // Stop polling
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
        setShowProgressDialog(false);
        setShowResultDialog(true);
      } else if (taskStatus.state === 'FAILURE') {
        // Stop polling on failure
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
        setShowProgressDialog(false);
        showSnackbar(taskStatus.error || 'Pipeline execution failed', 'error');
      }
    }
  }, [taskStatus, pipelineResult]);

  const handleCloseResultDialog = () => {
    setShowResultDialog(false);
    dispatch(clearPipelineResult());
  };

  const handleRefreshSteps = () => {
    if (currentDataset?.id) {
      dispatch(fetchPreprocessingSteps(currentDataset.id));
    }
  };

  const showSnackbar = (message: string, severity: 'success' | 'error') => {
    setSnackbar({ open: true, message, severity });
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
    dispatch(clearError());
  };

  if (isLoading) {
    return <LoadingState message="Loading preprocessing steps..." />;
  }

  if (!currentDataset) {
    return (
      <EmptyState
        title="No Dataset Selected"
        message="Please upload a dataset first to start preprocessing"
      />
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4, height: 'calc(100vh - 100px)' }}>
      {/* Page Header */}
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Transform sx={{ fontSize: 40, color: 'primary.main' }} />
            <Box>
              <Typography variant="h4" component="h1" fontWeight={600}>
                Data Preprocessing
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Build your preprocessing pipeline for <strong>{currentDataset.name}</strong>
              </Typography>
            </Box>
          </Box>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={handleAddStep}
            disabled={isProcessing || columns.length === 0}
          >
            Add Step
          </Button>
        </Box>

        {error && (
          <Alert severity="error" onClose={() => dispatch(clearError())} sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {columns.length === 0 && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            Loading dataset information...
          </Alert>
        )}
      </Box>

      {/* Main Content - Step History Sidebar */}
      <Box sx={{ display: 'flex', gap: 3, height: 'calc(100% - 120px)', flexDirection: { xs: 'column', md: 'row' } }}>
        <Box sx={{ width: { xs: '100%', md: '33.33%' }, height: '100%' }}>
          <StepHistory
            steps={steps}
            onEdit={handleEditStep}
            onDelete={handleDeleteStep}
            onReorder={handleReorderSteps}
            onExecute={handleExecutePipeline}
            onRefresh={handleRefreshSteps}
            isProcessing={isProcessing}
          />
        </Box>

        {/* Main Area - Dataset Preview or Instructions */}
        <Box sx={{ width: { xs: '100%', md: '66.67%' }, height: '100%' }}>
          <Box
            sx={{
              height: '100%',
              border: '1px solid',
              borderColor: 'divider',
              borderRadius: 2,
              p: 4,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: 'background.paper',
            }}
          >
            {steps.length === 0 ? (
              <EmptyState
                title="Build Your Preprocessing Pipeline"
                message="Click 'Add Step' to start building your data preprocessing pipeline. You can add multiple steps and drag to reorder them."
              />
            ) : (
              <Box sx={{ textAlign: 'center', maxWidth: 600 }}>
                <Typography variant="h6" gutterBottom fontWeight={600}>
                  Pipeline Preview
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Your preprocessing pipeline has <strong>{steps.length}</strong> step{steps.length !== 1 ? 's' : ''}.
                  Steps will be executed in the order shown in the sidebar.
                </Typography>
                <Box sx={{ mt: 4, textAlign: 'left' }}>
                  <Typography variant="subtitle2" gutterBottom fontWeight={600}>
                    Pipeline Steps:
                  </Typography>
                  {steps.map((step, index) => (
                    <Box
                      key={step.id}
                      sx={{
                        p: 2,
                        mb: 1,
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: 1,
                        bgcolor: 'grey.50',
                      }}
                    >
                      <Typography variant="body2">
                        <strong>{index + 1}.</strong> {step.step_type}
                        {step.column_name && ` on column "${step.column_name}"`}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </Box>
            )}
          </Box>
        </Box>
      </Box>

      {/* Step Builder Dialog */}
      <StepBuilder
        open={isBuilderOpen}
        onClose={() => {
          setIsBuilderOpen(false);
          setEditingStep(null);
        }}
        onSave={handleSaveStep}
        editStep={editingStep}
      />

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>

      {/* Progress Dialog */}
      <Dialog
        open={showProgressDialog}
        maxWidth="sm"
        fullWidth
        disableEscapeKeyDown
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Transform />
            <Typography variant="h6">Executing Preprocessing Pipeline</Typography>
          </Box>
        </DialogTitle>
        <DialogContent>
          <Box sx={{ py: 2 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {taskStatus?.status || 'Starting preprocessing...'}
            </Typography>

            <Box sx={{ mt: 2, mb: 1 }}>
              <LinearProgress
                variant="determinate"
                value={taskStatus?.progress || 0}
                sx={{ height: 8, borderRadius: 4 }}
              />
            </Box>

            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                {taskStatus?.progress || 0}% complete
              </Typography>
              {taskStatus?.current_step && (
                <Chip
                  label={taskStatus.current_step}
                  size="small"
                  color="primary"
                  variant="outlined"
                />
              )}
            </Box>

            <Box sx={{ mt: 3 }}>
              <Typography variant="caption" color="text.secondary">
                Task ID: {currentTaskId}
              </Typography>
            </Box>
          </Box>
        </DialogContent>
      </Dialog>

      {/* Result Dialog */}
      <Dialog
        open={showResultDialog}
        onClose={handleCloseResultDialog}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CheckCircle color="success" />
            <Typography variant="h6">Pipeline Execution Complete</Typography>
          </Box>
        </DialogTitle>
        <DialogContent>
          {pipelineResult && (
            <Box sx={{ py: 2 }}>
              <Alert severity="success" sx={{ mb: 3 }}>
                {pipelineResult.message}
              </Alert>

              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2 }}>
                <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                  <Typography variant="caption" color="text.secondary" gutterBottom>
                    Steps Applied
                  </Typography>
                  <Typography variant="h5" fontWeight={600}>
                    {pipelineResult.steps_applied}
                  </Typography>
                </Paper>
                <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                  <Typography variant="caption" color="text.secondary" gutterBottom>
                    Rows Changed
                  </Typography>
                  <Typography variant="h5" fontWeight={600}>
                    {pipelineResult.original_shape[0]} → {pipelineResult.transformed_shape[0]}
                  </Typography>
                </Paper>
                <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                  <Typography variant="caption" color="text.secondary" gutterBottom>
                    Columns Changed
                  </Typography>
                  <Typography variant="h5" fontWeight={600}>
                    {pipelineResult.original_shape[1]} → {pipelineResult.transformed_shape[1]}
                  </Typography>
                </Paper>
                <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                  <Typography variant="caption" color="text.secondary" gutterBottom>
                    Output Dataset
                  </Typography>
                  <Typography variant="body2" fontWeight={600}>
                    {pipelineResult.output_dataset_id ? 'Created' : 'Not Saved'}
                  </Typography>
                </Paper>
              </Box>

              {pipelineResult.statistics && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle2" gutterBottom fontWeight={600}>
                    Statistics:
                  </Typography>
                  <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                    {Object.entries(pipelineResult.statistics).map(([key, value]) => (
                      <Box key={key} sx={{ display: 'flex', justifyContent: 'space-between', py: 0.5 }}>
                        <Typography variant="body2" color="text.secondary">
                          {key.replace(/_/g, ' ')}:
                        </Typography>
                        <Typography variant="body2" fontWeight={500}>
                          {String(value)}
                        </Typography>
                      </Box>
                    ))}
                  </Paper>
                </Box>
              )}

              {pipelineResult.preview && pipelineResult.preview.length > 0 && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle2" gutterBottom fontWeight={600}>
                    Preview (first {pipelineResult.preview.length} rows):
                  </Typography>
                  <Box sx={{ overflowX: 'auto' }}>
                    <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                      <pre style={{ fontSize: '12px', margin: 0 }}>
                        {JSON.stringify(pipelineResult.preview, null, 2)}
                      </pre>
                    </Paper>
                  </Box>
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseResultDialog} variant="contained">
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default PreprocessingPage;
