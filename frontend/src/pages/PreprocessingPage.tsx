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
  IconButton,
  Tooltip,
  Divider,
} from '@mui/material';
import { Transform, Add, CheckCircle, Undo, Redo, History } from '@mui/icons-material';
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
  applyPreprocessingPipeline,
  clearError,
  clearPipelineResult,
  undo,
  redo,
  moveStepUp,
  moveStepDown,
  removeStepLocal,
} from '../store/slices/preprocessingSlice';
import { fetchDatasetStats, setCurrentDataset, fetchDataset, fetchDatasetPreview } from '../store/slices/datasetSlice';
import type { PreprocessingStep, PreprocessingStepCreate } from '../types/preprocessing';
import { validateDatasetForPreprocessing, getErrorMessage } from '../utils/preprocessingValidation';
import PreviewPanel from '../components/preprocessing/PreviewPanel';

const PreprocessingPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const {
    steps,
    isLoading,
    isProcessing,
    isExecuting,
    error,
    pipelineResult,
    history,
    historyIndex,
  } = useAppSelector((state) => state.preprocessing);
  const { currentDataset, columns } = useAppSelector((state) => state.dataset);

  const [isBuilderOpen, setIsBuilderOpen] = useState(false);
  const [editingStep, setEditingStep] = useState<PreprocessingStep | null>(null);
  const [showResultDialog, setShowResultDialog] = useState(false);
  const [refetchTrigger, setRefetchTrigger] = useState(0);
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({
    open: false,
    message: '',
    severity: 'success',
  });

  // Refetch preprocessing steps whenever component mounts or dataset changes
  // This ensures we always have the latest data from the server
  useEffect(() => {
    if (currentDataset?.id) {
      // Always fetch preprocessing steps to ensure we have the latest data
      dispatch(fetchPreprocessingSteps(currentDataset.id));

      // Fetch dataset stats if not already loaded
      if (!columns || columns.length === 0) {
        dispatch(fetchDatasetStats(currentDataset.id));
      }
    }
  }, [currentDataset?.id, dispatch, columns, refetchTrigger]);

  // Trigger refetch when component becomes visible (user navigates back to this page)
  useEffect(() => {
    const handleFocus = () => {
      setRefetchTrigger(prev => prev + 1);
    };

    window.addEventListener('focus', handleFocus);
    
    // Also trigger on mount to ensure fresh data
    setRefetchTrigger(prev => prev + 1);
    
    return () => {
      window.removeEventListener('focus', handleFocus);
    };
  }, []);

  // Keyboard shortcuts for undo/redo
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+Z or Cmd+Z for Undo
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        if (canUndo) {
          dispatch(undo());
          showSnackbar('Undo successful', 'success');
        }
      }
      // Ctrl+Shift+Z or Cmd+Shift+Z or Ctrl+Y for Redo
      if (((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'z') || (e.ctrlKey && e.key === 'y')) {
        e.preventDefault();
        if (canRedo) {
          dispatch(redo());
          showSnackbar('Redo successful', 'success');
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [dispatch, historyIndex, history.length]);

  const canUndo = historyIndex > 0;
  const canRedo = historyIndex < history.length - 1;

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
      setIsBuilderOpen(false);
      setEditingStep(null);
    } catch (error: any) {
      const errorMessage = getErrorMessage(error);
      showSnackbar(errorMessage, 'error');
    }
  };

  const handleDeleteStep = async (stepId: string) => {
    if (confirm('Are you sure you want to delete this step?')) {
      try {
        await dispatch(deletePreprocessingStep(stepId)).unwrap();
        showSnackbar('Step deleted successfully', 'success');
      } catch (error: any) {
        const errorMessage = getErrorMessage(error);
        showSnackbar(errorMessage, 'error');
      }
    }
  };

  const handleReorderSteps = async (reorderedSteps: PreprocessingStep[]) => {
    try {
      const stepIds = reorderedSteps.map(step => step.id);
      await dispatch(reorderPreprocessingSteps(stepIds)).unwrap();
      showSnackbar('Steps reordered successfully', 'success');
    } catch (error: any) {
      const errorMessage = getErrorMessage(error);
      showSnackbar(errorMessage, 'error');
    }
  };

  const handleExecutePipeline = async () => {
    // Validate dataset and steps before execution
    const validation = validateDatasetForPreprocessing(currentDataset, steps);

    if (!validation.isValid) {
      const errorMessages = validation.errorList.map(e => e.message).join('; ');
      showSnackbar(errorMessages, 'error');
      return;
    }

    // Additional validation: check if dataset has data
    if (currentDataset?.shape) {
      const [rows, cols] = currentDataset.shape;
      if (rows === 0) {
        showSnackbar('Cannot execute pipeline: Dataset has no rows', 'error');
        return;
      }
      if (cols === 0) {
        showSnackbar('Cannot execute pipeline: Dataset has no columns', 'error');
        return;
      }
    }

    try {
      // Execute preprocessing synchronously
      const result = await dispatch(applyPreprocessingPipeline({
        dataset_id: currentDataset!.id,
        save_output: true,
        output_name: `${currentDataset!.name}_preprocessed`,
      })).unwrap();

      showSnackbar('Preprocessing pipeline completed successfully', 'success');
    } catch (error: any) {
      const errorMessage = getErrorMessage(error);
      showSnackbar(errorMessage, 'error');
    }
  };

  // Show result dialog when pipeline completes
  useEffect(() => {
    if (pipelineResult) {
      setShowResultDialog(true);
    }
  }, [pipelineResult]);

  const handleCloseResultDialog = () => {
    setShowResultDialog(false);
    dispatch(clearPipelineResult());
  };

  const handleLoadPreprocessedDataset = async () => {
    if (pipelineResult?.output_dataset_id) {
      try {
        // Clear the pipeline result first to prevent dialog from reopening
        dispatch(clearPipelineResult());
        setShowResultDialog(false);
        
        // Fetch dataset details
        const dataset = await dispatch(fetchDataset(pipelineResult.output_dataset_id)).unwrap();
        dispatch(setCurrentDataset(dataset));
        
        // Fetch all dataset information in parallel (including steps for new dataset)
        await Promise.all([
          dispatch(fetchDatasetStats(pipelineResult.output_dataset_id)),
          dispatch(fetchDatasetPreview(pipelineResult.output_dataset_id)),
          dispatch(fetchPreprocessingSteps(pipelineResult.output_dataset_id)),
        ]);
        
        showSnackbar('Preprocessed dataset loaded successfully', 'success');
      } catch (error: any) {
        showSnackbar('Failed to load preprocessed dataset: ' + (error.message || 'Unknown error'), 'error');
      }
    }
  };

  const handleRefreshSteps = () => {
    if (currentDataset?.id) {
      dispatch(fetchPreprocessingSteps(currentDataset.id));
    }
  };

  const handleUndo = () => {
    if (canUndo) {
      dispatch(undo());
      showSnackbar('Undo successful', 'success');
    }
  };

  const handleRedo = () => {
    if (canRedo) {
      dispatch(redo());
      showSnackbar('Redo successful', 'success');
    }
  };

  const handleMoveUp = (index: number) => {
    dispatch(moveStepUp(index));
  };

  const handleMoveDown = (index: number) => {
    dispatch(moveStepDown(index));
  };

  const handleRemoveStep = (stepId: string) => {
    dispatch(removeStepLocal(stepId));
    showSnackbar('Step removed', 'success');
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

          {/* Action Buttons */}
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            {/* Undo/Redo Controls */}
            <Box sx={{ display: 'flex', border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
              <Tooltip title={`Undo (${navigator.platform.includes('Mac') ? 'Cmd' : 'Ctrl'}+Z)`}>
                <span>
                  <IconButton
                    size="small"
                    onClick={handleUndo}
                    disabled={!canUndo || isProcessing}
                    sx={{ borderRadius: 0 }}
                  >
                    <Undo fontSize="small" />
                  </IconButton>
                </span>
              </Tooltip>
              <Divider orientation="vertical" flexItem />
              <Tooltip title={`Redo (${navigator.platform.includes('Mac') ? 'Cmd+Shift+Z' : 'Ctrl+Y'})`}>
                <span>
                  <IconButton
                    size="small"
                    onClick={handleRedo}
                    disabled={!canRedo || isProcessing}
                    sx={{ borderRadius: 0 }}
                  >
                    <Redo fontSize="small" />
                  </IconButton>
                </span>
              </Tooltip>
            </Box>

            {/* History Info */}
            {history.length > 0 && (
              <Tooltip title={`${history.length} actions in history`}>
                <Chip
                  icon={<History />}
                  label={`${historyIndex + 1}/${history.length}`}
                  size="small"
                  variant="outlined"
                />
              </Tooltip>
            )}

            <Button
              variant="contained"
              startIcon={<Add />}
              onClick={handleAddStep}
              disabled={isProcessing || !columns || columns.length === 0}
            >
              Add Step
            </Button>
          </Box>
        </Box>

        {error && (
          <Alert severity="error" onClose={() => dispatch(clearError())} sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {(!columns || columns.length === 0) && (
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
            onMoveUp={handleMoveUp}
            onMoveDown={handleMoveDown}
            onRemove={handleRemoveStep}
            onExecute={handleExecutePipeline}
            onRefresh={handleRefreshSteps}
            isProcessing={isProcessing}
          />
        </Box>

        {/* Main Area - Preview Panel or Instructions */}
        <Box sx={{ width: { xs: '100%', md: '66.67%' }, height: '100%' }}>
          {pipelineResult ? (
            <PreviewPanel
              previewData={pipelineResult.preview as any}
              isLoading={isExecuting}
              error={error}
            />
          ) : (
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
                // Use center alignment only when empty, otherwise top
                justifyContent: steps.length === 0 ? 'center' : 'flex-start',
                bgcolor: 'background.paper',
                overflowY: 'auto', // Enable scrolling
              }}
            >
              {steps.length === 0 ? (
                <EmptyState
                  title="Build Your Preprocessing Pipeline"
                  message="Click 'Add Step' to start building your data preprocessing pipeline. You can add multiple steps and drag to reorder them."
                />
              ) : (
                <Box sx={{ textAlign: 'center', maxWidth: 600, width: '100%' }}>
                  <Typography variant="h6" gutterBottom fontWeight={600} sx={{ mt: 2 }}>
                    Pipeline Preview
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    Your preprocessing pipeline has <strong>{steps.length}</strong> step{steps.length !== 1 ? 's' : ''}.
                    Steps will be executed in the order shown in the sidebar.
                  </Typography>
                  <Box sx={{ mt: 4, textAlign: 'left', width: '100%' }}>
                    <Typography variant="subtitle2" gutterBottom fontWeight={600}>
                      Pipeline Steps:
                    </Typography>
                    {steps.map((step, index) => (
                      <Paper
                        key={step.id}
                        elevation={0}
                        sx={{
                          p: 2,
                          mb: 2,
                          border: '1px solid',
                          borderColor: 'divider',
                          borderRadius: 2,
                          bgcolor: 'grey.50',
                          transition: 'all 0.2s',
                          '&:hover': {
                            bgcolor: 'grey.100',
                            borderColor: 'primary.light',
                            transform: 'translateY(-2px)',
                            boxShadow: 2,
                          }
                        }}
                      >
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                          <Chip 
                            label={index + 1} 
                            size="small" 
                            color="primary" 
                            variant="outlined" 
                            sx={{ fontWeight: 'bold' }} 
                          />
                          <Box>
                            <Typography variant="body1" fontWeight={500}>
                              {step.step_type}
                            </Typography>
                            {step.column_name && (
                              <Typography variant="body2" color="text.secondary">
                                on column "<strong>{step.column_name}</strong>"
                              </Typography>
                            )}
                          </Box>
                        </Box>
                      </Paper>
                    ))}
                  </Box>
                </Box>
              )}
            </Box>
          )}
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

      {/* Progress Dialog - removed for synchronous execution */}
      {/* Synchronous execution shows loading state in button and completes immediately */}

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
          {pipelineResult?.output_dataset_id && (
            <Button 
              onClick={handleLoadPreprocessedDataset} 
              variant="contained"
              color="primary"
            >
              Load Preprocessed Data
            </Button>
          )}
          <Button onClick={handleCloseResultDialog} variant="outlined">
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default PreprocessingPage;
