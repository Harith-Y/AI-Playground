import React, { useEffect, useState } from 'react';
import { Typography, Box, Container, Grid, Button, Alert, Snackbar } from '@mui/material';
import { Transform, Add } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import LoadingState from '../components/common/LoadingState';
import ErrorState from '../components/common/ErrorState';
import EmptyState from '../components/common/EmptyState';
import StepBuilder from '../components/preprocessing/StepBuilder';
import StepHistory from '../components/preprocessing/StepHistory';
import {
  fetchPreprocessingSteps,
  createPreprocessingStep,
  updatePreprocessingStep,
  deletePreprocessingStep,
  reorderPreprocessingSteps,
  clearError,
} from '../store/slices/preprocessingSlice';
import { fetchDatasetStats } from '../store/slices/datasetSlice';
import type { PreprocessingStep, PreprocessingStepCreate } from '../types/preprocessing';

const PreprocessingPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { steps, isLoading, isProcessing, error } = useAppSelector((state) => state.preprocessing);
  const { currentDataset, columns } = useAppSelector((state) => state.dataset);

  const [isBuilderOpen, setIsBuilderOpen] = useState(false);
  const [editingStep, setEditingStep] = useState<PreprocessingStep | null>(null);
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({
    open: false,
    message: '',
    severity: 'success',
  });

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

  const handleExecutePipeline = () => {
    // TODO: Implement pipeline execution
    showSnackbar('Pipeline execution not yet implemented', 'info' as any);
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
      <Grid container spacing={3} sx={{ height: 'calc(100% - 120px)' }}>
        <Grid item xs={12} md={4} sx={{ height: '100%' }}>
          <StepHistory
            steps={steps}
            onEdit={handleEditStep}
            onDelete={handleDeleteStep}
            onReorder={handleReorderSteps}
            onExecute={handleExecutePipeline}
            onRefresh={handleRefreshSteps}
            isProcessing={isProcessing}
          />
        </Grid>

        {/* Main Area - Dataset Preview or Instructions */}
        <Grid item xs={12} md={8} sx={{ height: '100%' }}>
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
        </Grid>
      </Grid>

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
    </Container>
  );
};

export default PreprocessingPage;
