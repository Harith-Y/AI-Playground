import React, { useEffect, useState } from 'react';
import {
  Typography,
  Box,
  Button,
  Container,
  Grid,
  Card,
  CardContent,
  Stepper,
  Step,
  StepLabel,
  Divider,
  Alert,
  Chip,
  Stack,
} from '@mui/material';
import {
  ModelTraining,
  PlayArrow,
  Stop,
  Storage,
  Psychology,
  Tune,
  Timeline,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import { ModelSelector, HyperparameterEditor } from '../components/model';
import {
  fetchModels,
  clearModelError,
  setSelectedModel,
  setHyperparameters,
} from '../store/slices/modelingSlice';
import LoadingState from '../components/common/LoadingState';
import ErrorState from '../components/common/ErrorState';
import EmptyState from '../components/common/EmptyState';
import TrainingProgress from '../components/modeling/TrainingProgress';

// Workflow steps
const workflowSteps = [
  'Select Data',
  'Choose Model',
  'Configure Hyperparameters',
  'Train & Evaluate',
];

const ModelingPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const {
    models,
    selectedModel,
    isLoading,
    isTraining,
    error,
    trainingProgress,
    trainingMetrics,
    logs,
  } = useAppSelector((state) => state.modeling);
  const { currentDataset } = useAppSelector((state) => state.dataset);
  const { taskType } = useAppSelector((state) => state.feature);

  const [activeStep, setActiveStep] = useState(0);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [_hyperparameters, _setHyperparameters] = useState<Record<string, any>>({});

  useEffect(() => {
    dispatch(fetchModels());
    return () => {
      dispatch(clearModelError());
    };
  }, [dispatch]);

  useEffect(() => {
    // Auto-select current dataset if available
    if (currentDataset?.id && !selectedDatasetId) {
      setSelectedDatasetId(currentDataset.id);
    }
  }, [currentDataset, selectedDatasetId]);

  const handleRetry = () => {
    dispatch(clearModelError());
    dispatch(fetchModels());
  };

  const handleNextStep = () => {
    if (activeStep < workflowSteps.length - 1) {
      setActiveStep((prev) => prev + 1);
    }
  };

  const handlePreviousStep = () => {
    if (activeStep > 0) {
      setActiveStep((prev) => prev - 1);
    }
  };

  const canProceedToNextStep = () => {
    switch (activeStep) {
      case 0: // Data selection
        return selectedDatasetId !== null;
      case 1: // Model selection
        return selectedModel !== null;
      case 2: // Hyperparameters
        return true; // Can always proceed with default hyperparameters
      default:
        return false;
    }
  };

  const handleModelSelect = (selection: any) => {
    dispatch(setSelectedModel(selection.modelId));
    dispatch(setHyperparameters(selection.hyperparameters));
  };
  
  if (isLoading && models.length === 0) {
    return <LoadingState message="Loading model configurations..." />;
  }

  if (error && models.length === 0) {
    return (
      <ErrorState
        title="Failed to Load Models"
        message={error}
        onRetry={handleRetry}
      />
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Page Header */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <ModelTraining sx={{ fontSize: 40, color: 'primary.main' }} />
          <Box>
            <Typography variant="h4" component="h1" fontWeight={600}>
              Model Training
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Train and evaluate machine learning models on your datasets
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* Error Alert */}
      {error && models.length > 0 && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => dispatch(clearModelError())}>
          {error}
        </Alert>
      )}

      {/* Workflow Stepper */}
      <Card sx={{ mb: 3, border: '1px solid #e2e8f0' }}>
        <CardContent>
          <Stepper activeStep={activeStep} alternativeLabel>
            {workflowSteps.map((label) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>
        </CardContent>
      </Card>

      <Grid container spacing={3}>
        {/* Left Column: Configuration Sections */}
        <Grid size={{ xs: 12, lg: 8 }}>
          <Stack spacing={3}>
            {/* Section 1: Data Selection */}
            <Card
              sx={{
                border: activeStep === 0 ? '2px solid' : '1px solid #e2e8f0',
                borderColor: activeStep === 0 ? 'primary.main' : '#e2e8f0',
              }}
            >
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                  <Storage color={activeStep === 0 ? 'primary' : 'action'} />
                  <Typography variant="h6" fontWeight={600}>
                    1. Data Selection
                  </Typography>
                  {selectedDatasetId && (
                    <Chip label="Selected" color="success" size="small" />
                  )}
                </Box>
                <Divider sx={{ mb: 2 }} />
                {currentDataset ? (
                  <Box>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Current Dataset
                    </Typography>
                    <Alert severity="info" icon={<Storage />}>
                      <Typography variant="body2" fontWeight={600}>
                        {currentDataset.name}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {currentDataset.rows} rows × {currentDataset.columns} columns
                      </Typography>
                    </Alert>
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                      This dataset will be used for training. You can change it from the Datasets page.
                    </Typography>
                  </Box>
                ) : (
                  <EmptyState
                    title="No Dataset Selected"
                    message="Please upload and select a dataset from the Datasets page to begin training"
                    icon={<Storage sx={{ fontSize: 48 }} />}
                  />
                )}
              </CardContent>
            </Card>

            {/* Section 2: Model Selection */}
            <Card
              sx={{
                border: activeStep === 1 ? '2px solid' : '1px solid #e2e8f0',
                borderColor: activeStep === 1 ? 'primary.main' : '#e2e8f0',
              }}
            >
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                  <Psychology color={activeStep === 1 ? 'primary' : 'action'} />
                  <Typography variant="h6" fontWeight={600}>
                    2. Model Selection
                  </Typography>
                  {selectedModel && (
                    <Chip label="Selected" color="success" size="small" />
                  )}
                </Box>
                <Divider sx={{ mb: 2 }} />
                
                <ModelSelector
                  taskType={(taskType as any) || 'classification'}
                  selectedModel={selectedModel ? { 
                    modelId: selectedModel, 
                    taskType: (taskType as any) || 'classification', 
                    hyperparameters: {} 
                  } : null}
                  onModelSelect={handleModelSelect}
                  disabled={!currentDataset} // Always enable model selection if we have a dataset
                />
              </CardContent>
            </Card>

            {/* Section 3: Hyperparameters */}
            <Card
              sx={{
                border: activeStep === 2 ? '2px solid' : '1px solid #e2e8f0',
                borderColor: activeStep === 2 ? 'primary.main' : '#e2e8f0',
              }}
            >
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                  <Tune color={activeStep === 2 ? 'primary' : 'action'} />
                  <Typography variant="h6" fontWeight={600}>
                    3. Hyperparameters
                  </Typography>
                </Box>
                <Divider sx={{ mb: 2 }} />
                {selectedModel ? (
                  <Box>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Configure model hyperparameters
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Hyperparameter configuration interface will be implemented in the next phase
                    </Typography>
                  </Box>
                ) : (
                  <EmptyState
                    title="No Model Selected"
                    message="Select a model first to configure hyperparameters"
                    icon={<Tune sx={{ fontSize: 48 }} />}
                  />
                )}
              </CardContent>
            </Card>
          </Stack>

          {/* Navigation Buttons */}
          <Box sx={{ mt: 3, display: 'flex', gap: 2, justifyContent: 'space-between' }}>
            <Button
              onClick={handlePreviousStep}
              disabled={activeStep === 0 || isTraining}
              variant="outlined"
            >
              Previous
            </Button>
            <Box sx={{ display: 'flex', gap: 2 }}>
              {activeStep < workflowSteps.length - 1 ? (
                <Button
                  onClick={handleNextStep}
                  disabled={!canProceedToNextStep() || isTraining}
                  variant="contained"
                >
                  Next
                </Button>
              ) : (
                <>
                  <Button
                    variant="contained"
                    startIcon={<PlayArrow />}
                    disabled={!selectedModel || isTraining || !selectedDatasetId}
                  >
                    Start Training
                  </Button>
                  <Button
                    variant="outlined"
                    color="error"
                    startIcon={<Stop />}
                    disabled={!isTraining}
                  >
                    Stop Training
                  </Button>
                </>
              )}
            </Box>
          </Box>
        </Grid>

        {/* Right Column: Run Status & Progress */}
        <Grid size={{ xs: 12, lg: 4 }}>
          <Card sx={{ border: '1px solid #e2e8f0', position: 'sticky', top: 24 }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                <Timeline color="primary" />
                <Typography variant="h6" fontWeight={600}>
                  Training Status
                </Typography>
              </Box>
              <Divider sx={{ mb: 2 }} />

              {isTraining || trainingProgress > 0 ? (
                <TrainingProgress
                  progress={trainingProgress}
                  isTraining={isTraining}
                  metrics={trainingMetrics}
                  logs={logs}
                />
              ) : (
                <EmptyState
                  title="No Training in Progress"
                  message="Configure your model and click 'Start Training' to begin"
                  icon={<Timeline sx={{ fontSize: 48 }} />}
                />
              )}

              {/* Current Configuration Summary */}
              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                  Configuration Summary
                </Typography>
                <Stack spacing={1}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="caption" color="text.secondary">
                      Dataset:
                    </Typography>
                    <Typography variant="caption" fontWeight={500}>
                      {currentDataset?.name || 'Not selected'}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="caption" color="text.secondary">
                      Model:
                    </Typography>
                    <Typography variant="caption" fontWeight={500}>
                      {typeof selectedModel === 'string' ? selectedModel : (selectedModel as any)?.name || 'Not selected'}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="caption" color="text.secondary">
                      Status:
                    </Typography>
                    <Chip
                      label={isTraining ? 'Training' : 'Ready'}
                      color={isTraining ? 'warning' : 'default'}
                      size="small"
                    />
                  </Box>
                </Stack>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default ModelingPage;
