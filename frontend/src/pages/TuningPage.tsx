/**
 * Tuning Page
 * 
 * Dedicated hyperparameter tuning screen with sections for:
 * - Tuning configuration
 * - Progress tracking
 * - Results display
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Breadcrumbs,
  Link,
  Stepper,
  Step,
  StepLabel,
  Alert,
  Divider,
  Chip,
  IconButton,
  Tooltip,
  Stack,
  Button,
  Snackbar,
} from '@mui/material';
import {
  NavigateNext as NavigateNextIcon,
  Home as HomeIcon,
  Tune as TuneIcon,
  PlayArrow as PlayArrowIcon,
  Refresh as RefreshIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { useNavigate, useParams } from 'react-router-dom';
import TuningConfiguration from '../components/tuning/TuningConfiguration';
import TuningProgress from '../components/tuning/TuningProgress';
import TuningResults from '../components/tuning/TuningResults';
import { useTuning } from '../hooks/useTuning';
import type { TuningConfig } from '../services/tuningService';

// Placeholder components (to be implemented in subsequent tasks)

const TuningPage: React.FC = () => {
  const navigate = useNavigate();
  const { modelRunId } = useParams<{ modelRunId?: string }>();
  
  // State
  const [activeStep, setActiveStep] = useState(0);
  const [selectedModel] = useState<string>('random_forest_classifier');
  const [tuningConfig, setTuningConfig] = useState<TuningConfig | null>(null);
  const [snackbarMessage, setSnackbarMessage] = useState<string>('');
  const [showSnackbar, setShowSnackbar] = useState(false);

  // Tuning hook
  const {
    tuningId,
    progress,
    results,
    isLoading,
    error,
    start,
    applyConfig,
    exportResults,
    refresh,
    clearError,
  } = useTuning({
    onComplete: (results) => {
      setSnackbarMessage(`Tuning completed! Best score: ${results.best_score.toFixed(4)}`);
      setShowSnackbar(true);
      setActiveStep(2);
    },
    onError: (error) => {
      setSnackbarMessage(`Error: ${error.message}`);
      setShowSnackbar(true);
    },
  });

  // Derive tuning status from progress
  const tuningStatus = progress?.status || (results ? 'completed' : 'idle');
  
  // Steps for the tuning workflow
  const steps = [
    'Configure Tuning',
    'Run Tuning',
    'View Results',
  ];
  
  // Handle navigation
  const handleBreadcrumbClick = (path: string) => {
    navigate(path);
  };
  
  // Handle step change
  const handleStepClick = (step: number) => {
    if (step <= activeStep || tuningStatus === 'completed') {
      setActiveStep(step);
    }
  };

  // Handle start tuning
  const handleStartTuning = useCallback(async () => {
    if (!tuningConfig || !modelRunId) {
      setSnackbarMessage('Please configure tuning parameters first');
      setShowSnackbar(true);
      return;
    }

    try {
      await start({
        model_run_id: modelRunId,
        model_type: selectedModel,
        tuning_config: tuningConfig,
      });
      setActiveStep(1);
      setSnackbarMessage('Tuning started successfully');
      setShowSnackbar(true);
    } catch (err) {
      // Error handled by hook
    }
  }, [tuningConfig, modelRunId, selectedModel, start]);

  // Handle apply configuration
  const handleApplyConfig = useCallback(async (parameters: Record<string, any>) => {
    try {
      await applyConfig(parameters, true);
      setSnackbarMessage('Configuration applied successfully');
      setShowSnackbar(true);
    } catch (err) {
      // Error handled by hook
    }
  }, [applyConfig]);

  // Handle export
  const handleExport = useCallback(async () => {
    try {
      await exportResults();
      setSnackbarMessage('Results exported successfully');
      setShowSnackbar(true);
    } catch (err) {
      // Error handled by hook
    }
  }, [exportResults]);

  // Handle status change from progress component
  const handleStatusChange = useCallback((status: string) => {
    if (status === 'completed') {
      setActiveStep(2);
    }
  }, []);
  
  // Status color mapping
  const getStatusColor = (status: typeof tuningStatus) => {
    switch (status) {
      case 'running':
        return 'primary';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'paused':
        return 'warning';
      default:
        return 'default';
    }
  };
  
  // Status label mapping
  const getStatusLabel = (status: typeof tuningStatus) => {
    switch (status) {
      case 'running':
        return 'Tuning in Progress';
      case 'completed':
        return 'Tuning Completed';
      case 'failed':
        return 'Tuning Failed';
      case 'paused':
        return 'Paused';
      default:
        return 'Ready to Start';
    }
  };
  
  return (
    <Box sx={{ flexGrow: 1, bgcolor: 'background.default', minHeight: '100vh', py: 3 }}>
      <Container maxWidth="xl">
        {/* Header Section */}
        <Box sx={{ mb: 3 }}>
          {/* Breadcrumbs */}
          <Breadcrumbs
            separator={<NavigateNextIcon fontSize="small" />}
            aria-label="breadcrumb"
            sx={{ mb: 2 }}
          >
            <Link
              color="inherit"
              href="#"
              onClick={(e) => {
                e.preventDefault();
                handleBreadcrumbClick('/');
              }}
              sx={{ display: 'flex', alignItems: 'center', textDecoration: 'none' }}
            >
              <HomeIcon sx={{ mr: 0.5 }} fontSize="small" />
              Home
            </Link>
            <Link
              color="inherit"
              href="#"
              onClick={(e) => {
                e.preventDefault();
                handleBreadcrumbClick('/modeling');
              }}
              sx={{ textDecoration: 'none' }}
            >
              Modeling
            </Link>
            <Typography color="text.primary" sx={{ display: 'flex', alignItems: 'center' }}>
              <TuneIcon sx={{ mr: 0.5 }} fontSize="small" />
              Hyperparameter Tuning
            </Typography>
          </Breadcrumbs>
          
          {/* Page Title and Actions */}
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Box>
              <Typography variant="h4" component="h1" gutterBottom>
                Hyperparameter Tuning
              </Typography>
              <Typography variant="body1" color="text.secondary">
                Optimize model hyperparameters to improve performance
              </Typography>
            </Box>
            
            <Box display="flex" gap={1}>
              <Chip
                label={getStatusLabel(tuningStatus)}
                color={getStatusColor(tuningStatus)}
                icon={<TuneIcon />}
              />
              <Tooltip title="Refresh">
                <IconButton size="small" color="primary" onClick={refresh} disabled={isLoading}>
                  <RefreshIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Help">
                <IconButton size="small">
                  <InfoIcon />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        </Box>
        
        {/* Info Alert */}
        {tuningStatus === 'idle' && (
          <Alert severity="info" sx={{ mb: 3 }}>
            Configure your tuning parameters below and start the hyperparameter optimization process.
            The system will automatically search for the best parameter combination.
          </Alert>
        )}
        
        {/* Workflow Stepper */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Stepper activeStep={activeStep} alternativeLabel>
            {steps.map((label, index) => (
              <Step
                key={label}
                onClick={() => handleStepClick(index)}
                sx={{ cursor: index <= activeStep ? 'pointer' : 'default' }}
              >
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>
        </Paper>
        
        {/* Main Content Grid */}
        <Box sx={{ display: 'flex', gap: 3, flexDirection: { xs: 'column', lg: 'row' } }}>
          {/* Left Column - Configuration */}
          <Box sx={{ flex: { xs: '1 1 100%', lg: '0 0 33.333%' } }}>
            <Box sx={{ position: 'sticky', top: 80 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <TuneIcon sx={{ mr: 1 }} />
                Configuration
              </Typography>
              <Divider sx={{ mb: 2 }} />
              
              {/* Configuration Component */}
              <TuningConfiguration 
                modelType={selectedModel || 'random_forest_classifier'}
                onConfigChange={setTuningConfig}
                disabled={tuningStatus === 'running'}
              />
              
              {/* Quick Stats */}
              <Paper sx={{ p: 2, mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Quick Stats
                </Typography>
                <Box display="flex" flexDirection="column" gap={1}>
                  <Box display="flex" justifyContent="space-between">
                    <Typography variant="body2" color="text.secondary">
                      Model:
                    </Typography>
                    <Typography variant="body2" fontWeight="medium">
                      {selectedModel || 'Not selected'}
                    </Typography>
                  </Box>
                  <Box display="flex" justifyContent="space-between">
                    <Typography variant="body2" color="text.secondary">
                      Method:
                    </Typography>
                    <Typography variant="body2" fontWeight="medium">
                      {tuningConfig?.method?.replace(/_/g, ' ') || 'Random Search'}
                    </Typography>
                  </Box>
                  <Box display="flex" justifyContent="space-between">
                    <Typography variant="body2" color="text.secondary">
                      Iterations:
                    </Typography>
                    <Typography variant="body2" fontWeight="medium">
                      {tuningConfig?.n_iter || 50}
                    </Typography>
                  </Box>
                  <Box display="flex" justifyContent="space-between">
                    <Typography variant="body2" color="text.secondary">
                      CV Folds:
                    </Typography>
                    <Typography variant="body2" fontWeight="medium">
                      {tuningConfig?.cv_folds || 5}
                    </Typography>
                  </Box>
                  <Box display="flex" justifyContent="space-between">
                    <Typography variant="body2" color="text.secondary">
                      Parameters:
                    </Typography>
                    <Typography variant="body2" fontWeight="medium">
                      {tuningConfig?.parameters?.length || 0}
                    </Typography>
                  </Box>
                </Box>
              </Paper>
            </Box>
          </Box>
          
          {/* Right Column - Progress and Results */}
          <Box sx={{ flex: { xs: '1 1 100%', lg: '1 1 66.666%' } }}>
            {/* Progress Section */}
            {(activeStep === 1 || tuningStatus === 'running') && (
              <Box sx={{ mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Progress
                </Typography>
                <Divider sx={{ mb: 2 }} />
                <TuningProgress 
                  tuningId={tuningId || undefined}
                  onStatusChange={handleStatusChange}
                />
              </Box>
            )}
            
            {/* Results Section */}
            {(activeStep === 2 || tuningStatus === 'completed') && results && (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Results
                </Typography>
                <Divider sx={{ mb: 2 }} />
                <TuningResults 
                  resultsData={results}
                  onApplyConfig={handleApplyConfig}
                  onExport={handleExport}
                />
              </Box>
            )}
            
            {/* Initial State */}
            {activeStep === 0 && tuningStatus === 'idle' && (
              <Paper
                sx={{
                  p: 6,
                  textAlign: 'center',
                  bgcolor: 'background.default',
                  border: '2px dashed',
                  borderColor: 'divider',
                }}
              >
                <TuneIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  Ready to Start Tuning
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  Configure your tuning parameters in the left panel and click "Start Tuning" to begin
                  the hyperparameter optimization process.
                </Typography>
                <Box display="flex" gap={2} justifyContent="center">
                  <Button
                    variant="contained"
                    color="primary"
                    size="large"
                    startIcon={<PlayArrowIcon />}
                    onClick={handleStartTuning}
                    disabled={!tuningConfig || isLoading}
                  >
                    Start Tuning
                  </Button>
                </Box>
              </Paper>
            )}
          </Box>
        </Box>
        
        {/* Help Section */}
        <Paper sx={{ p: 3, mt: 3, bgcolor: 'info.lighter' }}>
          <Typography variant="h6" gutterBottom>
            💡 Tuning Tips
          </Typography>
          <Stack spacing={2} direction={{ xs: 'column', md: 'row' }}>
            <Box sx={{ flex: 1 }}>
              <Typography variant="subtitle2" gutterBottom>
                Choose the Right Method
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Grid Search for small search spaces, Random Search for larger spaces,
                and Bayesian Optimization for expensive evaluations.
              </Typography>
            </Box>
            <Box sx={{ flex: 1 }}>
              <Typography variant="subtitle2" gutterBottom>
                Define Search Space
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Focus on the most important hyperparameters first. Start with wider ranges
                and narrow down based on initial results.
              </Typography>
            </Box>
            <Box sx={{ flex: 1 }}>
              <Typography variant="subtitle2" gutterBottom>
                Monitor Progress
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Watch the progress in real-time and stop early if you find good parameters.
                You can always resume tuning later.
              </Typography>
            </Box>
          </Stack>
        </Paper>

        {/* Error Alert */}
        {error && (
          <Alert severity="error" sx={{ mt: 3 }} onClose={clearError}>
            {error.message}
          </Alert>
        )}

        {/* Loading Indicator */}
        {isLoading && (
          <Alert severity="info" sx={{ mt: 3 }}>
            Processing...
          </Alert>
        )}

        {/* Snackbar for notifications */}
        <Snackbar
          open={showSnackbar}
          autoHideDuration={6000}
          onClose={() => setShowSnackbar(false)}
          message={snackbarMessage}
        />
      </Container>
    </Box>
  );
};

export default TuningPage;
