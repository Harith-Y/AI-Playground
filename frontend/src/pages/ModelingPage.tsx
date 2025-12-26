import React, { useEffect } from 'react';
import {
  Typography,
  Box,
  Paper,
  Button,
  Container,
  Grid,
  Card,
  CardContent,
} from '@mui/material';
import { ModelTraining, PlayArrow, Stop } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import {
  fetchModels,
  clearModelError,
  setSelectedModel,
} from '../store/slices/modelingSlice';
import LoadingState from '../components/common/LoadingState';
import ErrorState from '../components/common/ErrorState';
import EmptyState from '../components/common/EmptyState';
import TrainingProgress from '../components/modeling/TrainingProgress';

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

  useEffect(() => {
    dispatch(fetchModels());
    return () => {
      dispatch(clearModelError());
    };
  }, [dispatch]);

  const handleRetry = () => {
    dispatch(clearModelError());
    dispatch(fetchModels());
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
              Train and evaluate machine learning models
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* Error Alert */}
      {error && models.length > 0 && (
        <ErrorState
          message={error}
          onRetry={handleRetry}
          variant="alert"
        />
      )}

      <Grid container spacing={3}>
        {/* Model Configuration */}
        <Grid item xs={12} md={6}>
          <Card sx={{ border: '1px solid #e2e8f0' }}>
            <CardContent>
              <Typography variant="h6" fontWeight={600} gutterBottom>
                Model Configuration
              </Typography>
              {models.length === 0 ? (
                <EmptyState
                  title="No Models Available"
                  message="Upload a dataset to see available models"
                />
              ) : (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Model configuration UI will be implemented here
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Training Controls */}
        <Grid item xs={12} md={6}>
          <Card sx={{ border: '1px solid #e2e8f0' }}>
            <CardContent>
              <Typography variant="h6" fontWeight={600} gutterBottom>
                Training Controls
              </Typography>
              <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<PlayArrow />}
                  disabled={!selectedModel || isTraining}
                  fullWidth
                >
                  Start Training
                </Button>
                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<Stop />}
                  disabled={!isTraining}
                  fullWidth
                >
                  Stop Training
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Training Progress */}
        {(isTraining || trainingProgress > 0) && (
          <Grid item xs={12}>
            <Card sx={{ border: '1px solid #e2e8f0' }}>
              <CardContent>
                <TrainingProgress
                  progress={trainingProgress}
                  isTraining={isTraining}
                  metrics={trainingMetrics}
                  logs={logs}
                />
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Container>
  );
};

export default ModelingPage;
