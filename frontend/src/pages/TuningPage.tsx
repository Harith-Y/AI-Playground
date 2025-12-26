import React, { useEffect } from 'react';
import { Typography, Box, Container, Grid, Card, CardContent, Button, LinearProgress } from '@mui/material';
import { Tune, PlayArrow, Stop } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import LoadingState from '../components/common/LoadingState';
import ErrorState from '../components/common/ErrorState';
import EmptyState from '../components/common/EmptyState';

const TuningPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { results, isTuning, progress, isLoading, error } = useAppSelector(
    (state) => state.tuning
  );
  const { currentModel } = useAppSelector((state) => state.modeling);

  const handleRetry = () => {
    // TODO: Implement retry logic
  };

  if (isLoading) {
    return <LoadingState message="Loading tuning configurations..." />;
  }

  if (error) {
    return (
      <ErrorState
        title="Tuning Failed"
        message={error}
        onRetry={handleRetry}
      />
    );
  }

  if (!currentModel) {
    return (
      <EmptyState
        title="No Model Selected"
        message="Please train a model first to optimize its hyperparameters"
      />
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Page Header */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <Tune sx={{ fontSize: 40, color: 'primary.main' }} />
          <Box>
            <Typography variant="h4" component="h1" fontWeight={600}>
              Hyperparameter Tuning
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Optimize model performance through hyperparameter search
            </Typography>
          </Box>
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* Tuning Configuration */}
        <Grid item xs={12} md={6}>
          <Card sx={{ border: '1px solid #e2e8f0' }}>
            <CardContent>
              <Typography variant="h6" fontWeight={600} gutterBottom>
                Tuning Configuration
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                Configure hyperparameter search space and method
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Tuning Controls */}
        <Grid item xs={12} md={6}>
          <Card sx={{ border: '1px solid #e2e8f0' }}>
            <CardContent>
              <Typography variant="h6" fontWeight={600} gutterBottom>
                Tuning Controls
              </Typography>
              <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<PlayArrow />}
                  disabled={isTuning}
                  fullWidth
                >
                  Start Tuning
                </Button>
                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<Stop />}
                  disabled={!isTuning}
                  fullWidth
                >
                  Stop Tuning
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Tuning Progress */}
        {isTuning && (
          <Grid item xs={12}>
            <Card sx={{ border: '1px solid #e2e8f0' }}>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Tuning Progress
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={progress}
                  sx={{ height: 8, borderRadius: 1, mt: 2 }}
                />
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                  {progress}% Complete
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Tuning Results */}
        {results.length > 0 && (
          <Grid item xs={12}>
            <Card sx={{ border: '1px solid #e2e8f0' }}>
              <CardContent>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  Tuning Results
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Best hyperparameter combinations will be displayed here
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Container>
  );
};

export default TuningPage;
