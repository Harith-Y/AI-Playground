import React, { useEffect } from 'react';
import { Typography, Box, Container, Grid, Card, CardContent } from '@mui/material';
import { Assessment } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import LoadingState from '../components/common/LoadingState';
import ErrorState from '../components/common/ErrorState';
import EmptyState from '../components/common/EmptyState';
import EvaluationMetrics from '../components/evaluation/EvaluationMetrics';
import ConfusionMatrix from '../components/evaluation/ConfusionMatrix';

const ExplorationPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { metrics, confusionMatrix, isEvaluating, error } = useAppSelector(
    (state) => state.evaluation
  );
  const { currentModel } = useAppSelector((state) => state.modeling);

  const handleRetry = () => {
    // TODO: Implement retry logic
  };

  if (isEvaluating) {
    return <LoadingState message="Evaluating model performance..." />;
  }

  if (error) {
    return (
      <ErrorState
        title="Evaluation Failed"
        message={error}
        onRetry={handleRetry}
      />
    );
  }

  if (!currentModel) {
    return (
      <EmptyState
        title="No Model Selected"
        message="Please train a model first to evaluate its performance"
      />
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Page Header */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <Assessment sx={{ fontSize: 40, color: 'primary.main' }} />
          <Box>
            <Typography variant="h4" component="h1" fontWeight={600}>
              Model Evaluation
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Analyze and evaluate model performance
            </Typography>
          </Box>
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* Evaluation Metrics */}
        <Grid item xs={12}>
          <Card sx={{ border: '1px solid #e2e8f0' }}>
            <CardContent>
              <EvaluationMetrics metrics={metrics} />
            </CardContent>
          </Card>
        </Grid>

        {/* Confusion Matrix */}
        {confusionMatrix && (
          <Grid item xs={12} md={6}>
            <Card sx={{ border: '1px solid #e2e8f0' }}>
              <CardContent>
                <ConfusionMatrix matrix={confusionMatrix} />
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Container>
  );
};

export default ExplorationPage;
