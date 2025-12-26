import React, { useEffect } from 'react';
import { Typography, Box, Container, Grid, Card, CardContent } from '@mui/material';
import { TrendingUp } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import {
  fetchCorrelationMatrix,
  fetchFeatureImportance,
  clearFeatureError,
} from '../store/slices/featureSlice';
import LoadingState from '../components/common/LoadingState';
import ErrorState from '../components/common/ErrorState';
import EmptyState from '../components/common/EmptyState';
import CorrelationMatrix from '../components/features/CorrelationMatrix';
import FeatureImportance from '../components/features/FeatureImportance';

const FeatureEngineeringPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { correlationMatrix, featureImportance, isLoading, error } = useAppSelector(
    (state) => state.feature
  );
  const { currentDataset } = useAppSelector((state) => state.dataset);

  useEffect(() => {
    return () => {
      dispatch(clearFeatureError());
    };
  }, [dispatch]);

  const handleRetry = () => {
    dispatch(clearFeatureError());
    if (currentDataset?.id) {
      dispatch(fetchCorrelationMatrix(currentDataset.id));
      dispatch(fetchFeatureImportance(currentDataset.id));
    }
  };

  const handleRefreshCorrelation = () => {
    if (currentDataset?.id) {
      dispatch(fetchCorrelationMatrix(currentDataset.id));
    }
  };

  const handleRefreshImportance = () => {
    if (currentDataset?.id) {
      dispatch(fetchFeatureImportance(currentDataset.id));
    }
  };

  if (isLoading && !correlationMatrix && !featureImportance.length) {
    return <LoadingState message="Analyzing features..." />;
  }

  if (error && !correlationMatrix && !featureImportance.length) {
    return (
      <ErrorState
        title="Feature Analysis Failed"
        message={error}
        onRetry={handleRetry}
      />
    );
  }

  if (!currentDataset) {
    return (
      <EmptyState
        title="No Dataset Selected"
        message="Please upload a dataset first to analyze features"
      />
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Page Header */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <TrendingUp sx={{ fontSize: 40, color: 'primary.main' }} />
          <Box>
            <Typography variant="h4" component="h1" fontWeight={600}>
              Feature Engineering
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Analyze and select the most important features
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* Error Alert (non-blocking) */}
      {error && (correlationMatrix || featureImportance.length > 0) && (
        <ErrorState
          message={error}
          onRetry={handleRetry}
          variant="alert"
        />
      )}

      <Grid container spacing={3}>
        {/* Correlation Matrix */}
        <Grid item xs={12} lg={6}>
          <Card sx={{ border: '1px solid #e2e8f0' }}>
            <CardContent>
              <CorrelationMatrix
                matrix={correlationMatrix}
                isLoading={isLoading}
                onRefresh={handleRefreshCorrelation}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Feature Importance */}
        <Grid item xs={12} lg={6}>
          <Card sx={{ border: '1px solid #e2e8f0' }}>
            <CardContent>
              <FeatureImportance
                features={featureImportance}
                isLoading={isLoading}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default FeatureEngineeringPage;
