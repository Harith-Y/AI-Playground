import React, { useEffect } from 'react';
import { Typography, Box, Container, Grid, Card, CardContent, Button } from '@mui/material';
import { Transform, Add } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import LoadingState from '../components/common/LoadingState';
import ErrorState from '../components/common/ErrorState';
import EmptyState from '../components/common/EmptyState';

const PreprocessingPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { steps, isProcessing, error } = useAppSelector((state) => state.preprocessing);
  const { currentDataset } = useAppSelector((state) => state.dataset);

  const handleRetry = () => {
    // TODO: Implement retry logic
  };

  if (isProcessing) {
    return <LoadingState message="Processing data transformations..." />;
  }

  if (error) {
    return (
      <ErrorState
        title="Preprocessing Failed"
        message={error}
        onRetry={handleRetry}
        variant="alert"
      />
    );
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
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Page Header */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <Transform sx={{ fontSize: 40, color: 'primary.main' }} />
          <Box>
            <Typography variant="h4" component="h1" fontWeight={600}>
              Data Preprocessing
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Clean and transform your dataset
            </Typography>
          </Box>
        </Box>
      </Box>

      {error && (
        <ErrorState message={error} onRetry={handleRetry} variant="alert" />
      )}

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card sx={{ border: '1px solid #e2e8f0' }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" fontWeight={600}>
                  Preprocessing Steps
                </Typography>
                <Button variant="contained" startIcon={<Add />} size="small">
                  Add Step
                </Button>
              </Box>
              {steps.length === 0 ? (
                <EmptyState
                  title="No Preprocessing Steps"
                  message="Add preprocessing steps to clean and transform your data"
                />
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Preprocessing steps will be displayed here
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default PreprocessingPage;
