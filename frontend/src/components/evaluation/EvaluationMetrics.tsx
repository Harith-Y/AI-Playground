import React from 'react';
import { Box, Typography, Grid, Card, CardContent } from '@mui/material';
import { Assessment } from '@mui/icons-material';
import LoadingState from '../common/LoadingState';
import EmptyState from '../common/EmptyState';

interface EvaluationMetricsProps {
  metrics?: any;
  isLoading?: boolean;
}

const EvaluationMetrics: React.FC<EvaluationMetricsProps> = ({
  metrics,
  isLoading = false,
}) => {
  if (isLoading) {
    return <LoadingState message="Calculating evaluation metrics..." variant="linear" />;
  }

  if (!metrics) {
    return (
      <EmptyState
        icon={<Assessment sx={{ fontSize: 64, color: 'text.secondary', opacity: 0.5 }} />}
        title="No Evaluation Metrics"
        message="Model evaluation metrics will appear here after evaluation"
      />
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
        <Assessment sx={{ color: 'primary.main' }} />
        <Typography variant="h6" fontWeight={600}>
          Evaluation Metrics
        </Typography>
      </Box>

      <Grid container spacing={2}>
        {Object.entries(metrics).map(([key, value]) => (
          <Grid size={{ xs: 12, sm: 6, md: 4 }} key={key}>
            <Card sx={{ border: '1px solid #e2e8f0', background: '#f8fafc' }}>
              <CardContent>
                <Typography variant="caption" color="text.secondary">
                  {key.replace(/_/g, ' ').toUpperCase()}
                </Typography>
                <Typography variant="h5" fontWeight={600} sx={{ mt: 1 }}>
                  {typeof value === 'number' ? value.toFixed(4) : String(value)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default EvaluationMetrics;
