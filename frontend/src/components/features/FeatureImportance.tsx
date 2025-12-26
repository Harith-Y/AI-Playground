import React from 'react';
import { Box, Typography } from '@mui/material';
import { BarChart } from '@mui/icons-material';
import LoadingState from '../common/LoadingState';
import EmptyState from '../common/EmptyState';

interface FeatureImportanceProps {
  features?: any[];
  isLoading?: boolean;
}

const FeatureImportance: React.FC<FeatureImportanceProps> = ({
  features,
  isLoading = false,
}) => {
  if (isLoading) {
    return <LoadingState message="Calculating feature importance..." variant="linear" />;
  }

  if (!features || features.length === 0) {
    return (
      <EmptyState
        icon={<BarChart sx={{ fontSize: 64, color: 'text.secondary', opacity: 0.5 }} />}
        title="No Feature Importance Data"
        message="Feature importance scores will appear here after model training"
      />
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <BarChart sx={{ color: 'primary.main' }} />
        <Typography variant="h6" fontWeight={600}>
          Feature Importance
        </Typography>
      </Box>
      <Typography variant="body2" color="text.secondary">
        Feature importance chart will be implemented here
      </Typography>
    </Box>
  );
};

export default FeatureImportance;
