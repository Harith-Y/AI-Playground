import React from 'react';
import { Box, Typography } from '@mui/material';
import { GridOn } from '@mui/icons-material';
import LoadingState from '../common/LoadingState';
import EmptyState from '../common/EmptyState';

interface CorrelationMatrixProps {
  matrix?: any;
  isLoading?: boolean;
}

const CorrelationMatrix: React.FC<CorrelationMatrixProps> = ({
  matrix,
  isLoading = false,
}) => {
  if (isLoading) {
    return <LoadingState message="Calculating correlations..." variant="linear" />;
  }

  if (!matrix) {
    return (
      <EmptyState
        icon={<GridOn sx={{ fontSize: 64, color: 'text.secondary', opacity: 0.5 }} />}
        title="No Correlation Data"
        message="Feature correlation analysis will appear here once computed"
      />
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <GridOn sx={{ color: 'primary.main' }} />
        <Typography variant="h6" fontWeight={600}>
          Correlation Matrix
        </Typography>
      </Box>
      <Typography variant="body2" color="text.secondary">
        Correlation matrix visualization will be implemented here
      </Typography>
    </Box>
  );
};

export default CorrelationMatrix;
