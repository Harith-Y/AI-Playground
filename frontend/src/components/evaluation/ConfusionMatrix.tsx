import React from 'react';
import { Box, Typography } from '@mui/material';
import { GridOn } from '@mui/icons-material';
import LoadingState from '../common/LoadingState';
import EmptyState from '../common/EmptyState';

interface ConfusionMatrixProps {
  matrix?: any;
  isLoading?: boolean;
}

const ConfusionMatrix: React.FC<ConfusionMatrixProps> = ({
  matrix,
  isLoading = false,
}) => {
  if (isLoading) {
    return <LoadingState message="Generating confusion matrix..." variant="linear" />;
  }

  if (!matrix) {
    return (
      <EmptyState
        icon={<GridOn sx={{ fontSize: 64, color: 'text.secondary', opacity: 0.5 }} />}
        title="No Confusion Matrix"
        message="Confusion matrix will appear here after model evaluation"
      />
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <GridOn sx={{ color: 'primary.main' }} />
        <Typography variant="h6" fontWeight={600}>
          Confusion Matrix
        </Typography>
      </Box>
      <Typography variant="body2" color="text.secondary">
        Confusion matrix visualization will be implemented here
      </Typography>
    </Box>
  );
};

export default ConfusionMatrix;
