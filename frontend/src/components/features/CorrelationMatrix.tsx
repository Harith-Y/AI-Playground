import React from 'react';
import { Box, Typography, Button } from '@mui/material';
import { GridOn, Refresh } from '@mui/icons-material';
import LoadingState from '../common/LoadingState';
import EmptyState from '../common/EmptyState';
import CorrelationHeatmap from './CorrelationHeatmap';

interface CorrelationMatrixData {
  features: string[];
  matrix: number[][];
}

interface CorrelationMatrixProps {
  matrix?: CorrelationMatrixData | null;
  isLoading?: boolean;
  onRefresh?: () => void;
}

const CorrelationMatrix: React.FC<CorrelationMatrixProps> = ({
  matrix,
  isLoading = false,
  onRefresh,
}) => {
  if (isLoading) {
    return <LoadingState message="Calculating correlations..." variant="linear" />;
  }

  if (!matrix || !matrix.features || !matrix.matrix || matrix.features.length === 0) {
    return (
      <Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2, justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <GridOn sx={{ color: 'primary.main' }} />
            <Typography variant="h6" fontWeight={600}>
              Correlation Matrix
            </Typography>
          </Box>
          {onRefresh && (
            <Button
              size="small"
              startIcon={<Refresh />}
              onClick={onRefresh}
            >
              Calculate
            </Button>
          )}
        </Box>
        <EmptyState
          icon={<GridOn sx={{ fontSize: 64, color: 'text.secondary', opacity: 0.5 }} />}
          title="No Correlation Data"
          message="Feature correlation analysis will appear here once computed"
        />
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3, justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <GridOn sx={{ color: 'primary.main' }} />
          <Typography variant="h6" fontWeight={600}>
            Correlation Matrix
          </Typography>
        </Box>
        {onRefresh && (
          <Button
            size="small"
            startIcon={<Refresh />}
            onClick={onRefresh}
            variant="outlined"
          >
            Refresh
          </Button>
        )}
      </Box>

      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Pearson correlation coefficients between numerical features. Hover over cells for details.
      </Typography>

      <CorrelationHeatmap
        features={matrix.features}
        matrix={matrix.matrix}
      />
    </Box>
  );
};

export default CorrelationMatrix;
