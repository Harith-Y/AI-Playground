/**
 * InertiaPlot Component
 *
 * Displays inertia (within-cluster sum of squares) metric
 * Lower inertia values indicate tighter clusters
 */

import React from 'react';
import { Box, Card, CardContent, Typography, Chip } from '@mui/material';
import { TrendingDown, Info } from '@mui/icons-material';
import Tooltip from '@mui/material/Tooltip';

/**
 * Props for InertiaPlot component
 */
export interface InertiaPlotProps {
  inertia: number;
  nClusters: number;
  previousInertia?: number;
}

/**
 * Format large numbers with K, M suffixes
 */
const formatInertia = (value: number): string => {
  if (value >= 1000000) {
    return `${(value / 1000000).toFixed(2)}M`;
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(2)}K`;
  }
  return value.toFixed(2);
};

const InertiaPlot: React.FC<InertiaPlotProps> = ({
  inertia,
  nClusters,
  previousInertia,
}) => {
  // Calculate improvement if previous inertia is available
  const hasImprovement = previousInertia !== undefined && previousInertia > 0;
  const improvement = hasImprovement
    ? ((previousInertia! - inertia) / previousInertia!) * 100
    : 0;
  const isImprovement = improvement > 0;

  return (
    <Card sx={{ height: '100%', border: '1px solid #e2e8f0' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6" fontWeight={600}>
            Inertia (WCSS)
          </Typography>
          <Tooltip
            title="Within-Cluster Sum of Squares. Measures how internally coherent clusters are. Lower values indicate tighter, more compact clusters."
            arrow
          >
            <Info sx={{ color: 'text.secondary', fontSize: 20, cursor: 'help' }} />
          </Tooltip>
        </Box>

        {/* Inertia Value Display */}
        <Box sx={{ textAlign: 'center', my: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
            <TrendingDown sx={{ fontSize: 32, color: '#3b82f6' }} />
            <Typography
              variant="h2"
              fontWeight={700}
              sx={{ color: '#3b82f6' }}
            >
              {formatInertia(inertia)}
            </Typography>
          </Box>
          <Typography variant="subtitle2" color="text.secondary" sx={{ mt: 1 }}>
            Within-Cluster Sum of Squares
          </Typography>
        </Box>

        {/* Improvement Indicator */}
        {hasImprovement && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
            <Chip
              label={`${isImprovement ? '↓' : '↑'} ${Math.abs(improvement).toFixed(1)}% ${
                isImprovement ? 'improvement' : 'increase'
              }`}
              size="small"
              color={isImprovement ? 'success' : 'error'}
              sx={{ fontWeight: 600 }}
            />
          </Box>
        )}

        {/* Cluster Info */}
        <Box
          sx={{
            mt: 3,
            p: 2,
            bgcolor: '#f8fafc',
            borderRadius: 1,
            border: '1px solid #e2e8f0',
          }}
        >
          <Typography variant="body2" color="text.secondary" gutterBottom>
            <strong>Number of Clusters:</strong> {nClusters}
          </Typography>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            <strong>Raw Inertia:</strong> {inertia.toFixed(2)}
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
            Lower inertia indicates more compact clusters. Use the Elbow Method to find the optimal
            number of clusters by plotting inertia vs. k.
          </Typography>
        </Box>

        {/* Visual Indicator */}
        <Box sx={{ mt: 3 }}>
          <Typography variant="caption" color="text.secondary" fontWeight={600} gutterBottom>
            Compactness Level
          </Typography>
          <Box
            sx={{
              mt: 1,
              height: 8,
              borderRadius: 1,
              background: 'linear-gradient(to right, #22c55e, #f59e0b, #ef4444)',
              position: 'relative',
            }}
          >
            {/* Indicator marker - simplified as we don't have baseline */}
            <Box
              sx={{
                position: 'absolute',
                top: -4,
                left: '50%',
                width: 16,
                height: 16,
                borderRadius: '50%',
                bgcolor: '#3b82f6',
                border: '2px solid white',
                boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
              }}
            />
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
            <Typography variant="caption" color="text.secondary">
              Tight
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Loose
            </Typography>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default InertiaPlot;
