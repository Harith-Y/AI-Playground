/**
 * ClusteringCharts Component
 *
 * Container for all clustering evaluation charts
 * Displays Silhouette Score, Inertia/Elbow Plot, and Cluster Distribution together
 */

import React from 'react';
import { Box, Alert, Typography, CircularProgress } from '@mui/material';
import SilhouetteScorePlot from './SilhouetteScorePlot';
import InertiaPlot from './InertiaPlot';
import ClusterDistributionPlot from './ClusterDistributionPlot';
import ClusterProjection2D from './ClusterProjection2D';
import type { ClusteringMetrics } from '../../../types/evaluation';

/**
 * Props for ClusteringCharts component
 */
export interface ClusteringChartsProps {
  metrics: ClusteringMetrics;
  projectionData?: {
    x: number[];
    y: number[];
    labels: number[];
  };
  isLoading?: boolean;
  error?: string;
  showSilhouette?: boolean;
  showInertia?: boolean;
  showDistribution?: boolean;
  showProjection?: boolean;
}

const ClusteringCharts: React.FC<ClusteringChartsProps> = ({
  metrics,
  projectionData,
  isLoading = false,
  error,
  showSilhouette = true,
  showInertia = true,
  showDistribution = true,
  showProjection = true,
}) => {
  // Loading state
  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" py={8}>
        <CircularProgress />
        <Typography variant="body1" color="text.secondary" sx={{ ml: 2 }}>
          Loading clustering charts...
        </Typography>
      </Box>
    );
  }

  // Error state
  if (error) {
    return (
      <Alert severity="error" sx={{ my: 2 }}>
        <Typography variant="body2" fontWeight="500">
          Error loading clustering charts
        </Typography>
        <Typography variant="body2" sx={{ mt: 0.5 }}>
          {error}
        </Typography>
      </Alert>
    );
  }

  // Check if we have required data
  if (!metrics || metrics.nClusters === undefined) {
    return (
      <Alert severity="info" sx={{ my: 2 }}>
        <Typography variant="body2">
          No clustering data available. Charts require clustering metrics from model evaluation.
        </Typography>
      </Alert>
    );
  }

  // Check which charts to display
  const hasAnything = showSilhouette || showInertia || showDistribution || showProjection;

  if (!hasAnything) {
    return (
      <Alert severity="info" sx={{ my: 2 }}>
        <Typography variant="body2">No charts enabled for display</Typography>
      </Alert>
    );
  }

  return (
    <Box sx={{ width: '100%' }}>
      {/* First Row: Silhouette and Inertia */}
      <Box
        display="grid"
        gridTemplateColumns={{ xs: '1fr', md: 'repeat(2, 1fr)' }}
        gap={3}
        mb={3}
      >
        {/* Silhouette Score Display */}
        {showSilhouette && (
          <SilhouetteScorePlot
            silhouetteScore={metrics.silhouetteScore}
            nClusters={metrics.nClusters}
          />
        )}

        {/* Inertia/Elbow Plot */}
        {showInertia && metrics.inertia !== undefined && (
          <InertiaPlot inertia={metrics.inertia} nClusters={metrics.nClusters} />
        )}
      </Box>

      {/* Second Row: Distribution and Projection */}
      <Box
        display="grid"
        gridTemplateColumns={{ xs: '1fr', md: 'repeat(2, 1fr)' }}
        gap={3}
        mb={3}
      >
        {/* Cluster Distribution */}
        {showDistribution && metrics.clusterSizes && metrics.clusterSizes.length > 0 && (
          <ClusterDistributionPlot
            clusterSizes={metrics.clusterSizes}
            nClusters={metrics.nClusters}
          />
        )}

        {/* 2D Projection Plot */}
        {showProjection && projectionData && (
          <ClusterProjection2D data={projectionData} nClusters={metrics.nClusters} />
        )}
      </Box>

      {/* Additional Metrics Display */}
      {(metrics.daviesBouldinScore !== undefined ||
        metrics.calinskiHarabaszScore !== undefined) && (
        <Alert severity="info" sx={{ my: 1 }}>
          <Typography variant="body2" fontWeight="500" gutterBottom>
            Additional Clustering Metrics
          </Typography>
          {metrics.daviesBouldinScore !== undefined && (
            <Typography variant="body2">
              Davies-Bouldin Score: {metrics.daviesBouldinScore.toFixed(4)} (lower is better)
            </Typography>
          )}
          {metrics.calinskiHarabaszScore !== undefined && (
            <Typography variant="body2">
              Calinski-Harabasz Score: {metrics.calinskiHarabaszScore.toFixed(2)} (higher is
              better)
            </Typography>
          )}
        </Alert>
      )}
    </Box>
  );
};

export default ClusteringCharts;
