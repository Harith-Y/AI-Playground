/**
 * RegressionCharts Component
 *
 * Container for all regression evaluation charts
 * Displays Residual Plot, Actual vs Predicted, and Error Distribution together
 */

import React from 'react';
import { Box, Alert, Typography, CircularProgress } from '@mui/material';
import ResidualPlot from './ResidualPlot';
import ActualVsPredictedPlot from './ActualVsPredictedPlot';
import ErrorDistributionPlot from './ErrorDistributionPlot';
import type { RegressionMetrics } from '../../../types/evaluation';

/**
 * Props for RegressionCharts component
 */
export interface RegressionChartsProps {
  metrics: RegressionMetrics;
  isLoading?: boolean;
  error?: string;
  showResidualPlot?: boolean;
  showActualVsPredicted?: boolean;
  showErrorDistribution?: boolean;
}

const RegressionCharts: React.FC<RegressionChartsProps> = ({
  metrics,
  isLoading = false,
  error,
  showResidualPlot = true,
  showActualVsPredicted = true,
  showErrorDistribution = true,
}) => {
  // Loading state
  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" py={8}>
        <CircularProgress />
        <Typography variant="body1" color="text.secondary" sx={{ ml: 2 }}>
          Loading charts...
        </Typography>
      </Box>
    );
  }

  // Error state
  if (error) {
    return (
      <Alert severity="error" sx={{ my: 2 }}>
        <Typography variant="body2" fontWeight="500">
          Error loading charts
        </Typography>
        <Typography variant="body2" sx={{ mt: 0.5 }}>
          {error}
        </Typography>
      </Alert>
    );
  }

  // Check if we have required data
  if (!metrics.predicted || !metrics.actual) {
    return (
      <Alert severity="info" sx={{ my: 2 }}>
        <Typography variant="body2">
          No visualization data available. Regression charts require predicted and actual values
          from the model evaluation.
        </Typography>
      </Alert>
    );
  }

  // Validate data lengths
  if (metrics.predicted.length !== metrics.actual.length) {
    return (
      <Alert severity="error" sx={{ my: 2 }}>
        <Typography variant="body2">
          Invalid data: Predicted and Actual arrays must have the same length
        </Typography>
      </Alert>
    );
  }

  if (metrics.predicted.length === 0) {
    return (
      <Alert severity="warning" sx={{ my: 2 }}>
        <Typography variant="body2">
          No data points available for visualization
        </Typography>
      </Alert>
    );
  }

  // Check which charts to display
  const hasResidualPlot = showResidualPlot;
  const hasActualVsPredicted = showActualVsPredicted;
  const hasErrorDistribution = showErrorDistribution;

  if (!hasResidualPlot && !hasActualVsPredicted && !hasErrorDistribution) {
    return (
      <Alert severity="info" sx={{ my: 2 }}>
        <Typography variant="body2">
          No charts enabled. Please enable at least one chart to visualize regression metrics.
        </Typography>
      </Alert>
    );
  }

  return (
    <Box>
      {/* Top Row: Residual Plot and Actual vs Predicted */}
      <Box
        display="grid"
        gridTemplateColumns={{
          xs: '1fr',
          md:
            hasResidualPlot && hasActualVsPredicted
              ? 'repeat(2, 1fr)'
              : '1fr',
        }}
        gap={3}
      >
        {/* Residual Plot */}
        {hasResidualPlot && (
          <Box>
            <ResidualPlot
              predicted={metrics.predicted}
              actual={metrics.actual}
              showZeroLine={true}
              showTrendLine={true}
            />
          </Box>
        )}

        {/* Actual vs Predicted Plot */}
        {hasActualVsPredicted && (
          <Box>
            <ActualVsPredictedPlot
              predicted={metrics.predicted}
              actual={metrics.actual}
              showDiagonal={true}
              showRegressionLine={true}
            />
          </Box>
        )}
      </Box>

      {/* Bottom Row: Error Distribution - Full Width */}
      {hasErrorDistribution && (
        <Box mt={3}>
          <ErrorDistributionPlot
            predicted={metrics.predicted}
            actual={metrics.actual}
            defaultView="both"
            showNormalCurve={true}
          />
        </Box>
      )}

      {/* Info Box */}
      <Alert severity="info" sx={{ mt: 3 }}>
        <Typography variant="body2" fontWeight="500" gutterBottom>
          Regression Chart Interpretation Tips:
        </Typography>
        <Box display="flex" flexDirection="column" gap={0.5}>
          {hasResidualPlot && (
            <Typography variant="caption" color="text.secondary">
              • <strong>Residual Plot</strong>: Check for patterns (should be random scatter
              around zero). Patterns indicate model bias or missing features.
            </Typography>
          )}
          {hasActualVsPredicted && (
            <Typography variant="caption" color="text.secondary">
              • <strong>Actual vs Predicted</strong>: Points should cluster near diagonal line.
              Systematic deviations indicate bias.
            </Typography>
          )}
          {hasErrorDistribution && (
            <Typography variant="caption" color="text.secondary">
              • <strong>Error Distribution</strong>: Should be approximately normal (bell-shaped)
              and centered at zero for unbiased predictions.
            </Typography>
          )}
          <Typography variant="caption" color="text.secondary">
            • <strong>Color coding</strong>: Green = good/low error, Orange = moderate, Red =
            high error or outliers
          </Typography>
        </Box>
      </Alert>
    </Box>
  );
};

export default RegressionCharts;
