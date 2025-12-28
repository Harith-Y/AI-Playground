/**
 * RegressionMetrics Component
 *
 * Displays regression-specific metrics (MAE, MSE, RMSE, R², MAPE)
 */

import React from 'react';
import { Box, Typography, Paper, Divider, Chip } from '@mui/material';
import {
  ShowChart as ShowChartIcon,
  TrendingUp as TrendingUpIcon,
  Analytics as AnalyticsIcon,
  Assessment as AssessmentIcon,
  Percent as PercentIcon,
} from '@mui/icons-material';
import MetricCard, { MetricVariant } from './MetricCard';
import type { RegressionMetrics as RegressionMetricsType } from '../../../types/evaluation';

/**
 * Props for RegressionMetrics component
 */
export interface RegressionMetricsProps {
  metrics: RegressionMetricsType;
  showDescription?: boolean;
  compact?: boolean;
}

const RegressionMetrics: React.FC<RegressionMetricsProps> = ({
  metrics,
  showDescription = true,
  compact = false,
}) => {
  // Determine R² variant
  const getR2Variant = (value: number): MetricVariant => {
    if (value >= 0.9) return MetricVariant.SUCCESS;
    if (value >= 0.7) return MetricVariant.DEFAULT;
    if (value >= 0.5) return MetricVariant.WARNING;
    return MetricVariant.ERROR;
  };

  // Determine MAPE variant (lower is better)
  const getMAPEVariant = (value: number): MetricVariant => {
    if (value <= 10) return MetricVariant.SUCCESS;
    if (value <= 20) return MetricVariant.DEFAULT;
    if (value <= 30) return MetricVariant.WARNING;
    return MetricVariant.ERROR;
  };

  // Format large numbers
  const formatLargeNumber = (value: number): string => {
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(2)}M`;
    } else if (value >= 1000) {
      return `${(value / 1000).toFixed(2)}K`;
    }
    return value.toFixed(2);
  };

  return (
    <Box>
      {/* Header */}
      {!compact && (
        <Box mb={3}>
          <Typography variant="h6" fontWeight="bold" gutterBottom>
            Regression Metrics
          </Typography>
          {showDescription && (
            <Typography variant="body2" color="text.secondary">
              Key performance indicators for regression model evaluation
            </Typography>
          )}
        </Box>
      )}

      {/* Error Metrics Grid */}
      <Box mb={3}>
        <Typography variant="subtitle2" fontWeight="bold" gutterBottom sx={{ mb: 1.5 }}>
          Error Metrics
          <Chip label="Lower is better" size="small" sx={{ ml: 1 }} />
        </Typography>

        <Box
          display="grid"
          gridTemplateColumns={{
            xs: '1fr',
            sm: 'repeat(2, 1fr)',
            md: 'repeat(3, 1fr)',
          }}
          gap={2}
        >
          {/* MAE */}
          <MetricCard
            label="MAE"
            value={metrics.mae}
            format="number"
            precision={2}
            icon={<ShowChartIcon />}
            tooltip="Mean Absolute Error - average magnitude of errors"
            description={compact ? undefined : "Avg absolute error"}
          />

          {/* MSE */}
          <MetricCard
            label="MSE"
            value={metrics.mse}
            format="number"
            precision={2}
            icon={<AnalyticsIcon />}
            tooltip="Mean Squared Error - average of squared errors"
            description={compact ? undefined : "Avg squared error"}
          />

          {/* RMSE */}
          <MetricCard
            label="RMSE"
            value={metrics.rmse}
            format="number"
            precision={2}
            icon={<AssessmentIcon />}
            tooltip="Root Mean Squared Error - square root of MSE"
            description={compact ? undefined : "Standard deviation"}
          />
        </Box>
      </Box>

      {/* Goodness of Fit Metrics */}
      <Box mb={3}>
        <Typography variant="subtitle2" fontWeight="bold" gutterBottom sx={{ mb: 1.5 }}>
          Goodness of Fit
          <Chip label="Higher is better" size="small" color="primary" sx={{ ml: 1 }} />
        </Typography>

        <Box
          display="grid"
          gridTemplateColumns={{
            xs: '1fr',
            sm: 'repeat(2, 1fr)',
            md: metrics.mape !== undefined ? 'repeat(2, 1fr)' : '1fr',
          }}
          gap={2}
        >
          {/* R² Score */}
          <MetricCard
            label="R² Score"
            value={metrics.r2}
            format="percentage"
            precision={2}
            variant={getR2Variant(metrics.r2)}
            icon={<TrendingUpIcon />}
            tooltip="Coefficient of Determination - proportion of variance explained by the model"
            showProgress
            min={0}
            max={1}
            description={compact ? undefined : "Variance explained"}
          />

          {/* MAPE (if available) */}
          {metrics.mape !== undefined && (
            <MetricCard
              label="MAPE"
              value={metrics.mape}
              format="decimal"
              precision={2}
              unit="%"
              variant={getMAPEVariant(metrics.mape)}
              icon={<PercentIcon />}
              tooltip="Mean Absolute Percentage Error - average percentage error"
              description={compact ? undefined : "Avg % error"}
            />
          )}
        </Box>
      </Box>

      {/* Metrics Summary */}
      {!compact && (
        <Paper elevation={1} sx={{ p: 2, bgcolor: 'grey.50' }}>
          <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
            Metrics Summary
          </Typography>
          <Divider sx={{ mb: 1.5 }} />

          <Box display="flex" flexDirection="column" gap={1}>
            <Box display="flex" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">
                Model Quality:
              </Typography>
              <Typography variant="body2" fontWeight="500">
                {getModelQuality(metrics.r2)}
              </Typography>
            </Box>

            <Box display="flex" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">
                R² Score:
              </Typography>
              <Typography variant="body2" fontWeight="500" color={getR2Variant(metrics.r2)}>
                {(metrics.r2 * 100).toFixed(2)}% variance explained
              </Typography>
            </Box>

            <Box display="flex" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">
                Error Range:
              </Typography>
              <Typography variant="body2" fontWeight="500">
                MAE: {formatLargeNumber(metrics.mae)} | RMSE: {formatLargeNumber(metrics.rmse)}
              </Typography>
            </Box>

            {metrics.mape !== undefined && (
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">
                  Percentage Error:
                </Typography>
                <Typography variant="body2" fontWeight="500" color={getMAPEVariant(metrics.mape)}>
                  {metrics.mape.toFixed(2)}% (MAPE)
                </Typography>
              </Box>
            )}

            {metrics.residuals && (
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">
                  Residuals Data:
                </Typography>
                <Typography variant="body2" fontWeight="500">
                  {metrics.residuals.predicted.length} predictions
                </Typography>
              </Box>
            )}

            {metrics.actualVsPredicted && (
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">
                  Prediction Data:
                </Typography>
                <Typography variant="body2" fontWeight="500">
                  {metrics.actualVsPredicted.actual.length} samples
                </Typography>
              </Box>
            )}
          </Box>
        </Paper>
      )}

      {/* Error Interpretation Guide */}
      {!compact && (
        <Paper elevation={0} sx={{ p: 2, mt: 2, bgcolor: 'info.lighter', border: '1px solid', borderColor: 'info.light' }}>
          <Typography variant="caption" fontWeight="bold" color="info.dark" display="block" gutterBottom>
            Understanding Error Metrics
          </Typography>
          <Box display="flex" flexDirection="column" gap={0.5}>
            <Typography variant="caption" color="text.secondary">
              • <strong>MAE</strong>: Average absolute difference between predicted and actual values
            </Typography>
            <Typography variant="caption" color="text.secondary">
              • <strong>MSE</strong>: Penalizes larger errors more heavily than MAE
            </Typography>
            <Typography variant="caption" color="text.secondary">
              • <strong>RMSE</strong>: Same units as target variable, easier to interpret
            </Typography>
            <Typography variant="caption" color="text.secondary">
              • <strong>R²</strong>: 1.0 = perfect fit, 0.0 = no better than mean, negative = worse than mean
            </Typography>
            {metrics.mape !== undefined && (
              <Typography variant="caption" color="text.secondary">
                • <strong>MAPE</strong>: Scale-independent, useful for comparing across datasets
              </Typography>
            )}
          </Box>
        </Paper>
      )}
    </Box>
  );
};

// Helper function to determine model quality based on R²
const getModelQuality = (r2: number): string => {
  if (r2 >= 0.9) return 'Excellent';
  if (r2 >= 0.8) return 'Very Good';
  if (r2 >= 0.7) return 'Good';
  if (r2 >= 0.5) return 'Moderate';
  if (r2 >= 0.3) return 'Fair';
  return 'Poor';
};

export default RegressionMetrics;
