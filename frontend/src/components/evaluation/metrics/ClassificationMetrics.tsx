/**
 * ClassificationMetrics Component
 *
 * Displays classification-specific metrics (accuracy, precision, recall, F1, AUC)
 */

import React from 'react';
import { Box, Typography, Paper, Divider } from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  BarChart as BarChartIcon,
  Timeline as TimelineIcon,
  TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';
import MetricCard, { MetricVariant } from './MetricCard';
import type { ClassificationMetrics as ClassificationMetricsType } from '../../../types/evaluation';

/**
 * Props for ClassificationMetrics component
 */
export interface ClassificationMetricsProps {
  metrics: ClassificationMetricsType;
  showDescription?: boolean;
  compact?: boolean;
}

const ClassificationMetrics: React.FC<ClassificationMetricsProps> = ({
  metrics,
  showDescription = true,
  compact = false,
}) => {
  // Determine variant based on metric value
  const getAccuracyVariant = (value: number): MetricVariant => {
    if (value >= 0.9) return MetricVariant.SUCCESS;
    if (value >= 0.7) return MetricVariant.DEFAULT;
    if (value >= 0.5) return MetricVariant.WARNING;
    return MetricVariant.ERROR;
  };

  const getF1Variant = (value: number): MetricVariant => {
    if (value >= 0.85) return MetricVariant.SUCCESS;
    if (value >= 0.65) return MetricVariant.DEFAULT;
    if (value >= 0.45) return MetricVariant.WARNING;
    return MetricVariant.ERROR;
  };

  const getAUCVariant = (value: number): MetricVariant => {
    if (value >= 0.95) return MetricVariant.SUCCESS;
    if (value >= 0.8) return MetricVariant.DEFAULT;
    if (value >= 0.6) return MetricVariant.WARNING;
    return MetricVariant.ERROR;
  };

  return (
    <Box>
      {/* Header */}
      {!compact && (
        <Box mb={3}>
          <Typography variant="h6" fontWeight="bold" gutterBottom>
            Classification Metrics
          </Typography>
          {showDescription && (
            <Typography variant="body2" color="text.secondary">
              Key performance indicators for classification model evaluation
            </Typography>
          )}
        </Box>
      )}

      {/* Primary Metrics Grid */}
      <Box
        display="grid"
        gridTemplateColumns={{
          xs: '1fr',
          sm: 'repeat(2, 1fr)',
          md: metrics.auc ? 'repeat(3, 1fr)' : 'repeat(4, 1fr)',
        }}
        gap={2}
        mb={3}
      >
        {/* Accuracy */}
        <MetricCard
          label="Accuracy"
          value={metrics.accuracy}
          format="percentage"
          precision={2}
          variant={getAccuracyVariant(metrics.accuracy)}
          icon={<CheckCircleIcon />}
          tooltip="Percentage of correct predictions out of all predictions"
          showProgress
          min={0}
          max={1}
          description={compact ? undefined : "Overall correctness"}
        />

        {/* Precision */}
        <MetricCard
          label="Precision"
          value={metrics.precision}
          format="percentage"
          precision={2}
          variant={metrics.precision >= 0.8 ? MetricVariant.SUCCESS : MetricVariant.DEFAULT}
          icon={<BarChartIcon />}
          tooltip="Percentage of true positives among all positive predictions"
          showProgress
          min={0}
          max={1}
          description={compact ? undefined : "Positive predictive value"}
        />

        {/* Recall */}
        <MetricCard
          label="Recall"
          value={metrics.recall}
          format="percentage"
          precision={2}
          variant={metrics.recall >= 0.8 ? MetricVariant.SUCCESS : MetricVariant.DEFAULT}
          icon={<TimelineIcon />}
          tooltip="Percentage of actual positives correctly identified"
          showProgress
          min={0}
          max={1}
          description={compact ? undefined : "True positive rate"}
        />

        {/* F1 Score */}
        <MetricCard
          label="F1 Score"
          value={metrics.f1Score}
          format="percentage"
          precision={2}
          variant={getF1Variant(metrics.f1Score)}
          icon={<TrendingUpIcon />}
          tooltip="Harmonic mean of precision and recall"
          showProgress
          min={0}
          max={1}
          description={compact ? undefined : "Balanced measure"}
        />
      </Box>

      {/* AUC (if available) */}
      {metrics.auc !== undefined && (
        <Box mb={3}>
          <Box
            display="grid"
            gridTemplateColumns={{
              xs: '1fr',
              sm: 'repeat(2, 1fr)',
              md: 'repeat(3, 1fr)',
            }}
            gap={2}
          >
            <MetricCard
              label="AUC-ROC"
              value={metrics.auc}
              format="percentage"
              precision={2}
              variant={getAUCVariant(metrics.auc)}
              tooltip="Area Under the ROC Curve - measures model's ability to distinguish between classes"
              showProgress
              min={0}
              max={1}
              description={compact ? undefined : "Discriminative power"}
            />
          </Box>
        </Box>
      )}

      {/* Metrics Summary */}
      {!compact && (
        <Paper elevation={1} sx={{ p: 2, bgcolor: 'action.hover' }}>
          <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
            Metrics Summary
          </Typography>
          <Divider sx={{ mb: 1.5 }} />

          <Box display="flex" flexDirection="column" gap={1}>
            <Box display="flex" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">
                Best Metric:
              </Typography>
              <Typography variant="body2" fontWeight="500">
                {getBestMetric(metrics)}
              </Typography>
            </Box>

            <Box display="flex" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">
                Average Score:
              </Typography>
              <Typography variant="body2" fontWeight="500">
                {getAverageScore(metrics).toFixed(2)}%
              </Typography>
            </Box>

            {metrics.confusionMatrix && (
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">
                  Confusion Matrix:
                </Typography>
                <Typography variant="body2" fontWeight="500">
                  {metrics.confusionMatrix.length}x{metrics.confusionMatrix[0]?.length || 0} Available
                </Typography>
              </Box>
            )}

            {metrics.rocCurve && (
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">
                  ROC Data Points:
                </Typography>
                <Typography variant="body2" fontWeight="500">
                  {metrics.rocCurve.fpr.length} points
                </Typography>
              </Box>
            )}

            {metrics.prCurve && (
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">
                  PR Curve Data:
                </Typography>
                <Typography variant="body2" fontWeight="500">
                  {metrics.prCurve.precision.length} points
                </Typography>
              </Box>
            )}
          </Box>
        </Paper>
      )}
    </Box>
  );
};

// Helper function to get best metric
const getBestMetric = (metrics: ClassificationMetricsType): string => {
  const scores = {
    'Accuracy': metrics.accuracy,
    'Precision': metrics.precision,
    'Recall': metrics.recall,
    'F1 Score': metrics.f1Score,
    ...(metrics.auc && { 'AUC': metrics.auc }),
  };

  const best = Object.entries(scores).reduce((a, b) => (a[1] > b[1] ? a : b));
  return `${best[0]} (${(best[1] * 100).toFixed(2)}%)`;
};

// Helper function to get average score
const getAverageScore = (metrics: ClassificationMetricsType): number => {
  const scores = [
    metrics.accuracy,
    metrics.precision,
    metrics.recall,
    metrics.f1Score,
  ];

  if (metrics.auc !== undefined) {
    scores.push(metrics.auc);
  }

  const sum = scores.reduce((a, b) => a + b, 0);
  return (sum / scores.length) * 100;
};

export default ClassificationMetrics;
