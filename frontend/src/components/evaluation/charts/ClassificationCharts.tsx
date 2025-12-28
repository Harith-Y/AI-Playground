/**
 * ClassificationCharts Component
 *
 * Container for all classification evaluation charts
 * Displays Confusion Matrix, ROC Curve, and PR Curve together
 */

import React from 'react';
import { Box, Alert, Typography, CircularProgress } from '@mui/material';
import ConfusionMatrixHeatmap from './ConfusionMatrixHeatmap';
import ROCCurve from './ROCCurve';
import PRCurve from './PRCurve';
import type { ClassificationMetrics } from '../../../types/evaluation';

/**
 * Props for ClassificationCharts component
 */
export interface ClassificationChartsProps {
  metrics: ClassificationMetrics;
  isLoading?: boolean;
  error?: string;
  showConfusionMatrix?: boolean;
  showROC?: boolean;
  showPR?: boolean;
}

const ClassificationCharts: React.FC<ClassificationChartsProps> = ({
  metrics,
  isLoading = false,
  error,
  showConfusionMatrix = true,
  showROC = true,
  showPR = true,
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

  // Check if we have any chart data
  const hasConfusionMatrix = showConfusionMatrix && metrics.confusionMatrix;
  const hasROC = showROC && metrics.rocCurve;
  const hasPR = showPR && metrics.prCurve;

  if (!hasConfusionMatrix && !hasROC && !hasPR) {
    return (
      <Alert severity="info" sx={{ my: 2 }}>
        <Typography variant="body2">
          No visualization data available. Confusion matrix, ROC curve, and PR curve require
          additional data from the model evaluation.
        </Typography>
      </Alert>
    );
  }

  return (
    <Box>
      <Box
        display="grid"
        gridTemplateColumns={{
          xs: '1fr',
          md: hasConfusionMatrix && (hasROC || hasPR) ? 'repeat(2, 1fr)' : '1fr',
        }}
        gap={3}
      >
        {/* Confusion Matrix */}
        {hasConfusionMatrix && (
          <Box>
            <ConfusionMatrixHeatmap
              confusionMatrix={metrics.confusionMatrix!}
              classNames={metrics.classNames}
              showPercentages={true}
              showCounts={true}
            />
          </Box>
        )}

        {/* ROC Curve */}
        {hasROC && (
          <Box>
            <ROCCurve
              fpr={metrics.rocCurve!.fpr}
              tpr={metrics.rocCurve!.tpr}
              thresholds={metrics.rocCurve!.thresholds}
              auc={metrics.auc}
              showDiagonal={true}
              showThresholds={false}
            />
          </Box>
        )}
      </Box>

      {/* PR Curve - Full Width */}
      {hasPR && (
        <Box mt={3}>
          <PRCurve
            precision={metrics.prCurve!.precision}
            recall={metrics.prCurve!.recall}
            thresholds={metrics.prCurve!.thresholds}
            showBaseline={true}
            showThresholds={false}
          />
        </Box>
      )}

      {/* Info Box */}
      {(hasROC || hasPR) && (
        <Alert severity="info" sx={{ mt: 3 }}>
          <Typography variant="body2" fontWeight="500" gutterBottom>
            Curve Interpretation Tips:
          </Typography>
          <Box display="flex" flexDirection="column" gap={0.5}>
            {hasROC && (
              <Typography variant="caption" color="text.secondary">
                • <strong>ROC Curve</strong>: Use when classes are balanced. Higher AUC =
                better discrimination ability.
              </Typography>
            )}
            {hasPR && (
              <Typography variant="caption" color="text.secondary">
                • <strong>PR Curve</strong>: Use when classes are imbalanced. Higher AP =
                better precision-recall trade-off.
              </Typography>
            )}
            <Typography variant="caption" color="text.secondary">
              • <strong>Green star</strong>: Marks the optimal threshold point for each curve.
            </Typography>
          </Box>
        </Alert>
      )}
    </Box>
  );
};

export default ClassificationCharts;
