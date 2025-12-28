/**
 * MetricsDisplay Component
 *
 * Generic container for displaying metrics based on task type
 * Automatically renders the appropriate metrics component
 */

import React from 'react';
import { Box, Alert, CircularProgress, Typography } from '@mui/material';
import ClassificationMetrics from './ClassificationMetrics';
import RegressionMetrics from './RegressionMetrics';
import ClusteringMetrics from './ClusteringMetrics';
import type {
  ModelMetrics,
  ClassificationMetrics as ClassificationMetricsType,
  RegressionMetrics as RegressionMetricsType,
  ClusteringMetrics as ClusteringMetricsType,
  TaskType,
} from '../../../types/evaluation';
import { TaskType as TaskTypeEnum } from '../../../types/evaluation';

/**
 * Props for MetricsDisplay component
 */
export interface MetricsDisplayProps {
  taskType: TaskType;
  metrics: ModelMetrics | null;
  isLoading?: boolean;
  error?: string;
  showDescription?: boolean;
  compact?: boolean;
  emptyMessage?: string;
}

const MetricsDisplay: React.FC<MetricsDisplayProps> = ({
  taskType,
  metrics,
  isLoading = false,
  error,
  showDescription = true,
  compact = false,
  emptyMessage = 'No metrics available. Train a model to see evaluation metrics.',
}) => {
  // Loading state
  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" py={8}>
        <CircularProgress />
        <Typography variant="body1" color="text.secondary" sx={{ ml: 2 }}>
          Loading metrics...
        </Typography>
      </Box>
    );
  }

  // Error state
  if (error) {
    return (
      <Alert severity="error" sx={{ my: 2 }}>
        <Typography variant="body2" fontWeight="500">
          Error loading metrics
        </Typography>
        <Typography variant="body2" sx={{ mt: 0.5 }}>
          {error}
        </Typography>
      </Alert>
    );
  }

  // Empty state
  if (!metrics) {
    return (
      <Alert severity="info" sx={{ my: 2 }}>
        <Typography variant="body2">{emptyMessage}</Typography>
      </Alert>
    );
  }

  // Render appropriate metrics component based on task type
  try {
    switch (taskType) {
      case TaskTypeEnum.CLASSIFICATION:
        return (
          <ClassificationMetrics
            metrics={metrics as ClassificationMetricsType}
            showDescription={showDescription}
            compact={compact}
          />
        );

      case TaskTypeEnum.REGRESSION:
        return (
          <RegressionMetrics
            metrics={metrics as RegressionMetricsType}
            showDescription={showDescription}
            compact={compact}
          />
        );

      case TaskTypeEnum.CLUSTERING:
        return (
          <ClusteringMetrics
            metrics={metrics as ClusteringMetricsType}
            showDescription={showDescription}
            compact={compact}
          />
        );

      default:
        return (
          <Alert severity="warning" sx={{ my: 2 }}>
            <Typography variant="body2">
              Unsupported task type: {taskType}
            </Typography>
          </Alert>
        );
    }
  } catch (err) {
    return (
      <Alert severity="error" sx={{ my: 2 }}>
        <Typography variant="body2" fontWeight="500">
          Error rendering metrics
        </Typography>
        <Typography variant="body2" sx={{ mt: 0.5 }}>
          {err instanceof Error ? err.message : 'Unknown error occurred'}
        </Typography>
      </Alert>
    );
  }
};

export default MetricsDisplay;
