/**
 * Experiment Summary Component
 * 
 * Displays a summary of the experiment used for code generation
 */

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Divider,
  Stack,
  LinearProgress,
} from '@mui/material';
import {
  Science as ScienceIcon,
  Dataset as DatasetIcon,
  TrendingUp as TrendingUpIcon,
  AccessTime as AccessTimeIcon,
} from '@mui/icons-material';

interface ExperimentSummaryProps {
  experimentName: string;
  datasetName: string;
  modelType: string;
  score: number;
  metrics?: Record<string, number>;
  createdAt?: string;
  parameters?: Record<string, any>;
}

const ExperimentSummary: React.FC<ExperimentSummaryProps> = ({
  experimentName,
  datasetName,
  modelType,
  score,
  metrics,
  createdAt,
  parameters,
}) => {
  const formatDate = (dateString?: string) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.9) return 'success';
    if (score >= 0.7) return 'warning';
    return 'error';
  };

  return (
    <Card>
      <CardContent>
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <ScienceIcon color="primary" />
          <Typography variant="h6">Experiment Summary</Typography>
        </Box>
        <Divider sx={{ mb: 2 }} />

        {/* Basic Info */}
        <Stack spacing={2}>
          <Box>
            <Typography variant="caption" color="text.secondary">
              Experiment Name
            </Typography>
            <Typography variant="body1" fontWeight="medium">
              {experimentName}
            </Typography>
          </Box>

          <Box>
            <Box display="flex" alignItems="center" gap={0.5} mb={0.5}>
              <DatasetIcon fontSize="small" color="action" />
              <Typography variant="caption" color="text.secondary">
                Dataset
              </Typography>
            </Box>
            <Typography variant="body2">{datasetName}</Typography>
          </Box>

          <Box>
            <Typography variant="caption" color="text.secondary">
              Model Type
            </Typography>
            <Box mt={0.5}>
              <Chip label={modelType} size="small" color="primary" variant="outlined" />
            </Box>
          </Box>

          {/* Score */}
          <Box>
            <Box display="flex" alignItems="center" gap={0.5} mb={0.5}>
              <TrendingUpIcon fontSize="small" color="action" />
              <Typography variant="caption" color="text.secondary">
                Performance Score
              </Typography>
            </Box>
            <Box display="flex" alignItems="center" gap={2}>
              <Typography variant="h5" color={`${getScoreColor(score)}.main`}>
                {(score * 100).toFixed(1)}%
              </Typography>
              <Box flex={1}>
                <LinearProgress
                  variant="determinate"
                  value={score * 100}
                  color={getScoreColor(score)}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>
            </Box>
          </Box>

          {/* Additional Metrics */}
          {metrics && Object.keys(metrics).length > 0 && (
            <Box>
              <Typography variant="caption" color="text.secondary" gutterBottom>
                Additional Metrics
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={1} mt={1}>
                {Object.entries(metrics).map(([key, value]) => (
                  <Chip
                    key={key}
                    label={`${key}: ${typeof value === 'number' ? value.toFixed(3) : value}`}
                    size="small"
                    variant="outlined"
                  />
                ))}
              </Box>
            </Box>
          )}

          {/* Parameters */}
          {parameters && Object.keys(parameters).length > 0 && (
            <Box>
              <Typography variant="caption" color="text.secondary" gutterBottom>
                Model Parameters
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={1} mt={1}>
                {Object.entries(parameters).slice(0, 5).map(([key, value]) => (
                  <Chip
                    key={key}
                    label={`${key}: ${value}`}
                    size="small"
                  />
                ))}
                {Object.keys(parameters).length > 5 && (
                  <Chip
                    label={`+${Object.keys(parameters).length - 5} more`}
                    size="small"
                    variant="outlined"
                  />
                )}
              </Box>
            </Box>
          )}

          {/* Created Date */}
          {createdAt && (
            <Box>
              <Box display="flex" alignItems="center" gap={0.5}>
                <AccessTimeIcon fontSize="small" color="action" />
                <Typography variant="caption" color="text.secondary">
                  Created: {formatDate(createdAt)}
                </Typography>
              </Box>
            </Box>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
};

export default ExperimentSummary;
