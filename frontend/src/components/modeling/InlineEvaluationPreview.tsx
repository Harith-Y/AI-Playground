import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  Button,
  Alert,
  Skeleton,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  Assessment as AssessmentIcon,
  ArrowForward as ArrowForwardIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import Plot from 'react-plotly.js';

export interface InlineEvaluationMetrics {
  // Classification metrics
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1Score?: number;
  rocAuc?: number;
  
  // Regression metrics
  rmse?: number;
  mae?: number;
  r2Score?: number;
  mse?: number;
  
  // Training metrics
  trainScore?: number;
  valScore?: number;
  testScore?: number;
}

export interface InlineEvaluationData {
  runId: string;
  modelName: string;
  modelType: string;
  status: 'completed' | 'failed' | 'running';
  metrics: InlineEvaluationMetrics;
  trainingHistory?: {
    epoch: number[];
    trainLoss?: number[];
    valLoss?: number[];
    trainAccuracy?: number[];
    valAccuracy?: number[];
  };
  taskType?: 'classification' | 'regression' | 'clustering';
}

export interface InlineEvaluationPreviewProps {
  data: InlineEvaluationData | null;
  loading?: boolean;
  error?: string | null;
  onViewDetails?: () => void;
}

const InlineEvaluationPreview: React.FC<InlineEvaluationPreviewProps> = ({
  data,
  loading = false,
  error = null,
  onViewDetails,
}) => {
  const navigate = useNavigate();

  const handleViewDetails = () => {
    if (onViewDetails) {
      onViewDetails();
    } else if (data) {
      navigate(`/evaluation?runId=${data.runId}`);
    }
  };

  const formatMetric = (value: number | undefined, isPercentage = false): string => {
    if (value === undefined) return 'N/A';
    if (isPercentage) {
      return `${(value * 100).toFixed(2)}%`;
    }
    return value.toFixed(4);
  };

  const getPrimaryMetrics = (): { label: string; value: string; color: string }[] => {
    if (!data) return [];

    const metrics: { label: string; value: string; color: string }[] = [];

    // Classification metrics
    if (data.metrics.accuracy !== undefined) {
      metrics.push({
        label: 'Accuracy',
        value: formatMetric(data.metrics.accuracy, true),
        color: data.metrics.accuracy > 0.8 ? 'success' : data.metrics.accuracy > 0.6 ? 'warning' : 'error',
      });
    }
    if (data.metrics.f1Score !== undefined) {
      metrics.push({
        label: 'F1 Score',
        value: formatMetric(data.metrics.f1Score, true),
        color: data.metrics.f1Score > 0.8 ? 'success' : data.metrics.f1Score > 0.6 ? 'warning' : 'error',
      });
    }

    // Regression metrics
    if (data.metrics.r2Score !== undefined) {
      metrics.push({
        label: 'R² Score',
        value: formatMetric(data.metrics.r2Score),
        color: data.metrics.r2Score > 0.8 ? 'success' : data.metrics.r2Score > 0.6 ? 'warning' : 'error',
      });
    }
    if (data.metrics.mae !== undefined) {
      metrics.push({
        label: 'MAE',
        value: formatMetric(data.metrics.mae),
        color: 'primary',
      });
    }

    return metrics.slice(0, 4); // Show max 4 metrics
  };

  const createMiniChart = () => {
    if (!data?.trainingHistory) return null;

    const { epoch, trainLoss, valLoss, trainAccuracy, valAccuracy } = data.trainingHistory;

    if (!epoch || epoch.length === 0) return null;

    const traces: any[] = [];

    // Prefer accuracy over loss for classification
    if (trainAccuracy && trainAccuracy.length > 0) {
      traces.push({
        x: epoch,
        y: trainAccuracy,
        name: 'Train',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#1976d2', width: 2 },
      });
    }

    if (valAccuracy && valAccuracy.length > 0) {
      traces.push({
        x: epoch,
        y: valAccuracy,
        name: 'Val',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#dc004e', width: 2 },
      });
    }

    // If no accuracy, show loss
    if (traces.length === 0 && trainLoss && trainLoss.length > 0) {
      traces.push({
        x: epoch,
        y: trainLoss,
        name: 'Train Loss',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#1976d2', width: 2 },
      });

      if (valLoss && valLoss.length > 0) {
        traces.push({
          x: epoch,
          y: valLoss,
          name: 'Val Loss',
          type: 'scatter',
          mode: 'lines',
          line: { color: '#dc004e', width: 2 },
        });
      }
    }

    if (traces.length === 0) return null;

    return (
      <Plot
        data={traces}
        layout={{
          height: 150,
          margin: { l: 40, r: 20, t: 20, b: 30 },
          xaxis: {
            title: 'Epoch',
            titlefont: { size: 10 },
            tickfont: { size: 9 },
          },
          yaxis: {
            title: trainAccuracy ? 'Score' : 'Loss',
            titlefont: { size: 10 },
            tickfont: { size: 9 },
          },
          showlegend: true,
          legend: {
            x: 1,
            xanchor: 'right',
            y: 1,
            font: { size: 9 },
          },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%' }}
      />
    );
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" gap={1} mb={2}>
            <AssessmentIcon color="primary" />
            <Typography variant="h6">Latest Training Results</Typography>
          </Box>
          <Skeleton variant="rectangular" height={100} />
          <Box mt={2}>
            <Skeleton variant="rectangular" height={150} />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error">{error}</Alert>
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" gap={1} mb={2}>
            <AssessmentIcon color="disabled" />
            <Typography variant="h6" color="text.secondary">
              Latest Training Results
            </Typography>
          </Box>
          <Alert severity="info">
            No completed training runs yet. Start a training to see results here.
          </Alert>
        </CardContent>
      </Card>
    );
  }

  const primaryMetrics = getPrimaryMetrics();

  return (
    <Card>
      <CardContent>
        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Box display="flex" alignItems="center" gap={1}>
            <AssessmentIcon color="primary" />
            <Typography variant="h6">Latest Training Results</Typography>
          </Box>
          <Button
            size="small"
            endIcon={<ArrowForwardIcon />}
            onClick={handleViewDetails}
          >
            View Full Evaluation
          </Button>
        </Box>

        {/* Model Info */}
        <Box mb={2}>
          <Typography variant="body2" color="text.secondary">
            Model: <strong>{data.modelName}</strong>
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Type: {data.modelType}
          </Typography>
        </Box>

        {/* Primary Metrics */}
        {primaryMetrics.length > 0 && (
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: {
                xs: 'repeat(2, 1fr)',
                sm: 'repeat(4, 1fr)',
              },
              gap: 2,
              mb: 2,
            }}
          >
            {primaryMetrics.map((metric, index) => (
              <Box
                key={index}
                sx={{
                  p: 1.5,
                  bgcolor: 'action.hover',
                  borderRadius: 1,
                  textAlign: 'center',
                }}
              >
                <Typography variant="caption" color="text.secondary" display="block">
                  {metric.label}
                </Typography>
                <Typography variant="h6" color={`${metric.color}.main`}>
                  {metric.value}
                </Typography>
              </Box>
            ))}
          </Box>
        )}

        {/* Secondary Metrics */}
        <Box display="flex" flexWrap="wrap" gap={1} mb={2}>
          {data.metrics.precision !== undefined && (
            <Chip
              label={`Precision: ${formatMetric(data.metrics.precision, true)}`}
              size="small"
              variant="outlined"
            />
          )}
          {data.metrics.recall !== undefined && (
            <Chip
              label={`Recall: ${formatMetric(data.metrics.recall, true)}`}
              size="small"
              variant="outlined"
            />
          )}
          {data.metrics.rocAuc !== undefined && (
            <Chip
              label={`ROC AUC: ${formatMetric(data.metrics.rocAuc, true)}`}
              size="small"
              variant="outlined"
            />
          )}
          {data.metrics.rmse !== undefined && (
            <Chip
              label={`RMSE: ${formatMetric(data.metrics.rmse)}`}
              size="small"
              variant="outlined"
            />
          )}
          {data.metrics.mse !== undefined && (
            <Chip
              label={`MSE: ${formatMetric(data.metrics.mse)}`}
              size="small"
              variant="outlined"
            />
          )}
        </Box>

        {/* Training History Chart */}
        {data.trainingHistory && (
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Training Progress
            </Typography>
            {createMiniChart()}
          </Box>
        )}

        {/* Performance Indicator */}
        {data.metrics.accuracy !== undefined && (
          <Box display="flex" alignItems="center" gap={1} mt={2}>
            <TrendingUpIcon
              color={data.metrics.accuracy > 0.8 ? 'success' : 'warning'}
              fontSize="small"
            />
            <Typography variant="body2" color="text.secondary">
              {data.metrics.accuracy > 0.8
                ? 'Excellent performance'
                : data.metrics.accuracy > 0.6
                ? 'Good performance'
                : 'Needs improvement'}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default InlineEvaluationPreview;
