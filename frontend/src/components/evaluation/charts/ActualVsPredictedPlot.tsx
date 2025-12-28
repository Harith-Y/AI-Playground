/**
 * ActualVsPredictedPlot Component
 *
 * Visualizes actual vs predicted values with diagonal reference line
 * Shows how well predictions align with actual values
 */

import React, { useMemo } from 'react';
import { Box, Paper, Typography, Chip, Alert, Stack } from '@mui/material';
import Plot from 'react-plotly.js';
import type { Data, Layout } from 'plotly.js';

/**
 * Props for ActualVsPredictedPlot component
 */
export interface ActualVsPredictedPlotProps {
  predicted: number[];
  actual: number[];
  title?: string;
  className?: string;
  showDiagonal?: boolean;
  showRegressionLine?: boolean;
  height?: number;
}

const ActualVsPredictedPlot: React.FC<ActualVsPredictedPlotProps> = ({
  predicted,
  actual,
  title = 'Actual vs Predicted',
  className,
  showDiagonal = true,
  showRegressionLine = true,
  height = 500,
}) => {
  // Validate data
  if (!predicted || !actual || predicted.length === 0 || actual.length === 0) {
    return (
      <Alert severity="warning">
        <Typography variant="body2">
          No actual vs predicted data available
        </Typography>
      </Alert>
    );
  }

  if (predicted.length !== actual.length) {
    return (
      <Alert severity="error">
        <Typography variant="body2">
          Invalid data: Predicted and Actual arrays must have the same length
        </Typography>
      </Alert>
    );
  }

  // Calculate statistics
  const stats = useMemo(() => {
    const errors = actual.map((a, i) => a - predicted[i]);
    const mae = errors.reduce((sum, e) => sum + Math.abs(e), 0) / errors.length;
    const mse = errors.reduce((sum, e) => sum + e * e, 0) / errors.length;
    const rmse = Math.sqrt(mse);

    // R² calculation
    const actualMean = actual.reduce((sum, a) => sum + a, 0) / actual.length;
    const ssTot = actual.reduce((sum, a) => sum + Math.pow(a - actualMean, 2), 0);
    const ssRes = errors.reduce((sum, e) => sum + e * e, 0);
    const r2 = 1 - ssRes / ssTot;

    const minValue = Math.min(...actual, ...predicted);
    const maxValue = Math.max(...actual, ...predicted);

    return { mae, rmse, r2, minValue, maxValue };
  }, [actual, predicted]);

  // Calculate regression line (best fit line for actual vs predicted)
  const regressionLine = useMemo(() => {
    if (!showRegressionLine) return null;

    const n = predicted.length;
    const sumX = predicted.reduce((sum, x) => sum + x, 0);
    const sumY = actual.reduce((sum, y) => sum + y, 0);
    const sumXY = predicted.reduce((sum, x, i) => sum + x * actual[i], 0);
    const sumX2 = predicted.reduce((sum, x) => sum + x * x, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    return {
      x: [stats.minValue, stats.maxValue],
      y: [slope * stats.minValue + intercept, slope * stats.maxValue + intercept],
      slope,
      intercept,
    };
  }, [predicted, actual, showRegressionLine, stats.minValue, stats.maxValue]);

  // Calculate absolute errors for coloring
  const absoluteErrors = useMemo(() => {
    return actual.map((a, i) => Math.abs(a - predicted[i]));
  }, [actual, predicted]);

  const maxError = Math.max(...absoluteErrors);

  // Prepare plot data
  const plotData: Data[] = useMemo(() => {
    const data: Data[] = [];

    // Scatter plot of actual vs predicted
    const scatterTrace: Data = {
      x: predicted,
      y: actual,
      type: 'scatter',
      mode: 'markers',
      name: 'Data Points',
      marker: {
        color: absoluteErrors,
        colorscale: [
          [0, '#4caf50'],
          [0.5, '#ff9800'],
          [1, '#f44336'],
        ],
        size: 8,
        opacity: 0.7,
        line: {
          color: '#fff',
          width: 1,
        },
        colorbar: {
          title: { text: 'Absolute Error' },
          thickness: 15,
          len: 0.7,
        },
        cmin: 0,
        cmax: maxError,
      },
      text: absoluteErrors.map((e) => `Error: ${e.toFixed(3)}`),
      hovertemplate:
        '<b>Data Point</b><br>' +
        'Predicted: %{x:.3f}<br>' +
        'Actual: %{y:.3f}<br>' +
        '%{text}<br>' +
        '<extra></extra>',
    };
    data.push(scatterTrace);

    // Diagonal line (perfect predictions: y = x)
    if (showDiagonal) {
      const diagonalTrace: Data = {
        x: [stats.minValue, stats.maxValue],
        y: [stats.minValue, stats.maxValue],
        type: 'scatter',
        mode: 'lines',
        name: 'Perfect Fit (y=x)',
        line: {
          color: '#4caf50',
          width: 2,
          dash: 'dash',
        },
        hoverinfo: 'skip',
      };
      data.push(diagonalTrace);
    }

    // Regression line (best fit)
    if (showRegressionLine && regressionLine) {
      const regressionTrace: Data = {
        x: regressionLine.x,
        y: regressionLine.y,
        type: 'scatter',
        mode: 'lines',
        name: `Best Fit (y=${regressionLine.slope.toFixed(2)}x + ${regressionLine.intercept.toFixed(2)})`,
        line: {
          color: '#1976d2',
          width: 2,
        },
        hoverinfo: 'skip',
      };
      data.push(regressionTrace);
    }

    return data;
  }, [
    predicted,
    actual,
    absoluteErrors,
    maxError,
    stats.minValue,
    stats.maxValue,
    showDiagonal,
    showRegressionLine,
    regressionLine,
  ]);

  // Plot layout
  const layout: Partial<Layout> = useMemo(
    () => ({
      title: {
        text: title,
        font: { size: 16, weight: 'bold' as any },
      },
      xaxis: {
        title: { text: 'Predicted Values' },
        gridcolor: '#e0e0e0',
        zeroline: true,
      },
      yaxis: {
        title: { text: 'Actual Values' },
        gridcolor: '#e0e0e0',
        zeroline: true,
      },
      height,
      margin: { l: 60, r: 100, t: 60, b: 60 },
      hovermode: 'closest',
      showlegend: true,
      legend: {
        x: 0,
        y: 1,
        xanchor: 'left',
        yanchor: 'top',
        bgcolor: 'rgba(255, 255, 255, 0.9)',
        bordercolor: '#e0e0e0',
        borderwidth: 1,
      },
      plot_bgcolor: '#fafafa',
      paper_bgcolor: '#ffffff',
    }),
    [title, height]
  );

  // Get R² quality label
  const getR2Quality = (r2Value: number): { label: string; color: string } => {
    if (r2Value >= 0.9) return { label: 'Excellent', color: 'success' };
    if (r2Value >= 0.8) return { label: 'Very Good', color: 'success' };
    if (r2Value >= 0.7) return { label: 'Good', color: 'primary' };
    if (r2Value >= 0.5) return { label: 'Fair', color: 'warning' };
    if (r2Value >= 0.3) return { label: 'Poor', color: 'warning' };
    return { label: 'Very Poor', color: 'error' };
  };

  const r2Quality = getR2Quality(stats.r2);

  // Check alignment with diagonal
  const alignmentCheck = useMemo(() => {
    if (!regressionLine) return { aligned: true, message: '' };

    const slopeDiff = Math.abs(1 - regressionLine.slope);
    const avgValue = (stats.minValue + stats.maxValue) / 2;
    const interceptDiff = Math.abs(regressionLine.intercept) / avgValue;

    if (slopeDiff > 0.2 || interceptDiff > 0.2) {
      return {
        aligned: false,
        message: 'Predictions systematically deviate from actual values. Consider model recalibration.',
      };
    }

    return { aligned: true, message: 'Predictions well-aligned with actual values!' };
  }, [regressionLine, stats.minValue, stats.maxValue]);

  return (
    <Paper elevation={2} sx={{ p: 3 }} className={className}>
      {/* Header */}
      <Box mb={2}>
        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
          <Typography variant="h6" fontWeight="bold">
            {title}
          </Typography>
          <Chip
            label={`R²: ${stats.r2.toFixed(3)}`}
            color={r2Quality.color as any}
            size="small"
          />
          <Chip
            label={r2Quality.label}
            variant="outlined"
            color={r2Quality.color as any}
            size="small"
          />
          <Chip label={`RMSE: ${stats.rmse.toFixed(3)}`} variant="outlined" size="small" />
          <Chip label={`${predicted.length} Points`} variant="outlined" size="small" />
        </Stack>
      </Box>

      {/* Alignment Check Alert */}
      {!alignmentCheck.aligned && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          <Typography variant="body2">{alignmentCheck.message}</Typography>
        </Alert>
      )}
      {alignmentCheck.aligned && (
        <Alert severity="success" sx={{ mb: 2 }}>
          <Typography variant="body2">{alignmentCheck.message}</Typography>
        </Alert>
      )}

      {/* Plot */}
      <Box sx={{ width: '100%', overflowX: 'auto' }}>
        <Plot
          data={plotData}
          layout={layout}
          config={{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
          }}
          style={{ width: '100%' }}
        />
      </Box>

      {/* Interpretation Guide */}
      <Box mt={2} p={2} bgcolor="grey.50" borderRadius={1}>
        <Typography variant="caption" fontWeight="bold" display="block" gutterBottom>
          Understanding Actual vs Predicted Plot:
        </Typography>
        <Box display="flex" flexDirection="column" gap={0.5}>
          <Typography variant="caption" color="text.secondary">
            • <strong>Perfect model</strong>: All points on green diagonal line (y=x)
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Good fit</strong>: Points clustered tightly around diagonal
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Systematic bias</strong>: Points consistently above/below diagonal
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Color coding</strong>: Green = low error, Red = high error
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Blue line</strong>: Best fit regression line through points
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>R² close to 1.0</strong>: Strong correlation, accurate predictions
          </Typography>
        </Box>
      </Box>
    </Paper>
  );
};

export default ActualVsPredictedPlot;
