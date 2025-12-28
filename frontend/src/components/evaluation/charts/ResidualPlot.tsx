/**
 * ResidualPlot Component
 *
 * Visualizes residuals (prediction errors) vs predicted values
 * Helps identify patterns in model errors and check assumptions
 */

import React, { useMemo } from 'react';
import { Box, Paper, Typography, Chip, Alert, Stack } from '@mui/material';
import Plot from 'react-plotly.js';
import type { Data, Layout } from 'plotly.js';

/**
 * Props for ResidualPlot component
 */
export interface ResidualPlotProps {
  predicted: number[];
  actual: number[];
  title?: string;
  className?: string;
  showZeroLine?: boolean;
  showTrendLine?: boolean;
  height?: number;
}

const ResidualPlot: React.FC<ResidualPlotProps> = ({
  predicted,
  actual,
  title = 'Residual Plot',
  className,
  showZeroLine = true,
  showTrendLine = true,
  height = 500,
}) => {
  // Validate data
  if (!predicted || !actual || predicted.length === 0 || actual.length === 0) {
    return (
      <Alert severity="warning">
        <Typography variant="body2">
          No residual plot data available
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

  // Calculate residuals (actual - predicted)
  const residuals = useMemo(() => {
    return actual.map((actualVal, idx) => actualVal - predicted[idx]);
  }, [actual, predicted]);

  // Calculate residual statistics
  const residualStats = useMemo(() => {
    const mean = residuals.reduce((sum, r) => sum + r, 0) / residuals.length;
    const variance =
      residuals.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / residuals.length;
    const stdDev = Math.sqrt(variance);
    const min = Math.min(...residuals);
    const max = Math.max(...residuals);

    return { mean, stdDev, min, max };
  }, [residuals]);

  // Calculate trend line (linear regression of residuals vs predicted)
  const trendLine = useMemo(() => {
    if (!showTrendLine) return null;

    const n = predicted.length;
    const sumX = predicted.reduce((sum, x) => sum + x, 0);
    const sumY = residuals.reduce((sum, y) => sum + y, 0);
    const sumXY = predicted.reduce((sum, x, i) => sum + x * residuals[i], 0);
    const sumX2 = predicted.reduce((sum, x) => sum + x * x, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    const minX = Math.min(...predicted);
    const maxX = Math.max(...predicted);

    return {
      x: [minX, maxX],
      y: [slope * minX + intercept, slope * maxX + intercept],
      slope,
      intercept,
    };
  }, [predicted, residuals, showTrendLine]);

  // Detect pattern in residuals
  const patternDetection = useMemo(() => {
    if (!trendLine) return { hasPattern: false, message: '' };

    const absSlope = Math.abs(trendLine.slope);
    const avgPredicted = predicted.reduce((sum, x) => sum + x, 0) / predicted.length;
    const relativeSlope = absSlope / Math.abs(avgPredicted);

    if (relativeSlope > 0.1) {
      return {
        hasPattern: true,
        message: 'Pattern detected: Residuals show systematic trend. Model may be biased.',
      };
    }

    // Check for heteroscedasticity (increasing variance)
    const midIdx = Math.floor(predicted.length / 2);
    const sortedByPredicted = predicted
      .map((p, i) => ({ pred: p, res: residuals[i] }))
      .sort((a, b) => a.pred - b.pred);

    const firstHalfVar =
      sortedByPredicted
        .slice(0, midIdx)
        .reduce((sum, d) => sum + d.res * d.res, 0) / midIdx;
    const secondHalfVar =
      sortedByPredicted
        .slice(midIdx)
        .reduce((sum, d) => sum + d.res * d.res, 0) /
      (sortedByPredicted.length - midIdx);

    const varianceRatio = Math.max(firstHalfVar, secondHalfVar) / Math.min(firstHalfVar, secondHalfVar);

    if (varianceRatio > 2) {
      return {
        hasPattern: true,
        message: 'Heteroscedasticity detected: Error variance increases with predicted values.',
      };
    }

    return { hasPattern: false, message: 'No obvious pattern detected. Good!' };
  }, [predicted, residuals, trendLine]);

  // Prepare plot data
  const plotData: Data[] = useMemo(() => {
    const data: Data[] = [];

    // Scatter plot of residuals
    const scatterTrace: Data = {
      x: predicted,
      y: residuals,
      type: 'scatter',
      mode: 'markers',
      name: 'Residuals',
      marker: {
        color: residuals.map((r) => (Math.abs(r) > 2 * residualStats.stdDev ? '#f44336' : '#1976d2')),
        size: 8,
        opacity: 0.6,
        line: {
          color: '#fff',
          width: 1,
        },
      },
      hovertemplate:
        '<b>Residual</b><br>' +
        'Predicted: %{x:.3f}<br>' +
        'Residual: %{y:.3f}<br>' +
        '<extra></extra>',
    };
    data.push(scatterTrace);

    // Zero line (perfect predictions)
    if (showZeroLine) {
      const minX = Math.min(...predicted);
      const maxX = Math.max(...predicted);

      const zeroLineTrace: Data = {
        x: [minX, maxX],
        y: [0, 0],
        type: 'scatter',
        mode: 'lines',
        name: 'Zero Line',
        line: {
          color: '#4caf50',
          width: 2,
          dash: 'dash',
        },
        hoverinfo: 'skip',
      };
      data.push(zeroLineTrace);
    }

    // Trend line
    if (showTrendLine && trendLine) {
      const trendLineTrace: Data = {
        x: trendLine.x,
        y: trendLine.y,
        type: 'scatter',
        mode: 'lines',
        name: 'Trend Line',
        line: {
          color: '#ff9800',
          width: 2,
          dash: 'dot',
        },
        hoverinfo: 'skip',
      };
      data.push(trendLineTrace);
    }

    // Standard deviation bands (±2σ)
    const minX = Math.min(...predicted);
    const maxX = Math.max(...predicted);

    const upperBand: Data = {
      x: [minX, maxX],
      y: [2 * residualStats.stdDev, 2 * residualStats.stdDev],
      type: 'scatter',
      mode: 'lines',
      name: '+2σ',
      line: {
        color: '#9e9e9e',
        width: 1,
        dash: 'dot',
      },
      showlegend: false,
      hoverinfo: 'skip',
    };
    data.push(upperBand);

    const lowerBand: Data = {
      x: [minX, maxX],
      y: [-2 * residualStats.stdDev, -2 * residualStats.stdDev],
      type: 'scatter',
      mode: 'lines',
      name: '-2σ',
      line: {
        color: '#9e9e9e',
        width: 1,
        dash: 'dot',
      },
      showlegend: false,
      hoverinfo: 'skip',
    };
    data.push(lowerBand);

    return data;
  }, [predicted, residuals, residualStats, showZeroLine, showTrendLine, trendLine]);

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
        zeroline: false,
      },
      yaxis: {
        title: { text: 'Residuals (Actual - Predicted)' },
        gridcolor: '#e0e0e0',
        zeroline: true,
        zerolinecolor: '#4caf50',
        zerolinewidth: 2,
      },
      height,
      margin: { l: 60, r: 40, t: 60, b: 60 },
      hovermode: 'closest',
      showlegend: true,
      legend: {
        x: 1,
        y: 1,
        xanchor: 'right',
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

  // Count outliers (beyond 2 standard deviations)
  const outlierCount = residuals.filter((r) => Math.abs(r) > 2 * residualStats.stdDev).length;
  const outlierPercentage = (outlierCount / residuals.length) * 100;

  return (
    <Paper elevation={2} sx={{ p: 3 }} className={className}>
      {/* Header */}
      <Box mb={2}>
        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
          <Typography variant="h6" fontWeight="bold">
            {title}
          </Typography>
          <Chip
            label={`Mean: ${residualStats.mean.toFixed(3)}`}
            color={Math.abs(residualStats.mean) < 0.1 ? 'success' : 'warning'}
            size="small"
          />
          <Chip
            label={`Std Dev: ${residualStats.stdDev.toFixed(3)}`}
            variant="outlined"
            size="small"
          />
          <Chip
            label={`Outliers: ${outlierCount} (${outlierPercentage.toFixed(1)}%)`}
            color={outlierPercentage < 5 ? 'success' : 'warning'}
            size="small"
          />
          <Chip label={`${predicted.length} Points`} variant="outlined" size="small" />
        </Stack>
      </Box>

      {/* Pattern Detection Alert */}
      {patternDetection.hasPattern && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          <Typography variant="body2">{patternDetection.message}</Typography>
        </Alert>
      )}
      {!patternDetection.hasPattern && (
        <Alert severity="success" sx={{ mb: 2 }}>
          <Typography variant="body2">{patternDetection.message}</Typography>
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
          Understanding Residual Plot:
        </Typography>
        <Box display="flex" flexDirection="column" gap={0.5}>
          <Typography variant="caption" color="text.secondary">
            • <strong>Good model</strong>: Residuals randomly scattered around zero line
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Pattern/trend</strong>: Indicates model bias or missing features
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Increasing spread</strong>: Heteroscedasticity (variance increases with
            predicted values)
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Red points</strong>: Outliers beyond ±2σ (may need investigation)
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Zero line</strong>: Perfect predictions (residual = 0)
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Dotted lines</strong>: ±2σ bounds (95% of residuals should fall within)
          </Typography>
        </Box>
      </Box>
    </Paper>
  );
};

export default ResidualPlot;
