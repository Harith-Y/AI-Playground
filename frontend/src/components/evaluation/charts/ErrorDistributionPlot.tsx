/**
 * ErrorDistributionPlot Component
 *
 * Visualizes the distribution of prediction errors
 * Includes histogram and box plot to check normality and identify outliers
 */

import React, { useMemo } from 'react';
import { Box, Paper, Typography, Chip, Alert, Stack, ToggleButton, ToggleButtonGroup } from '@mui/material';
import Plot from 'react-plotly.js';
import type { Data, Layout } from 'plotly.js';

/**
 * Props for ErrorDistributionPlot component
 */
export interface ErrorDistributionPlotProps {
  predicted: number[];
  actual: number[];
  title?: string;
  className?: string;
  defaultView?: 'histogram' | 'box' | 'both';
  showNormalCurve?: boolean;
  height?: number;
}

const ErrorDistributionPlot: React.FC<ErrorDistributionPlotProps> = ({
  predicted,
  actual,
  title = 'Error Distribution',
  className,
  defaultView = 'both',
  showNormalCurve = true,
  height = 500,
}) => {
  const [viewMode, setViewMode] = React.useState<'histogram' | 'box' | 'both'>(defaultView);

  // Validate data
  if (!predicted || !actual || predicted.length === 0 || actual.length === 0) {
    return (
      <Alert severity="warning">
        <Typography variant="body2">
          No error distribution data available
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

  // Calculate errors (residuals)
  const errors = useMemo(() => {
    return actual.map((actualVal, idx) => actualVal - predicted[idx]);
  }, [actual, predicted]);

  // Calculate error statistics
  const errorStats = useMemo(() => {
    const n = errors.length;
    const mean = errors.reduce((sum, e) => sum + e, 0) / n;
    const variance = errors.reduce((sum, e) => sum + Math.pow(e - mean, 2), 0) / n;
    const stdDev = Math.sqrt(variance);
    const min = Math.min(...errors);
    const max = Math.max(...errors);

    // Calculate quartiles for box plot
    const sorted = [...errors].sort((a, b) => a - b);
    const q1 = sorted[Math.floor(n * 0.25)];
    const median = sorted[Math.floor(n * 0.5)];
    const q3 = sorted[Math.floor(n * 0.75)];
    const iqr = q3 - q1;

    // Skewness (measure of asymmetry)
    const skewness =
      errors.reduce((sum, e) => sum + Math.pow((e - mean) / stdDev, 3), 0) / n;

    // Kurtosis (measure of tail heaviness)
    const kurtosis =
      errors.reduce((sum, e) => sum + Math.pow((e - mean) / stdDev, 4), 0) / n - 3;

    return { mean, stdDev, min, max, q1, median, q3, iqr, skewness, kurtosis };
  }, [errors]);

  // Generate normal distribution curve
  const normalCurve = useMemo(() => {
    if (!showNormalCurve) return null;

    const { mean, stdDev, min, max } = errorStats;
    const points = 100;
    const step = (max - min) / points;
    const x = Array.from({ length: points }, (_, i) => min + i * step);

    // Normal distribution PDF: f(x) = (1 / (σ * √(2π))) * e^(-0.5 * ((x - μ) / σ)^2)
    const y = x.map((val) => {
      const z = (val - mean) / stdDev;
      return (1 / (stdDev * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * z * z);
    });

    // Scale to match histogram (approximate)
    const binWidth = (max - min) / 30; // Assume ~30 bins
    const scaleFactor = errors.length * binWidth;
    const scaledY = y.map((val) => val * scaleFactor);

    return { x, y: scaledY };
  }, [errors, errorStats, showNormalCurve]);

  // Check normality (simple test based on skewness and kurtosis)
  const normalityCheck = useMemo(() => {
    const { skewness, kurtosis } = errorStats;

    // Generally, |skewness| < 1 and |kurtosis| < 1 indicate approximate normality
    const isNormal = Math.abs(skewness) < 1 && Math.abs(kurtosis) < 1;
    const isModeratelyNormal = Math.abs(skewness) < 2 && Math.abs(kurtosis) < 3;

    if (isNormal) {
      return {
        level: 'good',
        message: 'Errors appear approximately normally distributed (good assumption for many models).',
      };
    } else if (isModeratelyNormal) {
      return {
        level: 'moderate',
        message: 'Errors show some deviation from normality. Model assumptions may be partially violated.',
      };
    } else {
      return {
        level: 'poor',
        message: 'Errors significantly deviate from normality. Consider data transformation or different model.',
      };
    }
  }, [errorStats]);

  // Prepare plot data
  const plotData: Data[] = useMemo(() => {
    const data: Data[] = [];

    // Histogram
    if (viewMode === 'histogram' || viewMode === 'both') {
      const histogramTrace: Data = {
        x: errors,
        type: 'histogram',
        name: 'Error Distribution',
        marker: {
          color: '#1976d2',
          line: {
            color: '#fff',
            width: 1,
          },
        },
        opacity: 0.7,
        autobinx: false,
        xbins: {
          start: errorStats.min,
          end: errorStats.max,
          size: (errorStats.max - errorStats.min) / 30,
        },
        hovertemplate: 'Error Range: %{x}<br>Count: %{y}<extra></extra>',
      };
      data.push(histogramTrace);

      // Normal curve overlay
      if (showNormalCurve && normalCurve) {
        const normalTrace: Data = {
          x: normalCurve.x,
          y: normalCurve.y,
          type: 'scatter',
          mode: 'lines',
          name: 'Normal Distribution',
          line: {
            color: '#f44336',
            width: 3,
            dash: 'dash',
          },
          yaxis: 'y',
          hoverinfo: 'skip',
        };
        data.push(normalTrace);
      }
    }

    // Box plot
    if (viewMode === 'box' || viewMode === 'both') {
      const boxTrace: Data = {
        y: errors,
        type: 'box',
        name: 'Error Box Plot',
        marker: {
          color: '#1976d2',
        },
        boxmean: 'sd', // Show mean and standard deviation
        hovertemplate:
          '<b>Error Statistics</b><br>' +
          'Value: %{y:.3f}<br>' +
          '<extra></extra>',
      };
      data.push(boxTrace);
    }

    return data;
  }, [errors, viewMode, showNormalCurve, normalCurve]);

  // Plot layout
  const layout: Partial<Layout> = useMemo(() => {
    const baseLayout: Partial<Layout> = {
      title: {
        text: title,
        font: { size: 16, weight: 'bold' as any },
      },
      height,
      margin: { l: 60, r: 40, t: 60, b: 60 },
      hovermode: 'closest',
      showlegend: viewMode === 'histogram' || viewMode === 'both',
      legend: {
        x: 1,
        y: 1,
        xanchor: 'right',
        yanchor: 'top',
        bgcolor: 'rgba(255, 255, 255, 0.9)',
        bordercolor: '#e0e0e0',
        borderwidth: 1,
      },
      plot_bgcolor: 'transparent',
      paper_bgcolor: 'transparent',
    };

    if (viewMode === 'histogram') {
      return {
        ...baseLayout,
        xaxis: {
          title: { text: 'Prediction Error (Actual - Predicted)' },
          gridcolor: '#e0e0e0',
          zeroline: true,
          zerolinecolor: '#4caf50',
          zerolinewidth: 2,
        },
        yaxis: {
          title: { text: 'Frequency' },
          gridcolor: '#e0e0e0',
        },
      };
    } else if (viewMode === 'box') {
      return {
        ...baseLayout,
        yaxis: {
          title: { text: 'Prediction Error (Actual - Predicted)' },
          gridcolor: '#e0e0e0',
          zeroline: true,
          zerolinecolor: '#4caf50',
          zerolinewidth: 2,
        },
      };
    } else {
      // both
      return {
        ...baseLayout,
        xaxis: {
          title: { text: 'Prediction Error' },
          gridcolor: '#e0e0e0',
          zeroline: true,
          zerolinecolor: '#4caf50',
          zerolinewidth: 2,
        },
        yaxis: {
          title: { text: 'Frequency / Value' },
          gridcolor: '#e0e0e0',
        },
      };
    }
  }, [title, height, viewMode]);

  // Count outliers (beyond 1.5 * IQR)
  const outlierBounds = {
    lower: errorStats.q1 - 1.5 * errorStats.iqr,
    upper: errorStats.q3 + 1.5 * errorStats.iqr,
  };
  const outlierCount = errors.filter(
    (e) => e < outlierBounds.lower || e > outlierBounds.upper
  ).length;
  const outlierPercentage = (outlierCount / errors.length) * 100;

  return (
    <Paper elevation={2} sx={{ p: 3 }} className={className}>
      {/* Header */}
      <Box mb={2}>
        <Stack
          direction="row"
          spacing={1}
          alignItems="center"
          flexWrap="wrap"
          justifyContent="space-between"
        >
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
            <Typography variant="h6" fontWeight="bold">
              {title}
            </Typography>
            <Chip
              label={`Mean: ${errorStats.mean.toFixed(3)}`}
              color={Math.abs(errorStats.mean) < 0.1 ? 'success' : 'warning'}
              size="small"
            />
            <Chip
              label={`Std Dev: ${errorStats.stdDev.toFixed(3)}`}
              variant="outlined"
              size="small"
            />
            <Chip
              label={`Median: ${errorStats.median.toFixed(3)}`}
              variant="outlined"
              size="small"
            />
            <Chip
              label={`Outliers: ${outlierCount} (${outlierPercentage.toFixed(1)}%)`}
              color={outlierPercentage < 5 ? 'success' : 'warning'}
              size="small"
            />
          </Stack>

          {/* View Mode Toggle */}
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={(_, newMode) => {
              if (newMode !== null) setViewMode(newMode);
            }}
            size="small"
          >
            <ToggleButton value="histogram">Histogram</ToggleButton>
            <ToggleButton value="box">Box Plot</ToggleButton>
            <ToggleButton value="both">Both</ToggleButton>
          </ToggleButtonGroup>
        </Stack>
      </Box>

      {/* Normality Check Alert */}
      <Alert
        severity={
          normalityCheck.level === 'good'
            ? 'success'
            : normalityCheck.level === 'moderate'
              ? 'info'
              : 'warning'
        }
        sx={{ mb: 2 }}
      >
        <Typography variant="body2">{normalityCheck.message}</Typography>
        <Typography variant="caption" sx={{ mt: 0.5, display: 'block' }}>
          Skewness: {errorStats.skewness.toFixed(3)}, Kurtosis: {errorStats.kurtosis.toFixed(3)}
        </Typography>
      </Alert>

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
          Understanding Error Distribution:
        </Typography>
        <Box display="flex" flexDirection="column" gap={0.5}>
          <Typography variant="caption" color="text.secondary">
            • <strong>Normal distribution</strong>: Bell-shaped curve indicates well-behaved errors
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Centered at zero</strong>: Mean close to 0 means unbiased predictions
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Skewness</strong>: Measures asymmetry (|skew| &lt; 1 is good)
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Kurtosis</strong>: Measures tail heaviness (|kurt| &lt; 1 is good)
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Box plot</strong>: Shows median, quartiles, and outliers visually
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Outliers</strong>: Points beyond 1.5×IQR from quartiles (need investigation)
          </Typography>
        </Box>
      </Box>
    </Paper>
  );
};

export default ErrorDistributionPlot;
