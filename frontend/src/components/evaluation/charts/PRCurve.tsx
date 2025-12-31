/**
 * PRCurve Component
 *
 * Visualizes Precision-Recall Curve
 */

import React, { useMemo } from 'react';
import { Box, Paper, Typography, Chip, Alert, Stack } from '@mui/material';
import Plot from 'react-plotly.js';
import type { Data, Layout } from 'plotly.js';

/**
 * Props for PRCurve component
 */
export interface PRCurveProps {
  precision: number[];
  recall: number[];
  thresholds?: number[];
  averagePrecision?: number;
  title?: string;
  className?: string;
  showBaseline?: boolean;
  showThresholds?: boolean;
  height?: number;
}

const PRCurve: React.FC<PRCurveProps> = ({
  precision,
  recall,
  thresholds,
  averagePrecision,
  title = 'Precision-Recall Curve',
  className,
  showBaseline = true,
  showThresholds = false,
  height = 500,
}) => {
  // Validate data
  if (!precision || !recall || precision.length === 0 || recall.length === 0) {
    return (
      <Alert severity="warning">
        <Typography variant="body2">
          No Precision-Recall curve data available
        </Typography>
      </Alert>
    );
  }

  if (precision.length !== recall.length) {
    return (
      <Alert severity="error">
        <Typography variant="body2">
          Invalid PR data: Precision and Recall arrays must have the same length
        </Typography>
      </Alert>
    );
  }

  // Calculate Average Precision if not provided (using trapezoidal rule)
  const calculatedAP = useMemo(() => {
    if (averagePrecision !== undefined) return averagePrecision;

    // Sort by recall (ascending)
    const sorted = recall
      .map((r, i) => ({ recall: r, precision: precision[i] }))
      .sort((a, b) => a.recall - b.recall);

    let sum = 0;
    for (let i = 1; i < sorted.length; i++) {
      const dx = sorted[i].recall - sorted[i - 1].recall;
      const avgY = (sorted[i].precision + sorted[i - 1].precision) / 2;
      sum += dx * avgY;
    }
    return Math.max(0, Math.min(1, sum));
  }, [precision, recall, averagePrecision]);

  // Calculate baseline (random classifier)
  const baseline = useMemo(() => {
    // For balanced classes, baseline = 0.5
    // For imbalanced, baseline ≈ positive class ratio
    // We'll use average precision as approximation
    return 0.5;
  }, []);

  // Calculate optimal point (highest F1 score)
  const optimalIndex = useMemo(() => {
    let maxF1 = -Infinity;
    let optimalIdx = 0;

    precision.forEach((p, idx) => {
      const r = recall[idx];
      const f1 = p + r > 0 ? (2 * p * r) / (p + r) : 0;
      if (f1 > maxF1) {
        maxF1 = f1;
        optimalIdx = idx;
      }
    });

    return optimalIdx;
  }, [precision, recall]);

  // Prepare plot data
  const plotData: Data[] = useMemo(() => {
    const data: Data[] = [];

    // PR Curve
    const prTrace: Data = {
      x: recall,
      y: precision,
      type: 'scatter',
      mode: 'lines',
      name: `PR Curve (AP = ${calculatedAP.toFixed(3)})`,
      line: {
        color: '#9c27b0',
        width: 3,
      },
      hovertemplate:
        '<b>PR Curve</b><br>' +
        'Recall: %{x:.3f}<br>' +
        'Precision: %{y:.3f}<br>' +
        (showThresholds && thresholds
          ? 'Threshold: %{customdata:.3f}<br>'
          : '') +
        '<extra></extra>',
      customdata: showThresholds && thresholds ? thresholds : undefined,
      fill: 'tozeroy',
      fillcolor: 'rgba(156, 39, 176, 0.1)',
    };
    data.push(prTrace);

    // Baseline (random classifier)
    if (showBaseline) {
      const baselineTrace: Data = {
        x: [0, 1],
        y: [baseline, baseline],
        type: 'scatter',
        mode: 'lines',
        name: `Baseline (AP = ${baseline.toFixed(3)})`,
        line: {
          color: '#9e9e9e',
          width: 2,
          dash: 'dash',
        },
        hoverinfo: 'skip',
      };
      data.push(baselineTrace);
    }

    // F1 Score iso-lines (optional - show F1 = 0.5, 0.7, 0.9)
    const f1Values = [0.5, 0.7, 0.9];
    f1Values.forEach((f1) => {
      const recallRange = Array.from({ length: 100 }, (_, i) => i / 99);
      const precisionForF1 = recallRange.map((r) => {
        // F1 = 2 * (P * R) / (P + R)
        // Solving for P: P = (F1 * R) / (2 * R - F1)
        if (r === 0) return 1;
        const p = (f1 * r) / (2 * r - f1);
        return p > 0 && p <= 1 ? p : null;
      });

      const validPoints = recallRange
        .map((r, i) => ({ x: r, y: precisionForF1[i] }))
        .filter((p) => p.y !== null && p.y! >= 0 && p.y! <= 1);

      if (validPoints.length > 0) {
        const isoLine: Data = {
          x: validPoints.map((p) => p.x),
          y: validPoints.map((p) => p.y!),
          type: 'scatter',
          mode: 'lines',
          name: `F1 = ${f1.toFixed(1)}`,
          line: {
            color: '#e0e0e0',
            width: 1,
            dash: 'dot',
          },
          showlegend: false,
          hoverinfo: 'skip',
        };
        data.push(isoLine);
      }
    });

    // Optimal point
    const optimalF1 =
      precision[optimalIndex] + recall[optimalIndex] > 0
        ? (2 * precision[optimalIndex] * recall[optimalIndex]) /
          (precision[optimalIndex] + recall[optimalIndex])
        : 0;

    const optimalPoint: Data = {
      x: [recall[optimalIndex]],
      y: [precision[optimalIndex]],
      type: 'scatter',
      mode: 'markers',
      name: 'Optimal Point',
      marker: {
        color: '#4caf50',
        size: 12,
        symbol: 'star',
        line: {
          color: '#fff',
          width: 2,
        },
      },
      hovertemplate:
        '<b>Optimal Point</b><br>' +
        'Recall: %{x:.3f}<br>' +
        'Precision: %{y:.3f}<br>' +
        `F1: ${optimalF1.toFixed(3)}<br>` +
        (showThresholds && thresholds
          ? `Threshold: ${thresholds[optimalIndex]?.toFixed(3)}<br>`
          : '') +
        '<extra></extra>',
    };
    data.push(optimalPoint);

    return data;
  }, [precision, recall, thresholds, calculatedAP, baseline, showBaseline, showThresholds, optimalIndex]);

  // Plot layout
  const layout: Partial<Layout> = useMemo(
    () => ({
      title: {
        text: title,
        font: { size: 16, weight: 'bold' as any },
      },
      xaxis: {
        title: { text: 'Recall (Sensitivity)' },
        range: [0, 1],
        gridcolor: '#e0e0e0',
        zeroline: true,
      },
      yaxis: {
        title: { text: 'Precision' },
        range: [0, 1],
        gridcolor: '#e0e0e0',
        zeroline: true,
      },
      height,
      margin: { l: 60, r: 40, t: 60, b: 60 },
      hovermode: 'closest',
      showlegend: true,
      legend: {
        x: 0.6,
        y: 0.95,
        bgcolor: 'rgba(255, 255, 255, 0.9)',
        bordercolor: '#e0e0e0',
        borderwidth: 1,
      },
      plot_bgcolor: '#fafafa',
      paper_bgcolor: '#ffffff',
    }),
    [title, height]
  );

  // Get AP quality label
  const getAPQuality = (apValue: number): { label: string; color: string } => {
    if (apValue >= 0.9) return { label: 'Excellent', color: 'success' };
    if (apValue >= 0.8) return { label: 'Very Good', color: 'success' };
    if (apValue >= 0.7) return { label: 'Good', color: 'primary' };
    if (apValue >= 0.6) return { label: 'Fair', color: 'warning' };
    if (apValue >= 0.5) return { label: 'Poor', color: 'warning' };
    return { label: 'Very Poor', color: 'error' };
  };

  const apQuality = getAPQuality(calculatedAP);

  return (
    <Paper elevation={2} sx={{ p: 3 }} className={className}>
      {/* Header */}
      <Box mb={2}>
        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
          <Typography variant="h6" fontWeight="bold">
            {title}
          </Typography>
          <Chip
            label={`AP: ${calculatedAP.toFixed(3)}`}
            color={apQuality.color as any}
            size="small"
          />
          <Chip
            label={apQuality.label}
            variant="outlined"
            color={apQuality.color as any}
            size="small"
          />
          <Chip
            label={`${precision.length} Points`}
            variant="outlined"
            size="small"
          />
        </Stack>
      </Box>

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
          Understanding PR Curve:
        </Typography>
        <Box display="flex" flexDirection="column" gap={0.5}>
          <Typography variant="caption" color="text.secondary">
            • <strong>Average Precision (AP)</strong>: Area under the PR curve
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>AP = 1.0</strong>: Perfect classifier
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Higher curve</strong>: Better performance
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Precision</strong>: Of all positive predictions, how many are correct
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Recall</strong>: Of all actual positives, how many are found
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Green star</strong>: Optimal threshold (highest F1 score)
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Dotted lines</strong>: F1 score iso-lines (0.5, 0.7, 0.9)
          </Typography>
        </Box>
      </Box>
    </Paper>
  );
};

export default PRCurve;
