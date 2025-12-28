/**
 * ROCCurve Component
 *
 * Visualizes ROC (Receiver Operating Characteristic) Curve
 */

import React, { useMemo } from 'react';
import { Box, Paper, Typography, Chip, Alert, Stack } from '@mui/material';
import Plot from 'react-plotly.js';
import type { Data, Layout } from 'plotly.js';

/**
 * Props for ROCCurve component
 */
export interface ROCCurveProps {
  fpr: number[];
  tpr: number[];
  thresholds?: number[];
  auc?: number;
  title?: string;
  className?: string;
  showDiagonal?: boolean;
  showThresholds?: boolean;
  height?: number;
}

const ROCCurve: React.FC<ROCCurveProps> = ({
  fpr,
  tpr,
  thresholds,
  auc,
  title = 'ROC Curve',
  className,
  showDiagonal = true,
  showThresholds = false,
  height = 500,
}) => {
  // Validate data
  if (!fpr || !tpr || fpr.length === 0 || tpr.length === 0) {
    return (
      <Alert severity="warning">
        <Typography variant="body2">
          No ROC curve data available
        </Typography>
      </Alert>
    );
  }

  if (fpr.length !== tpr.length) {
    return (
      <Alert severity="error">
        <Typography variant="body2">
          Invalid ROC data: FPR and TPR arrays must have the same length
        </Typography>
      </Alert>
    );
  }

  // Calculate AUC if not provided (using trapezoidal rule)
  const calculatedAUC = useMemo(() => {
    if (auc !== undefined) return auc;

    let sum = 0;
    for (let i = 1; i < fpr.length; i++) {
      const dx = fpr[i] - fpr[i - 1];
      const avgY = (tpr[i] + tpr[i - 1]) / 2;
      sum += dx * avgY;
    }
    return Math.max(0, Math.min(1, sum));
  }, [fpr, tpr, auc]);

  // Prepare plot data
  const plotData: Data[] = useMemo(() => {
    const data: Data[] = [];

    // ROC Curve
    const rocTrace: Data = {
      x: fpr,
      y: tpr,
      type: 'scatter',
      mode: 'lines',
      name: `ROC Curve (AUC = ${calculatedAUC.toFixed(3)})`,
      line: {
        color: '#1976d2',
        width: 3,
      },
      hovertemplate:
        '<b>ROC Curve</b><br>' +
        'FPR: %{x:.3f}<br>' +
        'TPR: %{y:.3f}<br>' +
        (showThresholds && thresholds
          ? 'Threshold: %{customdata:.3f}<br>'
          : '') +
        '<extra></extra>',
      customdata: showThresholds && thresholds ? thresholds : undefined,
      fill: 'tonexty',
      fillcolor: 'rgba(25, 118, 210, 0.1)',
    };
    data.push(rocTrace);

    // Diagonal line (random classifier)
    if (showDiagonal) {
      const diagonalTrace: Data = {
        x: [0, 1],
        y: [0, 1],
        type: 'scatter',
        mode: 'lines',
        name: 'Random Classifier (AUC = 0.500)',
        line: {
          color: '#9e9e9e',
          width: 2,
          dash: 'dash',
        },
        hoverinfo: 'skip',
      };
      data.push(diagonalTrace);
    }

    // Optimal point (closest to top-left corner)
    const optimalIndex = useMemo(() => {
      let minDist = Infinity;
      let optimalIdx = 0;

      fpr.forEach((fprVal, idx) => {
        const tprVal = tpr[idx];
        // Distance to (0, 1) - perfect classifier
        const dist = Math.sqrt(Math.pow(fprVal, 2) + Math.pow(1 - tprVal, 2));
        if (dist < minDist) {
          minDist = dist;
          optimalIdx = idx;
        }
      });

      return optimalIdx;
    }, [fpr, tpr]);

    const optimalPoint: Data = {
      x: [fpr[optimalIndex]],
      y: [tpr[optimalIndex]],
      type: 'scatter',
      mode: 'markers',
      name: 'Optimal Threshold',
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
        'FPR: %{x:.3f}<br>' +
        'TPR: %{y:.3f}<br>' +
        (showThresholds && thresholds
          ? `Threshold: ${thresholds[optimalIndex]?.toFixed(3)}<br>`
          : '') +
        '<extra></extra>',
    };
    data.push(optimalPoint);

    return data;
  }, [fpr, tpr, thresholds, calculatedAUC, showDiagonal, showThresholds]);

  // Plot layout
  const layout: Partial<Layout> = useMemo(
    () => ({
      title: {
        text: title,
        font: { size: 16, weight: 'bold' as any },
      },
      xaxis: {
        title: { text: 'False Positive Rate (FPR)' },
        range: [0, 1],
        gridcolor: '#e0e0e0',
        zeroline: true,
      },
      yaxis: {
        title: { text: 'True Positive Rate (TPR)' },
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
        y: 0.1,
        bgcolor: 'rgba(255, 255, 255, 0.9)',
        bordercolor: '#e0e0e0',
        borderwidth: 1,
      },
      plot_bgcolor: '#fafafa',
      paper_bgcolor: '#ffffff',
    }),
    [title, height]
  );

  // Get AUC quality label
  const getAUCQuality = (aucValue: number): { label: string; color: string } => {
    if (aucValue >= 0.95) return { label: 'Excellent', color: 'success' };
    if (aucValue >= 0.9) return { label: 'Very Good', color: 'success' };
    if (aucValue >= 0.8) return { label: 'Good', color: 'primary' };
    if (aucValue >= 0.7) return { label: 'Fair', color: 'warning' };
    if (aucValue >= 0.6) return { label: 'Poor', color: 'warning' };
    return { label: 'Very Poor', color: 'error' };
  };

  const aucQuality = getAUCQuality(calculatedAUC);

  return (
    <Paper elevation={2} sx={{ p: 3 }} className={className}>
      {/* Header */}
      <Box mb={2}>
        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
          <Typography variant="h6" fontWeight="bold">
            {title}
          </Typography>
          <Chip
            label={`AUC: ${calculatedAUC.toFixed(3)}`}
            color={aucQuality.color as any}
            size="small"
          />
          <Chip
            label={aucQuality.label}
            variant="outlined"
            color={aucQuality.color as any}
            size="small"
          />
          <Chip
            label={`${fpr.length} Points`}
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
          Understanding ROC Curve:
        </Typography>
        <Box display="flex" flexDirection="column" gap={0.5}>
          <Typography variant="caption" color="text.secondary">
            • <strong>AUC = 1.0</strong>: Perfect classifier (all predictions correct)
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>AUC = 0.5</strong>: Random classifier (no better than chance)
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>AUC &gt; 0.9</strong>: Excellent discrimination ability
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Closer to top-left</strong>: Better performance (high TPR, low FPR)
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Green star</strong>: Optimal threshold point (best balance of TPR/FPR)
          </Typography>
        </Box>
      </Box>
    </Paper>
  );
};

export default ROCCurve;
