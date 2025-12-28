/**
 * PermutationImportancePlot Component
 *
 * Visualizes permutation importance with uncertainty estimates
 * Shows the decrease in model performance when feature values are randomly shuffled
 */

import React, { useMemo, useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Chip,
  Alert,
  Stack,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import Plot from 'react-plotly.js';
import type { Data, Layout } from 'plotly.js';

/**
 * Permutation importance entry
 */
export interface PermutationImportance {
  feature: string;
  importance: number;
  importances?: number[]; // Multiple runs for uncertainty
  mean?: number;
  std?: number;
  rank?: number;
}

/**
 * Props for PermutationImportancePlot component
 */
export interface PermutationImportancePlotProps {
  features: PermutationImportance[];
  title?: string;
  className?: string;
  showTopN?: number;
  metric?: string;
  height?: number;
}

const PermutationImportancePlot: React.FC<PermutationImportancePlotProps> = ({
  features,
  title = 'Permutation Feature Importance',
  className,
  showTopN = 20,
  metric = 'Accuracy',
  height = 600,
}) => {
  const [viewMode, setViewMode] = useState<'bar' | 'box'>('bar');

  // Validate data
  if (!features || features.length === 0) {
    return (
      <Alert severity="warning">
        <Typography variant="body2">
          No permutation importance data available
        </Typography>
      </Alert>
    );
  }

  // Process and sort features
  const processedFeatures = useMemo(() => {
    let sorted = [...features];

    // Calculate mean and std if not provided but importances array exists
    sorted = sorted.map((f) => {
      if (f.importances && f.importances.length > 0) {
        const mean = f.mean ?? f.importances.reduce((sum, val) => sum + val, 0) / f.importances.length;
        const variance =
          f.importances.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) /
          f.importances.length;
        const std = f.std ?? Math.sqrt(variance);
        return { ...f, mean, std, importance: mean };
      }
      return { ...f, mean: f.importance, std: 0 };
    });

    // Sort by importance (mean) descending
    sorted = sorted.sort((a, b) => b.importance - a.importance);

    // Add ranks
    sorted = sorted.map((f, idx) => ({
      ...f,
      rank: f.rank ?? idx + 1,
    }));

    // Limit to top N
    if (sorted.length > showTopN) {
      sorted = sorted.slice(0, showTopN);
    }

    return sorted;
  }, [features, showTopN]);

  // Calculate statistics
  const stats = useMemo(() => {
    const importances = processedFeatures.map((f) => f.importance);
    const mean = importances.reduce((sum, val) => sum + val, 0) / importances.length;
    const max = Math.max(...importances);
    const min = Math.min(...importances);

    // Count significant features (importance > 0)
    const significant = processedFeatures.filter((f) => f.importance > 0).length;

    return { mean, max, min, significant, total: processedFeatures.length };
  }, [processedFeatures]);

  // Prepare plot data
  const plotData: Data[] = useMemo(() => {
    const data: Data[] = [];

    if (viewMode === 'bar') {
      // Bar chart with error bars
      const barTrace: Data = {
        x: processedFeatures.map((f) => f.importance),
        y: processedFeatures.map((f) => f.feature),
        type: 'bar',
        orientation: 'h',
        name: 'Importance',
        marker: {
          color: processedFeatures.map((f) => {
            // Color based on significance
            if (f.importance > 0.01) return '#4caf50'; // Significant positive
            if (f.importance > 0) return '#ff9800'; // Slightly positive
            if (f.importance > -0.01) return '#9e9e9e'; // Near zero
            return '#f44336'; // Negative (bad for model)
          }),
          line: {
            color: '#fff',
            width: 1,
          },
        },
        error_x: {
          type: 'data',
          array: processedFeatures.map((f) => f.std || 0),
          visible: true,
          color: '#666',
          thickness: 1.5,
        },
        hovertemplate:
          '<b>%{y}</b><br>' +
          'Mean Importance: %{x:.4f}<br>' +
          'Std Dev: %{customdata:.4f}<br>' +
          '<extra></extra>',
        customdata: processedFeatures.map((f) => f.std || 0),
      };
      data.push(barTrace);

      // Add zero line
      const zeroLine: Data = {
        x: [0, 0],
        y: [processedFeatures[processedFeatures.length - 1]?.feature, processedFeatures[0]?.feature],
        type: 'scatter',
        mode: 'lines',
        name: 'Zero',
        line: {
          color: '#000',
          width: 2,
          dash: 'dash',
        },
        hoverinfo: 'skip',
        showlegend: false,
      };
      data.push(zeroLine);
    } else {
      // Box plot showing distribution
      processedFeatures.forEach((f) => {
        if (f.importances && f.importances.length > 0) {
          const boxTrace: Data = {
            y: f.importances,
            type: 'box',
            name: f.feature,
            marker: {
              color: f.importance > 0 ? '#4caf50' : '#f44336',
            },
            boxmean: 'sd',
            hovertemplate:
              `<b>${f.feature}</b><br>` +
              'Value: %{y:.4f}<br>' +
              '<extra></extra>',
          };
          data.push(boxTrace);
        }
      });
    }

    return data;
  }, [processedFeatures, viewMode]);

  // Plot layout
  const layout: Partial<Layout> = useMemo(() => {
    if (viewMode === 'bar') {
      return {
        title: {
          text: title,
          font: { size: 16, weight: 'bold' as any },
        },
        xaxis: {
          title: { text: `Decrease in ${metric}` },
          gridcolor: '#e0e0e0',
          zeroline: true,
          zerolinecolor: '#000',
          zerolinewidth: 2,
        },
        yaxis: {
          title: { text: 'Feature' },
          automargin: true,
          categoryorder: 'trace',
        },
        height,
        margin: { l: 150, r: 40, t: 60, b: 60 },
        hovermode: 'closest',
        showlegend: false,
        plot_bgcolor: '#fafafa',
        paper_bgcolor: '#ffffff',
      };
    } else {
      return {
        title: {
          text: title,
          font: { size: 16, weight: 'bold' as any },
        },
        xaxis: {
          title: { text: 'Feature' },
          tickangle: -45,
          automargin: true,
        },
        yaxis: {
          title: { text: `Decrease in ${metric}` },
          gridcolor: '#e0e0e0',
          zeroline: true,
          zerolinecolor: '#000',
          zerolinewidth: 2,
        },
        height,
        margin: { l: 60, r: 40, t: 60, b: 150 },
        hovermode: 'closest',
        showlegend: false,
        plot_bgcolor: '#fafafa',
        paper_bgcolor: '#ffffff',
      };
    }
  }, [title, height, viewMode, metric]);

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
              label={`${stats.significant}/${stats.total} Significant`}
              color="primary"
              size="small"
            />
            <Chip
              label={`Metric: ${metric}`}
              variant="outlined"
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
            <ToggleButton value="bar">Bar Chart</ToggleButton>
            <ToggleButton value="box">Box Plot</ToggleButton>
          </ToggleButtonGroup>
        </Stack>
      </Box>

      {/* Significance Alert */}
      {stats.significant < stats.total * 0.5 && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          <Typography variant="body2">
            <strong>Low significance:</strong> Only {stats.significant} out of {stats.total}{' '}
            features show positive importance. Consider feature engineering or model selection.
          </Typography>
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

      {/* Statistics Summary */}
      <Box mt={2} p={2} bgcolor="grey.50" borderRadius={1}>
        <Typography variant="caption" fontWeight="bold" display="block" gutterBottom>
          Permutation Importance Statistics:
        </Typography>
        <Box display="flex" flexDirection="column" gap={0.5}>
          <Typography variant="caption" color="text.secondary">
            • <strong>Significant Features:</strong> {stats.significant} ({((stats.significant / stats.total) * 100).toFixed(1)}%)
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Mean Importance:</strong> {stats.mean.toFixed(4)}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Max Importance:</strong> {stats.max.toFixed(4)} (
            {processedFeatures[0]?.feature})
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Min Importance:</strong> {stats.min.toFixed(4)}
          </Typography>
        </Box>
      </Box>

      {/* Interpretation Guide */}
      <Box mt={2} p={2} bgcolor="grey.50" borderRadius={1}>
        <Typography variant="caption" fontWeight="bold" display="block" gutterBottom>
          Understanding Permutation Importance:
        </Typography>
        <Box display="flex" flexDirection="column" gap={0.5}>
          <Typography variant="caption" color="text.secondary">
            • <strong>Positive values</strong>: Feature contributes to model performance. Higher =
            more important.
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Zero/negative values</strong>: Feature does not improve (or hurts) model
            performance.
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Error bars</strong>: Uncertainty in importance estimate. Large bars =
            unstable importance.
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Box plot</strong>: Shows full distribution of importance across multiple
            permutations.
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Green</strong>: Significant positive importance | <strong>Red</strong>:
            Negative importance
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>How it works</strong>: Randomly shuffle feature values and measure drop in{' '}
            {metric}. Larger drop = more important.
          </Typography>
        </Box>
      </Box>
    </Paper>
  );
};

export default PermutationImportancePlot;
