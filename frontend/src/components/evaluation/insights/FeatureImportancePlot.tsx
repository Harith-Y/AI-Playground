/**
 * FeatureImportancePlot Component
 *
 * Visualizes feature importance scores from trained models
 * Shows which features contribute most to model predictions
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
  FormControlLabel,
  Switch,
} from '@mui/material';
import Plot from 'react-plotly.js';
import type { Data, Layout } from 'plotly.js';

/**
 * Feature importance entry
 */
export interface FeatureImportance {
  feature: string;
  importance: number;
  rank?: number;
  stdDev?: number;
}

/**
 * Props for FeatureImportancePlot component
 */
export interface FeatureImportancePlotProps {
  features: FeatureImportance[];
  title?: string;
  className?: string;
  showTopN?: number;
  sortByImportance?: boolean;
  showErrorBars?: boolean;
  height?: number;
  importanceType?: 'gain' | 'split' | 'weight' | 'cover';
}

const FeatureImportancePlot: React.FC<FeatureImportancePlotProps> = ({
  features,
  title = 'Feature Importance',
  className,
  showTopN = 20,
  sortByImportance = true,
  showErrorBars = true,
  height = 600,
  importanceType = 'gain',
}) => {
  const [viewMode, setViewMode] = useState<'bar' | 'horizontal'>('horizontal');
  const [showAll, setShowAll] = useState(false);

  // Validate data
  if (!features || features.length === 0) {
    return (
      <Alert severity="warning">
        <Typography variant="body2">
          No feature importance data available
        </Typography>
      </Alert>
    );
  }

  // Process and sort features
  const processedFeatures = useMemo(() => {
    let sorted = [...features];

    // Sort by importance if enabled
    if (sortByImportance) {
      sorted = sorted.sort((a, b) => b.importance - a.importance);
    }

    // Add ranks if not present
    sorted = sorted.map((f, idx) => ({
      ...f,
      rank: f.rank ?? idx + 1,
    }));

    // Limit to top N if not showing all
    if (!showAll && sorted.length > showTopN) {
      sorted = sorted.slice(0, showTopN);
    }

    return sorted;
  }, [features, sortByImportance, showAll, showTopN]);

  // Calculate statistics
  const stats = useMemo(() => {
    const importances = processedFeatures.map((f) => f.importance);
    const total = importances.reduce((sum, val) => sum + val, 0);
    const mean = total / importances.length;
    const max = Math.max(...importances);
    const min = Math.min(...importances);

    // Calculate cumulative importance
    let cumulative = 0;
    const cumulativeImportances = processedFeatures.map((f) => {
      cumulative += f.importance;
      return cumulative;
    });
    const cumulativePercentages = cumulativeImportances.map((c) => (c / total) * 100);

    return {
      total,
      mean,
      max,
      min,
      count: processedFeatures.length,
      cumulativePercentages,
    };
  }, [processedFeatures]);

  // Find how many features account for 80% importance
  const features80Percent = useMemo(() => {
    const idx = stats.cumulativePercentages.findIndex((p) => p >= 80);
    return idx !== -1 ? idx + 1 : processedFeatures.length;
  }, [stats.cumulativePercentages, processedFeatures.length]);

  // Prepare plot data
  const plotData: Data[] = useMemo(() => {
    const data: Data[] = [];

    if (viewMode === 'horizontal') {
      // Horizontal bar chart (default, easier to read feature names)
      const barTrace: Data = {
        x: processedFeatures.map((f) => f.importance),
        y: processedFeatures.map((f) => f.feature),
        type: 'bar',
        orientation: 'h',
        name: 'Importance',
        marker: {
          color: processedFeatures.map((f, idx) => {
            // Color gradient based on importance
            const ratio = f.importance / stats.max;
            if (ratio > 0.7) return '#4caf50'; // High importance - green
            if (ratio > 0.4) return '#ff9800'; // Medium importance - orange
            return '#1976d2'; // Low importance - blue
          }),
          line: {
            color: '#fff',
            width: 1,
          },
        },
        error_x: showErrorBars && processedFeatures.some((f) => f.stdDev)
          ? {
              type: 'data',
              array: processedFeatures.map((f) => f.stdDev || 0),
              visible: true,
              color: '#666',
            }
          : undefined,
        hovertemplate:
          '<b>%{y}</b><br>' +
          `${importanceType.charAt(0).toUpperCase() + importanceType.slice(1)}: %{x:.4f}<br>` +
          'Rank: %{customdata}<br>' +
          '<extra></extra>',
        customdata: processedFeatures.map((f) => f.rank || 0),
      };
      data.push(barTrace);
    } else {
      // Vertical bar chart
      const barTrace: Data = {
        x: processedFeatures.map((f) => f.feature),
        y: processedFeatures.map((f) => f.importance),
        type: 'bar',
        name: 'Importance',
        marker: {
          color: processedFeatures.map((f, idx) => {
            const ratio = f.importance / stats.max;
            if (ratio > 0.7) return '#4caf50';
            if (ratio > 0.4) return '#ff9800';
            return '#1976d2';
          }),
          line: {
            color: '#fff',
            width: 1,
          },
        },
        error_y: showErrorBars && processedFeatures.some((f) => f.stdDev)
          ? {
              type: 'data',
              array: processedFeatures.map((f) => f.stdDev || 0),
              visible: true,
              color: '#666',
            }
          : undefined,
        hovertemplate:
          '<b>%{x}</b><br>' +
          `${importanceType.charAt(0).toUpperCase() + importanceType.slice(1)}: %{y:.4f}<br>` +
          'Rank: %{customdata}<br>' +
          '<extra></extra>',
        customdata: processedFeatures.map((f) => f.rank || 0),
      };
      data.push(barTrace);
    }

    return data;
  }, [processedFeatures, viewMode, stats.max, showErrorBars, importanceType]);

  // Plot layout
  const layout: Partial<Layout> = useMemo(() => {
    if (viewMode === 'horizontal') {
      return {
        title: {
          text: title,
          font: { size: 16, weight: 'bold' as any },
        },
        xaxis: {
          title: { text: `Importance (${importanceType})` },
          gridcolor: '#e0e0e0',
          zeroline: true,
        },
        yaxis: {
          title: { text: 'Feature' },
          automargin: true,
          categoryorder: 'trace', // Keep order from data
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
          title: { text: `Importance (${importanceType})` },
          gridcolor: '#e0e0e0',
          zeroline: true,
        },
        height,
        margin: { l: 60, r: 40, t: 60, b: 150 },
        hovermode: 'closest',
        showlegend: false,
        plot_bgcolor: '#fafafa',
        paper_bgcolor: '#ffffff',
      };
    }
  }, [title, height, viewMode, importanceType]);

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
              label={`${processedFeatures.length} Features`}
              variant="outlined"
              size="small"
            />
            <Chip
              label={`Top ${features80Percent} = 80%`}
              color="primary"
              size="small"
            />
            <Chip
              label={`Type: ${importanceType}`}
              variant="outlined"
              size="small"
            />
          </Stack>

          {/* View Mode Toggle */}
          <Stack direction="row" spacing={2} alignItems="center">
            <FormControlLabel
              control={
                <Switch
                  checked={showAll}
                  onChange={(e) => setShowAll(e.target.checked)}
                  size="small"
                />
              }
              label={
                <Typography variant="caption">
                  Show All ({features.length})
                </Typography>
              }
            />
            <ToggleButtonGroup
              value={viewMode}
              exclusive
              onChange={(_, newMode) => {
                if (newMode !== null) setViewMode(newMode);
              }}
              size="small"
            >
              <ToggleButton value="horizontal">Horizontal</ToggleButton>
              <ToggleButton value="bar">Vertical</ToggleButton>
            </ToggleButtonGroup>
          </Stack>
        </Stack>
      </Box>

      {/* 80/20 Rule Alert */}
      {features80Percent < processedFeatures.length && (
        <Alert severity="info" sx={{ mb: 2 }}>
          <Typography variant="body2">
            <strong>80/20 Rule:</strong> The top {features80Percent} features (
            {((features80Percent / features.length) * 100).toFixed(1)}%) account for 80% of
            total importance. Consider focusing on these for model interpretation.
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
          Importance Statistics:
        </Typography>
        <Box display="flex" flexDirection="column" gap={0.5}>
          <Typography variant="caption" color="text.secondary">
            • <strong>Total Importance:</strong> {stats.total.toFixed(4)}
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
          Understanding Feature Importance:
        </Typography>
        <Box display="flex" flexDirection="column" gap={0.5}>
          <Typography variant="caption" color="text.secondary">
            • <strong>Green bars</strong>: High importance (&gt;70% of max) - critical features
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Orange bars</strong>: Medium importance (40-70% of max) - useful features
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Blue bars</strong>: Low importance (&lt;40% of max) - minor features
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Error bars</strong>: Variability in importance across folds/trees (if
            available)
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Importance Type:</strong> {importanceType === 'gain' && 'Average gain when feature is used for splitting'}
            {importanceType === 'split' && 'Number of times feature is used to split data'}
            {importanceType === 'weight' && 'Number of times feature appears in trees'}
            {importanceType === 'cover' && 'Average coverage (samples) when feature is used'}
          </Typography>
        </Box>
      </Box>
    </Paper>
  );
};

export default FeatureImportancePlot;
