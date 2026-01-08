/**
 * ClusterProjection2D Component
 *
 * Displays a 2D projection (e.g., PCA, t-SNE) of clustered data points
 * Shows how well clusters are separated in 2D space
 */

import React from 'react';
import Plot from 'react-plotly.js';
import { Box, Card, CardContent, Typography, Chip } from '@mui/material';
import { ScatterPlot, Info } from '@mui/icons-material';
import Tooltip from '@mui/material/Tooltip';

/**
 * Props for ClusterProjection2D component
 */
export interface ClusterProjection2DProps {
  data: {
    x: number[];
    y: number[];
    labels: number[];
  };
  nClusters: number;
  method?: 'PCA' | 't-SNE' | 'UMAP' | 'Other';
}

/**
 * Generate distinct colors for clusters
 */
const generateClusterColors = (nClusters: number): string[] => {
  const baseColors = [
    '#3b82f6', // Blue
    '#ef4444', // Red
    '#22c55e', // Green
    '#f59e0b', // Orange
    '#8b5cf6', // Purple
    '#ec4899', // Pink
    '#14b8a6', // Teal
    '#f97316', // Deep Orange
    '#6366f1', // Indigo
    '#84cc16', // Lime
  ];

  // If we need more colors than base palette, generate them
  if (nClusters <= baseColors.length) {
    return baseColors.slice(0, nClusters);
  }

  // Generate additional colors using HSL
  const colors = [...baseColors];
  for (let i = baseColors.length; i < nClusters; i++) {
    const hue = (i * 137.5) % 360; // Golden angle for color distribution
    colors.push(`hsl(${hue}, 70%, 60%)`);
  }
  return colors;
};

const ClusterProjection2D: React.FC<ClusterProjection2DProps> = ({
  data,
  nClusters,
  method = 'PCA',
}) => {
  const { x, y, labels } = data;

  // Validate data
  if (!x || !y || !labels || x.length === 0 || y.length === 0) {
    return (
      <Card sx={{ height: '100%', border: '1px solid', borderColor: 'divider' }}>
        <CardContent>
          <Typography variant="body2" color="text.secondary">
            No projection data available
          </Typography>
        </CardContent>
      </Card>
    );
  }

  // Generate colors
  const clusterColors = generateClusterColors(nClusters);

  // Prepare data for each cluster
  const traces = Array.from({ length: nClusters }, (_, clusterIdx) => {
    const clusterIndices = labels
      .map((label, idx) => (label === clusterIdx ? idx : -1))
      .filter((idx) => idx !== -1);

    return {
      type: 'scatter' as const,
      mode: 'markers' as const,
      name: `Cluster ${clusterIdx}`,
      x: clusterIndices.map((idx) => x[idx]),
      y: clusterIndices.map((idx) => y[idx]),
      marker: {
        color: clusterColors[clusterIdx],
        size: 8,
        opacity: 0.7,
        line: {
          color: 'white',
          width: 1,
        },
      },
      hovertemplate: `<b>Cluster ${clusterIdx}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>`,
    };
  });

  return (
    <Card sx={{ height: '100%', border: '1px solid', borderColor: 'divider' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ScatterPlot sx={{ color: '#3b82f6' }} />
            <Typography variant="h6" fontWeight={600}>
              2D Cluster Projection
            </Typography>
          </Box>
          <Tooltip
            title="2D visualization of clusters using dimensionality reduction. Shows how well-separated clusters are in reduced space."
            arrow
          >
            <Info sx={{ color: 'text.secondary', fontSize: 20, cursor: 'help' }} />
          </Tooltip>
        </Box>

        {/* Method Badge */}
        <Box sx={{ mb: 2 }}>
          <Chip label={`Method: ${method}`} size="small" color="primary" variant="outlined" />
        </Box>

        {/* Scatter Plot */}
        <Box sx={{ mt: 2 }}>
          <Plot
            data={traces}
            layout={{
              autosize: true,
              height: 400,
              margin: { l: 50, r: 30, t: 30, b: 50 },
              xaxis: {
                title: `${method} Component 1`,
                titlefont: { size: 12 },
                zeroline: true,
                zerolinecolor: '#e2e8f0',
                gridcolor: '#f1f5f9',
              },
              yaxis: {
                title: `${method} Component 2`,
                titlefont: { size: 12 },
                zeroline: true,
                zerolinecolor: '#e2e8f0',
                gridcolor: '#f1f5f9',
              },
              hovermode: 'closest',
              showlegend: true,
              legend: {
                x: 1.05,
                y: 1,
                xanchor: 'left',
                yanchor: 'top',
                bgcolor: 'rgba(255, 255, 255, 0.9)',
                bordercolor: '#e2e8f0',
                borderwidth: 1,
              },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
            }}
            config={{
              displayModeBar: true,
              displaylogo: false,
              modeBarButtonsToRemove: ['lasso2d', 'select2d'],
              responsive: true,
            }}
            style={{ width: '100%' }}
          />
        </Box>

        {/* Info Box */}
        <Box
          sx={{
            mt: 2,
            p: 2,
            bgcolor: '#f8fafc',
            borderRadius: 1,
            border: '1px solid', borderColor: 'divider',
          }}
        >
          <Typography variant="body2" color="text.secondary" gutterBottom>
            <strong>Samples:</strong> {x.length.toLocaleString()}
          </Typography>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            <strong>Clusters:</strong> {nClusters}
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
            Well-separated clusters in this projection suggest good clustering quality. Overlapping
            clusters may indicate ambiguous groupings or the need for more dimensions.
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ClusterProjection2D;
