/**
 * ClusterDistributionPlot Component
 *
 * Displays the distribution of samples across clusters as a bar chart
 */

import React from 'react';
import Plot from 'react-plotly.js';
import { Box, Card, CardContent, Typography } from '@mui/material';
import { PieChart, Info } from '@mui/icons-material';
import Tooltip from '@mui/material/Tooltip';

/**
 * Props for ClusterDistributionPlot component
 */
export interface ClusterDistributionPlotProps {
  clusterSizes: number[];
  nClusters: number;
}

const ClusterDistributionPlot: React.FC<ClusterDistributionPlotProps> = ({
  clusterSizes,
  nClusters,
}) => {
  // Prepare data for the bar chart
  const clusterLabels = Array.from({ length: nClusters }, (_, i) => `Cluster ${i}`);
  const totalSamples = clusterSizes.reduce((sum, size) => sum + size, 0);
  const percentages = clusterSizes.map((size) => ((size / totalSamples) * 100).toFixed(1));

  // Check for imbalance (if any cluster has >50% or <5% of samples)
  const hasImbalance = clusterSizes.some(
    (size) => size / totalSamples > 0.5 || size / totalSamples < 0.05
  );

  return (
    <Card sx={{ height: '100%', border: '1px solid #e2e8f0' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <PieChart sx={{ color: '#3b82f6' }} />
            <Typography variant="h6" fontWeight={600}>
              Cluster Distribution
            </Typography>
          </Box>
          <Tooltip
            title="Shows how samples are distributed across clusters. Balanced clusters have similar sizes, while imbalanced clusters may indicate outliers or poor clustering."
            arrow
          >
            <Info sx={{ color: 'text.secondary', fontSize: 20, cursor: 'help' }} />
          </Tooltip>
        </Box>

        {/* Bar Chart */}
        <Box sx={{ mt: 2 }}>
          <Plot
            data={[
              {
                type: 'bar',
                x: clusterLabels,
                y: clusterSizes,
                text: percentages.map((p) => `${p}%`),
                textposition: 'auto',
                marker: {
                  color: '#3b82f6',
                  line: {
                    color: '#1e40af',
                    width: 1,
                  },
                },
                hovertemplate: '<b>%{x}</b><br>Samples: %{y}<br>Percentage: %{text}<extra></extra>',
              },
            ]}
            layout={{
              autosize: true,
              height: 300,
              margin: { l: 50, r: 30, t: 30, b: 80 },
              xaxis: {
                title: 'Cluster',
                titlefont: { size: 12 },
              },
              yaxis: {
                title: 'Number of Samples',
                titlefont: { size: 12 },
              },
              hovermode: 'closest',
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
            }}
            config={{
              displayModeBar: false,
              responsive: true,
            }}
            style={{ width: '100%' }}
          />
        </Box>

        {/* Summary Statistics */}
        <Box
          sx={{
            mt: 2,
            p: 2,
            bgcolor: hasImbalance ? '#fef3c7' : '#f0f9ff',
            borderRadius: 1,
            border: `1px solid ${hasImbalance ? '#fbbf24' : '#bae6fd'}`,
          }}
        >
          <Typography variant="body2" color="text.secondary" gutterBottom>
            <strong>Total Samples:</strong> {totalSamples.toLocaleString()}
          </Typography>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            <strong>Clusters:</strong> {nClusters}
          </Typography>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            <strong>Average per Cluster:</strong> {(totalSamples / nClusters).toFixed(0)}
          </Typography>
          {hasImbalance && (
            <Typography
              variant="caption"
              sx={{ display: 'block', mt: 1, color: '#d97706', fontWeight: 600 }}
            >
              ⚠️ Imbalanced distribution detected. Some clusters have significantly more/fewer samples.
            </Typography>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default ClusterDistributionPlot;
