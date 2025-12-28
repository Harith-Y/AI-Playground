/**
 * ClusteringMetrics Component
 *
 * Displays clustering-specific metrics (silhouette, inertia, Davies-Bouldin, Calinski-Harabasz)
 */

import React from 'react';
import { Box, Typography, Paper, Divider, Chip } from '@mui/material';
import {
  BubbleChart as BubbleChartIcon,
  Speed as SpeedIcon,
  CompareArrows as CompareArrowsIcon,
  Analytics as AnalyticsIcon,
  Groups as GroupsIcon,
} from '@mui/icons-material';
import MetricCard, { MetricVariant } from './MetricCard';
import type { ClusteringMetrics as ClusteringMetricsType } from '../../../types/evaluation';

/**
 * Props for ClusteringMetrics component
 */
export interface ClusteringMetricsProps {
  metrics: ClusteringMetricsType;
  showDescription?: boolean;
  compact?: boolean;
}

const ClusteringMetrics: React.FC<ClusteringMetricsProps> = ({
  metrics,
  showDescription = true,
  compact = false,
}) => {
  // Determine Silhouette variant (range: -1 to 1, higher is better)
  const getSilhouetteVariant = (value: number): MetricVariant => {
    if (value >= 0.7) return MetricVariant.SUCCESS;
    if (value >= 0.5) return MetricVariant.DEFAULT;
    if (value >= 0.3) return MetricVariant.WARNING;
    return MetricVariant.ERROR;
  };

  // Determine Davies-Bouldin variant (lower is better, typically 0+)
  const getDBVariant = (value: number): MetricVariant => {
    if (value <= 0.5) return MetricVariant.SUCCESS;
    if (value <= 1.0) return MetricVariant.DEFAULT;
    if (value <= 2.0) return MetricVariant.WARNING;
    return MetricVariant.ERROR;
  };

  // Format large numbers
  const formatLargeNumber = (value: number): string => {
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(2)}M`;
    } else if (value >= 1000) {
      return `${(value / 1000).toFixed(2)}K`;
    }
    return value.toFixed(2);
  };

  // Calculate average cluster size
  const getAverageClusterSize = (): number | null => {
    if (!metrics.clusterSizes || metrics.clusterSizes.length === 0) return null;
    const sum = metrics.clusterSizes.reduce((a, b) => a + b, 0);
    return sum / metrics.clusterSizes.length;
  };

  return (
    <Box>
      {/* Header */}
      {!compact && (
        <Box mb={3}>
          <Typography variant="h6" fontWeight="bold" gutterBottom>
            Clustering Metrics
          </Typography>
          {showDescription && (
            <Typography variant="body2" color="text.secondary">
              Key performance indicators for clustering model evaluation
            </Typography>
          )}
        </Box>
      )}

      {/* Primary Metrics */}
      <Box mb={3}>
        <Typography variant="subtitle2" fontWeight="bold" gutterBottom sx={{ mb: 1.5 }}>
          Clustering Quality
        </Typography>

        <Box
          display="grid"
          gridTemplateColumns={{
            xs: '1fr',
            sm: 'repeat(2, 1fr)',
            md: 'repeat(3, 1fr)',
          }}
          gap={2}
        >
          {/* Number of Clusters */}
          <MetricCard
            label="Clusters"
            value={metrics.nClusters}
            format="integer"
            icon={<BubbleChartIcon />}
            tooltip="Number of clusters identified by the algorithm"
            description={compact ? undefined : "Total groups"}
          />

          {/* Silhouette Score */}
          <MetricCard
            label="Silhouette Score"
            value={metrics.silhouetteScore}
            format="decimal"
            precision={3}
            variant={getSilhouetteVariant(metrics.silhouetteScore)}
            icon={<SpeedIcon />}
            tooltip="Measures how similar points are within clusters vs. between clusters (range: -1 to 1)"
            showProgress
            min={-1}
            max={1}
            description={compact ? undefined : "Cluster separation"}
          />

          {/* Inertia (if available) */}
          {metrics.inertia !== undefined && (
            <MetricCard
              label="Inertia"
              value={metrics.inertia}
              format="number"
              precision={2}
              icon={<AnalyticsIcon />}
              tooltip="Sum of squared distances of samples to their closest cluster center (lower is better)"
              description={compact ? undefined : "Within-cluster sum"}
            />
          )}
        </Box>
      </Box>

      {/* Additional Quality Metrics */}
      {(metrics.daviesBouldinScore !== undefined || metrics.calinskiHarabaszScore !== undefined) && (
        <Box mb={3}>
          <Typography variant="subtitle2" fontWeight="bold" gutterBottom sx={{ mb: 1.5 }}>
            Advanced Metrics
          </Typography>

          <Box
            display="grid"
            gridTemplateColumns={{
              xs: '1fr',
              sm: 'repeat(2, 1fr)',
            }}
            gap={2}
          >
            {/* Davies-Bouldin Index */}
            {metrics.daviesBouldinScore !== undefined && (
              <MetricCard
                label="Davies-Bouldin Index"
                value={metrics.daviesBouldinScore}
                format="decimal"
                precision={3}
                variant={getDBVariant(metrics.daviesBouldinScore)}
                icon={<CompareArrowsIcon />}
                tooltip="Average similarity ratio of clusters (lower is better, 0 is perfect)"
                description={compact ? undefined : "Cluster similarity"}
              />
            )}

            {/* Calinski-Harabasz Index */}
            {metrics.calinskiHarabaszScore !== undefined && (
              <MetricCard
                label="Calinski-Harabasz Score"
                value={metrics.calinskiHarabaszScore}
                format="number"
                precision={2}
                variant={
                  metrics.calinskiHarabaszScore >= 100
                    ? MetricVariant.SUCCESS
                    : MetricVariant.DEFAULT
                }
                icon={<AnalyticsIcon />}
                tooltip="Ratio of between-cluster to within-cluster dispersion (higher is better)"
                description={compact ? undefined : "Variance ratio"}
              />
            )}
          </Box>
        </Box>
      )}

      {/* Cluster Distribution */}
      {metrics.clusterSizes && metrics.clusterSizes.length > 0 && !compact && (
        <Box mb={3}>
          <Typography variant="subtitle2" fontWeight="bold" gutterBottom sx={{ mb: 1.5 }}>
            Cluster Distribution
          </Typography>

          <Paper elevation={1} sx={{ p: 2 }}>
            <Box display="flex" flexDirection="column" gap={1.5}>
              {metrics.clusterSizes.map((size, index) => {
                const totalPoints = metrics.clusterSizes!.reduce((a, b) => a + b, 0);
                const percentage = ((size / totalPoints) * 100).toFixed(1);

                return (
                  <Box key={index}>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
                      <Typography variant="body2" fontWeight="500">
                        Cluster {index + 1}
                      </Typography>
                      <Box display="flex" gap={1} alignItems="center">
                        <Typography variant="body2" color="text.secondary">
                          {size.toLocaleString()} points
                        </Typography>
                        <Chip label={`${percentage}%`} size="small" />
                      </Box>
                    </Box>
                    <Box
                      sx={{
                        width: '100%',
                        height: 8,
                        bgcolor: 'grey.200',
                        borderRadius: 1,
                        overflow: 'hidden',
                      }}
                    >
                      <Box
                        sx={{
                          width: `${percentage}%`,
                          height: '100%',
                          bgcolor: `hsl(${(index * 360) / metrics.nClusters}, 70%, 60%)`,
                        }}
                      />
                    </Box>
                  </Box>
                );
              })}
            </Box>
          </Paper>
        </Box>
      )}

      {/* Metrics Summary */}
      {!compact && (
        <Paper elevation={1} sx={{ p: 2, bgcolor: 'grey.50' }}>
          <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
            Metrics Summary
          </Typography>
          <Divider sx={{ mb: 1.5 }} />

          <Box display="flex" flexDirection="column" gap={1}>
            <Box display="flex" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">
                Clustering Quality:
              </Typography>
              <Typography variant="body2" fontWeight="500">
                {getClusteringQuality(metrics.silhouetteScore)}
              </Typography>
            </Box>

            <Box display="flex" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">
                Number of Clusters:
              </Typography>
              <Typography variant="body2" fontWeight="500">
                {metrics.nClusters} groups
              </Typography>
            </Box>

            {metrics.clusterSizes && (
              <>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2" color="text.secondary">
                    Total Points:
                  </Typography>
                  <Typography variant="body2" fontWeight="500">
                    {metrics.clusterSizes.reduce((a, b) => a + b, 0).toLocaleString()}
                  </Typography>
                </Box>

                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2" color="text.secondary">
                    Avg Cluster Size:
                  </Typography>
                  <Typography variant="body2" fontWeight="500">
                    {getAverageClusterSize()?.toFixed(0) || 'N/A'} points
                  </Typography>
                </Box>

                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2" color="text.secondary">
                    Largest Cluster:
                  </Typography>
                  <Typography variant="body2" fontWeight="500">
                    {Math.max(...metrics.clusterSizes).toLocaleString()} points
                  </Typography>
                </Box>

                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2" color="text.secondary">
                    Smallest Cluster:
                  </Typography>
                  <Typography variant="body2" fontWeight="500">
                    {Math.min(...metrics.clusterSizes).toLocaleString()} points
                  </Typography>
                </Box>
              </>
            )}

            {metrics.inertia !== undefined && (
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">
                  Inertia:
                </Typography>
                <Typography variant="body2" fontWeight="500">
                  {formatLargeNumber(metrics.inertia)}
                </Typography>
              </Box>
            )}

            {metrics.clusterCenters && (
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">
                  Cluster Centers:
                </Typography>
                <Typography variant="body2" fontWeight="500">
                  {metrics.clusterCenters.length}x{metrics.clusterCenters[0]?.length || 0} Available
                </Typography>
              </Box>
            )}
          </Box>
        </Paper>
      )}

      {/* Interpretation Guide */}
      {!compact && (
        <Paper elevation={0} sx={{ p: 2, mt: 2, bgcolor: 'info.lighter', border: '1px solid', borderColor: 'info.light' }}>
          <Typography variant="caption" fontWeight="bold" color="info.dark" display="block" gutterBottom>
            Understanding Clustering Metrics
          </Typography>
          <Box display="flex" flexDirection="column" gap={0.5}>
            <Typography variant="caption" color="text.secondary">
              • <strong>Silhouette Score</strong>: Close to 1 = well-separated clusters, close to 0 = overlapping,
              negative = misclassified
            </Typography>
            {metrics.inertia !== undefined && (
              <Typography variant="caption" color="text.secondary">
                • <strong>Inertia</strong>: Lower values indicate tighter, more compact clusters
              </Typography>
            )}
            {metrics.daviesBouldinScore !== undefined && (
              <Typography variant="caption" color="text.secondary">
                • <strong>Davies-Bouldin</strong>: Lower values indicate better separation between clusters
              </Typography>
            )}
            {metrics.calinskiHarabaszScore !== undefined && (
              <Typography variant="caption" color="text.secondary">
                • <strong>Calinski-Harabasz</strong>: Higher values indicate denser, well-separated clusters
              </Typography>
            )}
          </Box>
        </Paper>
      )}
    </Box>
  );
};

// Helper function to determine clustering quality based on Silhouette score
const getClusteringQuality = (silhouette: number): string => {
  if (silhouette >= 0.7) return 'Excellent';
  if (silhouette >= 0.5) return 'Good';
  if (silhouette >= 0.3) return 'Fair';
  if (silhouette >= 0) return 'Weak';
  return 'Poor';
};

export default ClusteringMetrics;
