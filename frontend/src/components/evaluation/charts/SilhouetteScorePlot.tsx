/**
 * SilhouetteScorePlot Component
 *
 * Displays the silhouette score as a gauge/indicator chart
 * Silhouette score ranges from -1 to 1, with higher values indicating better clustering
 */

import React from 'react';
import { Box, Card, CardContent, Typography, LinearProgress } from '@mui/material';
import { Info } from '@mui/icons-material';
import Tooltip from '@mui/material/Tooltip';

/**
 * Props for SilhouetteScorePlot component
 */
export interface SilhouetteScorePlotProps {
  silhouetteScore: number;
  nClusters: number;
}

/**
 * Get color based on silhouette score value
 */
const getScoreColor = (score: number): string => {
  if (score >= 0.7) return '#22c55e'; // Excellent - Green
  if (score >= 0.5) return '#3b82f6'; // Good - Blue
  if (score >= 0.3) return '#f59e0b'; // Fair - Orange
  return '#ef4444'; // Poor - Red
};

/**
 * Get quality label based on silhouette score
 */
const getQualityLabel = (score: number): string => {
  if (score >= 0.7) return 'Excellent';
  if (score >= 0.5) return 'Good';
  if (score >= 0.3) return 'Fair';
  if (score >= 0) return 'Poor';
  return 'Invalid';
};

const SilhouetteScorePlot: React.FC<SilhouetteScorePlotProps> = ({
  silhouetteScore,
  nClusters,
}) => {
  // Normalize score from [-1, 1] to [0, 100] for progress bar
  const normalizedScore = ((silhouetteScore + 1) / 2) * 100;
  const scoreColor = getScoreColor(silhouetteScore);
  const qualityLabel = getQualityLabel(silhouetteScore);

  return (
    <Card sx={{ height: '100%', border: '1px solid', borderColor: 'divider' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6" fontWeight={600}>
            Silhouette Score
          </Typography>
          <Tooltip
            title="Measures how similar an object is to its own cluster compared to other clusters. Range: -1 to 1, where 1 means perfect clustering."
            arrow
          >
            <Info sx={{ color: 'text.secondary', fontSize: 20, cursor: 'help' }} />
          </Tooltip>
        </Box>

        {/* Large Score Display */}
        <Box sx={{ textAlign: 'center', my: 3 }}>
          <Typography
            variant="h2"
            fontWeight={700}
            sx={{ color: scoreColor, mb: 1 }}
          >
            {silhouetteScore.toFixed(3)}
          </Typography>
          <Typography
            variant="subtitle1"
            fontWeight={600}
            sx={{ color: scoreColor }}
          >
            {qualityLabel} Clustering
          </Typography>
        </Box>

        {/* Progress Bar Visualization */}
        <Box sx={{ mt: 3 }}>
          <LinearProgress
            variant="determinate"
            value={normalizedScore}
            sx={{
              height: 12,
              borderRadius: 1,
              backgroundColor: '#e2e8f0',
              '& .MuiLinearProgress-bar': {
                backgroundColor: scoreColor,
                borderRadius: 1,
              },
            }}
          />
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
            <Typography variant="caption" color="text.secondary">
              -1.0 (Worst)
            </Typography>
            <Typography variant="caption" color="text.secondary">
              0.0 (Neutral)
            </Typography>
            <Typography variant="caption" color="text.secondary">
              1.0 (Best)
            </Typography>
          </Box>
        </Box>

        {/* Additional Info */}
        <Box
          sx={{
            mt: 3,
            p: 2,
            bgcolor: '#f8fafc',
            borderRadius: 1,
            border: '1px solid', borderColor: 'divider',
          }}
        >
          <Typography variant="body2" color="text.secondary" gutterBottom>
            <strong>Number of Clusters:</strong> {nClusters}
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
            Silhouette scores closer to 1 indicate well-separated clusters. Scores near 0 suggest
            overlapping clusters, and negative scores indicate misclassified samples.
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default SilhouetteScorePlot;
