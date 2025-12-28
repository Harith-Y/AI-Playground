/**
 * ModelInsights Component
 *
 * Container for all model interpretation and feature importance visualizations
 * Combines feature importance, permutation importance, and feature ranking displays
 */

import React, { useState } from 'react';
import {
  Box,
  Alert,
  Typography,
  CircularProgress,
  Tabs,
  Tab,
  Paper,
} from '@mui/material';
import FeatureImportancePlot from './FeatureImportancePlot';
import PermutationImportancePlot from './PermutationImportancePlot';
import FeatureRankingTable from './FeatureRankingTable';
import type { FeatureImportance } from './FeatureImportancePlot';
import type { PermutationImportance } from './PermutationImportancePlot';
import type { FeatureRanking } from './FeatureRankingTable';

/**
 * Model insights data
 */
export interface ModelInsightsData {
  featureImportance?: FeatureImportance[];
  permutationImportance?: PermutationImportance[];
  featureRankings?: FeatureRanking[];
  importanceType?: 'gain' | 'split' | 'weight' | 'cover';
  metric?: string;
}

/**
 * Props for ModelInsights component
 */
export interface ModelInsightsProps {
  data: ModelInsightsData | null;
  isLoading?: boolean;
  error?: string;
  showFeatureImportance?: boolean;
  showPermutationImportance?: boolean;
  showRankingTable?: boolean;
}

type InsightTab = 'importance' | 'permutation' | 'ranking';

const ModelInsights: React.FC<ModelInsightsProps> = ({
  data,
  isLoading = false,
  error,
  showFeatureImportance = true,
  showPermutationImportance = true,
  showRankingTable = true,
}) => {
  const [currentTab, setCurrentTab] = useState<InsightTab>('importance');

  // Loading state
  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" py={8}>
        <CircularProgress />
        <Typography variant="body1" color="text.secondary" sx={{ ml: 2 }}>
          Loading model insights...
        </Typography>
      </Box>
    );
  }

  // Error state
  if (error) {
    return (
      <Alert severity="error" sx={{ my: 2 }}>
        <Typography variant="body2" fontWeight="500">
          Error loading model insights
        </Typography>
        <Typography variant="body2" sx={{ mt: 0.5 }}>
          {error}
        </Typography>
      </Alert>
    );
  }

  // No data state
  if (!data) {
    return (
      <Alert severity="info" sx={{ my: 2 }}>
        <Typography variant="body2">
          No model insights data available. Feature importance and interpretability metrics
          will be displayed here after model training.
        </Typography>
      </Alert>
    );
  }

  // Check which insights are available
  const hasFeatureImportance = showFeatureImportance && data.featureImportance && data.featureImportance.length > 0;
  const hasPermutationImportance = showPermutationImportance && data.permutationImportance && data.permutationImportance.length > 0;
  const hasRankingTable = showRankingTable && data.featureRankings && data.featureRankings.length > 0;

  if (!hasFeatureImportance && !hasPermutationImportance && !hasRankingTable) {
    return (
      <Alert severity="info" sx={{ my: 2 }}>
        <Typography variant="body2">
          No feature importance data available. This model may not support feature importance
          calculation, or the data has not been computed yet.
        </Typography>
      </Alert>
    );
  }

  // Determine default tab
  const defaultTab = hasFeatureImportance
    ? 'importance'
    : hasPermutationImportance
      ? 'permutation'
      : 'ranking';

  // Set initial tab if current tab is not available
  React.useEffect(() => {
    if (
      (currentTab === 'importance' && !hasFeatureImportance) ||
      (currentTab === 'permutation' && !hasPermutationImportance) ||
      (currentTab === 'ranking' && !hasRankingTable)
    ) {
      setCurrentTab(defaultTab);
    }
  }, [currentTab, hasFeatureImportance, hasPermutationImportance, hasRankingTable, defaultTab]);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: InsightTab) => {
    setCurrentTab(newValue);
  };

  return (
    <Box>
      {/* Info Box */}
      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2" fontWeight="500" gutterBottom>
          Model Interpretability & Feature Insights:
        </Typography>
        <Box display="flex" flexDirection="column" gap={0.5}>
          {hasFeatureImportance && (
            <Typography variant="caption" color="text.secondary">
              • <strong>Feature Importance</strong>: Model-specific importance scores based on{' '}
              {data.importanceType || 'gain'}
            </Typography>
          )}
          {hasPermutationImportance && (
            <Typography variant="caption" color="text.secondary">
              • <strong>Permutation Importance</strong>: Model-agnostic importance via feature
              shuffling (metric: {data.metric || 'accuracy'})
            </Typography>
          )}
          {hasRankingTable && (
            <Typography variant="caption" color="text.secondary">
              • <strong>Feature Rankings</strong>: Detailed comparison and statistics across
              multiple metrics
            </Typography>
          )}
        </Box>
      </Alert>

      {/* Tabs */}
      <Paper elevation={2}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={currentTab}
            onChange={handleTabChange}
            aria-label="model insights tabs"
            variant="fullWidth"
          >
            {hasFeatureImportance && (
              <Tab
                label="Feature Importance"
                value="importance"
                id="insights-tab-importance"
                aria-controls="insights-tabpanel-importance"
              />
            )}
            {hasPermutationImportance && (
              <Tab
                label="Permutation Importance"
                value="permutation"
                id="insights-tab-permutation"
                aria-controls="insights-tabpanel-permutation"
              />
            )}
            {hasRankingTable && (
              <Tab
                label="Feature Rankings"
                value="ranking"
                id="insights-tab-ranking"
                aria-controls="insights-tabpanel-ranking"
              />
            )}
          </Tabs>
        </Box>

        {/* Tab Panels */}
        <Box sx={{ p: 3 }}>
          {/* Feature Importance Tab */}
          {hasFeatureImportance && currentTab === 'importance' && (
            <FeatureImportancePlot
              features={data.featureImportance!}
              importanceType={data.importanceType}
              showTopN={20}
              sortByImportance={true}
              showErrorBars={true}
            />
          )}

          {/* Permutation Importance Tab */}
          {hasPermutationImportance && currentTab === 'permutation' && (
            <PermutationImportancePlot
              features={data.permutationImportance!}
              metric={data.metric}
              showTopN={20}
            />
          )}

          {/* Feature Rankings Tab */}
          {hasRankingTable && currentTab === 'ranking' && (
            <FeatureRankingTable
              features={data.featureRankings!}
              showSearch={true}
              showPermutation={hasPermutationImportance}
              showCorrelation={true}
            />
          )}
        </Box>
      </Paper>
    </Box>
  );
};

export default ModelInsights;
