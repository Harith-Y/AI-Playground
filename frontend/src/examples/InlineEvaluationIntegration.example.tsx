/**
 * Example: How to integrate FE-41 Inline Evaluation Preview into ModelingPage
 * 
 * This shows how to display a compact evaluation preview
 * for the latest training run on the modeling page.
 */

import React from 'react';
import { Box, Container, Typography, Divider } from '@mui/material';

// Import the new components and hooks
import InlineEvaluationPreview from '../components/modeling/InlineEvaluationPreview';
import { useLatestTrainingResults } from '../hooks/useLatestTrainingResults';

const ModelingPageWithInlineEvaluation: React.FC = () => {
  // Fetch latest training results
  const {
    data: latestResults,
    loading: resultsLoading,
    error: resultsError,
    refresh: _refreshResults,
  } = useLatestTrainingResults({
    autoRefresh: true, // Auto-refresh every 10 seconds
    refreshInterval: 10000,
    // Optional filters:
    // datasetId: 'dataset-123',
    // modelType: 'random_forest',
  });

  return (
    <Container maxWidth="xl">
      <Box py={4}>
        <Typography variant="h4" gutterBottom>
          Model Training
        </Typography>

        {/* Your existing ModelingPage content */}
        {/* Model selection, hyperparameters, data split, etc. */}
        
        <Box my={4}>
          <Typography variant="body1" color="text.secondary">
            Configure your model and start training...
          </Typography>
        </Box>

        <Divider sx={{ my: 4 }} />

        {/* Inline Evaluation Preview */}
        <InlineEvaluationPreview
          data={latestResults}
          loading={resultsLoading}
          error={resultsError}
          onViewDetails={() => {
            // Navigate to full evaluation page
            console.log('View full evaluation');
          }}
        />

        {/* You can also manually refresh */}
        {/* <Button onClick={_refreshResults}>Refresh Results</Button> */}
      </Box>
    </Container>
  );
};

export default ModelingPageWithInlineEvaluation;

/**
 * USAGE NOTES:
 * 
 * 1. Basic Integration:
 *    - Import InlineEvaluationPreview component
 *    - Import useLatestTrainingResults hook
 *    - Place component anywhere on ModelingPage
 * 
 * 2. Auto-Refresh:
 *    - Enable autoRefresh to update automatically
 *    - Configurable refresh interval (default 10s)
 *    - Useful for monitoring latest training
 * 
 * 3. Filtering:
 *    - Filter by datasetId to show results for current dataset
 *    - Filter by modelType to show specific model results
 *    - Omit filters to show latest overall result
 * 
 * 4. Navigation:
 *    - onViewDetails callback for custom navigation
 *    - Default navigates to /evaluation?runId=xxx
 *    - Can integrate with your routing logic
 * 
 * 5. Display Features:
 *    - Shows 4 primary metrics (accuracy, F1, R², MAE)
 *    - Additional metrics as chips
 *    - Mini training history chart
 *    - Performance indicator
 *    - Model name and type
 * 
 * 6. States:
 *    - Loading: Shows skeleton
 *    - Error: Shows error alert
 *    - No data: Shows info message
 *    - Success: Shows full preview
 * 
 * 7. Responsive:
 *    - Grid layout adapts to screen size
 *    - 2 columns on mobile, 4 on desktop
 *    - Chart scales to container
 * 
 * 8. Performance:
 *    - Lightweight component
 *    - Only fetches latest run
 *    - Minimal API calls
 *    - Efficient rendering
 */

/**
 * ALTERNATIVE: Show results for specific run
 */
/*
import { useState } from 'react';

const ModelingPageWithSpecificRun: React.FC = () => {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  
  // Fetch specific run results
  const fetchRunResults = async (runId: string) => {
    const results = await modelingService.getTrainingResults(runId);
    const metrics = await modelingService.getTrainingMetrics(runId);
    
    return {
      runId,
      modelName: 'Model Name',
      modelType: 'random_forest',
      status: 'completed' as const,
      metrics: results.metrics,
      trainingHistory: metrics.metrics,
    };
  };

  return (
    <Box>
      {selectedRunId && (
        <InlineEvaluationPreview
          data={await fetchRunResults(selectedRunId)}
          loading={false}
          error={null}
        />
      )}
    </Box>
  );
};
*/

/**
 * ALTERNATIVE: Multiple previews
 */
/*
const ModelingPageWithMultiplePreviews: React.FC = () => {
  // Show latest for each model type
  const rfResults = useLatestTrainingResults({ modelType: 'random_forest' });
  const lrResults = useLatestTrainingResults({ modelType: 'logistic_regression' });
  
  return (
    <Box>
      <Typography variant="h6">Random Forest</Typography>
      <InlineEvaluationPreview {...rfResults} />
      
      <Typography variant="h6" mt={2}>Logistic Regression</Typography>
      <InlineEvaluationPreview {...lrResults} />
    </Box>
  );
};
*/
