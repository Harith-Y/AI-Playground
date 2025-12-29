/**
 * Example: How to integrate FE-40 Training Error Handling
 * 
 * This shows how to use error handling components and hooks
 * for robust training with user-friendly error messages.
 */

import React from 'react';
import { Box, Container, Button, Typography } from '@mui/material';
import { useNavigate } from 'react-router-dom';

// Import error handling components and hooks
import TrainingErrorDisplay from '../components/modeling/TrainingErrorDisplay';
import ValidationErrorsList from '../components/modeling/ValidationErrorsList';
import { useTrainingWithErrorHandling } from '../hooks/useTrainingWithErrorHandling';

// Import types
import type { TrainingConfig } from '../types/training';

const ModelingPageWithErrorHandling: React.FC = () => {
  const navigate = useNavigate();

  // Use the error-handling hook
  const {
    startTraining,
    validateConfig,
    training,
    error,
    validationErrors,
    retryCount,
    canRetry,
    retry,
    clearError,
  } = useTrainingWithErrorHandling({
    onSuccess: (runId) => {
      console.log('Training started successfully:', runId);
      navigate(`/training/${runId}`);
    },
    onError: (error) => {
      console.error('Training error:', error);
    },
    maxRetries: 3,
    autoRetry: false, // Set to true for automatic retries
  });

  // Example training configuration
  const exampleConfig: TrainingConfig = {
    modelType: 'random_forest',
    datasetId: 'dataset-123',
    hyperparameters: {
      n_estimators: 100,
      max_depth: 10,
    },
    trainTestSplit: 0.7,
    validationSplit: 0.15,
    epochs: 100,
    batchSize: 32,
  };

  const handleStartTraining = async () => {
    // Validate before starting
    const errors = validateConfig(exampleConfig);
    if (errors.some(e => e.severity === 'error')) {
      console.log('Validation failed:', errors);
      return;
    }

    // Start training
    await startTraining(exampleConfig);
  };

  return (
    <Container maxWidth="xl">
      <Box py={4}>
        <Typography variant="h4" gutterBottom>
          Model Training with Error Handling
        </Typography>

        {/* Validation Errors */}
        {validationErrors.length > 0 && (
          <Box mb={2}>
            <ValidationErrorsList
              errors={validationErrors}
              onDismiss={clearError}
            />
          </Box>
        )}

        {/* Training Error */}
        {error && (
          <Box mb={2}>
            <TrainingErrorDisplay
              error={error}
              onRetry={canRetry ? retry : undefined}
              onDismiss={clearError}
              showTechnicalDetails={true}
            />
          </Box>
        )}

        {/* Training Controls */}
        <Box display="flex" gap={2} alignItems="center">
          <Button
            variant="contained"
            onClick={handleStartTraining}
            disabled={training}
          >
            {training ? 'Training...' : 'Start Training'}
          </Button>

          {canRetry && (
            <Button
              variant="outlined"
              onClick={retry}
              disabled={training}
            >
              Retry ({retryCount}/{3})
            </Button>
          )}

          {error && (
            <Button
              variant="text"
              onClick={clearError}
            >
              Clear Error
            </Button>
          )}
        </Box>

        {/* Status Info */}
        <Box mt={2}>
          <Typography variant="body2" color="text.secondary">
            Status: {training ? 'Training...' : 'Ready'}
          </Typography>
          {retryCount > 0 && (
            <Typography variant="body2" color="text.secondary">
              Retry attempts: {retryCount}
            </Typography>
          )}
        </Box>
      </Box>
    </Container>
  );
};

export default ModelingPageWithErrorHandling;

/**
 * USAGE NOTES:
 * 
 * 1. Error Handling Hook:
 *    - Use `useTrainingWithErrorHandling` for automatic error handling
 *    - Provides validation, error parsing, and retry logic
 *    - Configurable max retries and auto-retry
 * 
 * 2. Validation:
 *    - Call `validateConfig` before starting training
 *    - Shows validation errors in ValidationErrorsList
 *    - Prevents training if errors exist
 * 
 * 3. Error Display:
 *    - TrainingErrorDisplay shows user-friendly messages
 *    - Includes suggestions for fixing errors
 *    - Optional technical details
 *    - Retry button for retryable errors
 * 
 * 4. Retry Logic:
 *    - Automatic retry with exponential backoff
 *    - Manual retry button
 *    - Configurable max retries
 *    - Tracks retry count
 * 
 * 5. Error Types Handled:
 *    - Configuration errors (invalid model, hyperparameters)
 *    - Data errors (insufficient data, missing values)
 *    - Runtime errors (timeout, out of memory)
 *    - System errors (network, server)
 * 
 * 6. User Experience:
 *    - Clear error messages
 *    - Actionable suggestions
 *    - Retry capability
 *    - Dismissible errors
 *    - Loading states
 */

/**
 * ALTERNATIVE: Manual Error Handling
 */
/*
import { handleTrainingError, validateTrainingConfig } from '../utils/trainingErrorHandler';

const ManualErrorHandling: React.FC = () => {
  const [error, setError] = useState(null);

  const handleTrain = async (config: TrainingConfig) => {
    try {
      // Validate
      const validationErrors = validateTrainingConfig(config);
      if (validationErrors.some(e => e.severity === 'error')) {
        // Handle validation errors
        return;
      }

      // Start training
      await modelingService.startTraining(config);
    } catch (err) {
      // Handle error
      const errorResult = handleTrainingError(err);
      setError(errorResult);
    }
  };

  return (
    <Box>
      {error && (
        <TrainingErrorDisplay
          error={error}
          onDismiss={() => setError(null)}
        />
      )}
    </Box>
  );
};
*/

/**
 * ALTERNATIVE: With Auto-Retry
 */
/*
const WithAutoRetry: React.FC = () => {
  const {
    startTraining,
    training,
    error,
    retryCount,
  } = useTrainingWithErrorHandling({
    maxRetries: 5,
    autoRetry: true, // Enable automatic retries
  });

  return (
    <Box>
      {training && retryCount > 0 && (
        <Alert severity="info">
          Retrying... (Attempt {retryCount}/5)
        </Alert>
      )}
      
      {error && !error.retryable && (
        <TrainingErrorDisplay error={error} />
      )}
    </Box>
  );
};
*/
