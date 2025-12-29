import { useState, useCallback } from 'react';
import modelingService from '../services/modelingService';
import type { TrainingConfig } from '../types/training';

export interface UseModelTrainingOptions {
  onSuccess?: (runId: string) => void;
  onError?: (error: string) => void;
  validateBeforeStart?: boolean;
}

export interface UseModelTrainingReturn {
  startTraining: (config: TrainingConfig) => Promise<string | null>;
  validateConfig: (config: TrainingConfig) => Promise<boolean>;
  training: boolean;
  validating: boolean;
  error: string | null;
  validationErrors: string[];
  validationWarnings: string[];
}

export const useModelTraining = (
  options: UseModelTrainingOptions = {}
): UseModelTrainingReturn => {
  const {
    onSuccess,
    onError,
    validateBeforeStart = true,
  } = options;

  const [training, setTraining] = useState(false);
  const [validating, setValidating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [validationWarnings, setValidationWarnings] = useState<string[]>([]);

  const validateConfig = useCallback(async (config: TrainingConfig): Promise<boolean> => {
    try {
      setValidating(true);
      setValidationErrors([]);
      setValidationWarnings([]);

      const response = await modelingService.validateTrainingConfig(config);

      if (response.errors && response.errors.length > 0) {
        setValidationErrors(response.errors);
      }

      if (response.warnings && response.warnings.length > 0) {
        setValidationWarnings(response.warnings);
      }

      return response.valid;
    } catch (err) {
      console.error('[ModelTraining] Error validating config:', err);
      const errorMessage = err instanceof Error ? err.message : 'Validation failed';
      setValidationErrors([errorMessage]);
      return false;
    } finally {
      setValidating(false);
    }
  }, []);

  const startTraining = useCallback(async (config: TrainingConfig): Promise<string | null> => {
    try {
      setTraining(true);
      setError(null);
      setValidationErrors([]);
      setValidationWarnings([]);

      // Validate config if required
      if (validateBeforeStart) {
        const isValid = await validateConfig(config);
        if (!isValid) {
          const errorMsg = 'Training configuration is invalid';
          setError(errorMsg);
          if (onError) {
            onError(errorMsg);
          }
          return null;
        }
      }

      // Start training
      const response = await modelingService.startTraining(config);
      
      if (onSuccess) {
        onSuccess(response.runId);
      }

      return response.runId;
    } catch (err) {
      console.error('[ModelTraining] Error starting training:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to start training';
      setError(errorMessage);
      
      if (onError) {
        onError(errorMessage);
      }

      return null;
    } finally {
      setTraining(false);
    }
  }, [validateBeforeStart, validateConfig, onSuccess, onError]);

  return {
    startTraining,
    validateConfig,
    training,
    validating,
    error,
    validationErrors,
    validationWarnings,
  };
};

export default useModelTraining;
