import { useState, useCallback } from 'react';
import modelingService from '../services/modelingService';
import {
  handleTrainingError,
  validateTrainingConfig,
  getRetryDelay,
  TrainingErrorType,
} from '../utils/trainingErrorHandler';
import type { TrainingConfig } from '../types/training';
import type { ErrorHandlerResult, ValidationError } from '../utils/trainingErrorHandler';

export interface UseTrainingWithErrorHandlingOptions {
  onSuccess?: (runId: string) => void;
  onError?: (error: ErrorHandlerResult) => void;
  maxRetries?: number;
  autoRetry?: boolean;
}

export interface UseTrainingWithErrorHandlingReturn {
  startTraining: (config: TrainingConfig) => Promise<string | null>;
  validateConfig: (config: TrainingConfig) => ValidationError[];
  training: boolean;
  error: ErrorHandlerResult | null;
  validationErrors: ValidationError[];
  retryCount: number;
  canRetry: boolean;
  retry: () => Promise<void>;
  clearError: () => void;
}

export const useTrainingWithErrorHandling = (
  options: UseTrainingWithErrorHandlingOptions = {}
): UseTrainingWithErrorHandlingReturn => {
  const {
    onSuccess,
    onError,
    maxRetries = 3,
    autoRetry = false,
  } = options;

  const [training, setTraining] = useState(false);
  const [error, setError] = useState<ErrorHandlerResult | null>(null);
  const [validationErrors, setValidationErrors] = useState<ValidationError[]>([]);
  const [retryCount, setRetryCount] = useState(0);
  const [lastConfig, setLastConfig] = useState<TrainingConfig | null>(null);

  const validateConfig = useCallback((config: TrainingConfig): ValidationError[] => {
    const errors = validateTrainingConfig(config);
    setValidationErrors(errors);
    return errors;
  }, []);

  const handleError = useCallback((err: any) => {
    const errorResult = handleTrainingError(err);
    setError(errorResult);
    
    if (onError) {
      onError(errorResult);
    }
    
    return errorResult;
  }, [onError]);

  const attemptTraining = useCallback(async (
    config: TrainingConfig,
    attempt: number = 1
  ): Promise<string | null> => {
    try {
      setTraining(true);
      setError(null);

      // Validate configuration
      const errors = validateConfig(config);
      const hasErrors = errors.some(e => e.severity === 'error');
      
      if (hasErrors) {
        const errorResult: ErrorHandlerResult = {
          userMessage: 'Configuration validation failed',
          technicalMessage: 'Please fix all validation errors before starting training',
          suggestions: errors.map(e => `${e.field}: ${e.message}`),
          recoverable: true,
          retryable: false,
          severity: 'error',
        };
        setError(errorResult);
        return null;
      }

      // Start training
      const response = await modelingService.startTraining(config);
      
      // Reset retry count on success
      setRetryCount(0);
      setLastConfig(null);
      
      if (onSuccess) {
        onSuccess(response.runId);
      }

      return response.runId;
    } catch (err: any) {
      console.error('[Training] Error:', err);
      
      const errorResult = handleError(err);
      
      // Auto-retry logic
      if (autoRetry && errorResult.retryable && attempt < maxRetries) {
        const delay = getRetryDelay(
          err.type || TrainingErrorType.UNKNOWN,
          attempt
        );
        
        console.log(`[Training] Retrying in ${delay}ms (attempt ${attempt + 1}/${maxRetries})`);
        
        await new Promise(resolve => setTimeout(resolve, delay));
        
        setRetryCount(attempt);
        return attemptTraining(config, attempt + 1);
      }
      
      setRetryCount(attempt);
      return null;
    } finally {
      setTraining(false);
    }
  }, [validateConfig, handleError, onSuccess, autoRetry, maxRetries]);

  const startTraining = useCallback(async (config: TrainingConfig): Promise<string | null> => {
    setLastConfig(config);
    setRetryCount(0);
    return attemptTraining(config, 1);
  }, [attemptTraining]);

  const retry = useCallback(async (): Promise<void> => {
    if (!lastConfig) {
      console.warn('[Training] No previous configuration to retry');
      return;
    }
    
    if (!error?.retryable) {
      console.warn('[Training] Current error is not retryable');
      return;
    }
    
    await attemptTraining(lastConfig, retryCount + 1);
  }, [lastConfig, error, retryCount, attemptTraining]);

  const clearError = useCallback(() => {
    setError(null);
    setValidationErrors([]);
  }, []);

  const canRetry = Boolean(
    lastConfig &&
    error?.retryable &&
    retryCount < maxRetries
  );

  return {
    startTraining,
    validateConfig,
    training,
    error,
    validationErrors,
    retryCount,
    canRetry,
    retry,
    clearError,
  };
};

export default useTrainingWithErrorHandling;
