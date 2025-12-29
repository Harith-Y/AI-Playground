/**
 * Training Error Handler Utilities
 * 
 * Provides user-friendly error messages and recovery suggestions
 * for various training errors and edge cases.
 */

export const TrainingErrorType = {
  // Configuration Errors
  INVALID_MODEL: 'INVALID_MODEL',
  INVALID_CONFIG: 'INVALID_CONFIG',
  INVALID_HYPERPARAMETERS: 'INVALID_HYPERPARAMETERS',
  INVALID_DATASET: 'INVALID_DATASET',
  MISSING_TARGET: 'MISSING_TARGET',
  MISSING_FEATURES: 'MISSING_FEATURES',
  
  // Data Errors
  INSUFFICIENT_DATA: 'INSUFFICIENT_DATA',
  DATA_TYPE_MISMATCH: 'DATA_TYPE_MISMATCH',
  MISSING_VALUES: 'MISSING_VALUES',
  IMBALANCED_CLASSES: 'IMBALANCED_CLASSES',
  
  // Runtime Errors
  TIMEOUT: 'TIMEOUT',
  OUT_OF_MEMORY: 'OUT_OF_MEMORY',
  CONVERGENCE_FAILURE: 'CONVERGENCE_FAILURE',
  NUMERICAL_INSTABILITY: 'NUMERICAL_INSTABILITY',
  
  // System Errors
  NETWORK_ERROR: 'NETWORK_ERROR',
  SERVER_ERROR: 'SERVER_ERROR',
  RESOURCE_UNAVAILABLE: 'RESOURCE_UNAVAILABLE',
  
  // Unknown
  UNKNOWN: 'UNKNOWN',
} as const;

export type TrainingErrorType = typeof TrainingErrorType[keyof typeof TrainingErrorType];

export interface TrainingError {
  type: TrainingErrorType;
  message: string;
  details?: string;
  field?: string;
  suggestions?: string[];
  recoverable: boolean;
  retryable: boolean;
}

export interface ErrorHandlerResult {
  userMessage: string;
  technicalMessage: string;
  suggestions: string[];
  recoverable: boolean;
  retryable: boolean;
  severity: 'error' | 'warning' | 'info';
}

/**
 * Parse error from API response
 */
export function parseTrainingError(error: any): TrainingError {
  // Handle axios errors
  if (error.response) {
    const { status, data } = error.response;
    
    // Parse error type from response
    if (data.errorType) {
      return {
        type: data.errorType as TrainingErrorType,
        message: data.message || 'Training failed',
        details: data.details,
        field: data.field,
        suggestions: data.suggestions,
        recoverable: data.recoverable !== false,
        retryable: data.retryable !== false,
      };
    }
    
    // Infer error type from status code
    if (status === 400) {
      return {
        type: TrainingErrorType.INVALID_CONFIG,
        message: data.message || 'Invalid training configuration',
        details: data.details,
        recoverable: true,
        retryable: false,
      };
    }
    
    if (status === 404) {
      return {
        type: TrainingErrorType.INVALID_DATASET,
        message: 'Dataset not found',
        recoverable: true,
        retryable: false,
      };
    }
    
    if (status === 408 || status === 504) {
      return {
        type: TrainingErrorType.TIMEOUT,
        message: 'Training request timed out',
        recoverable: true,
        retryable: true,
      };
    }
    
    if (status >= 500) {
      return {
        type: TrainingErrorType.SERVER_ERROR,
        message: 'Server error occurred',
        details: data.message,
        recoverable: true,
        retryable: true,
      };
    }
  }
  
  // Handle network errors
  if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
    return {
      type: TrainingErrorType.TIMEOUT,
      message: 'Request timed out',
      recoverable: true,
      retryable: true,
    };
  }
  
  if (error.code === 'ERR_NETWORK' || !navigator.onLine) {
    return {
      type: TrainingErrorType.NETWORK_ERROR,
      message: 'Network connection lost',
      recoverable: true,
      retryable: true,
    };
  }
  
  // Parse error message for known patterns
  const errorMessage = error.message?.toLowerCase() || '';
  
  if (errorMessage.includes('memory')) {
    return {
      type: TrainingErrorType.OUT_OF_MEMORY,
      message: 'Out of memory',
      recoverable: true,
      retryable: false,
    };
  }
  
  if (errorMessage.includes('convergence')) {
    return {
      type: TrainingErrorType.CONVERGENCE_FAILURE,
      message: 'Model failed to converge',
      recoverable: true,
      retryable: false,
    };
  }
  
  if (errorMessage.includes('invalid') || errorMessage.includes('validation')) {
    return {
      type: TrainingErrorType.INVALID_CONFIG,
      message: error.message || 'Invalid configuration',
      recoverable: true,
      retryable: false,
    };
  }
  
  // Default unknown error
  return {
    type: TrainingErrorType.UNKNOWN,
    message: error.message || 'An unexpected error occurred',
    details: error.stack,
    recoverable: true,
    retryable: true,
  };
}

/**
 * Get user-friendly error message and suggestions
 */
export function handleTrainingError(error: any): ErrorHandlerResult {
  const parsedError = parseTrainingError(error);
  
  const handlers: Record<TrainingErrorType, () => ErrorHandlerResult> = {
    [TrainingErrorType.INVALID_MODEL]: () => ({
      userMessage: 'The selected model is not valid or not supported.',
      technicalMessage: parsedError.message,
      suggestions: [
        'Choose a different model from the available options',
        'Check if the model is compatible with your task type',
        'Verify that the model is properly configured',
      ],
      recoverable: true,
      retryable: false,
      severity: 'error',
    }),
    
    [TrainingErrorType.INVALID_CONFIG]: () => ({
      userMessage: 'Training configuration is invalid.',
      technicalMessage: parsedError.message,
      suggestions: parsedError.suggestions || [
        'Review your hyperparameter settings',
        'Check that all required fields are filled',
        'Ensure values are within valid ranges',
        parsedError.field ? `Fix the "${parsedError.field}" field` : '',
      ].filter(Boolean),
      recoverable: true,
      retryable: false,
      severity: 'error',
    }),
    
    [TrainingErrorType.INVALID_HYPERPARAMETERS]: () => ({
      userMessage: 'One or more hyperparameters are invalid.',
      technicalMessage: parsedError.message,
      suggestions: [
        'Check hyperparameter values are within valid ranges',
        'Ensure numeric values are positive where required',
        'Verify that string values match expected options',
        'Try using default hyperparameters',
      ],
      recoverable: true,
      retryable: false,
      severity: 'error',
    }),
    
    [TrainingErrorType.INVALID_DATASET]: () => ({
      userMessage: 'The selected dataset is not available or invalid.',
      technicalMessage: parsedError.message,
      suggestions: [
        'Verify the dataset exists and is accessible',
        'Check that the dataset has been properly uploaded',
        'Ensure the dataset format is correct',
        'Try refreshing the dataset list',
      ],
      recoverable: true,
      retryable: false,
      severity: 'error',
    }),
    
    [TrainingErrorType.MISSING_TARGET]: () => ({
      userMessage: 'No target column selected for training.',
      technicalMessage: parsedError.message,
      suggestions: [
        'Select a target column from your dataset',
        'Ensure the target column contains valid data',
        'Check that the target column matches your task type',
      ],
      recoverable: true,
      retryable: false,
      severity: 'error',
    }),
    
    [TrainingErrorType.MISSING_FEATURES]: () => ({
      userMessage: 'No features selected for training.',
      technicalMessage: parsedError.message,
      suggestions: [
        'Select at least one feature column',
        'Ensure selected features contain valid data',
        'Remove features with too many missing values',
      ],
      recoverable: true,
      retryable: false,
      severity: 'error',
    }),
    
    [TrainingErrorType.INSUFFICIENT_DATA]: () => ({
      userMessage: 'Not enough data to train the model.',
      technicalMessage: parsedError.message,
      suggestions: [
        'Use a larger dataset with more samples',
        'Reduce the validation split size',
        'Consider using a simpler model',
        'Check for and remove duplicate rows',
      ],
      recoverable: true,
      retryable: false,
      severity: 'error',
    }),
    
    [TrainingErrorType.DATA_TYPE_MISMATCH]: () => ({
      userMessage: 'Data types do not match model requirements.',
      technicalMessage: parsedError.message,
      suggestions: [
        'Ensure numeric features contain only numbers',
        'Convert categorical features to appropriate format',
        'Check for mixed data types in columns',
        'Apply proper preprocessing steps',
      ],
      recoverable: true,
      retryable: false,
      severity: 'error',
    }),
    
    [TrainingErrorType.MISSING_VALUES]: () => ({
      userMessage: 'Dataset contains missing values that need to be handled.',
      technicalMessage: parsedError.message,
      suggestions: [
        'Apply imputation in the preprocessing step',
        'Remove rows with missing values',
        'Use a model that handles missing values',
        'Check data quality before training',
      ],
      recoverable: true,
      retryable: false,
      severity: 'warning',
    }),
    
    [TrainingErrorType.IMBALANCED_CLASSES]: () => ({
      userMessage: 'Dataset has imbalanced classes which may affect performance.',
      technicalMessage: parsedError.message,
      suggestions: [
        'Use class weighting in model configuration',
        'Apply oversampling or undersampling',
        'Consider using different evaluation metrics',
        'Collect more data for minority classes',
      ],
      recoverable: true,
      retryable: false,
      severity: 'warning',
    }),
    
    [TrainingErrorType.TIMEOUT]: () => ({
      userMessage: 'Training request timed out.',
      technicalMessage: parsedError.message,
      suggestions: [
        'Try again - the server may be busy',
        'Use a smaller dataset or simpler model',
        'Reduce the number of training iterations',
        'Check your network connection',
      ],
      recoverable: true,
      retryable: true,
      severity: 'error',
    }),
    
    [TrainingErrorType.OUT_OF_MEMORY]: () => ({
      userMessage: 'Training ran out of memory.',
      technicalMessage: parsedError.message,
      suggestions: [
        'Use a smaller dataset or batch size',
        'Reduce model complexity',
        'Try a simpler model type',
        'Contact support if the issue persists',
      ],
      recoverable: true,
      retryable: false,
      severity: 'error',
    }),
    
    [TrainingErrorType.CONVERGENCE_FAILURE]: () => ({
      userMessage: 'Model failed to converge during training.',
      technicalMessage: parsedError.message,
      suggestions: [
        'Increase the number of training iterations',
        'Adjust the learning rate',
        'Try different hyperparameters',
        'Scale your features properly',
        'Use a different optimization algorithm',
      ],
      recoverable: true,
      retryable: false,
      severity: 'warning',
    }),
    
    [TrainingErrorType.NUMERICAL_INSTABILITY]: () => ({
      userMessage: 'Numerical instability detected during training.',
      technicalMessage: parsedError.message,
      suggestions: [
        'Scale your features to similar ranges',
        'Reduce the learning rate',
        'Check for extreme values in your data',
        'Try a more stable model type',
      ],
      recoverable: true,
      retryable: false,
      severity: 'error',
    }),
    
    [TrainingErrorType.NETWORK_ERROR]: () => ({
      userMessage: 'Network connection lost.',
      technicalMessage: parsedError.message,
      suggestions: [
        'Check your internet connection',
        'Try again in a few moments',
        'Verify the server is accessible',
      ],
      recoverable: true,
      retryable: true,
      severity: 'error',
    }),
    
    [TrainingErrorType.SERVER_ERROR]: () => ({
      userMessage: 'Server error occurred while processing your request.',
      technicalMessage: parsedError.message,
      suggestions: [
        'Try again in a few moments',
        'Contact support if the issue persists',
        'Check the system status page',
      ],
      recoverable: true,
      retryable: true,
      severity: 'error',
    }),
    
    [TrainingErrorType.RESOURCE_UNAVAILABLE]: () => ({
      userMessage: 'Training resources are currently unavailable.',
      technicalMessage: parsedError.message,
      suggestions: [
        'Try again later when resources are available',
        'Contact support for resource allocation',
        'Consider using a smaller model or dataset',
      ],
      recoverable: true,
      retryable: true,
      severity: 'error',
    }),
    
    [TrainingErrorType.UNKNOWN]: () => ({
      userMessage: 'An unexpected error occurred during training.',
      technicalMessage: parsedError.message,
      suggestions: [
        'Try again',
        'Check your configuration',
        'Contact support if the issue persists',
      ],
      recoverable: true,
      retryable: true,
      severity: 'error',
    }),
  };
  
  const handler = handlers[parsedError.type] || handlers[TrainingErrorType.UNKNOWN];
  return handler();
}

/**
 * Validate training configuration before submission
 */
export interface ValidationError {
  field: string;
  message: string;
  severity: 'error' | 'warning';
}

export function validateTrainingConfig(config: any): ValidationError[] {
  const errors: ValidationError[] = [];
  
  // Model validation
  if (!config.modelType) {
    errors.push({
      field: 'modelType',
      message: 'Model type is required',
      severity: 'error',
    });
  }
  
  // Dataset validation
  if (!config.datasetId) {
    errors.push({
      field: 'datasetId',
      message: 'Dataset is required',
      severity: 'error',
    });
  }
  
  // Target validation
  if (!config.target && !config.targetColumn) {
    errors.push({
      field: 'target',
      message: 'Target column is required',
      severity: 'error',
    });
  }
  
  // Features validation
  if (!config.features || config.features.length === 0) {
    errors.push({
      field: 'features',
      message: 'At least one feature is required',
      severity: 'error',
    });
  }
  
  // Split validation
  if (config.splitConfig) {
    const { trainSize, valSize, testSize } = config.splitConfig;
    const total = (trainSize || 0) + (valSize || 0) + (testSize || 0);
    
    if (Math.abs(total - 1.0) > 0.01) {
      errors.push({
        field: 'splitConfig',
        message: 'Train, validation, and test splits must sum to 1.0',
        severity: 'error',
      });
    }
    
    if (trainSize && trainSize < 0.5) {
      errors.push({
        field: 'splitConfig.trainSize',
        message: 'Training set should be at least 50% of the data',
        severity: 'warning',
      });
    }
  }
  
  // Hyperparameters validation
  if (config.hyperparameters) {
    // Model-specific validation would go here
    // This is a placeholder for extensibility
  }
  
  return errors;
}

/**
 * Get retry delay based on error type and attempt number
 */
export function getRetryDelay(errorType: TrainingErrorType, attempt: number): number {
  const baseDelays: Partial<Record<TrainingErrorType, number>> = {
    [TrainingErrorType.NETWORK_ERROR]: 2000,
    [TrainingErrorType.SERVER_ERROR]: 5000,
    [TrainingErrorType.TIMEOUT]: 10000,
    [TrainingErrorType.RESOURCE_UNAVAILABLE]: 30000,
  };
  
  const baseDelay = baseDelays[errorType] || 5000;
  
  // Exponential backoff with jitter
  const exponentialDelay = baseDelay * Math.pow(2, attempt - 1);
  const jitter = Math.random() * 1000;
  
  return Math.min(exponentialDelay + jitter, 60000); // Max 60 seconds
}

export default {
  parseTrainingError,
  handleTrainingError,
  validateTrainingConfig,
  getRetryDelay,
  TrainingErrorType,
};
