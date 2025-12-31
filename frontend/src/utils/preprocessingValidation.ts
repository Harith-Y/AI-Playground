// Preprocessing validation utilities

import type { PreprocessingStepCreate, StepType, ParameterField } from '../types/preprocessing';
import { STEP_TYPE_CONFIGS } from '../types/preprocessing';

export interface ValidationError {
  field: string;
  message: string;
}

export interface ValidationResult {
  isValid: boolean;
  errors: Record<string, string>;
  errorList: ValidationError[];
}

/**
 * Validates a preprocessing step configuration
 */
export const validatePreprocessingStep = (
  step: Partial<PreprocessingStepCreate>,
  columns: { name: string; dtype: string }[]
): ValidationResult => {
  const errors: Record<string, string> = {};
  const errorList: ValidationError[] = [];

  // Validate step type
  if (!step.step_type) {
    errors.stepType = 'Step type is required';
    errorList.push({ field: 'stepType', message: 'Step type is required' });
  }

  const config = step.step_type ? STEP_TYPE_CONFIGS[step.step_type] : null;

  // Validate column selection
  if (config?.requiresColumn) {
    if (!step.column_name) {
      errors.column = 'Column selection is required for this step type';
      errorList.push({ field: 'column', message: 'Column selection is required for this step type' });
    } else {
      // Check if column exists in dataset
      const columnExists = columns.some(col => col.name === step.column_name);
      if (!columnExists) {
        errors.column = `Column "${step.column_name}" does not exist in the dataset`;
        errorList.push({ field: 'column', message: `Column "${step.column_name}" does not exist in the dataset` });
      }
    }
  }

  // Validate parameters
  if (config && step.parameters && step.step_type) {
    const paramErrors = validateParameters(step.parameters, config.parameterSchema, step.step_type);
    Object.assign(errors, paramErrors.errors);
    errorList.push(...paramErrors.errorList);
  }

  // Check for required parameters
  if (config) {
    config.parameterSchema.forEach((field: ParameterField) => {
      if (field.required && (!step.parameters || step.parameters[field.name] === undefined || step.parameters[field.name] === '')) {
        errors[field.name] = `${field.label} is required`;
        errorList.push({ field: field.name, message: `${field.label} is required` });
      }
    });
  }

  return {
    isValid: errorList.length === 0,
    errors,
    errorList,
  };
};

/**
 * Validates preprocessing step parameters
 */
export const validateParameters = (
  parameters: Record<string, any>,
  schema: ParameterField[],
  stepType: StepType
): { errors: Record<string, string>; errorList: ValidationError[] } => {
  const errors: Record<string, string> = {};
  const errorList: ValidationError[] = [];

  schema.forEach((field: ParameterField) => {
    const value = parameters[field.name];

    // Skip validation if value is undefined/null and not required
    if ((value === undefined || value === null || value === '') && !field.required) {
      return;
    }

    // Type-specific validation
    switch (field.type) {
      case 'number':
        if (value !== undefined && value !== '') {
          const numValue = Number(value);
          if (isNaN(numValue)) {
            errors[field.name] = 'Must be a valid number';
            errorList.push({ field: field.name, message: 'Must be a valid number' });
          } else {
            if (field.min !== undefined && numValue < field.min) {
              errors[field.name] = `Must be at least ${field.min}`;
              errorList.push({ field: field.name, message: `Must be at least ${field.min}` });
            }
            if (field.max !== undefined && numValue > field.max) {
              errors[field.name] = `Must be at most ${field.max}`;
              errorList.push({ field: field.name, message: `Must be at most ${field.max}` });
            }
          }
        }
        break;

      case 'select':
        if (value !== undefined && value !== '' && field.options) {
          const validOptions = field.options.map(opt => opt.value);
          if (!validOptions.includes(value)) {
            errors[field.name] = 'Invalid selection';
            errorList.push({ field: field.name, message: 'Invalid selection' });
          }
        }
        break;

      case 'text':
        if (field.required && (!value || value.trim() === '')) {
          errors[field.name] = `${field.label} is required`;
          errorList.push({ field: field.name, message: `${field.label} is required` });
        }
        break;
    }
  });

  // Step-specific validations
  if (stepType === 'missing_value_imputation') {
    if (parameters.strategy === 'constant' && !parameters.fill_value) {
      errors.fill_value = 'Fill value is required when using constant strategy';
      errorList.push({ field: 'fill_value', message: 'Fill value is required when using constant strategy' });
    }
  }

  if (stepType === 'scaling') {
    if (parameters.method === 'minmax') {
      const min = Number(parameters.feature_range_min);
      const max = Number(parameters.feature_range_max);
      if (!isNaN(min) && !isNaN(max) && min >= max) {
        errors.feature_range_max = 'Max value must be greater than min value';
        errorList.push({ field: 'feature_range_max', message: 'Max value must be greater than min value' });
      }
    }
  }

  if (stepType === 'feature_selection') {
    if (parameters.n_features !== undefined && parameters.n_features !== '') {
      const nFeatures = Number(parameters.n_features);
      if (nFeatures < 1) {
        errors.n_features = 'Must select at least 1 feature';
        errorList.push({ field: 'n_features', message: 'Must select at least 1 feature' });
      }
    }
  }

  return { errors, errorList };
};

/**
 * Validates dataset before preprocessing
 */
export const validateDatasetForPreprocessing = (
  dataset: { id: string; name: string; shape?: [number, number] } | null,
  steps: any[]
): ValidationResult => {
  const errors: Record<string, string> = {};
  const errorList: ValidationError[] = [];

  if (!dataset) {
    errors.dataset = 'No dataset selected';
    errorList.push({ field: 'dataset', message: 'No dataset selected' });
    return { isValid: false, errors, errorList };
  }

  if (steps.length === 0) {
    errors.steps = 'No preprocessing steps configured';
    errorList.push({ field: 'steps', message: 'No preprocessing steps configured' });
  }

  // Check if dataset is empty
  if (dataset.shape && dataset.shape[0] === 0) {
    errors.dataset = 'Dataset is empty (0 rows)';
    errorList.push({ field: 'dataset', message: 'Dataset is empty (0 rows)' });
  }

  if (dataset.shape && dataset.shape[1] === 0) {
    errors.dataset = 'Dataset has no columns';
    errorList.push({ field: 'dataset', message: 'Dataset has no columns' });
  }

  return {
    isValid: errorList.length === 0,
    errors,
    errorList,
  };
};

/**
 * Validates column compatibility with step type
 */
export const validateColumnForStep = (
  columnName: string,
  columnType: string,
  stepType: StepType
): { isValid: boolean; warning?: string } => {
  const isNumeric = columnType?.includes('int') || columnType?.includes('float');
  const isCategorical = columnType?.includes('object') || columnType?.includes('string') || columnType?.includes('category');

  switch (stepType) {
    case 'missing_value_imputation':
      // Works for both numeric and categorical
      if (!isNumeric && !isCategorical) {
        return {
          isValid: false,
          warning: `Column type "${columnType}" may not be suitable for imputation`,
        };
      }
      return { isValid: true };

    case 'scaling':
      // Only works for numeric
      if (!isNumeric) {
        return {
          isValid: false,
          warning: `Scaling requires numeric columns. "${columnName}" is ${columnType}`,
        };
      }
      return { isValid: true };

    case 'encoding':
      // Only works for categorical
      if (!isCategorical) {
        return {
          isValid: false,
          warning: `Encoding requires categorical columns. "${columnName}" is ${columnType}`,
        };
      }
      return { isValid: true };

    case 'outlier_detection':
      // Only works for numeric
      if (!isNumeric) {
        return {
          isValid: false,
          warning: `Outlier detection requires numeric columns. "${columnName}" is ${columnType}`,
        };
      }
      return { isValid: true };

    case 'transformation':
      // Only works for numeric
      if (!isNumeric) {
        return {
          isValid: false,
          warning: `Transformation requires numeric columns. "${columnName}" is ${columnType}`,
        };
      }
      return { isValid: true };

    default:
      return { isValid: true };
  }
};

/**
 * Get user-friendly error message for backend errors
 */
export const getErrorMessage = (error: any): string => {
  if (typeof error === 'string') {
    return error;
  }

  if (error?.response?.data?.detail) {
    // FastAPI error format
    if (typeof error.response.data.detail === 'string') {
      return error.response.data.detail;
    }
    if (Array.isArray(error.response.data.detail)) {
      // Validation errors
      return error.response.data.detail
        .map((e: any) => `${e.loc?.join('.') || 'Error'}: ${e.msg}`)
        .join('; ');
    }
  }

  if (error?.response?.data?.message) {
    return error.response.data.message;
  }

  if (error?.message) {
    return error.message;
  }

  // Default messages for common HTTP errors
  if (error?.response?.status) {
    switch (error.response.status) {
      case 400:
        return 'Invalid request. Please check your configuration.';
      case 404:
        return 'Resource not found. The dataset or step may have been deleted.';
      case 422:
        return 'Validation error. Please check your inputs.';
      case 500:
        return 'Server error. Please try again later.';
      default:
        return `Request failed with status ${error.response.status}`;
    }
  }

  return 'An unexpected error occurred';
};
