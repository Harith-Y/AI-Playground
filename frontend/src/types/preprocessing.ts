// Preprocessing types matching backend schema

export interface PreprocessingStep {
  id: string;
  dataset_id: string;
  step_type: StepType;
  parameters: Record<string, any>;
  column_name?: string;
  order: number;
}

export interface PreprocessingStepCreate {
  dataset_id: string;
  step_type: StepType;
  parameters: Record<string, any>;
  column_name?: string;
  order?: number;
}

export interface PreprocessingStepUpdate {
  step_type?: StepType;
  parameters?: Record<string, any>;
  column_name?: string;
  order?: number;
}

export type StepType =
  | 'missing_value_imputation'
  | 'scaling'
  | 'encoding'
  | 'outlier_detection'
  | 'feature_selection'
  | 'transformation';

export interface StepTypeConfig {
  type: StepType;
  label: string;
  description: string;
  icon: string;
  parameterSchema: ParameterField[];
  requiresColumn: boolean;
}

export interface ParameterField {
  name: string;
  label: string;
  type: 'select' | 'number' | 'text' | 'boolean' | 'multiselect';
  options?: { value: string; label: string }[];
  defaultValue?: any;
  min?: number;
  max?: number;
  step?: number;
  required?: boolean;
  tooltip?: string;
}

// Step type configurations
export const STEP_TYPE_CONFIGS: Record<StepType, StepTypeConfig> = {
  missing_value_imputation: {
    type: 'missing_value_imputation',
    label: 'Missing Value Imputation',
    description: 'Fill missing values with statistical methods',
    icon: 'AutoFixHigh',
    requiresColumn: true,
    parameterSchema: [
      {
        name: 'strategy',
        label: 'Strategy',
        type: 'select',
        required: true,
        options: [
          { value: 'mean', label: 'Mean' },
          { value: 'median', label: 'Median' },
          { value: 'mode', label: 'Mode' },
          { value: 'constant', label: 'Constant Value' },
          { value: 'ffill', label: 'Forward Fill' },
          { value: 'bfill', label: 'Backward Fill' },
        ],
        defaultValue: 'mean',
        tooltip: 'Method to use for filling missing values',
      },
      {
        name: 'fill_value',
        label: 'Fill Value',
        type: 'text',
        tooltip: 'Value to use when strategy is "constant"',
      },
    ],
  },
  scaling: {
    type: 'scaling',
    label: 'Feature Scaling',
    description: 'Normalize or standardize feature values',
    icon: 'LinearScale',
    requiresColumn: true,
    parameterSchema: [
      {
        name: 'method',
        label: 'Scaling Method',
        type: 'select',
        required: true,
        options: [
          { value: 'standard', label: 'Standard Scaler (z-score)' },
          { value: 'minmax', label: 'Min-Max Scaler (0-1)' },
          { value: 'robust', label: 'Robust Scaler (IQR)' },
          { value: 'maxabs', label: 'Max Abs Scaler' },
        ],
        defaultValue: 'standard',
        tooltip: 'Choose scaling algorithm',
      },
      {
        name: 'feature_range_min',
        label: 'Min Value',
        type: 'number',
        defaultValue: 0,
        tooltip: 'Minimum value for Min-Max scaling',
      },
      {
        name: 'feature_range_max',
        label: 'Max Value',
        type: 'number',
        defaultValue: 1,
        tooltip: 'Maximum value for Min-Max scaling',
      },
    ],
  },
  encoding: {
    type: 'encoding',
    label: 'Categorical Encoding',
    description: 'Convert categorical variables to numeric',
    icon: 'Tag',
    requiresColumn: true,
    parameterSchema: [
      {
        name: 'method',
        label: 'Encoding Method',
        type: 'select',
        required: true,
        options: [
          { value: 'onehot', label: 'One-Hot Encoding' },
          { value: 'label', label: 'Label Encoding' },
          { value: 'ordinal', label: 'Ordinal Encoding' },
          { value: 'target', label: 'Target Encoding' },
        ],
        defaultValue: 'onehot',
        tooltip: 'Choose encoding strategy for categorical variables',
      },
      {
        name: 'handle_unknown',
        label: 'Handle Unknown',
        type: 'select',
        options: [
          { value: 'error', label: 'Raise Error' },
          { value: 'ignore', label: 'Ignore' },
          { value: 'use_encoded_value', label: 'Use Encoded Value' },
        ],
        defaultValue: 'ignore',
        tooltip: 'How to handle unknown categories',
      },
    ],
  },
  outlier_detection: {
    type: 'outlier_detection',
    label: 'Outlier Detection',
    description: 'Identify and handle outliers in data',
    icon: 'BubbleChart',
    requiresColumn: true,
    parameterSchema: [
      {
        name: 'method',
        label: 'Detection Method',
        type: 'select',
        required: true,
        options: [
          { value: 'iqr', label: 'IQR (Interquartile Range)' },
          { value: 'zscore', label: 'Z-Score' },
          { value: 'isolation_forest', label: 'Isolation Forest' },
        ],
        defaultValue: 'iqr',
        tooltip: 'Statistical method for outlier detection',
      },
      {
        name: 'threshold',
        label: 'Threshold',
        type: 'number',
        defaultValue: 1.5,
        min: 0.1,
        max: 10,
        step: 0.1,
        tooltip: 'IQR multiplier (1.5 = standard, 3.0 = extreme)',
      },
      {
        name: 'action',
        label: 'Action',
        type: 'select',
        options: [
          { value: 'remove', label: 'Remove Outliers' },
          { value: 'cap', label: 'Cap Values' },
          { value: 'flag', label: 'Flag Only' },
        ],
        defaultValue: 'flag',
        tooltip: 'What to do with detected outliers',
      },
    ],
  },
  feature_selection: {
    type: 'feature_selection',
    label: 'Feature Selection',
    description: 'Select most important features',
    icon: 'FilterList',
    requiresColumn: false,
    parameterSchema: [
      {
        name: 'method',
        label: 'Selection Method',
        type: 'select',
        required: true,
        options: [
          { value: 'variance_threshold', label: 'Variance Threshold' },
          { value: 'correlation', label: 'Correlation Based' },
          { value: 'mutual_information', label: 'Mutual Information' },
          { value: 'recursive_elimination', label: 'Recursive Feature Elimination' },
        ],
        defaultValue: 'variance_threshold',
        tooltip: 'Algorithm for selecting features',
      },
      {
        name: 'threshold',
        label: 'Threshold',
        type: 'number',
        defaultValue: 0.01,
        min: 0,
        max: 1,
        step: 0.01,
        tooltip: 'Minimum variance/correlation threshold',
      },
      {
        name: 'n_features',
        label: 'Number of Features',
        type: 'number',
        min: 1,
        tooltip: 'Target number of features to select',
      },
    ],
  },
  transformation: {
    type: 'transformation',
    label: 'Data Transformation',
    description: 'Apply mathematical transformations',
    icon: 'Functions',
    requiresColumn: true,
    parameterSchema: [
      {
        name: 'method',
        label: 'Transformation',
        type: 'select',
        required: true,
        options: [
          { value: 'log', label: 'Log Transform' },
          { value: 'sqrt', label: 'Square Root' },
          { value: 'box-cox', label: 'Box-Cox' },
          { value: 'yeo-johnson', label: 'Yeo-Johnson' },
          { value: 'quantile', label: 'Quantile Transform' },
        ],
        defaultValue: 'log',
        tooltip: 'Mathematical transformation to apply',
      },
      {
        name: 'output_distribution',
        label: 'Output Distribution',
        type: 'select',
        options: [
          { value: 'uniform', label: 'Uniform' },
          { value: 'normal', label: 'Normal' },
        ],
        defaultValue: 'normal',
        tooltip: 'For quantile transform only',
      },
    ],
  },
};

// API Request/Response types
export interface PreprocessingApplyRequest {
  dataset_id: string;
  save_output?: boolean;
  output_name?: string;
}

export interface PreprocessingApplyResponse {
  success: boolean;
  message: string;
  steps_applied: number;
  original_shape: [number, number];
  transformed_shape: [number, number];
  output_dataset_id?: string;
  preview?: Record<string, any>[];
  statistics?: Record<string, any>;
}

export interface PreprocessingAsyncResponse {
  task_id: string;
  message: string;
  status: string;
}

export interface PreprocessingTaskStatus {
  task_id: string;
  state: 'PENDING' | 'STARTED' | 'PROGRESS' | 'SUCCESS' | 'FAILURE';
  status?: string;
  progress?: number;
  current_step?: string;
  result?: PreprocessingApplyResponse;
  error?: string;
}
