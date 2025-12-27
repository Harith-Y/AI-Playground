/**
 * Feature Selection Types
 *
 * Types for feature and target column selection UI components
 */

/**
 * Task type for model training
 */
export const TaskType = {
  CLASSIFICATION: 'classification',
  REGRESSION: 'regression',
  CLUSTERING: 'clustering',
  ANOMALY_DETECTION: 'anomaly_detection',
} as const;

export type TaskType = (typeof TaskType)[keyof typeof TaskType];

/**
 * Column data type
 */
export const ColumnDataType = {
  NUMERIC: 'numeric',
  CATEGORICAL: 'categorical',
  TEXT: 'text',
  DATETIME: 'datetime',
  BOOLEAN: 'boolean',
} as const;

export type ColumnDataType = (typeof ColumnDataType)[keyof typeof ColumnDataType];

/**
 * Column information with metadata
 */
export interface ColumnInfo {
  name: string;
  dataType: ColumnDataType;
  dtype: string; // Original pandas dtype
  missing_count?: number;
  missing_percentage?: number;
  unique_count?: number;
  sample_values?: any[];
}

/**
 * Feature selection configuration
 */
export interface FeatureSelectionConfig {
  inputFeatures: string[];
  targetColumn: string | null;
  taskType: TaskType | null;
  excludedColumns?: string[];
}

/**
 * Validation result for feature selection
 */
export interface FeatureSelectionValidation {
  isValid: boolean;
  errors: {
    inputFeatures?: string;
    targetColumn?: string;
    taskType?: string;
  };
  warnings: {
    inputFeatures?: string;
    targetColumn?: string;
    taskType?: string;
  };
}

/**
 * Feature selection state in Redux
 */
export interface FeatureSelectionState {
  config: FeatureSelectionConfig;
  validation: FeatureSelectionValidation;
  isConfigured: boolean;
}

/**
 * Props for FeatureSelector component
 */
export interface FeatureSelectorProps {
  datasetId: string;
  columns: ColumnInfo[];
  selectedFeatures: string[];
  excludedColumns?: string[];
  onChange: (features: string[]) => void;
  maxFeatures?: number;
  disabled?: boolean;
}

/**
 * Props for TargetColumnSelector component
 */
export interface TargetColumnSelectorProps {
  datasetId: string;
  columns: ColumnInfo[];
  selectedTarget: string | null;
  taskType: TaskType | null;
  onChange: (target: string | null) => void;
  onTaskTypeChange: (taskType: TaskType | null) => void;
  disabled?: boolean;
  excludedColumns?: string[];
}

/**
 * Props for combined FeatureSelection component
 */
export interface FeatureSelectionProps {
  datasetId: string;
  columns: ColumnInfo[];
  initialConfig?: Partial<FeatureSelectionConfig>;
  onConfigChange: (config: FeatureSelectionConfig) => void;
  onValidationChange?: (validation: FeatureSelectionValidation) => void;
  disabled?: boolean;
}

/**
 * Task type metadata
 */
export interface TaskTypeMetadata {
  label: string;
  description: string;
  icon: string;
  recommendedTargetTypes: ColumnDataType[];
  requiresTarget: boolean;
}

/**
 * Task type configurations
 */
export const TASK_TYPE_CONFIGS: Record<TaskType, TaskTypeMetadata> = {
  [TaskType.CLASSIFICATION]: {
    label: 'Classification',
    description: 'Predict categorical outcomes (e.g., spam detection, image classification)',
    icon: 'category',
    recommendedTargetTypes: [ColumnDataType.CATEGORICAL, ColumnDataType.BOOLEAN],
    requiresTarget: true,
  },
  [TaskType.REGRESSION]: {
    label: 'Regression',
    description: 'Predict continuous values (e.g., house prices, temperature)',
    icon: 'trending_up',
    recommendedTargetTypes: [ColumnDataType.NUMERIC],
    requiresTarget: true,
  },
  [TaskType.CLUSTERING]: {
    label: 'Clustering',
    description: 'Group similar data points (e.g., customer segmentation)',
    icon: 'scatter_plot',
    recommendedTargetTypes: [],
    requiresTarget: false,
  },
  [TaskType.ANOMALY_DETECTION]: {
    label: 'Anomaly Detection',
    description: 'Identify unusual patterns (e.g., fraud detection)',
    icon: 'warning',
    recommendedTargetTypes: [],
    requiresTarget: false,
  },
};

/**
 * Helper function to get column data type from pandas dtype
 */
export function getColumnDataType(dtype: string): ColumnDataType {
  const dtypeLower = dtype.toLowerCase();

  if (dtypeLower.includes('int') || dtypeLower.includes('float')) {
    return ColumnDataType.NUMERIC;
  }
  if (dtypeLower.includes('bool')) {
    return ColumnDataType.BOOLEAN;
  }
  if (dtypeLower.includes('datetime') || dtypeLower.includes('date')) {
    return ColumnDataType.DATETIME;
  }
  if (dtypeLower.includes('object') || dtypeLower.includes('string') || dtypeLower.includes('category')) {
    return ColumnDataType.CATEGORICAL;
  }

  return ColumnDataType.TEXT;
}

/**
 * Helper function to validate feature selection
 */
export function validateFeatureSelection(
  config: FeatureSelectionConfig
): FeatureSelectionValidation {
  const errors: FeatureSelectionValidation['errors'] = {};
  const warnings: FeatureSelectionValidation['warnings'] = {};

  // Validate input features
  if (config.inputFeatures.length === 0) {
    errors.inputFeatures = 'Please select at least one input feature';
  } else if (config.inputFeatures.length === 1) {
    warnings.inputFeatures = 'Consider selecting multiple features for better model performance';
  }

  // Validate task type
  if (!config.taskType) {
    errors.taskType = 'Please select a task type';
  }

  // Validate target column for supervised learning
  const taskConfig = config.taskType ? TASK_TYPE_CONFIGS[config.taskType] : null;

  if (taskConfig?.requiresTarget && !config.targetColumn) {
    errors.targetColumn = 'Target column is required for this task type';
  }

  // Validate target not in input features
  if (config.targetColumn && config.inputFeatures.includes(config.targetColumn)) {
    errors.targetColumn = 'Target column cannot be an input feature';
  }

  const isValid = Object.keys(errors).length === 0;

  return {
    isValid,
    errors,
    warnings,
  };
}
