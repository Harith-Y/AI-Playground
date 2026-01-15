/**
 * Model Selection Types
 *
 * Types for ML model selection with hyperparameter presets
 */

/**
 * ML task types
 */
export const TaskType = {
  REGRESSION: 'regression',
  CLASSIFICATION: 'classification',
  CLUSTERING: 'clustering',
  ANOMALY_DETECTION: 'anomaly_detection',
} as const;

export type TaskType = (typeof TaskType)[keyof typeof TaskType];

/**
 * Model categories for organization
 */
export const ModelCategory = {
  LINEAR: 'linear',
  TREE_BASED: 'tree_based',
  ENSEMBLE: 'ensemble',
  NEURAL_NETWORK: 'neural_network',
  CLUSTERING: 'clustering',
  ANOMALY: 'anomaly',
} as const;

export type ModelCategory = (typeof ModelCategory)[keyof typeof ModelCategory];

/**
 * Hyperparameter data types
 */
export const HyperparamType = {
  INTEGER: 'integer',
  FLOAT: 'float',
  CATEGORICAL: 'categorical',
  BOOLEAN: 'boolean',
} as const;

export type HyperparamType = (typeof HyperparamType)[keyof typeof HyperparamType];

/**
 * Hyperparameter definition
 */
export interface HyperparameterDef {
  name: string;
  displayName: string;
  type: HyperparamType;
  default: any;
  min?: number;
  max?: number;
  step?: number;
  options?: Array<{ value: any; label: string }>;
  description: string;
  required?: boolean;
}

/**
 * Hyperparameter preset
 */
export interface HyperparameterPreset {
  id: string;
  name: string;
  description: string;
  values: Record<string, any>;
  isDefault?: boolean;
}

/**
 * Model definition
 */
export interface ModelDefinition {
  id: string;
  name: string;
  displayName: string;
  description: string;
  category: ModelCategory;
  taskTypes: TaskType[];
  hyperparameters: HyperparameterDef[];
  presets: HyperparameterPreset[];
  pros: string[];
  cons: string[];
  useCases: string[];
  complexity: 'low' | 'medium' | 'high';
  trainingSpeed: 'fast' | 'medium' | 'slow';
  interpretability: 'high' | 'medium' | 'low';
}

/**
 * Selected model configuration
 */
export interface ModelSelection {
  modelId: string;
  taskType: TaskType;
  hyperparameters: Record<string, any>;
  presetId?: string;
}

/**
 * Props for ModelSelector component
 */
export interface ModelSelectorProps {
  taskType: TaskType;
  selectedModel: ModelSelection | null;
  onModelSelect: (selection: ModelSelection) => void;
  disabled?: boolean;
}

/**
 * Props for HyperparameterEditor component
 */
export interface HyperparameterEditorProps {
  model: ModelDefinition;
  values: Record<string, any>;
  onChange: (values: Record<string, any>) => void;
  onPresetSelect?: (presetId: string) => void;
  disabled?: boolean;
}

/**
 * Model registry - all available models
 */
export const MODEL_REGISTRY: Record<string, ModelDefinition> = {
  // ============================================================================
  // REGRESSION MODELS
  // ============================================================================

  linear_regression: {
    id: 'linear_regression',
    name: 'LinearRegression',
    displayName: 'Linear Regression',
    description: 'Simple linear model that fits a line to minimize squared errors',
    category: ModelCategory.LINEAR,
    taskTypes: [TaskType.REGRESSION],
    complexity: 'low',
    trainingSpeed: 'fast',
    interpretability: 'high',
    pros: [
      'Very fast training and prediction',
      'Highly interpretable coefficients',
      'Works well with linearly separable data',
      'No hyperparameters to tune'
    ],
    cons: [
      'Assumes linear relationships',
      'Sensitive to outliers',
      'Cannot capture complex patterns',
      'May underfit on non-linear data'
    ],
    useCases: [
      'Price prediction with linear features',
      'Sales forecasting',
      'Simple trend analysis',
      'Quick baseline model'
    ],
    hyperparameters: [
      {
        name: 'fit_intercept',
        displayName: 'Fit Intercept',
        type: HyperparamType.BOOLEAN,
        default: true,
        description: 'Whether to calculate the intercept for this model',
        required: false,
      },
      {
        name: 'normalize',
        displayName: 'Normalize',
        type: HyperparamType.BOOLEAN,
        default: false,
        description: 'Whether to normalize features before regression',
        required: false,
      },
    ],
    presets: [
      {
        id: 'default',
        name: 'Default',
        description: 'Standard linear regression with intercept',
        values: { fit_intercept: true, normalize: false },
        isDefault: true,
      },
    ],
  },

  ridge: {
    id: 'ridge',
    name: 'Ridge',
    displayName: 'Ridge Regression (L2)',
    description: 'Linear regression with L2 regularization to prevent overfitting',
    category: ModelCategory.LINEAR,
    taskTypes: [TaskType.REGRESSION],
    complexity: 'low',
    trainingSpeed: 'fast',
    interpretability: 'high',
    pros: [
      'Prevents overfitting through regularization',
      'Handles multicollinearity well',
      'Stable with correlated features',
      'Fast training'
    ],
    cons: [
      'Requires tuning alpha parameter',
      'Assumes linear relationships',
      'All features retained (not sparse)',
      'May underperform on very non-linear data'
    ],
    useCases: [
      'Regression with many correlated features',
      'Preventing overfitting in linear models',
      'When feature selection is not needed',
      'Multicollinearity scenarios'
    ],
    hyperparameters: [
      {
        name: 'alpha',
        displayName: 'Alpha (Regularization Strength)',
        type: HyperparamType.FLOAT,
        default: 1.0,
        min: 0.0001,
        max: 100,
        step: 0.1,
        description: 'Regularization strength. Higher values = more regularization',
        required: true,
      },
      {
        name: 'fit_intercept',
        displayName: 'Fit Intercept',
        type: HyperparamType.BOOLEAN,
        default: true,
        description: 'Whether to calculate the intercept',
        required: false,
      },
    ],
    presets: [
      {
        id: 'weak_regularization',
        name: 'Weak Regularization',
        description: 'Low regularization (alpha=0.1)',
        values: { alpha: 0.1, fit_intercept: true },
      },
      {
        id: 'default',
        name: 'Default',
        description: 'Moderate regularization (alpha=1.0)',
        values: { alpha: 1.0, fit_intercept: true },
        isDefault: true,
      },
      {
        id: 'strong_regularization',
        name: 'Strong Regularization',
        description: 'High regularization (alpha=10.0)',
        values: { alpha: 10.0, fit_intercept: true },
      },
    ],
  },

  random_forest_regressor: {
    id: 'random_forest_regressor',
    name: 'RandomForestRegressor',
    displayName: 'Random Forest Regressor',
    description: 'Ensemble of decision trees for robust regression',
    category: ModelCategory.ENSEMBLE,
    taskTypes: [TaskType.REGRESSION],
    complexity: 'medium',
    trainingSpeed: 'medium',
    interpretability: 'medium',
    pros: [
      'Handles non-linear relationships well',
      'Robust to outliers',
      'Feature importance available',
      'Requires little preprocessing'
    ],
    cons: [
      'Can be slow with many trees',
      'Less interpretable than linear models',
      'May overfit on noisy data',
      'Large model size'
    ],
    useCases: [
      'Complex non-linear regression',
      'Feature importance analysis',
      'Robust prediction with mixed data types',
      'When interpretability is moderate priority'
    ],
    hyperparameters: [
      {
        name: 'n_estimators',
        displayName: 'Number of Trees',
        type: HyperparamType.INTEGER,
        default: 100,
        min: 10,
        max: 500,
        step: 10,
        description: 'Number of trees in the forest',
        required: true,
      },
      {
        name: 'max_depth',
        displayName: 'Max Depth',
        type: HyperparamType.INTEGER,
        default: 10,
        min: 2,
        max: 50,
        step: 1,
        description: 'Maximum depth of each tree (None for unlimited)',
        required: false,
      },
      {
        name: 'min_samples_split',
        displayName: 'Min Samples Split',
        type: HyperparamType.INTEGER,
        default: 2,
        min: 2,
        max: 20,
        step: 1,
        description: 'Minimum samples required to split a node',
        required: false,
      },
    ],
    presets: [
      {
        id: 'fast',
        name: 'Fast Training',
        description: 'Fewer trees for quick training',
        values: { n_estimators: 50, max_depth: 10, min_samples_split: 2 },
      },
      {
        id: 'default',
        name: 'Balanced',
        description: 'Good balance of speed and accuracy',
        values: { n_estimators: 100, max_depth: 10, min_samples_split: 2 },
        isDefault: true,
      },
      {
        id: 'accurate',
        name: 'High Accuracy',
        description: 'More trees for better accuracy',
        values: { n_estimators: 200, max_depth: 15, min_samples_split: 2 },
      },
    ],
  },

  // ============================================================================
  // CLASSIFICATION MODELS
  // ============================================================================

  logistic_regression: {
    id: 'logistic_regression',
    name: 'LogisticRegression',
    displayName: 'Logistic Regression',
    description: 'Linear model for binary and multi-class classification',
    category: ModelCategory.LINEAR,
    taskTypes: [TaskType.CLASSIFICATION],
    complexity: 'low',
    trainingSpeed: 'fast',
    interpretability: 'high',
    pros: [
      'Fast training and prediction',
      'Probabilistic outputs',
      'Highly interpretable',
      'Works well with linearly separable classes'
    ],
    cons: [
      'Assumes linear decision boundary',
      'May underfit complex data',
      'Sensitive to feature scaling',
      'Requires feature engineering for non-linear patterns'
    ],
    useCases: [
      'Binary classification',
      'Spam detection',
      'Disease prediction',
      'Customer churn prediction'
    ],
    hyperparameters: [
      {
        name: 'C',
        displayName: 'Inverse Regularization Strength',
        type: HyperparamType.FLOAT,
        default: 1.0,
        min: 0.001,
        max: 100,
        step: 0.1,
        description: 'Inverse of regularization strength. Smaller values = stronger regularization',
        required: true,
      },
      {
        name: 'max_iter',
        displayName: 'Max Iterations',
        type: HyperparamType.INTEGER,
        default: 100,
        min: 50,
        max: 1000,
        step: 50,
        description: 'Maximum number of iterations for solver convergence',
        required: false,
      },
    ],
    presets: [
      {
        id: 'strong_regularization',
        name: 'Strong Regularization',
        description: 'Prevent overfitting (C=0.1)',
        values: { C: 0.1, max_iter: 100 },
      },
      {
        id: 'default',
        name: 'Balanced',
        description: 'Standard regularization (C=1.0)',
        values: { C: 1.0, max_iter: 100 },
        isDefault: true,
      },
      {
        id: 'weak_regularization',
        name: 'Weak Regularization',
        description: 'More flexible model (C=10.0)',
        values: { C: 10.0, max_iter: 200 },
      },
    ],
  },

  random_forest_classifier: {
    id: 'random_forest_classifier',
    name: 'RandomForestClassifier',
    displayName: 'Random Forest Classifier',
    description: 'Ensemble of decision trees for classification',
    category: ModelCategory.ENSEMBLE,
    taskTypes: [TaskType.CLASSIFICATION],
    complexity: 'medium',
    trainingSpeed: 'medium',
    interpretability: 'medium',
    pros: [
      'Handles non-linear decision boundaries',
      'Feature importance available',
      'Robust to outliers',
      'Works well with imbalanced data'
    ],
    cons: [
      'Can overfit on noisy data',
      'Less interpretable than simpler models',
      'Slower training than linear models',
      'Large memory footprint'
    ],
    useCases: [
      'Multi-class classification',
      'Feature selection via importance',
      'Imbalanced datasets',
      'Mixed data types'
    ],
    hyperparameters: [
      {
        name: 'n_estimators',
        displayName: 'Number of Trees',
        type: HyperparamType.INTEGER,
        default: 100,
        min: 10,
        max: 500,
        step: 10,
        description: 'Number of trees in the forest',
        required: true,
      },
      {
        name: 'max_depth',
        displayName: 'Max Depth',
        type: HyperparamType.INTEGER,
        default: 10,
        min: 2,
        max: 50,
        step: 1,
        description: 'Maximum depth of trees',
        required: false,
      },
      {
        name: 'class_weight',
        displayName: 'Class Weight',
        type: HyperparamType.CATEGORICAL,
        default: 'balanced',
        options: [
          { value: 'balanced', label: 'Balanced (auto-weight)' },
          { value: null, label: 'None (equal weight)' },
        ],
        description: 'Weights for class imbalance',
        required: false,
      },
    ],
    presets: [
      {
        id: 'default',
        name: 'Balanced',
        description: 'Standard configuration',
        values: { n_estimators: 100, max_depth: 10, class_weight: 'balanced' },
        isDefault: true,
      },
      {
        id: 'deep',
        name: 'Deep Trees',
        description: 'Capture complex patterns',
        values: { n_estimators: 150, max_depth: 20, class_weight: 'balanced' },
      },
    ],
  },

  // ============================================================================
  // CLUSTERING MODELS
  // ============================================================================

  kmeans: {
    id: 'kmeans',
    name: 'KMeans',
    displayName: 'K-Means Clustering',
    description: 'Partitions data into K clusters based on distance',
    category: ModelCategory.CLUSTERING,
    taskTypes: [TaskType.CLUSTERING],
    complexity: 'low',
    trainingSpeed: 'fast',
    interpretability: 'high',
    pros: [
      'Simple and fast',
      'Scales well to large datasets',
      'Works well with spherical clusters',
      'Easy to interpret'
    ],
    cons: [
      'Requires specifying K beforehand',
      'Sensitive to initialization',
      'Assumes spherical clusters',
      'Sensitive to outliers'
    ],
    useCases: [
      'Customer segmentation',
      'Image compression',
      'Document clustering',
      'Anomaly detection preprocessing'
    ],
    hyperparameters: [
      {
        name: 'n_clusters',
        displayName: 'Number of Clusters',
        type: HyperparamType.INTEGER,
        default: 3,
        min: 2,
        max: 20,
        step: 1,
        description: 'Number of clusters to form',
        required: true,
      },
      {
        name: 'init',
        displayName: 'Initialization Method',
        type: HyperparamType.CATEGORICAL,
        default: 'k-means++',
        options: [
          { value: 'k-means++', label: 'K-Means++ (smart)' },
          { value: 'random', label: 'Random' },
        ],
        description: 'Method for initialization',
        required: false,
      },
      {
        name: 'max_iter',
        displayName: 'Max Iterations',
        type: HyperparamType.INTEGER,
        default: 300,
        min: 50,
        max: 1000,
        step: 50,
        description: 'Maximum number of iterations',
        required: false,
      },
    ],
    presets: [
      {
        id: 'small',
        name: 'Small Clusters (K=3)',
        description: 'Few distinct groups',
        values: { n_clusters: 3, init: 'k-means++', max_iter: 300 },
        isDefault: true,
      },
      {
        id: 'medium',
        name: 'Medium Clusters (K=5)',
        description: 'Moderate number of groups',
        values: { n_clusters: 5, init: 'k-means++', max_iter: 300 },
      },
      {
        id: 'large',
        name: 'Large Clusters (K=10)',
        description: 'Many fine-grained groups',
        values: { n_clusters: 10, init: 'k-means++', max_iter: 300 },
      },
    ],
  },
};

/**
 * Get models filtered by task type
 */
export function getModelsByTaskType(taskType: TaskType): ModelDefinition[] {
  return Object.values(MODEL_REGISTRY).filter((model) =>
    model.taskTypes.includes(taskType)
  );
}

/**
 * Get models grouped by category
 */
export function getModelsGroupedByCategory(
  taskType: TaskType
): Record<ModelCategory, ModelDefinition[]> {
  const models = getModelsByTaskType(taskType);
  const grouped: Partial<Record<ModelCategory, ModelDefinition[]>> = {};

  models.forEach((model) => {
    if (!grouped[model.category]) {
      grouped[model.category] = [];
    }
    grouped[model.category]!.push(model);
  });

  return grouped as Record<ModelCategory, ModelDefinition[]>;
}

/**
 * Get category display name
 */
export function getCategoryDisplayName(category: ModelCategory): string {
  const names: Record<ModelCategory, string> = {
    linear: 'Linear Models',
    tree_based: 'Tree-Based Models',
    ensemble: 'Ensemble Models',
    neural_network: 'Neural Networks',
    clustering: 'Clustering Algorithms',
    anomaly: 'Anomaly Detection',
  };
  return names[category] || category;
}

/**
 * Create default model selection
 */
export function createDefaultModelSelection(
  modelId: string,
  taskType: TaskType
): ModelSelection {
  const model = MODEL_REGISTRY[modelId];
  if (!model) {
    throw new Error(`Model ${modelId} not found`);
  }

  const defaultPreset = model.presets.find((p) => p.isDefault) || model.presets[0];

  return {
    modelId,
    taskType,
    hyperparameters: defaultPreset ? { ...defaultPreset.values } : {},
    presetId: defaultPreset?.id,
  };
}
