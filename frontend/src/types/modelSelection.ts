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

  lasso: {
    id: 'lasso',
    name: 'Lasso',
    displayName: 'Lasso Regression (L1)',
    description: 'Linear regression with L1 regularization for feature selection',
    category: ModelCategory.LINEAR,
    taskTypes: [TaskType.REGRESSION],
    complexity: 'low',
    trainingSpeed: 'fast',
    interpretability: 'high',
    pros: [
      'Performs automatic feature selection',
      'Produces sparse models',
      'Good for high-dimensional data',
      'Prevents overfitting'
    ],
    cons: [
      'Cannot learn complex patterns',
      'May exclude useful correlated features',
      'Sensitive to outliers',
      'Requires feature scaling'
    ],
    useCases: [
      'Feature selection',
      'Sparse modeling',
      'High-dimensional datasets',
      'Baseline regression'
    ],
    hyperparameters: [
      {
        name: 'alpha',
        displayName: 'Alpha',
        type: HyperparamType.FLOAT,
        default: 1.0,
        min: 0.0001,
        max: 100,
        step: 0.1,
        description: 'Constant that multiplies the L1 term',
        required: true,
      },
    ],
    presets: [
      {
        id: 'default',
        name: 'Default',
        description: 'Standard Lasso',
        values: { alpha: 1.0 },
        isDefault: true,
      },
    ],
  },

  decision_tree_regressor: {
    id: 'decision_tree_regressor',
    name: 'DecisionTreeRegressor',
    displayName: 'Decision Tree Regressor',
    description: 'Tree-based model that learns simple decision rules',
    category: ModelCategory.TREE_BASED,
    taskTypes: [TaskType.REGRESSION],
    complexity: 'medium',
    trainingSpeed: 'fast',
    interpretability: 'high',
    pros: [
      'Easy to visualize and interpret',
      'Handles non-linear data',
      'No scaling required',
      'Fast training'
    ],
    cons: [
      'Prone to overfitting',
      'Instable (high variance)',
      'Cannot extrapolate',
      'Biased to dominant classes'
    ],
    useCases: [
      'Non-linear regression',
      'When interpretation is key',
      'Baseline for ensembles',
      'Mixed data types'
    ],
    hyperparameters: [
      {
        name: 'max_depth',
        displayName: 'Max Depth',
        type: HyperparamType.INTEGER,
        default: 5,
        min: 1,
        max: 50,
        step: 1,
        description: 'Maximum depth of the tree',
        required: false,
      },
    ],
    presets: [
      {
        id: 'default',
        name: 'Default',
        description: 'Standard Tree',
        values: { max_depth: 5 },
        isDefault: true,
      },
    ],
  },

  svr: {
    id: 'svr',
    name: 'SVR',
    displayName: 'Support Vector Regression',
    description: 'Epsilon-Support Vector Regression',
    category: ModelCategory.SUPPORT_VECTOR,
    taskTypes: [TaskType.REGRESSION],
    complexity: 'high',
    trainingSpeed: 'medium',
    interpretability: 'low',
    pros: [
      'Effective in high dimensions',
      'Robust to outliers (with correct kernel)',
      'Versatile kernels',
      'Memory efficient'
    ],
    cons: [
      'Hard to tune',
      'No probabilistic interpretation',
      'Sensitive to scaling',
      'Slow on large datasets'
    ],
    useCases: [
      'High-dimensional regression',
      'Financial prediction',
      'Complex non-linear bounds',
      'Small to medium datasets'
    ],
    hyperparameters: [
      {
        name: 'C',
        displayName: 'Regularization (C)',
        type: HyperparamType.FLOAT,
        default: 1.0,
        min: 0.01,
        max: 100,
        step: 0.1,
        description: 'Regularization parameter',
        required: true,
      },
      {
        name: 'kernel',
        displayName: 'Kernel',
        type: HyperparamType.CATEGORICAL,
        default: 'rbf',
        options: [
          { value: 'linear', label: 'Linear' },
          { value: 'rbf', label: 'RBF' },
          { value: 'poly', label: 'Polynomial' },
        ],
        description: 'Kernel type',
        required: true,
      },
    ],
    presets: [
      {
        id: 'default',
        name: 'Default',
        description: 'Standard SVR',
        values: { C: 1.0, kernel: 'rbf' },
        isDefault: true,
      },
    ],
  },
  
  knn_regressor: {
    id: 'knn_regressor',
    name: 'KNNRegressor',
    displayName: 'K-Nearest Neighbors Regressor',
    description: 'Regression based on k-nearest neighbors',
    category: ModelCategory.INSTANCE_BASED,
    taskTypes: [TaskType.REGRESSION],
    complexity: 'low',
    trainingSpeed: 'fast',
    interpretability: 'medium',
    pros: [
      'Simple and intuitive',
      'No training phase (lazy)',
      'Adapts to local structure',
      'Non-parametric'
    ],
    cons: [
      'Slow prediction on large data',
      'Sensitive to noise',
      'Requires scaling',
      'Curse of dimensionality'
    ],
    useCases: [
      'Spatial regression',
      'Recommendation systems',
      'Baseline comparison',
      'Small datasets'
    ],
    hyperparameters: [
      {
        name: 'n_neighbors',
        displayName: 'Neighbors (K)',
        type: HyperparamType.INTEGER,
        default: 5,
        min: 1,
        max: 50,
        step: 1,
        description: 'Number of neighbors',
        required: true,
      },
    ],
    presets: [
      {
        id: 'default',
        name: 'Default',
        description: '5 Neighbors',
        values: { n_neighbors: 5 },
        isDefault: true,
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

  decision_tree_classifier: {
    id: 'decision_tree_classifier',
    name: 'DecisionTreeClassifier',
    displayName: 'Decision Tree Classifier',
    description: 'Tree-based classifier using simple decision rules',
    category: ModelCategory.TREE_BASED,
    taskTypes: [TaskType.CLASSIFICATION],
    complexity: 'medium',
    trainingSpeed: 'fast',
    interpretability: 'high',
    pros: [
      'Easy to interpret/visualize',
      'Non-parametric',
      'No scaling needed',
      'Handles multi-output'
    ],
    cons: [
      'Overfitting',
      'Sensitive to small data changes',
      'Biased to dominant classes',
      'Not smooth boundaries'
    ],
    useCases: [
      'Medical diagnosis',
      'Credit scoring',
      'Customer churn',
      'Rule generation'
    ],
    hyperparameters: [
      {
        name: 'max_depth',
        displayName: 'Max Depth',
        type: HyperparamType.INTEGER,
        default: 5,
        min: 1,
        max: 50,
        step: 1,
        description: 'Global tree depth limit',
        required: false,
      },
      {
        name: 'criterion',
        displayName: 'Criterion',
        type: HyperparamType.CATEGORICAL,
        default: 'gini',
        options: [
            { value: 'gini', label: 'Gini Impurity' },
            { value: 'entropy', label: 'Entropy' }
        ],
        description: 'Split quality measure',
        required: true,
      },
    ],
    presets: [
      {
        id: 'default',
        name: 'Default',
        description: 'Standard Tree',
        values: { max_depth: 5, criterion: 'gini' },
        isDefault: true,
      },
    ],
  },

  svm: {
    id: 'svm',
    name: 'SVC',
    displayName: 'Support Vector Machine (SVC)',
    description: 'C-Support Vector Classification',
    category: ModelCategory.SUPPORT_VECTOR,
    taskTypes: [TaskType.CLASSIFICATION],
    complexity: 'high',
    trainingSpeed: 'medium',
    interpretability: 'low',
    pros: [
      'Effective in high dimensions',
      'Versatile kernels',
      'Memory efficient',
      'Robust'
    ],
    cons: [
      'Slow on large data',
      'Noise sensitive',
      'Probability estimates are slow',
      'Hard parameter tuning'
    ],
    useCases: [
      'Image classification',
      'Text categorization',
      'Bioinformatics',
      'Handwriting recognition'
    ],
    hyperparameters: [
      {
        name: 'C',
        displayName: 'Regularization (C)',
        type: HyperparamType.FLOAT,
        default: 1.0,
        min: 0.01,
        max: 100,
        step: 0.1,
        description: 'Regularization strength',
        required: true,
      },
      {
        name: 'kernel',
        displayName: 'Kernel',
        type: HyperparamType.CATEGORICAL,
        default: 'rbf',
        options: [
            { value: 'linear', label: 'Linear' },
            { value: 'rbf', label: 'RBF' },
            { value: 'poly', label: 'Polynomial' }
        ],
        description: 'Kernel type',
        required: true,
      },
    ],
    presets: [
      {
        id: 'default',
        name: 'Default',
        description: 'Standard SVM',
        values: { C: 1.0, kernel: 'rbf' },
        isDefault: true,
      },
    ],
  },

  knn_classifier: {
    id: 'knn_classifier',
    name: 'KNNClassifier',
    displayName: 'K-Nearest Neighbors Classifier',
    description: 'Classifier implementing the k-nearest neighbors vote',
    category: ModelCategory.INSTANCE_BASED,
    taskTypes: [TaskType.CLASSIFICATION],
    complexity: 'low',
    trainingSpeed: 'fast',
    interpretability: 'medium',
    pros: [
      'Simple to understand',
      'No assumption about data',
      'Versatile (binary/multiclass)',
      'Constantly evolving (lazy)'
    ],
    cons: [
      'Computationally expensive prediction',
      'High memory usage',
      'Sensitive to unrelated features',
      'Needs scaling'
    ],
    useCases: [
      'Pattern recognition',
      'Data mining',
      'Intrusion detection',
      'Recommendation'
    ],
    hyperparameters: [
      {
        name: 'n_neighbors',
        displayName: 'Neighbors (K)',
        type: HyperparamType.INTEGER,
        default: 5,
        min: 1,
        max: 50,
        step: 1,
        description: 'Number of neighbors',
        required: true,
      },
    ],
    presets: [
      {
        id: 'default',
        name: 'Default',
        description: '5 Neighbors',
        values: { n_neighbors: 5 },
        isDefault: true,
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
