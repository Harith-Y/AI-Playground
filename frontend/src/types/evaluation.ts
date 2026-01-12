/**
 * Evaluation Types
 *
 * Types for model evaluation, metrics, and visualization
 */

/**
 * Task types for evaluation
 */
export const TaskType = {
  REGRESSION: 'regression',
  CLASSIFICATION: 'classification',
  CLUSTERING: 'clustering',
} as const;

export type TaskType = (typeof TaskType)[keyof typeof TaskType];

/**
 * Evaluation tab identifiers
 */
export const EvaluationTab = {
  CLASSIFICATION: 'classification',
  REGRESSION: 'regression',
  CLUSTERING: 'clustering',
  COMPARISON: 'comparison',
} as const;

export type EvaluationTab = (typeof EvaluationTab)[keyof typeof EvaluationTab];

/**
 * Model run status
 */
export const RunStatus = {
  PENDING: 'pending',
  RUNNING: 'running',
  COMPLETED: 'completed',
  FAILED: 'failed',
  CANCELLED: 'cancelled',
} as const;

export type RunStatus = (typeof RunStatus)[keyof typeof RunStatus];

/**
 * Model run information
 */
export interface ModelRun {
  id: string;
  name: string;
  modelId: string;
  modelName: string;
  modelType?: string; // Type of model (e.g., 'RandomForest', 'XGBoost')
  taskType: TaskType;
  datasetId: string;
  datasetName: string;
  status: RunStatus;
  createdAt: string;
  completedAt?: string;
  hyperparameters: Record<string, any>;
  metrics?: ModelMetrics;
  insights?: ModelInsightsData;
}

/**
 * Classification metrics
 */
export interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc?: number;
  confusionMatrix?: number[][];
  classNames?: string[];
  rocCurve?: {
    fpr: number[];
    tpr: number[];
    thresholds: number[];
  };
  prCurve?: {
    precision: number[];
    recall: number[];
    thresholds: number[];
  };
}

/**
 * Regression metrics
 */
export interface RegressionMetrics {
  mae: number;
  mse: number;
  rmse: number;
  r2_score: number;
  mape?: number;
  predicted?: number[];
  actual?: number[];
  residuals?: {
    predicted: number[];
    actual: number[];
    residual: number[];
  };
  actualVsPredicted?: {
    actual: number[];
    predicted: number[];
  };
  residual_plot?: {
    x: number[];
    y: number[];
  };
  prediction_error_plot?: {
    x: number[];
    y: number[];
  };
}

/**
 * Clustering metrics
 */
export interface ClusteringMetrics {
  silhouetteScore: number;
  inertia?: number;
  daviesBouldinScore?: number;
  calinskiHarabaszScore?: number;
  nClusters: number;
  clusterSizes?: number[];
  clusterCenters?: number[][];
}

/**
 * Combined metrics type
 */
export type ModelMetrics =
  | ClassificationMetrics
  | RegressionMetrics
  | ClusteringMetrics;

/**
 * Feature importance entry
 */
export interface FeatureImportance {
  feature: string;
  importance: number;
  rank?: number;
  stdDev?: number;
}

/**
 * Permutation importance entry
 */
export interface PermutationImportance {
  feature: string;
  importance: number;
  importances?: number[];
  mean?: number;
  std?: number;
  rank?: number;
}

/**
 * Feature ranking with multiple metrics
 */
export interface FeatureRanking {
  feature: string;
  importance: number;
  permutationImportance?: number;
  rank?: number;
  permutationRank?: number;
  correlation?: number;
  type?: string;
  stdDev?: number;
}

/**
 * Model insights data
 */
export interface ModelInsightsData {
  featureImportance?: FeatureImportance[];
  permutationImportance?: PermutationImportance[];
  featureRankings?: FeatureRanking[];
  importanceType?: 'gain' | 'split' | 'weight' | 'cover';
  metric?: string;
}

/**
 * Evaluation data for a model run
 */
export interface EvaluationData {
  run: ModelRun;
  metrics: ModelMetrics;
  featureImportance?: FeatureImportance[];
  plots?: {
    [key: string]: any;
  };
}

/**
 * Comparison data for multiple runs
 */
export interface ComparisonData {
  runs: ModelRun[];
  metricsComparison: {
    runId: string;
    runName: string;
    metrics: Record<string, number>;
  }[];
  bestRun?: string;
}

/**
 * Evaluation filter options
 */
export interface EvaluationFilters {
  taskType?: TaskType;
  datasetId?: string;
  modelId?: string;
  status?: RunStatus;
  dateFrom?: string;
  dateTo?: string;
}

/**
 * Props for EvaluationPage component
 */
export interface EvaluationPageProps {
  initialTab?: EvaluationTab;
  runId?: string;
}

/**
 * Props for task-specific evaluation tabs
 */
export interface TaskEvaluationProps {
  runs: ModelRun[];
  selectedRunId?: string;
  onRunSelect: (runId: string) => void;
  isLoading?: boolean;
  error?: string;
}

/**
 * Props for comparison view
 */
export interface ComparisonViewProps {
  runs: ModelRun[];
  selectedRunIds: string[];
  onRunsSelect: (runIds: string[]) => void;
  isLoading?: boolean;
  error?: string;
}
