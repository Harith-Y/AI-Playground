/**
 * Model Comparison Types
 * 
 * TypeScript interfaces for model comparison feature
 * Matches backend API schemas from app/schemas/model.py
 */

// ============================================================================
// Request Schemas
// ============================================================================

export interface CompareModelsRequest {
  model_run_ids: string[]; // 2-10 UUIDs
  comparison_metrics?: string[]; // Auto-detected if null
  ranking_criteria?: string; // Primary metric for ranking
  include_statistical_tests?: boolean;
}

export interface ModelRankingRequest {
  model_run_ids: string[]; // 2-20 UUIDs
  ranking_weights: Record<string, number>; // Must sum to 1.0
  higher_is_better?: Record<string, boolean>;
}

// ============================================================================
// Response Schemas
// ============================================================================

export interface ModelComparisonItem {
  model_run_id: string;
  model_type: string;
  experiment_id: string;
  status: string;
  metrics: Record<string, any>;
  hyperparameters: Record<string, any>;
  training_time?: number;
  created_at: string;
  rank?: number;
  ranking_score?: number;
}

export interface MetricStatistics {
  metric_name: string;
  mean: number;
  std: number;
  min: number;
  max: number;
  best_model_id: string;
  worst_model_id: string;
}

export interface ModelComparisonResponse {
  comparison_id: string;
  task_type: string;
  total_models: number;
  compared_models: ModelComparisonItem[];
  best_model: ModelComparisonItem;
  metric_statistics: MetricStatistics[];
  ranking_criteria: string;
  recommendations: string[];
  timestamp: string;
}

export interface RankedModel {
  model_run_id: string;
  model_type: string;
  rank: number;
  composite_score: number;
  individual_scores: Record<string, number>;
  weighted_contributions: Record<string, number>;
}

export interface ModelRankingResponse {
  ranking_id: string;
  ranked_models: RankedModel[];
  ranking_weights: Record<string, number>;
  best_model: {
    model_run_id: string;
    rank: number;
    composite_score: number;
  };
  score_range: {
    min: number;
    max: number;
    spread: number;
  };
  timestamp: string;
}

// ============================================================================
// Legacy Types (for backward compatibility with existing components)
// ============================================================================

export interface ModelComparisonData {
  id: string;
  name: string;
  type: string;
  status: 'training' | 'completed' | 'failed';
  createdAt: string;
  trainingTime?: number; // in seconds
  
  // Model configuration
  hyperparameters: Record<string, any>;
  datasetId: string;
  datasetName: string;
  
  // Performance metrics
  metrics: {
    // Classification metrics
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1Score?: number;
    rocAuc?: number;
    
    // Regression metrics
    rmse?: number;
    mae?: number;
    r2Score?: number;
    mse?: number;
    
    // Cross-validation scores
    cvMean?: number;
    cvStd?: number;
    
    // Training metrics
    trainScore?: number;
    valScore?: number;
    testScore?: number;
    
    // Additional metrics
    [key: string]: number | undefined;
  };
  
  // Feature importance (if available)
  featureImportance?: {
    feature: string;
    importance: number;
  }[];
  
  // Training history (if available)
  trainingHistory?: {
    epoch: number[];
    trainLoss: number[];
    valLoss?: number[];
    trainAccuracy?: number[];
    valAccuracy?: number[];
  };
  
  // Model size and complexity
  modelSize?: {
    parameters?: number;
    memoryUsage?: number; // in MB
    inferenceTime?: number; // in ms
  };
  
  // Confusion matrix (for classification)
  confusionMatrix?: number[][];
  classNames?: string[];
}

export interface ModelComparisonFilter {
  modelType?: string;
  status?: string;
  datasetId?: string;
  dateRange?: {
    start: string;
    end: string;
  };
  minAccuracy?: number;
  maxTrainingTime?: number;
}

export interface ModelComparisonSort {
  field: keyof ModelComparisonData | 'metrics.accuracy' | 'metrics.f1Score' | 'metrics.rmse';
  direction: 'asc' | 'desc';
}

export interface ModelComparisonState {
  selectedModels: string[]; // Model IDs
  availableModels: ModelComparisonData[];
  loading: boolean;
  error: string | null;
  filters: ModelComparisonFilter;
  sort: ModelComparisonSort;
}

export interface ComparisonChart {
  type: 'bar' | 'radar' | 'line' | 'scatter';
  title: string;
  data: any;
  layout: any;
}
