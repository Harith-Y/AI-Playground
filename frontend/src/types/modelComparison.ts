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
