export interface TrainingRun {
  id: string;
  modelId: string;
  modelName: string;
  modelType: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number; // 0-100
  currentEpoch?: number;
  totalEpochs?: number;
  startTime: string;
  endTime?: string;
  duration?: number; // in seconds
  metrics: TrainingMetrics;
  logs: TrainingLog[];
  error?: string;
}

export interface TrainingMetrics {
  // Training metrics
  trainLoss?: number;
  trainAccuracy?: number;
  trainScore?: number;
  
  // Validation metrics
  valLoss?: number;
  valAccuracy?: number;
  valScore?: number;
  
  // Test metrics
  testLoss?: number;
  testAccuracy?: number;
  testScore?: number;
  
  // Additional metrics
  precision?: number;
  recall?: number;
  f1Score?: number;
  rmse?: number;
  mae?: number;
  r2Score?: number;
  
  // History for plotting
  history?: {
    epoch: number[];
    trainLoss: number[];
    valLoss?: number[];
    trainAccuracy?: number[];
    valAccuracy?: number[];
    [key: string]: number[] | undefined;
  };
  
  [key: string]: any;
}

export interface TrainingLog {
  id: string;
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'debug';
  message: string;
  details?: any;
}

export interface TrainingConfig {
  datasetId: string;
  modelType: string;
  hyperparameters: Record<string, any>;
  trainTestSplit?: number;
  validationSplit?: number;
  epochs?: number;
  batchSize?: number;
  earlyStoppingPatience?: number;
}

export interface TrainingProgressUpdate {
  runId: string;
  status: TrainingRun['status'];
  progress: number;
  currentEpoch?: number;
  metrics?: Partial<TrainingMetrics>;
  log?: TrainingLog;
}
