export interface Model {
  id: string;
  name: string;
  type: string;
  status: 'training' | 'completed' | 'failed';
  accuracy?: number;
  createdAt: string;
  updatedAt: string;
  hyperparameters: Hyperparameters;
  metrics?: ModelMetrics;
}

export interface Hyperparameters {
  [key: string]: any;
}

export interface TrainingMetrics {
  epoch?: number;
  loss?: number;
  accuracy?: number;
  valLoss?: number;
  valAccuracy?: number;
  [key: string]: any;
}

export interface ModelMetrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1Score?: number;
  rmse?: number;
  mae?: number;
  r2Score?: number;
  [key: string]: any;
}
