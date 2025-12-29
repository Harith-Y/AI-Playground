import api from './api';
import type { TrainingRun, TrainingConfig } from '../types/training';

export interface AvailableModel {
  id: string;
  name: string;
  displayName: string;
  category: 'classification' | 'regression' | 'clustering';
  description: string;
  hyperparameters: {
    name: string;
    type: 'number' | 'string' | 'boolean' | 'select';
    default: any;
    options?: any[];
    min?: number;
    max?: number;
    description?: string;
  }[];
  pros: string[];
  cons: string[];
  complexity: 'low' | 'medium' | 'high';
  trainingSpeed: 'fast' | 'medium' | 'slow';
  interpretability: 'high' | 'medium' | 'low';
}

export interface TrainingRunListItem {
  id: string;
  modelType: string;
  modelName: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  datasetId: string;
  datasetName: string;
  startTime: string;
  endTime?: string;
  metrics?: Record<string, number>;
  error?: string;
}

export interface TrainingRunsResponse {
  runs: TrainingRunListItem[];
  total: number;
  page: number;
  pageSize: number;
}

export interface TrainingResultsResponse {
  runId: string;
  modelType: string;
  status: string;
  metrics: Record<string, number>;
  predictions?: any[];
  featureImportance?: {
    feature: string;
    importance: number;
  }[];
  confusionMatrix?: number[][];
  classNames?: string[];
  modelPath?: string;
  artifactsPath?: string;
}

export interface ModelStatusResponse {
  runId: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  currentEpoch?: number;
  totalEpochs?: number;
  metrics?: Record<string, number>;
  error?: string;
  startTime: string;
  endTime?: string;
  estimatedTimeRemaining?: number;
}

const MODELING_BASE_URL = '/api/models';

export const modelingService = {
  /**
   * Get list of available models
   */
  async getAvailableModels(params?: {
    category?: 'classification' | 'regression' | 'clustering';
    taskType?: string;
  }): Promise<{ models: AvailableModel[] }> {
    const response = await api.get(`${MODELING_BASE_URL}/available`, { params });
    return response.data;
  },

  /**
   * Start a new training run
   */
  async startTraining(config: TrainingConfig): Promise<{ runId: string; message: string }> {
    const response = await api.post(`${MODELING_BASE_URL}/train`, config);
    return response.data;
  },

  /**
   * Get training run status
   */
  async getTrainingStatus(runId: string): Promise<ModelStatusResponse> {
    const response = await api.get(`${MODELING_BASE_URL}/status/${runId}`);
    return response.data;
  },

  /**
   * Get training run results
   */
  async getTrainingResults(runId: string): Promise<TrainingResultsResponse> {
    const response = await api.get(`${MODELING_BASE_URL}/results/${runId}`);
    return response.data;
  },

  /**
   * Get all training runs with pagination and filtering
   */
  async getTrainingRuns(params?: {
    page?: number;
    pageSize?: number;
    status?: string;
    modelType?: string;
    datasetId?: string;
    sortBy?: string;
    sortOrder?: 'asc' | 'desc';
  }): Promise<TrainingRunsResponse> {
    const response = await api.get(`${MODELING_BASE_URL}/runs`, { params });
    return response.data;
  },

  /**
   * Get a specific training run by ID
   */
  async getTrainingRun(runId: string): Promise<TrainingRun> {
    const response = await api.get(`${MODELING_BASE_URL}/runs/${runId}`);
    return response.data;
  },

  /**
   * Delete a training run
   */
  async deleteTrainingRun(runId: string): Promise<{ success: boolean; message: string }> {
    const response = await api.delete(`${MODELING_BASE_URL}/runs/${runId}`);
    return response.data;
  },

  /**
   * Delete multiple training runs
   */
  async deleteTrainingRuns(runIds: string[]): Promise<{ 
    success: boolean; 
    deleted: number;
    failed: number;
    message: string;
  }> {
    const response = await api.post(`${MODELING_BASE_URL}/runs/delete-batch`, { runIds });
    return response.data;
  },

  /**
   * Cancel a running training
   */
  async cancelTraining(runId: string): Promise<{ success: boolean; message: string }> {
    const response = await api.post(`${MODELING_BASE_URL}/runs/${runId}/cancel`);
    return response.data;
  },

  /**
   * Retry a failed training run
   */
  async retryTraining(runId: string): Promise<{ runId: string; message: string }> {
    const response = await api.post(`${MODELING_BASE_URL}/runs/${runId}/retry`);
    return response.data;
  },

  /**
   * Get model hyperparameters schema
   */
  async getModelHyperparameters(modelType: string): Promise<{
    modelType: string;
    hyperparameters: AvailableModel['hyperparameters'];
  }> {
    const response = await api.get(`${MODELING_BASE_URL}/${modelType}/hyperparameters`);
    return response.data;
  },

  /**
   * Validate training configuration
   */
  async validateTrainingConfig(config: TrainingConfig): Promise<{
    valid: boolean;
    errors?: string[];
    warnings?: string[];
  }> {
    const response = await api.post(`${MODELING_BASE_URL}/validate`, config);
    return response.data;
  },

  /**
   * Get training metrics history
   */
  async getTrainingMetrics(runId: string): Promise<{
    runId: string;
    metrics: {
      epoch: number[];
      trainLoss: number[];
      valLoss?: number[];
      trainAccuracy?: number[];
      valAccuracy?: number[];
      [key: string]: number[] | undefined;
    };
  }> {
    const response = await api.get(`${MODELING_BASE_URL}/runs/${runId}/metrics`);
    return response.data;
  },

  /**
   * Get training logs
   */
  async getTrainingLogs(runId: string, params?: {
    level?: 'debug' | 'info' | 'warning' | 'error';
    limit?: number;
    offset?: number;
  }): Promise<{
    runId: string;
    logs: {
      id: string;
      timestamp: string;
      level: string;
      message: string;
    }[];
    total: number;
  }> {
    const response = await api.get(`${MODELING_BASE_URL}/runs/${runId}/logs`, { params });
    return response.data;
  },

  /**
   * Export model
   */
  async exportModel(runId: string, format: 'pickle' | 'onnx' | 'pmml' = 'pickle'): Promise<Blob> {
    const response = await api.get(`${MODELING_BASE_URL}/runs/${runId}/export`, {
      params: { format },
      responseType: 'blob',
    });
    return response.data;
  },

  /**
   * Get model comparison data
   */
  async getModelComparison(runIds: string[]): Promise<{
    runs: TrainingResultsResponse[];
    comparison: {
      bestModel: string;
      metrics: Record<string, any>;
    };
  }> {
    const response = await api.post(`${MODELING_BASE_URL}/compare`, { runIds });
    return response.data;
  },
};

export default modelingService;
