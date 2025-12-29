import api from './api';
import type { TrainingRun, TrainingConfig } from '../types/training';

const TRAINING_BASE_URL = '/api/training';

export const trainingService = {
  /**
   * Start a new training run
   */
  async startTraining(config: TrainingConfig): Promise<{ runId: string }> {
    const response = await api.post(`${TRAINING_BASE_URL}/start`, config);
    return response.data;
  },

  /**
   * Get training run by ID
   */
  async getTrainingRun(runId: string): Promise<TrainingRun> {
    const response = await api.get(`${TRAINING_BASE_URL}/runs/${runId}`);
    return response.data;
  },

  /**
   * Get all training runs
   */
  async getAllTrainingRuns(params?: {
    status?: string;
    modelType?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ runs: TrainingRun[]; total: number }> {
    const response = await api.get(`${TRAINING_BASE_URL}/runs`, { params });
    return response.data;
  },

  /**
   * Cancel a running training
   */
  async cancelTraining(runId: string): Promise<{ success: boolean; message: string }> {
    const response = await api.post(`${TRAINING_BASE_URL}/runs/${runId}/cancel`);
    return response.data;
  },

  /**
   * Stop a running training (alias for cancel)
   */
  async stopTraining(runId: string): Promise<{ success: boolean; message: string }> {
    return this.cancelTraining(runId);
  },

  /**
   * Delete a training run
   */
  async deleteTrainingRun(runId: string): Promise<{ success: boolean }> {
    const response = await api.delete(`${TRAINING_BASE_URL}/runs/${runId}`);
    return response.data;
  },

  /**
   * Get training logs
   */
  async getTrainingLogs(runId: string, params?: {
    level?: string;
    limit?: number;
    offset?: number;
  }): Promise<any> {
    const response = await api.get(`${TRAINING_BASE_URL}/runs/${runId}/logs`, { params });
    return response.data;
  },

  /**
   * Get training metrics
   */
  async getTrainingMetrics(runId: string): Promise<any> {
    const response = await api.get(`${TRAINING_BASE_URL}/runs/${runId}/metrics`);
    return response.data;
  },

  /**
   * Get WebSocket URL for training run
   */
  getWebSocketUrl(runId: string): string {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = import.meta.env.VITE_WS_HOST || window.location.host;
    return `${wsProtocol}//${wsHost}/ws/training/${runId}`;
  },
};

export default trainingService;
