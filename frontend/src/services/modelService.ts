import api from './api';
import type { Model, Hyperparameters } from '../types/model';

class ModelService {
  /**
   * Get all models
   */
  async getModels(): Promise<Model[]> {
    return api.get<Model[]>('/api/v1/models');
  }

  /**
   * Get model by ID
   */
  async getModel(id: string): Promise<Model> {
    return api.get<Model>(`/api/v1/models/${id}`);
  }

  /**
   * Train a new model
   */
  async trainModel(
    datasetId: string,
    modelType: string,
    hyperparameters: Hyperparameters
  ): Promise<Model> {
    return api.post<Model>('/api/v1/models/train', {
      datasetId,
      modelType,
      hyperparameters,
    });
  }

  /**
   * Get model training status
   */
  async getTrainingStatus(modelId: string): Promise<any> {
    return api.get(`/api/v1/models/${modelId}/status`);
  }

  /**
   * Stop model training
   */
  async stopTraining(modelId: string): Promise<void> {
    return api.post(`/api/v1/models/${modelId}/stop`, {});
  }

  /**
   * Delete model
   */
  async deleteModel(id: string): Promise<void> {
    return api.delete(`/api/v1/models/${id}`);
  }

  /**
   * Download trained model
   */
  async downloadModel(id: string, filename: string): Promise<void> {
    return api.download(`/api/v1/models/${id}/download`, filename);
  }
}

export const modelService = new ModelService();
export default modelService;
