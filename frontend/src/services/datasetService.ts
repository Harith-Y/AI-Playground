import api from './api';
import type { Dataset, DatasetStats, ColumnInfo } from '../types/dataset';

class DatasetService {
  /**
   * Upload a dataset file
   */
  async uploadDataset(file: File): Promise<Dataset> {
    return api.upload<Dataset>('/api/v1/datasets/upload', file);
  }

  /**
   * Get all datasets
   */
  async getDatasets(): Promise<Dataset[]> {
    return api.get<Dataset[]>('/api/v1/datasets');
  }

  /**
   * Get dataset by ID
   */
  async getDataset(id: string): Promise<Dataset> {
    return api.get<Dataset>(`/api/v1/datasets/${id}`);
  }

  /**
   * Get dataset statistics
   */
  async getDatasetStats(id: string): Promise<{ stats: DatasetStats; columns: ColumnInfo[] }> {
    return api.get(`/api/v1/datasets/${id}/stats`);
  }

  /**
   * Get dataset preview (first N rows)
   */
  async getDatasetPreview(id: string, rows: number = 10): Promise<any[]> {
    return api.get(`/api/v1/datasets/${id}/preview?rows=${rows}`);
  }

  /**
   * Delete dataset
   */
  async deleteDataset(id: string): Promise<void> {
    return api.delete(`/api/v1/datasets/${id}`);
  }
}

export const datasetService = new DatasetService();
export default datasetService;
