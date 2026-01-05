import api from './api';
import type { Dataset, DatasetStats, ColumnInfo } from '../types/dataset';

class DatasetService {
  /**
   * Upload a dataset file
   */
  async uploadDataset(file: File): Promise<Dataset> {
    const response = await api.upload<any>('/api/v1/datasets/upload', file);
    return this.mapBackendDatasetToFrontend(response, file.size);
  }

  /**
   * Get all datasets
   */
  async getDatasets(): Promise<Dataset[]> {
    const response = await api.get<any[]>('/api/v1/datasets');
    return response.map(d => this.mapBackendDatasetToFrontend(d));
  }

  /**
   * Get dataset by ID
   */
  async getDataset(id: string): Promise<Dataset> {
    const response = await api.get<any>(`/api/v1/datasets/${id}`);
    return this.mapBackendDatasetToFrontend(response);
  }

  private mapBackendDatasetToFrontend(backendData: any, fileSize?: number): Dataset {
    return {
      id: backendData.id,
      name: backendData.name,
      filename: backendData.name, // Assuming name is filename
      size: fileSize || 0, // Backend doesn't return size yet
      rowCount: backendData.shape?.rows || 0,
      columnCount: backendData.shape?.cols || 0,
      createdAt: backendData.uploaded_at || new Date().toISOString(),
      updatedAt: backendData.uploaded_at || new Date().toISOString(),
      status: 'uploaded', // Default status
      rows: backendData.shape?.rows,
      columns: backendData.shape?.cols,
      shape: [backendData.shape?.rows || 0, backendData.shape?.cols || 0]
    };
  }

  /**
   * Get dataset statistics
   */
  async getDatasetStats(id: string): Promise<DatasetStats & { columns: ColumnInfo[] }> {
    const response = await api.get(`/api/v1/datasets/${id}/stats`);
    return response;
  }

  /**
   * Get dataset preview (first N rows)
   */
  async getDatasetPreview(id: string, rows: number = 10): Promise<{ preview: any[]; columns: ColumnInfo[]; totalRows: number; displayedRows: number }> {
    const response = await api.get(`/api/v1/datasets/${id}/preview?rows=${rows}`);
    return response;
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
