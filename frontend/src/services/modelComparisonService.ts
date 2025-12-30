/**
 * Model Comparison Service
 * 
 * Service for comparing multiple model runs and analyzing their performance
 */

import api from './api';
import type {
  CompareModelsRequest,
  ModelComparisonResponse,
  ModelRankingRequest,
  ModelRankingResponse,
} from '../types/modelComparison';

class ModelComparisonService {
  /**
   * Compare multiple model runs
   * 
   * @param request - Comparison request with model run IDs and options
   * @returns Comprehensive comparison analysis
   */
  async compareModels(request: CompareModelsRequest): Promise<ModelComparisonResponse> {
    try {
      const response = await api.post<ModelComparisonResponse>(
        '/api/v1/models/compare',
        request
      );
      return response;
    } catch (error) {
      console.error('Error comparing models:', error);
      throw error;
    }
  }

  /**
   * Rank models using custom weighted criteria
   * 
   * @param request - Ranking request with model IDs and weights
   * @returns Ranked models with composite scores
   */
  async rankModels(request: ModelRankingRequest): Promise<ModelRankingResponse> {
    try {
      const response = await api.post<ModelRankingResponse>(
        '/api/v1/models/rank',
        request
      );
      return response;
    } catch (error) {
      console.error('Error ranking models:', error);
      throw error;
    }
  }

  /**
   * Get list of model runs for selection
   * 
   * @param experimentId - Optional experiment ID to filter by
   * @param status - Optional status to filter by
   * @param modelType - Optional model type to filter by
   * @param limit - Maximum number of results
   * @param offset - Offset for pagination
   * @returns List of model runs
   */
  async listModelRuns(
    experimentId?: string,
    status?: string,
    modelType?: string,
    limit: number = 50,
    offset: number = 0
  ): Promise<any[]> {
    try {
      const params = new URLSearchParams();
      if (experimentId) params.append('experiment_id', experimentId);
      if (status) params.append('status', status);
      if (modelType) params.append('model_type', modelType);
      params.append('limit', limit.toString());
      params.append('offset', offset.toString());

      const response = await api.get<any[]>(
        `/api/v1/models/runs?${params.toString()}`
      );
      return response;
    } catch (error) {
      console.error('Error fetching model runs:', error);
      throw error;
    }
  }

  /**
   * Get model metrics by ID
   * 
   * @param modelRunId - UUID of the model run
   * @param useCache - Whether to use cached results
   * @returns Detailed metrics for the model
   */
  async getModelMetrics(modelRunId: string, useCache: boolean = true): Promise<any> {
    try {
      const params = new URLSearchParams();
      params.append('use_cache', useCache.toString());

      const response = await api.get<any>(
        `/api/v1/models/train/${modelRunId}/metrics?${params.toString()}`
      );
      return response;
    } catch (error) {
      console.error('Error fetching model metrics:', error);
      throw error;
    }
  }
}

export const modelComparisonService = new ModelComparisonService();
export default modelComparisonService;
