import { useState, useEffect, useCallback } from 'react';
import modelingService from '../services/modelingService';
import type { AvailableModel } from '../services/modelingService';

export interface UseAvailableModelsOptions {
  category?: 'classification' | 'regression' | 'clustering';
  taskType?: string;
  autoFetch?: boolean;
}

export interface UseAvailableModelsReturn {
  models: AvailableModel[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  getModelByType: (modelType: string) => AvailableModel | undefined;
  getModelsByCategory: (category: string) => AvailableModel[];
}

export const useAvailableModels = (
  options: UseAvailableModelsOptions = {}
): UseAvailableModelsReturn => {
  const {
    category,
    taskType,
    autoFetch = true,
  } = options;

  const [models, setModels] = useState<AvailableModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchModels = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await modelingService.getAvailableModels({
        category,
        taskType,
      });

      setModels(response.models);
    } catch (err) {
      console.error('[AvailableModels] Error fetching models:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to load available models';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [category, taskType]);

  useEffect(() => {
    if (autoFetch) {
      fetchModels();
    }
  }, [autoFetch, fetchModels]);

  const refresh = useCallback(async () => {
    await fetchModels();
  }, [fetchModels]);

  const getModelByType = useCallback((modelType: string) => {
    return models.find(m => m.id === modelType || m.name === modelType);
  }, [models]);

  const getModelsByCategory = useCallback((cat: string) => {
    return models.filter(m => m.category === cat);
  }, [models]);

  return {
    models,
    loading,
    error,
    refresh,
    getModelByType,
    getModelsByCategory,
  };
};

export default useAvailableModels;
