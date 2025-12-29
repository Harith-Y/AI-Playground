import { useState, useEffect, useCallback } from 'react';
import modelingService from '../services/modelingService';
import type { InlineEvaluationData } from '../components/modeling/InlineEvaluationPreview';

export interface UseLatestTrainingResultsOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
  datasetId?: string;
  modelType?: string;
}

export interface UseLatestTrainingResultsReturn {
  data: InlineEvaluationData | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export const useLatestTrainingResults = (
  options: UseLatestTrainingResultsOptions = {}
): UseLatestTrainingResultsReturn => {
  const {
    autoRefresh = false,
    refreshInterval = 10000,
    datasetId,
    modelType,
  } = options;

  const [data, setData] = useState<InlineEvaluationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchLatestResults = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Get the latest completed training run
      const runsResponse = await modelingService.getTrainingRuns({
        page: 1,
        pageSize: 1,
        status: 'completed',
        datasetId,
        modelType,
        sortBy: 'endTime',
        sortOrder: 'desc',
      });

      if (runsResponse.runs.length === 0) {
        setData(null);
        return;
      }

      const latestRun = runsResponse.runs[0];

      // Fetch detailed results
      const results = await modelingService.getTrainingResults(latestRun.id);

      // Fetch training metrics history if available
      let trainingHistory;
      try {
        const metricsResponse = await modelingService.getTrainingMetrics(latestRun.id);
        trainingHistory = metricsResponse.metrics;
      } catch (err) {
        // Training history might not be available for all models
        console.log('Training history not available:', err);
      }

      // Determine task type from metrics
      let taskType: 'classification' | 'regression' | 'clustering' = 'classification';
      if (results.metrics.rmse !== undefined || results.metrics.mae !== undefined) {
        taskType = 'regression';
      }

      const evaluationData: InlineEvaluationData = {
        runId: latestRun.id,
        modelName: latestRun.modelName,
        modelType: latestRun.modelType,
        status: 'completed',
        metrics: results.metrics,
        trainingHistory,
        taskType,
      };

      setData(evaluationData);
    } catch (err) {
      console.error('[LatestTrainingResults] Error fetching results:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to load training results';
      setError(errorMessage);
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [datasetId, modelType]);

  // Initial fetch
  useEffect(() => {
    fetchLatestResults();
  }, [fetchLatestResults]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      fetchLatestResults();
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, fetchLatestResults]);

  const refresh = useCallback(async () => {
    await fetchLatestResults();
  }, [fetchLatestResults]);

  return {
    data,
    loading,
    error,
    refresh,
  };
};

export default useLatestTrainingResults;
