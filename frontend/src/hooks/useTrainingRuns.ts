import { useState, useEffect, useCallback } from 'react';
import modelingService from '../services/modelingService';
import type { TrainingRunListItem, TrainingRunsResponse } from '../services/modelingService';

export interface UseTrainingRunsOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
  initialPage?: number;
  initialPageSize?: number;
  filters?: {
    status?: string;
    modelType?: string;
    datasetId?: string;
  };
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface UseTrainingRunsReturn {
  runs: TrainingRunListItem[];
  total: number;
  page: number;
  pageSize: number;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  deleteRun: (runId: string) => Promise<void>;
  deleteRuns: (runIds: string[]) => Promise<void>;
  cancelRun: (runId: string) => Promise<void>;
  retryRun: (runId: string) => Promise<void>;
  setPage: (page: number) => void;
  setPageSize: (pageSize: number) => void;
  setFilters: (filters: UseTrainingRunsOptions['filters']) => void;
  setSort: (sortBy: string, sortOrder: 'asc' | 'desc') => void;
}

export const useTrainingRuns = (
  options: UseTrainingRunsOptions = {}
): UseTrainingRunsReturn => {
  const {
    autoRefresh = false,
    refreshInterval = 5000,
    initialPage = 1,
    initialPageSize = 10,
    filters: initialFilters = {},
    sortBy: initialSortBy = 'startTime',
    sortOrder: initialSortOrder = 'desc',
  } = options;

  const [runs, setRuns] = useState<TrainingRunListItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(initialPage);
  const [pageSize, setPageSize] = useState(initialPageSize);
  const [filters, setFilters] = useState(initialFilters);
  const [sortBy, setSortBy] = useState(initialSortBy);
  const [sortOrder, setSortOrder] = useState(initialSortOrder);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchRuns = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response: TrainingRunsResponse = await modelingService.getTrainingRuns({
        page,
        pageSize,
        ...filters,
        sortBy,
        sortOrder,
      });

      setRuns(response.runs);
      setTotal(response.total);
    } catch (err) {
      console.error('[TrainingRuns] Error fetching runs:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to load training runs';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, filters, sortBy, sortOrder]);

  // Initial fetch
  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      fetchRuns();
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, fetchRuns]);

  const refresh = useCallback(async () => {
    await fetchRuns();
  }, [fetchRuns]);

  const deleteRun = useCallback(async (runId: string) => {
    try {
      await modelingService.deleteTrainingRun(runId);
      await fetchRuns();
    } catch (err) {
      console.error('[TrainingRuns] Error deleting run:', err);
      throw err;
    }
  }, [fetchRuns]);

  const deleteRuns = useCallback(async (runIds: string[]) => {
    try {
      await modelingService.deleteTrainingRuns(runIds);
      await fetchRuns();
    } catch (err) {
      console.error('[TrainingRuns] Error deleting runs:', err);
      throw err;
    }
  }, [fetchRuns]);

  const cancelRun = useCallback(async (runId: string) => {
    try {
      await modelingService.cancelTraining(runId);
      await fetchRuns();
    } catch (err) {
      console.error('[TrainingRuns] Error cancelling run:', err);
      throw err;
    }
  }, [fetchRuns]);

  const retryRun = useCallback(async (runId: string) => {
    try {
      await modelingService.retryTraining(runId);
      await fetchRuns();
    } catch (err) {
      console.error('[TrainingRuns] Error retrying run:', err);
      throw err;
    }
  }, [fetchRuns]);

  const setSort = useCallback((newSortBy: string, newSortOrder: 'asc' | 'desc') => {
    setSortBy(newSortBy);
    setSortOrder(newSortOrder);
  }, []);

  const handleSetFilters = useCallback((newFilters: UseTrainingRunsOptions['filters']) => {
    setFilters(newFilters || {});
  }, []);

  return {
    runs,
    total,
    page,
    pageSize,
    loading,
    error,
    refresh,
    deleteRun,
    deleteRuns,
    cancelRun,
    retryRun,
    setPage,
    setPageSize,
    setFilters: handleSetFilters,
    setSort,
  };
};

export default useTrainingRuns;
