/**
 * useTuning Hook
 * 
 * Custom React hook for managing hyperparameter tuning operations
 * Provides state management, polling, and API integration
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  startTuning,
  getTuningProgress,
  getTuningResults,
  pauseTuning,
  resumeTuning,
  stopTuning,
  applyBestConfig,
  exportTuningResults,
} from '../services/tuningService';
import type {
  StartTuningRequest,
  TuningProgress,
  TuningResults,
  ApplyConfigRequest,
} from '../services/tuningService';

interface UseTuningOptions {
  tuningId?: string;
  autoStart?: boolean;
  pollingInterval?: number;
  onComplete?: (results: TuningResults) => void;
  onError?: (error: Error) => void;
}

interface UseTuningReturn {
  // State
  tuningId: string | null;
  progress: TuningProgress | null;
  results: TuningResults | null;
  isLoading: boolean;
  error: Error | null;
  isPolling: boolean;

  // Actions
  start: (request: StartTuningRequest) => Promise<void>;
  pause: () => Promise<void>;
  resume: () => Promise<void>;
  stop: () => Promise<void>;
  applyConfig: (parameters: Record<string, any>, createNewRun?: boolean) => Promise<void>;
  exportResults: () => Promise<void>;
  refresh: () => Promise<void>;
  clearError: () => void;
}

export const useTuning = (options: UseTuningOptions = {}): UseTuningReturn => {
  const {
    tuningId: initialTuningId,
    autoStart = false,
    pollingInterval = 2000,
    onComplete,
    onError,
  } = options;

  // State
  const [tuningId, setTuningId] = useState<string | null>(initialTuningId || null);
  const [progress, setProgress] = useState<TuningProgress | null>(null);
  const [results, setResults] = useState<TuningResults | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [isPolling, setIsPolling] = useState(autoStart);

  const pollingIntervalRef = useRef<number | null>(null);
  const isMountedRef = useRef(true);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  // Fetch progress
  const fetchProgress = useCallback(async () => {
    if (!tuningId) return;

    try {
      const progressData = await getTuningProgress(tuningId);
      
      if (!isMountedRef.current) return;
      
      setProgress(progressData);

      // Check if completed
      if (progressData.status === 'completed') {
        setIsPolling(false);
        
        // Fetch final results
        const resultsData = await getTuningResults(tuningId);
        if (isMountedRef.current) {
          setResults(resultsData);
          onComplete?.(resultsData);
        }
      } else if (progressData.status === 'failed') {
        setIsPolling(false);
        const err = new Error('Tuning job failed');
        setError(err);
        onError?.(err);
      }
    } catch (err) {
      if (isMountedRef.current) {
        const error = err as Error;
        setError(error);
        setIsPolling(false);
        onError?.(error);
      }
    }
  }, [tuningId, onComplete, onError]);

  // Polling effect
  useEffect(() => {
    if (isPolling && tuningId) {
      // Initial fetch
      fetchProgress();

      // Set up polling
      pollingIntervalRef.current = window.setInterval(() => {
        fetchProgress();
      }, pollingInterval);

      return () => {
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
        }
      };
    }
  }, [isPolling, tuningId, pollingInterval, fetchProgress]);

  // Start tuning
  const start = useCallback(async (request: StartTuningRequest) => {
    setIsLoading(true);
    setError(null);

    try {
      const job = await startTuning(request);
      
      if (!isMountedRef.current) return;
      
      setTuningId(job.tuning_id);
      setIsPolling(true);
    } catch (err) {
      if (isMountedRef.current) {
        const error = err as Error;
        setError(error);
        onError?.(error);
      }
    } finally {
      if (isMountedRef.current) {
        setIsLoading(false);
      }
    }
  }, [onError]);

  // Pause tuning
  const pause = useCallback(async () => {
    if (!tuningId) return;

    setIsLoading(true);
    setError(null);

    try {
      await pauseTuning(tuningId);
      
      if (!isMountedRef.current) return;
      
      setIsPolling(false);
      
      // Update progress
      await fetchProgress();
    } catch (err) {
      if (isMountedRef.current) {
        const error = err as Error;
        setError(error);
        onError?.(error);
      }
    } finally {
      if (isMountedRef.current) {
        setIsLoading(false);
      }
    }
  }, [tuningId, fetchProgress, onError]);

  // Resume tuning
  const resume = useCallback(async () => {
    if (!tuningId) return;

    setIsLoading(true);
    setError(null);

    try {
      await resumeTuning(tuningId);
      
      if (!isMountedRef.current) return;
      
      setIsPolling(true);
    } catch (err) {
      if (isMountedRef.current) {
        const error = err as Error;
        setError(error);
        onError?.(error);
      }
    } finally {
      if (isMountedRef.current) {
        setIsLoading(false);
      }
    }
  }, [tuningId, onError]);

  // Stop tuning
  const stop = useCallback(async () => {
    if (!tuningId) return;

    setIsLoading(true);
    setError(null);

    try {
      await stopTuning(tuningId);
      
      if (!isMountedRef.current) return;
      
      setIsPolling(false);
      
      // Fetch final results
      const resultsData = await getTuningResults(tuningId);
      if (isMountedRef.current) {
        setResults(resultsData);
      }
    } catch (err) {
      if (isMountedRef.current) {
        const error = err as Error;
        setError(error);
        onError?.(error);
      }
    } finally {
      if (isMountedRef.current) {
        setIsLoading(false);
      }
    }
  }, [tuningId, onError]);

  // Apply configuration
  const applyConfig = useCallback(async (parameters: Record<string, any>, createNewRun = false) => {
    if (!tuningId) return;

    setIsLoading(true);
    setError(null);

    try {
      const request: ApplyConfigRequest = {
        tuning_id: tuningId,
        parameters,
        create_new_run: createNewRun,
      };
      
      await applyBestConfig(request);
    } catch (err) {
      if (isMountedRef.current) {
        const error = err as Error;
        setError(error);
        onError?.(error);
      }
    } finally {
      if (isMountedRef.current) {
        setIsLoading(false);
      }
    }
  }, [tuningId, onError]);

  // Export results
  const exportResults = useCallback(async () => {
    if (!tuningId) return;

    setIsLoading(true);
    setError(null);

    try {
      const blob = await exportTuningResults(tuningId);
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `tuning_results_${tuningId}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      if (isMountedRef.current) {
        const error = err as Error;
        setError(error);
        onError?.(error);
      }
    } finally {
      if (isMountedRef.current) {
        setIsLoading(false);
      }
    }
  }, [tuningId, onError]);

  // Refresh data
  const refresh = useCallback(async () => {
    if (!tuningId) return;
    await fetchProgress();
  }, [tuningId, fetchProgress]);

  // Clear error
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    tuningId,
    progress,
    results,
    isLoading,
    error,
    isPolling,
    start,
    pause,
    resume,
    stop,
    applyConfig,
    exportResults,
    refresh,
    clearError,
  };
};

export default useTuning;
