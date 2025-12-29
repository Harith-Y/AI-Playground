import { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from './useWebSocket';
import type { TrainingRun, TrainingProgressUpdate } from '../types/training';
import trainingService from '../services/trainingService';

export interface UseTrainingProgressOptions {
  runId: string;
  enableWebSocket?: boolean;
  pollingInterval?: number;
  onStatusChange?: (status: TrainingRun['status']) => void;
  onComplete?: (run: TrainingRun) => void;
  onError?: (error: string) => void;
}

export interface UseTrainingProgressReturn {
  trainingRun: TrainingRun | null;
  loading: boolean;
  error: string | null;
  isConnected: boolean;
  connectionState: 'connecting' | 'connected' | 'disconnected' | 'error';
  refresh: () => Promise<void>;
  stopTraining: () => Promise<void>;
  cancelTraining: () => Promise<void>;
}

export const useTrainingProgress = (
  options: UseTrainingProgressOptions
): UseTrainingProgressReturn => {
  const {
    runId,
    enableWebSocket = true,
    pollingInterval = 5000,
    onStatusChange,
    onComplete,
    onError,
  } = options;

  const [trainingRun, setTrainingRun] = useState<TrainingRun | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // WebSocket connection
  const wsUrl = enableWebSocket ? trainingService.getWebSocketUrl(runId) : null;
  
  const { isConnected, lastMessage, connectionState } = useWebSocket(wsUrl, {
    shouldReconnect: true,
    reconnectAttempts: 5,
    reconnectInterval: 3000,
    onMessage: (event) => {
      console.log('[Training] WebSocket message:', event.data);
    },
  });

  // Fetch initial training run data
  const fetchTrainingRun = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await trainingService.getTrainingRun(runId);
      setTrainingRun(data);
      
      // Check if training is complete
      if (data.status === 'completed' && onComplete) {
        onComplete(data);
      }
      
      // Check if training failed
      if (data.status === 'failed' && onError && data.error) {
        onError(data.error);
      }
    } catch (err) {
      console.error('[Training] Error fetching training run:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to load training run';
      setError(errorMessage);
      
      if (onError) {
        onError(errorMessage);
      }
    } finally {
      setLoading(false);
    }
  }, [runId, onComplete, onError]);

  // Initial fetch
  useEffect(() => {
    fetchTrainingRun();
  }, [fetchTrainingRun]);

  // Handle WebSocket updates
  useEffect(() => {
    if (lastMessage && trainingRun) {
      try {
        const update: TrainingProgressUpdate = JSON.parse(lastMessage);
        
        setTrainingRun((prev) => {
          if (!prev) return prev;

          const newRun: TrainingRun = {
            ...prev,
            status: update.status || prev.status,
            progress: update.progress !== undefined ? update.progress : prev.progress,
            currentEpoch: update.currentEpoch || prev.currentEpoch,
            metrics: {
              ...prev.metrics,
              ...update.metrics,
            },
            logs: update.log ? [...prev.logs, update.log] : prev.logs,
            endTime:
              update.status === 'completed' || update.status === 'failed'
                ? new Date().toISOString()
                : prev.endTime,
          };

          // Trigger status change callback
          if (update.status && update.status !== prev.status && onStatusChange) {
            onStatusChange(update.status);
          }

          // Trigger completion callback
          if (update.status === 'completed' && onComplete) {
            onComplete(newRun);
          }

          return newRun;
        });
      } catch (err) {
        console.error('[Training] Error parsing WebSocket message:', err);
      }
    }
  }, [lastMessage, trainingRun, onStatusChange, onComplete]);

  // Polling fallback when WebSocket is not connected
  useEffect(() => {
    if (!isConnected && trainingRun?.status === 'running') {
      const interval = setInterval(() => {
        console.log('[Training] Polling for updates (WebSocket disconnected)');
        fetchTrainingRun();
      }, pollingInterval);

      return () => clearInterval(interval);
    }
  }, [isConnected, trainingRun?.status, pollingInterval, fetchTrainingRun]);

  // Stop training
  const stopTraining = useCallback(async () => {
    try {
      await trainingService.stopTraining(runId);
      
      setTrainingRun((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          status: 'cancelled',
          endTime: new Date().toISOString(),
        };
      });
    } catch (err) {
      console.error('[Training] Error stopping training:', err);
      throw err;
    }
  }, [runId]);

  // Cancel training (alias)
  const cancelTraining = stopTraining;

  // Refresh
  const refresh = useCallback(async () => {
    await fetchTrainingRun();
  }, [fetchTrainingRun]);

  return {
    trainingRun,
    loading,
    error,
    isConnected,
    connectionState,
    refresh,
    stopTraining,
    cancelTraining,
  };
};

export default useTrainingProgress;
