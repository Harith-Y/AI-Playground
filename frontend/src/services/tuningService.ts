/**
 * Tuning Service
 * 
 * API service for hyperparameter tuning operations including:
 * - Starting tuning jobs
 * - Monitoring progress
 * - Retrieving results
 * - Applying best configurations
 */

import api from './api';

// Types
export interface TuningConfig {
  method: 'grid_search' | 'random_search' | 'bayesian_optimization';
  n_iter: number;
  cv_folds: number;
  scoring: string;
  random_state: number;
  n_jobs: number;
  parameters: ParameterRange[];
}

export interface ParameterRange {
  name: string;
  type: 'int' | 'float' | 'categorical' | 'boolean';
  min?: number;
  max?: number;
  step?: number;
  values?: string[];
  default?: any;
}

export interface StartTuningRequest {
  model_run_id: string;
  model_type: string;
  tuning_config: TuningConfig;
  dataset_id?: string;
}

export interface TuningJob {
  tuning_id: string;
  model_run_id: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed';
  created_at: string;
  started_at?: string;
  completed_at?: string;
  config: TuningConfig;
}

export interface TrialResult {
  trial_id: number;
  parameters: Record<string, any>;
  score: number;
  cv_scores: number[];
  mean_score: number;
  std_score: number;
  fit_time: number;
  score_time: number;
  rank: number;
}

export interface TuningProgress {
  tuning_id: string;
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed';
  current_trial: number;
  total_trials: number;
  best_score: number;
  best_parameters: Record<string, any>;
  current_parameters: Record<string, any>;
  elapsed_time: number;
  estimated_time_remaining: number;
  trials: TrialResult[];
  progress_history: Array<{ trial: number; score: number; best_score: number }>;
}

export interface TuningResults {
  tuning_id: string;
  best_trial: TrialResult;
  all_trials: TrialResult[];
  best_parameters: Record<string, any>;
  best_score: number;
  total_trials: number;
  tuning_time: number;
  cv_folds: number;
  scoring_metric: string;
  model_run_id?: string;
}

export interface ApplyConfigRequest {
  tuning_id: string;
  parameters: Record<string, any>;
  create_new_run?: boolean;
}

export interface ApplyConfigResponse {
  model_run_id: string;
  message: string;
  applied_parameters: Record<string, any>;
}

/**
 * Start a new hyperparameter tuning job
 */
export const startTuning = async (request: StartTuningRequest): Promise<TuningJob> => {
  const response = await api.post('/tuning/start', request);
  return response.data;
};

/**
 * Get tuning job status and basic info
 */
export const getTuningJob = async (tuningId: string): Promise<TuningJob> => {
  const response = await api.get(`/tuning/${tuningId}`);
  return response.data;
};

/**
 * Get real-time tuning progress
 */
export const getTuningProgress = async (tuningId: string): Promise<TuningProgress> => {
  const response = await api.get(`/tuning/${tuningId}/progress`);
  return response.data;
};

/**
 * Get tuning results (after completion)
 */
export const getTuningResults = async (tuningId: string): Promise<TuningResults> => {
  const response = await api.get(`/tuning/${tuningId}/results`);
  return response.data;
};

/**
 * Pause a running tuning job
 */
export const pauseTuning = async (tuningId: string): Promise<{ message: string }> => {
  const response = await api.post(`/tuning/${tuningId}/pause`);
  return response.data;
};

/**
 * Resume a paused tuning job
 */
export const resumeTuning = async (tuningId: string): Promise<{ message: string }> => {
  const response = await api.post(`/tuning/${tuningId}/resume`);
  return response.data;
};

/**
 * Stop a running tuning job
 */
export const stopTuning = async (tuningId: string): Promise<{ message: string }> => {
  const response = await api.post(`/tuning/${tuningId}/stop`);
  return response.data;
};

/**
 * Delete a tuning job and its results
 */
export const deleteTuning = async (tuningId: string): Promise<{ message: string }> => {
  const response = await api.delete(`/tuning/${tuningId}`);
  return response.data;
};

/**
 * Apply best configuration to a model
 */
export const applyBestConfig = async (request: ApplyConfigRequest): Promise<ApplyConfigResponse> => {
  const response = await api.post('/tuning/apply-config', request);
  return response.data;
};

/**
 * Get list of all tuning jobs for a model run
 */
export const getTuningJobsByModelRun = async (modelRunId: string): Promise<TuningJob[]> => {
  const response = await api.get(`/tuning/model-run/${modelRunId}`);
  return response.data;
};

/**
 * Get list of all tuning jobs
 */
export const getAllTuningJobs = async (): Promise<TuningJob[]> => {
  const response = await api.get('/tuning/jobs');
  return response.data;
};

/**
 * Export tuning results as CSV
 */
export const exportTuningResults = async (tuningId: string): Promise<Blob> => {
  const response = await api.get(`/tuning/${tuningId}/export`, {
    responseType: 'blob',
  });
  return response.data;
};

/**
 * Get trial details
 */
export const getTrialDetails = async (tuningId: string, trialId: number): Promise<TrialResult> => {
  const response = await api.get(`/tuning/${tuningId}/trials/${trialId}`);
  return response.data;
};

/**
 * Compare multiple tuning runs
 */
export const compareTuningRuns = async (tuningIds: string[]): Promise<any> => {
  const response = await api.post('/tuning/compare', { tuning_ids: tuningIds });
  return response.data;
};

/**
 * Get recommended parameter ranges for a model type
 */
export const getRecommendedParameters = async (modelType: string): Promise<ParameterRange[]> => {
  const response = await api.get(`/tuning/recommended-parameters/${modelType}`);
  return response.data;
};

/**
 * Validate tuning configuration
 */
export const validateTuningConfig = async (config: TuningConfig): Promise<{ valid: boolean; errors?: string[] }> => {
  const response = await api.post('/tuning/validate-config', config);
  return response.data;
};

export default {
  startTuning,
  getTuningJob,
  getTuningProgress,
  getTuningResults,
  pauseTuning,
  resumeTuning,
  stopTuning,
  deleteTuning,
  applyBestConfig,
  getTuningJobsByModelRun,
  getAllTuningJobs,
  exportTuningResults,
  getTrialDetails,
  compareTuningRuns,
  getRecommendedParameters,
  validateTuningConfig,
};
