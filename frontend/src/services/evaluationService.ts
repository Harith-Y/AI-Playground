/**
 * Evaluation Service
 * 
 * Service for model evaluation APIs including metrics, feature importance,
 * plot data, and comparison functionality
 */

import api from './api';
import type {
  ModelMetrics,
  FeatureImportance,
  EvaluationData,
  ComparisonData,
} from '../types/evaluation';

// ============================================================================
// Response Types
// ============================================================================

export interface ModelMetricsResponse {
  model_run_id: string;
  model_type: string;
  task_type: string;
  metrics: Record<string, any>;
  training_metadata: {
    training_time?: number;
    created_at: string;
    hyperparameters: Record<string, any>;
    train_samples?: number;
    test_samples?: number;
    n_features?: number;
  };
  feature_importance?: Record<string, number>;
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
  rank: number;
}

export interface FeatureImportanceResponse {
  model_run_id: string;
  model_type: string;
  task_type: string;
  has_feature_importance: boolean;
  feature_importance: FeatureImportanceItem[] | null;
  feature_importance_dict: Record<string, number> | null;
  total_features: number;
  top_features: FeatureImportanceItem[] | null;
  importance_method: string | null;
  message: string | null;
}

export interface PlotDataResponse {
  plot_type: string;
  plot_data: any;
  layout?: any;
  config?: any;
}

export interface EvaluationPlotsResponse {
  model_run_id: string;
  task_type: string;
  available_plots: string[];
  plots: {
    [key: string]: PlotDataResponse;
  };
}

// ============================================================================
// Evaluation Service Class
// ============================================================================

class EvaluationService {
  /**
   * Get comprehensive metrics for a model run
   * 
   * @param modelRunId - UUID of the model run
   * @param useCache - Whether to use cached results (default: true)
   * @returns Detailed metrics and training metadata
   */
  async getModelMetrics(
    modelRunId: string,
    useCache: boolean = true
  ): Promise<ModelMetricsResponse> {
    try {
      const params = new URLSearchParams();
      params.append('use_cache', useCache.toString());

      const response = await api.get<ModelMetricsResponse>(
        `/api/v1/models/train/${modelRunId}/metrics?${params.toString()}`
      );
      
      return response;
    } catch (error) {
      console.error('Error fetching model metrics:', error);
      throw error;
    }
  }

  /**
   * Get feature importance for a model run
   * 
   * @param modelRunId - UUID of the model run
   * @param topN - Optional number of top features (default: all)
   * @param useCache - Whether to use cached results (default: true)
   * @returns Feature importance data with rankings
   */
  async getFeatureImportance(
    modelRunId: string,
    topN?: number,
    useCache: boolean = true
  ): Promise<FeatureImportanceResponse> {
    try {
      const params = new URLSearchParams();
      if (topN !== undefined) {
        params.append('top_n', topN.toString());
      }
      params.append('use_cache', useCache.toString());

      const response = await api.get<FeatureImportanceResponse>(
        `/api/v1/models/train/${modelRunId}/feature-importance?${params.toString()}`
      );
      
      return response;
    } catch (error) {
      console.error('Error fetching feature importance:', error);
      throw error;
    }
  }

  /**
   * Get plot data for visualization
   * 
   * @param modelRunId - UUID of the model run
   * @param plotType - Type of plot (confusion_matrix, roc_curve, residual, etc.)
   * @returns Plot data formatted for Plotly
   */
  async getPlotData(
    modelRunId: string,
    plotType: string
  ): Promise<PlotDataResponse> {
    try {
      // Note: This endpoint would need to be implemented in backend
      const response = await api.get<PlotDataResponse>(
        `/api/v1/models/train/${modelRunId}/plots/${plotType}`
      );
      
      return response;
    } catch (error) {
      console.error('Error fetching plot data:', error);
      throw error;
    }
  }

  /**
   * Get all available plots for a model run
   * 
   * @param modelRunId - UUID of the model run
   * @returns All available plots and their data
   */
  async getAllPlots(modelRunId: string): Promise<EvaluationPlotsResponse> {
    try {
      // Note: This endpoint would need to be implemented in backend
      const response = await api.get<EvaluationPlotsResponse>(
        `/api/v1/models/train/${modelRunId}/plots`
      );
      
      return response;
    } catch (error) {
      console.error('Error fetching all plots:', error);
      throw error;
    }
  }

  /**
   * Get training status
   * 
   * @param modelRunId - UUID of the model run
   * @returns Training status and progress
   */
  async getTrainingStatus(modelRunId: string): Promise<any> {
    try {
      const response = await api.get<any>(
        `/api/v1/models/train/${modelRunId}/status`
      );
      
      return response;
    } catch (error) {
      console.error('Error fetching training status:', error);
      throw error;
    }
  }

  /**
   * Get complete training result
   * 
   * @param modelRunId - UUID of the model run
   * @returns Complete training results
   */
  async getTrainingResult(modelRunId: string): Promise<any> {
    try {
      const response = await api.get<any>(
        `/api/v1/models/train/${modelRunId}/result`
      );
      
      return response;
    } catch (error) {
      console.error('Error fetching training result:', error);
      throw error;
    }
  }

  /**
   * Get evaluation data combining metrics, feature importance, and plots
   * 
   * @param modelRunId - UUID of the model run
   * @param options - Configuration options
   * @returns Complete evaluation data
   */
  async getEvaluationData(
    modelRunId: string,
    options: {
      includeFeatureImportance?: boolean;
      includePlots?: boolean;
      topFeatures?: number;
      useCache?: boolean;
    } = {}
  ): Promise<{
    metrics: ModelMetricsResponse;
    featureImportance?: FeatureImportanceResponse;
    plots?: EvaluationPlotsResponse;
  }> {
    const {
      includeFeatureImportance = true,
      includePlots = false,
      topFeatures,
      useCache = true,
    } = options;

    try {
      // Fetch metrics
      const metrics = await this.getModelMetrics(modelRunId, useCache);

      // Fetch feature importance if requested
      let featureImportance: FeatureImportanceResponse | undefined;
      if (includeFeatureImportance) {
        try {
          featureImportance = await this.getFeatureImportance(
            modelRunId,
            topFeatures,
            useCache
          );
        } catch (error) {
          console.warn('Feature importance not available:', error);
        }
      }

      // Fetch plots if requested
      let plots: EvaluationPlotsResponse | undefined;
      if (includePlots) {
        try {
          plots = await this.getAllPlots(modelRunId);
        } catch (error) {
          console.warn('Plots not available:', error);
        }
      }

      return {
        metrics,
        featureImportance,
        plots,
      };
    } catch (error) {
      console.error('Error fetching evaluation data:', error);
      throw error;
    }
  }

  /**
   * Compare multiple model runs
   * 
   * @param modelRunIds - Array of model run UUIDs
   * @returns Comparison data
   */
  async compareModelRuns(modelRunIds: string[]): Promise<any> {
    try {
      // Use the comparison service we created earlier
      const { modelComparisonService } = await import('./modelComparisonService');
      
      return await modelComparisonService.compareModels({
        model_run_ids: modelRunIds,
      });
    } catch (error) {
      console.error('Error comparing model runs:', error);
      throw error;
    }
  }

  /**
   * Export evaluation data
   * 
   * @param modelRunId - UUID of the model run
   * @param format - Export format (json, csv)
   * @returns Exported data as blob or string
   */
  async exportEvaluationData(
    modelRunId: string,
    format: 'json' | 'csv' = 'json'
  ): Promise<Blob> {
    try {
      const data = await this.getEvaluationData(modelRunId, {
        includeFeatureImportance: true,
        includePlots: false,
      });

      if (format === 'json') {
        const jsonStr = JSON.stringify(data, null, 2);
        return new Blob([jsonStr], { type: 'application/json' });
      } else {
        // Convert to CSV format
        const csv = this.convertToCSV(data);
        return new Blob([csv], { type: 'text/csv' });
      }
    } catch (error) {
      console.error('Error exporting evaluation data:', error);
      throw error;
    }
  }

  /**
   * Helper: Convert evaluation data to CSV format
   */
  private convertToCSV(data: any): string {
    const { metrics, featureImportance } = data;

    // Build CSV for metrics
    const metricsRows: string[] = ['Metric,Value'];
    Object.entries(metrics.metrics).forEach(([key, value]) => {
      metricsRows.push(`${key},${value}`);
    });

    // Add feature importance section
    if (featureImportance?.feature_importance) {
      metricsRows.push('');
      metricsRows.push('Feature,Importance,Rank');
      featureImportance.feature_importance.forEach((item: FeatureImportanceItem) => {
        metricsRows.push(`${item.feature},${item.importance},${item.rank}`);
      });
    }

    return metricsRows.join('\n');
  }

  /**
   * Helper: Generate plot data for confusion matrix
   */
  generateConfusionMatrixPlot(confusionMatrix: number[][], classNames?: string[]) {
    const labels = classNames || confusionMatrix.map((_, i) => `Class ${i}`);

    return {
      data: [{
        type: 'heatmap',
        z: confusionMatrix,
        x: labels,
        y: labels,
        colorscale: 'Blues',
        showscale: true,
      }],
      layout: {
        title: 'Confusion Matrix',
        xaxis: { title: 'Predicted' },
        yaxis: { title: 'Actual' },
      },
    };
  }

  /**
   * Helper: Generate plot data for ROC curve
   */
  generateROCCurvePlot(fpr: number[], tpr: number[], auc: number) {
    return {
      data: [
        {
          type: 'scatter',
          mode: 'lines',
          x: fpr,
          y: tpr,
          name: `ROC Curve (AUC = ${auc.toFixed(3)})`,
          line: { color: 'blue', width: 2 },
        },
        {
          type: 'scatter',
          mode: 'lines',
          x: [0, 1],
          y: [0, 1],
          name: 'Random Classifier',
          line: { dash: 'dash', color: 'red' },
        },
      ],
      layout: {
        title: 'ROC Curve',
        xaxis: { title: 'False Positive Rate' },
        yaxis: { title: 'True Positive Rate' },
      },
    };
  }

  /**
   * Helper: Generate plot data for residuals
   */
  generateResidualPlot(predicted: number[], actual: number[]) {
    const residuals = predicted.map((pred, i) => actual[i] - pred);

    return {
      data: [{
        type: 'scatter',
        mode: 'markers',
        x: predicted,
        y: residuals,
        marker: { color: 'blue', opacity: 0.6 },
      }],
      layout: {
        title: 'Residual Plot',
        xaxis: { title: 'Predicted Values' },
        yaxis: { title: 'Residuals' },
        shapes: [{
          type: 'line',
          x0: Math.min(...predicted),
          x1: Math.max(...predicted),
          y0: 0,
          y1: 0,
          line: { dash: 'dash', color: 'red' },
        }],
      },
    };
  }
}

export const evaluationService = new EvaluationService();
export default evaluationService;
