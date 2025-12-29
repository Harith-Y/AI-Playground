/**
 * Metrics Utilities
 *
 * Helper functions for formatting, calculating, and interpreting metrics
 */

import type {
  ClassificationMetrics,
  RegressionMetrics,
  ClusteringMetrics,
} from '../types/evaluation';

/**
 * Format a number as a percentage
 */
export const formatPercentage = (value: number, precision: number = 2): string => {
  return `${(value * 100).toFixed(precision)}%`;
};

/**
 * Format a large number with K/M/B suffixes
 */
export const formatLargeNumber = (value: number, precision: number = 2): string => {
  if (value >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(precision)}B`;
  } else if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(precision)}M`;
  } else if (value >= 1_000) {
    return `${(value / 1_000).toFixed(precision)}K`;
  }
  return value.toFixed(precision);
};

/**
 * Format a decimal number
 */
export const formatDecimal = (value: number, precision: number = 2): string => {
  return value.toFixed(precision);
};

/**
 * Format a number with locale-specific formatting
 */
export const formatNumber = (value: number, precision: number = 2): string => {
  return value.toLocaleString(undefined, {
    minimumFractionDigits: precision,
    maximumFractionDigits: precision,
  });
};

/**
 * Get metric quality label based on value and thresholds
 */
export const getQualityLabel = (
  value: number,
  thresholds: { excellent: number; good: number; fair: number; poor: number }
): string => {
  if (value >= thresholds.excellent) return 'Excellent';
  if (value >= thresholds.good) return 'Good';
  if (value >= thresholds.fair) return 'Fair';
  if (value >= thresholds.poor) return 'Poor';
  return 'Very Poor';
};

/**
 * Classification Metrics Utilities
 */
export const classificationMetricsUtils = {
  /**
   * Get overall model quality based on classification metrics
   */
  getModelQuality: (metrics: ClassificationMetrics): string => {
    const avgScore =
      (metrics.accuracy + metrics.precision + metrics.recall + metrics.f1Score) / 4;

    return getQualityLabel(avgScore, {
      excellent: 0.9,
      good: 0.75,
      fair: 0.6,
      poor: 0.4,
    });
  },

  /**
   * Get the best performing metric
   */
  getBestMetric: (metrics: ClassificationMetrics): { name: string; value: number } => {
    const scores = {
      Accuracy: metrics.accuracy,
      Precision: metrics.precision,
      Recall: metrics.recall,
      'F1 Score': metrics.f1Score,
      ...(metrics.auc && { 'AUC-ROC': metrics.auc }),
    };

    const best = Object.entries(scores).reduce((a, b) => (a[1] > b[1] ? a : b));
    return { name: best[0], value: best[1] };
  },

  /**
   * Calculate balanced accuracy (useful for imbalanced datasets)
   */
  calculateBalancedAccuracy: (metrics: ClassificationMetrics): number => {
    // Simplified calculation using precision and recall
    return (metrics.precision + metrics.recall) / 2;
  },

  /**
   * Interpret confusion matrix
   */
  interpretConfusionMatrix: (
    confusionMatrix: number[][]
  ): {
    totalSamples: number;
    correctPredictions: number;
    accuracy: number;
  } => {
    let totalSamples = 0;
    let correctPredictions = 0;

    confusionMatrix.forEach((row, i) => {
      row.forEach((value, j) => {
        totalSamples += value;
        if (i === j) {
          correctPredictions += value;
        }
      });
    });

    return {
      totalSamples,
      correctPredictions,
      accuracy: totalSamples > 0 ? correctPredictions / totalSamples : 0,
    };
  },
};

/**
 * Regression Metrics Utilities
 */
export const regressionMetricsUtils = {
  /**
   * Get overall model quality based on R² score
   */
  getModelQuality: (metrics: RegressionMetrics): string => {
    return getQualityLabel(metrics.r2, {
      excellent: 0.9,
      good: 0.7,
      fair: 0.5,
      poor: 0.3,
    });
  },

  /**
   * Calculate relative error percentage
   */
  calculateRelativeError: (mae: number, meanValue: number): number => {
    if (meanValue === 0) return 0;
    return (mae / Math.abs(meanValue)) * 100;
  },

  /**
   * Determine if RMSE is acceptable based on data scale
   */
  isRMSEAcceptable: (rmse: number, dataRange: number): boolean => {
    // RMSE should be less than 10% of data range for good models
    return rmse < dataRange * 0.1;
  },

  /**
   * Get error interpretation
   */
  getErrorInterpretation: (
    mape: number
  ): { level: string; description: string } => {
    if (mape <= 10) {
      return { level: 'Excellent', description: 'High accuracy forecast' };
    } else if (mape <= 20) {
      return { level: 'Good', description: 'Acceptable forecast' };
    } else if (mape <= 30) {
      return { level: 'Fair', description: 'Reasonable forecast' };
    } else if (mape <= 50) {
      return { level: 'Poor', description: 'Inaccurate forecast' };
    } else {
      return { level: 'Very Poor', description: 'Unreliable forecast' };
    }
  },

  /**
   * Calculate coefficient of variation of RMSE
   */
  calculateCVRMSE: (rmse: number, meanValue: number): number => {
    if (meanValue === 0) return 0;
    return (rmse / Math.abs(meanValue)) * 100;
  },
};

/**
 * Clustering Metrics Utilities
 */
export const clusteringMetricsUtils = {
  /**
   * Get overall clustering quality based on silhouette score
   */
  getClusteringQuality: (metrics: ClusteringMetrics): string => {
    return getQualityLabel(metrics.silhouetteScore, {
      excellent: 0.7,
      good: 0.5,
      fair: 0.3,
      poor: 0,
    });
  },

  /**
   * Calculate cluster balance (how evenly distributed clusters are)
   */
  calculateClusterBalance: (clusterSizes: number[]): number => {
    if (!clusterSizes || clusterSizes.length === 0) return 0;

    const total = clusterSizes.reduce((a, b) => a + b, 0);
    const avgSize = total / clusterSizes.length;

    // Calculate coefficient of variation
    const variance =
      clusterSizes.reduce((sum, size) => sum + Math.pow(size - avgSize, 2), 0) /
      clusterSizes.length;
    const stdDev = Math.sqrt(variance);

    // Return normalized balance score (1 = perfectly balanced, 0 = completely imbalanced)
    return avgSize > 0 ? Math.max(0, 1 - stdDev / avgSize) : 0;
  },

  /**
   * Get cluster distribution statistics
   */
  getClusterStats: (
    clusterSizes: number[]
  ): {
    total: number;
    average: number;
    min: number;
    max: number;
    balance: number;
  } => {
    if (!clusterSizes || clusterSizes.length === 0) {
      return { total: 0, average: 0, min: 0, max: 0, balance: 0 };
    }

    const total = clusterSizes.reduce((a, b) => a + b, 0);
    const average = total / clusterSizes.length;
    const min = Math.min(...clusterSizes);
    const max = Math.max(...clusterSizes);
    const balance = clusteringMetricsUtils.calculateClusterBalance(clusterSizes);

    return { total, average, min, max, balance };
  },

  /**
   * Interpret Davies-Bouldin Index
   */
  interpretDaviesBouldin: (
    score: number
  ): { level: string; description: string } => {
    if (score <= 0.5) {
      return { level: 'Excellent', description: 'Very well-separated clusters' };
    } else if (score <= 1.0) {
      return { level: 'Good', description: 'Well-separated clusters' };
    } else if (score <= 2.0) {
      return { level: 'Fair', description: 'Moderately separated clusters' };
    } else {
      return { level: 'Poor', description: 'Poorly separated clusters' };
    }
  },

  /**
   * Interpret Calinski-Harabasz Score
   */
  interpretCalinskiHarabasz: (
    score: number
  ): { level: string; description: string } => {
    if (score >= 100) {
      return { level: 'Good', description: 'Dense and well-separated clusters' };
    } else if (score >= 50) {
      return { level: 'Fair', description: 'Moderately dense clusters' };
    } else {
      return { level: 'Poor', description: 'Sparse or overlapping clusters' };
    }
  },
};

/**
 * Compare two metric values and return trend
 */
export const compareMetrics = (
  current: number,
  previous: number
): { trend: 'up' | 'down' | 'neutral'; change: number; changePercent: number } => {
  const change = current - previous;
  const changePercent = previous !== 0 ? (change / Math.abs(previous)) * 100 : 0;

  let trend: 'up' | 'down' | 'neutral' = 'neutral';
  if (Math.abs(changePercent) > 0.5) {
    // Only show trend if change is > 0.5%
    trend = change > 0 ? 'up' : 'down';
  }

  return { trend, change, changePercent };
};

/**
 * Get metric color based on value and better direction
 */
export const getMetricColor = (
  value: number,
  thresholds: { good: number; warning: number },
  higherIsBetter: boolean = true
): 'success' | 'warning' | 'error' | 'default' => {
  if (higherIsBetter) {
    if (value >= thresholds.good) return 'success';
    if (value >= thresholds.warning) return 'warning';
    return 'error';
  } else {
    if (value <= thresholds.good) return 'success';
    if (value <= thresholds.warning) return 'warning';
    return 'error';
  }
};

/**
 * Round to significant figures
 */
export const roundToSignificant = (value: number, figures: number = 3): number => {
  if (value === 0) return 0;
  const magnitude = Math.floor(Math.log10(Math.abs(value)));
  const scale = Math.pow(10, figures - magnitude - 1);
  return Math.round(value * scale) / scale;
};
