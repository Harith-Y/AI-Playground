import type { VisualizationData } from '../types/dataset';

/**
 * Visualization generation utilities for creating Plotly-compatible chart data
 */

/**
 * Configuration options for histogram generation
 */
export interface HistogramOptions {
  bins?: number;
  color?: string;
  opacity?: number;
  showMean?: boolean;
  showMedian?: boolean;
  title?: string;
  xLabel?: string;
  yLabel?: string;
}

/**
 * Statistical summary for a distribution
 */
export interface DistributionStats {
  mean: number;
  median: number;
  std: number;
  min: number;
  max: number;
  q1: number;
  q3: number;
  count: number;
}

/**
 * Calculate statistical summary for a numeric array
 */
export const calculateStats = (data: number[]): DistributionStats => {
  const sorted = [...data].sort((a, b) => a - b);
  const n = sorted.length;

  const mean = data.reduce((sum, val) => sum + val, 0) / n;
  const median = n % 2 === 0
    ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
    : sorted[Math.floor(n / 2)];

  const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
  const std = Math.sqrt(variance);

  const q1 = sorted[Math.floor(n * 0.25)];
  const q3 = sorted[Math.floor(n * 0.75)];

  return {
    mean,
    median,
    std,
    min: sorted[0],
    max: sorted[n - 1],
    q1,
    q3,
    count: n,
  };
};

/**
 * Create a histogram visualization from numeric data
 *
 * @param data - Array of numeric values
 * @param columnName - Name of the column/feature
 * @param options - Histogram configuration options
 * @returns VisualizationData object compatible with VisualizationGallery
 *
 * @example
 * ```ts
 * const ages = [25, 30, 35, 40, 45, 28, 32, 38, 42, 50];
 * const histogram = createHistogram(ages, 'age', {
 *   bins: 10,
 *   showMean: true,
 *   showMedian: true,
 * });
 * ```
 */
export const createHistogram = (
  data: number[],
  columnName: string,
  options: HistogramOptions = {}
): VisualizationData => {
  const {
    bins,
    color = '#3B82F6',
    opacity = 0.7,
    showMean = false,
    showMedian = false,
    title,
    xLabel,
    yLabel = 'Frequency',
  } = options;

  const stats = calculateStats(data);

  // Main histogram trace
  const traces: any[] = [
    {
      x: data,
      type: 'histogram',
      name: columnName,
      marker: {
        color: color,
        opacity: opacity,
        line: {
          color: color,
          width: 1,
        },
      },
      ...(bins && { nbinsx: bins }),
    },
  ];

  // Add mean line if requested
  if (showMean) {
    traces.push({
      x: [stats.mean, stats.mean],
      y: [0, 1],
      type: 'scatter',
      mode: 'lines',
      name: `Mean: ${stats.mean.toFixed(2)}`,
      line: {
        color: '#EF4444',
        width: 2,
        dash: 'dash',
      },
      yaxis: 'y2',
    });
  }

  // Add median line if requested
  if (showMedian) {
    traces.push({
      x: [stats.median, stats.median],
      y: [0, 1],
      type: 'scatter',
      mode: 'lines',
      name: `Median: ${stats.median.toFixed(2)}`,
      line: {
        color: '#10B981',
        width: 2,
        dash: 'dot',
      },
      yaxis: 'y2',
    });
  }

  return {
    type: 'histogram',
    title: title || `Distribution of ${columnName}`,
    data: traces,
    xLabel: xLabel || columnName,
    yLabel,
  };
};

/**
 * Create multiple histograms for all numeric columns in a dataset
 *
 * @param dataset - Object with column names as keys and arrays of numeric values
 * @param options - Histogram configuration options
 * @returns Array of VisualizationData objects
 *
 * @example
 * ```ts
 * const dataset = {
 *   age: [25, 30, 35, 40, 45],
 *   income: [50000, 60000, 70000, 80000, 90000],
 *   score: [85, 90, 78, 92, 88],
 * };
 * const histograms = createMultipleHistograms(dataset, { bins: 15 });
 * ```
 */
export const createMultipleHistograms = (
  dataset: Record<string, number[]>,
  options: HistogramOptions = {}
): VisualizationData[] => {
  return Object.entries(dataset).map(([columnName, data]) =>
    createHistogram(data, columnName, options)
  );
};

/**
 * Create a grouped histogram (multiple distributions on same chart)
 *
 * @param datasets - Object with group names as keys and arrays of numeric values
 * @param columnName - Name of the feature being compared
 * @param options - Histogram configuration options
 * @returns VisualizationData object
 *
 * @example
 * ```ts
 * const groupedData = {
 *   'Group A': [25, 30, 35, 40, 45],
 *   'Group B': [28, 32, 38, 42, 50],
 *   'Group C': [22, 26, 30, 34, 38],
 * };
 * const grouped = createGroupedHistogram(groupedData, 'Age', { bins: 10 });
 * ```
 */
export const createGroupedHistogram = (
  datasets: Record<string, number[]>,
  columnName: string,
  options: HistogramOptions = {}
): VisualizationData => {
  const {
    bins,
    opacity = 0.6,
    title,
    xLabel,
    yLabel = 'Frequency',
  } = options;

  const colors = ['#3B82F6', '#10B981', '#8B5CF6', '#F59E0B', '#EF4444', '#EC4899'];

  const traces = Object.entries(datasets).map(([groupName, data], idx) => ({
    x: data,
    type: 'histogram',
    name: groupName,
    marker: {
      color: colors[idx % colors.length],
      opacity: opacity,
    },
    ...(bins && { nbinsx: bins }),
  }));

  return {
    type: 'histogram',
    title: title || `Distribution of ${columnName} by Group`,
    data: traces,
    xLabel: xLabel || columnName,
    yLabel,
  };
};

/**
 * Create a histogram with overlaid normal distribution curve
 *
 * @param data - Array of numeric values
 * @param columnName - Name of the column/feature
 * @param options - Histogram configuration options
 * @returns VisualizationData object
 *
 * @example
 * ```ts
 * const heights = [160, 165, 170, 175, 180, 168, 172, 178, 182, 175];
 * const histWithNormal = createHistogramWithNormal(heights, 'Height (cm)');
 * ```
 */
export const createHistogramWithNormal = (
  data: number[],
  columnName: string,
  options: HistogramOptions = {}
): VisualizationData => {
  const {
    bins,
    color = '#3B82F6',
    opacity = 0.7,
    title,
    xLabel,
    yLabel = 'Frequency',
  } = options;

  const stats = calculateStats(data);

  // Generate normal distribution curve
  const numPoints = 100;
  const xMin = stats.min - stats.std;
  const xMax = stats.max + stats.std;
  const step = (xMax - xMin) / numPoints;

  const normalX: number[] = [];
  const normalY: number[] = [];

  for (let i = 0; i <= numPoints; i++) {
    const x = xMin + i * step;
    normalX.push(x);

    // Normal distribution formula
    const exponent = -Math.pow(x - stats.mean, 2) / (2 * Math.pow(stats.std, 2));
    const y = (1 / (stats.std * Math.sqrt(2 * Math.PI))) * Math.exp(exponent);

    // Scale by bin width and number of samples to match histogram
    normalY.push(y * data.length * step);
  }

  const traces = [
    {
      x: data,
      type: 'histogram',
      name: columnName,
      marker: {
        color: color,
        opacity: opacity,
      },
      ...(bins && { nbinsx: bins }),
    },
    {
      x: normalX,
      y: normalY,
      type: 'scatter',
      mode: 'lines',
      name: 'Normal Distribution',
      line: {
        color: '#EF4444',
        width: 2,
      },
    },
  ];

  return {
    type: 'histogram',
    title: title || `${columnName} Distribution with Normal Curve`,
    data: traces,
    xLabel: xLabel || columnName,
    yLabel,
  };
};

/**
 * Create a histogram with custom bin edges
 *
 * @param data - Array of numeric values
 * @param columnName - Name of the column/feature
 * @param binEdges - Array of bin edge values
 * @param options - Histogram configuration options
 * @returns VisualizationData object
 *
 * @example
 * ```ts
 * const scores = [65, 70, 75, 80, 85, 90, 95];
 * const bins = [0, 60, 70, 80, 90, 100]; // Grade boundaries
 * const customHist = createCustomBinHistogram(scores, 'Exam Score', bins);
 * ```
 */
export const createCustomBinHistogram = (
  data: number[],
  columnName: string,
  binEdges: number[],
  options: HistogramOptions = {}
): VisualizationData => {
  const {
    color = '#3B82F6',
    opacity = 0.7,
    title,
    xLabel,
    yLabel = 'Frequency',
  } = options;

  return {
    type: 'histogram',
    title: title || `Distribution of ${columnName}`,
    data: [
      {
        x: data,
        type: 'histogram',
        name: columnName,
        xbins: {
          start: binEdges[0],
          end: binEdges[binEdges.length - 1],
          size: binEdges[1] - binEdges[0],
        },
        marker: {
          color: color,
          opacity: opacity,
        },
      },
    ],
    xLabel: xLabel || columnName,
    yLabel,
  };
};

/**
 * Determine optimal number of bins using Sturges' rule
 *
 * @param dataLength - Number of data points
 * @returns Optimal number of bins
 */
export const calculateOptimalBins = (dataLength: number): number => {
  return Math.ceil(Math.log2(dataLength) + 1);
};

/**
 * Determine optimal number of bins using Freedman-Diaconis rule
 *
 * @param data - Array of numeric values
 * @returns Optimal number of bins
 */
export const calculateOptimalBinsFD = (data: number[]): number => {
  const stats = calculateStats(data);
  const iqr = stats.q3 - stats.q1;
  const binWidth = 2 * iqr / Math.pow(data.length, 1/3);

  if (binWidth === 0) return calculateOptimalBins(data.length);

  const numBins = Math.ceil((stats.max - stats.min) / binWidth);
  return Math.max(1, Math.min(numBins, 100)); // Cap between 1 and 100
};

/**
 * Create auto-binned histogram using optimal bin calculation
 *
 * @param data - Array of numeric values
 * @param columnName - Name of the column/feature
 * @param method - Binning method ('sturges' or 'fd')
 * @param options - Histogram configuration options
 * @returns VisualizationData object
 */
export const createAutoBinnedHistogram = (
  data: number[],
  columnName: string,
  method: 'sturges' | 'fd' = 'fd',
  options: HistogramOptions = {}
): VisualizationData => {
  const optimalBins = method === 'sturges'
    ? calculateOptimalBins(data.length)
    : calculateOptimalBinsFD(data);

  return createHistogram(data, columnName, {
    ...options,
    bins: optimalBins,
  });
};
