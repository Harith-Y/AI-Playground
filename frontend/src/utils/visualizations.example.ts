/**
 * Examples of using visualization utilities
 *
 * This file demonstrates various ways to create histograms and distribution plots
 * that can be used with the VisualizationGallery component.
 */

import {
  createHistogram,
  createMultipleHistograms,
  createGroupedHistogram,
  createHistogramWithNormal,
  createCustomBinHistogram,
  createAutoBinnedHistogram,
  calculateStats,
} from './visualizations';

// ============================================================================
// Example 1: Basic Histogram
// ============================================================================

export const basicHistogramExample = () => {
  const ages = [25, 30, 35, 40, 45, 28, 32, 38, 42, 50, 27, 33, 36, 41, 48];

  const histogram = createHistogram(ages, 'Age', {
    bins: 10,
    color: '#3B82F6',
    title: 'Employee Age Distribution',
  });

  // Use with VisualizationGallery:
  // <VisualizationGallery visualizations={[histogram]} />

  return histogram;
};

// ============================================================================
// Example 2: Histogram with Statistical Lines
// ============================================================================

export const histogramWithStatsExample = () => {
  const salaries = [
    45000, 52000, 48000, 65000, 71000, 58000, 62000, 55000,
    68000, 72000, 51000, 59000, 64000, 70000, 75000, 80000,
  ];

  const histogram = createHistogram(salaries, 'Salary', {
    bins: 15,
    color: '#10B981',
    showMean: true,
    showMedian: true,
    title: 'Salary Distribution with Mean & Median',
    xLabel: 'Annual Salary ($)',
  });

  return histogram;
};

// ============================================================================
// Example 3: Multiple Histograms (Separate Charts)
// ============================================================================

export const multipleHistogramsExample = () => {
  const dataset = {
    age: [25, 30, 35, 40, 45, 28, 32, 38, 42, 50],
    income: [50000, 60000, 70000, 80000, 90000, 55000, 65000, 75000, 85000, 95000],
    score: [85, 90, 78, 92, 88, 82, 94, 86, 91, 89],
    'years_experience': [2, 5, 8, 12, 15, 3, 6, 10, 14, 18],
  };

  const histograms = createMultipleHistograms(dataset, {
    bins: 12,
    showMean: true,
  });

  // Returns array of 4 separate histogram visualizations
  // <VisualizationGallery visualizations={histograms} />

  return histograms;
};

// ============================================================================
// Example 4: Grouped Histogram (Compare Multiple Groups)
// ============================================================================

export const groupedHistogramExample = () => {
  const testScores = {
    'Class A': [75, 82, 88, 91, 85, 79, 92, 87, 90, 84],
    'Class B': [68, 74, 80, 77, 72, 85, 88, 76, 81, 79],
    'Class C': [88, 92, 95, 91, 89, 93, 90, 94, 87, 91],
  };

  const grouped = createGroupedHistogram(testScores, 'Test Score', {
    bins: 10,
    title: 'Test Score Distribution by Class',
    xLabel: 'Score',
  });

  return grouped;
};

// ============================================================================
// Example 5: Histogram with Normal Distribution Overlay
// ============================================================================

export const histogramWithNormalExample = () => {
  // Generate sample data (approximately normal)
  const heights = [
    165, 168, 170, 172, 175, 168, 171, 169, 173, 176,
    167, 170, 172, 174, 169, 171, 173, 175, 170, 172,
    168, 171, 174, 176, 169, 172, 175, 177, 170, 173,
  ];

  const histogram = createHistogramWithNormal(heights, 'Height', {
    bins: 12,
    color: '#8B5CF6',
    title: 'Height Distribution with Normal Curve',
    xLabel: 'Height (cm)',
  });

  return histogram;
};

// ============================================================================
// Example 6: Custom Bin Edges (Grade Boundaries)
// ============================================================================

export const customBinHistogramExample = () => {
  const examScores = [
    45, 58, 62, 67, 71, 75, 78, 82, 85, 88, 91, 94,
    52, 65, 69, 73, 77, 81, 84, 87, 90, 93, 96, 98,
  ];

  // Define grade boundaries
  const gradeBoundaries = [0, 60, 70, 80, 90, 100];

  const histogram = createCustomBinHistogram(
    examScores,
    'Exam Score',
    gradeBoundaries,
    {
      color: '#F59E0B',
      title: 'Exam Score Distribution by Grade',
      xLabel: 'Score (F: 0-60, D: 60-70, C: 70-80, B: 80-90, A: 90-100)',
    }
  );

  return histogram;
};

// ============================================================================
// Example 7: Auto-Binned Histogram
// ============================================================================

export const autoBinnedHistogramExample = () => {
  const responseTime = [
    120, 145, 132, 158, 167, 141, 156, 149, 163, 172,
    138, 151, 164, 177, 143, 155, 168, 181, 147, 159,
    171, 185, 152, 165, 178, 191, 157, 169, 182, 195,
  ];

  // Using Freedman-Diaconis rule for optimal binning
  const histogram = createAutoBinnedHistogram(
    responseTime,
    'Response Time',
    'fd',
    {
      color: '#EC4899',
      title: 'API Response Time Distribution (Auto-binned)',
      xLabel: 'Response Time (ms)',
    }
  );

  return histogram;
};

// ============================================================================
// Example 8: Calculate Statistics
// ============================================================================

export const calculateStatsExample = () => {
  const data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

  const stats = calculateStats(data);

  console.log('Statistics:', {
    mean: stats.mean,      // 55
    median: stats.median,  // 55
    std: stats.std,        // ~28.72
    min: stats.min,        // 10
    max: stats.max,        // 100
    q1: stats.q1,          // 30
    q3: stats.q3,          // 80
    count: stats.count,    // 10
  });

  return stats;
};

// ============================================================================
// Example 9: Real-World Usage in Component
// ============================================================================

export const componentUsageExample = () => {
  /*
  import React, { useState, useEffect } from 'react';
  import { Box } from '@mui/material';
  import VisualizationGallery from '../components/dataset/VisualizationGallery';
  import { createMultipleHistograms } from '../utils/visualizations';

  function DatasetExplorationPage() {
    const [visualizations, setVisualizations] = useState([]);
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
      const fetchDataAndCreateVisualizations = async () => {
        setIsLoading(true);
        try {
          // Fetch dataset from API
          const response = await fetch('/api/v1/datasets/123/data');
          const dataset = await response.json();

          // Extract numeric columns
          const numericData = {
            age: dataset.data.map(row => row.age),
            income: dataset.data.map(row => row.income),
            score: dataset.data.map(row => row.score),
          };

          // Generate histograms
          const histograms = createMultipleHistograms(numericData, {
            bins: 15,
            showMean: true,
            showMedian: true,
          });

          setVisualizations(histograms);
        } catch (error) {
          console.error('Failed to create visualizations:', error);
        } finally {
          setIsLoading(false);
        }
      };

      fetchDataAndCreateVisualizations();
    }, []);

    return (
      <Box sx={{ p: 4 }}>
        <VisualizationGallery
          visualizations={visualizations}
          isLoading={isLoading}
          maxColumns={3}
        />
      </Box>
    );
  }
  */
};

// ============================================================================
// Example 10: Combining Multiple Visualization Types
// ============================================================================

export const combinedVisualizationsExample = () => {
  const ages = [25, 30, 35, 40, 45, 28, 32, 38, 42, 50, 27, 33, 36, 41, 48];
  const salaries = [50000, 60000, 70000, 80000, 90000, 55000, 65000, 75000, 85000, 95000, 52000, 68000, 73000, 83000, 93000];

  const visualizations = [
    // Basic histogram
    createHistogram(ages, 'Age', {
      bins: 10,
      color: '#3B82F6',
    }),

    // Histogram with stats
    createHistogram(salaries, 'Salary', {
      bins: 12,
      color: '#10B981',
      showMean: true,
      showMedian: true,
    }),

    // Auto-binned
    createAutoBinnedHistogram(ages, 'Age (Auto-binned)', 'fd', {
      color: '#8B5CF6',
    }),

    // With normal curve
    createHistogramWithNormal(salaries, 'Salary', {
      bins: 15,
      color: '#F59E0B',
    }),
  ];

  return visualizations;
};

// Export all examples
export const allExamples = {
  basic: basicHistogramExample,
  withStats: histogramWithStatsExample,
  multiple: multipleHistogramsExample,
  grouped: groupedHistogramExample,
  withNormal: histogramWithNormalExample,
  customBins: customBinHistogramExample,
  autoBinned: autoBinnedHistogramExample,
  stats: calculateStatsExample,
  combined: combinedVisualizationsExample,
};
