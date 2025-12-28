/**
 * EvaluationPage
 *
 * Standalone evaluation screen with tabs for classification, regression, clustering, and comparison
 */

import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Tabs,
  Tab,
  Paper,
  Alert,
  Chip,
  Stack,
  Button,
  CircularProgress,
} from '@mui/material';
import {
  Assessment as AssessmentIcon,
  BarChart as BarChartIcon,
  ScatterPlot as ScatterPlotIcon,
  BubbleChart as BubbleChartIcon,
  CompareArrows as CompareArrowsIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import type {
  EvaluationTab,
  ModelRun,
  TaskType,
  EvaluationPageProps,
} from '../types/evaluation';
import { EvaluationTab as EvaluationTabEnum, TaskType as TaskTypeEnum } from '../types/evaluation';
import { MetricsDisplay } from '../components/evaluation/metrics';
import { ClassificationCharts, RegressionCharts } from '../components/evaluation/charts';

// Tab panel component
interface TabPanelProps {
  children?: React.ReactNode;
  value: EvaluationTab;
  currentValue: EvaluationTab;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, currentValue }) => {
  return (
    <Box
      role="tabpanel"
      hidden={value !== currentValue}
      id={`evaluation-tabpanel-${value}`}
      aria-labelledby={`evaluation-tab-${value}`}
    >
      {value === currentValue && <Box sx={{ py: 3 }}>{children}</Box>}
    </Box>
  );
};

const EvaluationPage: React.FC<EvaluationPageProps> = ({
  initialTab = EvaluationTabEnum.CLASSIFICATION,
  runId,
}) => {
  const [currentTab, setCurrentTab] = useState<EvaluationTab>(initialTab);
  const [runs, setRuns] = useState<ModelRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | undefined>(runId);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load runs from sessionStorage or API
  useEffect(() => {
    loadRuns();
  }, []);

  // Update tab when task type changes
  useEffect(() => {
    if (selectedRunId) {
      const selectedRun = runs.find((r) => r.id === selectedRunId);
      if (selectedRun) {
        // Auto-switch to appropriate tab based on task type
        const tabMap: Record<TaskType, EvaluationTab> = {
          [TaskTypeEnum.CLASSIFICATION]: EvaluationTabEnum.CLASSIFICATION,
          [TaskTypeEnum.REGRESSION]: EvaluationTabEnum.REGRESSION,
          [TaskTypeEnum.CLUSTERING]: EvaluationTabEnum.CLUSTERING,
        };
        const newTab = tabMap[selectedRun.taskType];
        if (newTab && newTab !== currentTab) {
          setCurrentTab(newTab);
        }
      }
    }
  }, [selectedRunId, runs]);

  const loadRuns = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // TODO: Replace with actual API call
      // const response = await api.get('/models/runs');
      // setRuns(response.data);

      // For now, load from sessionStorage or use mock data
      const savedRuns = sessionStorage.getItem('modelRuns');
      if (savedRuns) {
        setRuns(JSON.parse(savedRuns));
      } else {
        // Mock data for development
        setRuns(getMockRuns());
      }
    } catch (err) {
      setError('Failed to load model runs. Please try again.');
      console.error('Error loading runs:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTabChange = (_event: React.SyntheticEvent, newValue: EvaluationTab) => {
    setCurrentTab(newValue);
  };

  const handleRefresh = () => {
    loadRuns();
  };

  // Filter runs by task type for each tab
  const classificationRuns = runs.filter((r) => r.taskType === TaskTypeEnum.CLASSIFICATION);
  const regressionRuns = runs.filter((r) => r.taskType === TaskTypeEnum.REGRESSION);
  const clusteringRuns = runs.filter((r) => r.taskType === TaskTypeEnum.CLUSTERING);

  // Get icon for tab
  const getTabIcon = (tab: EvaluationTab) => {
    switch (tab) {
      case EvaluationTabEnum.CLASSIFICATION:
        return <BarChartIcon />;
      case EvaluationTabEnum.REGRESSION:
        return <ScatterPlotIcon />;
      case EvaluationTabEnum.CLUSTERING:
        return <BubbleChartIcon />;
      case EvaluationTabEnum.COMPARISON:
        return <CompareArrowsIcon />;
      default:
        return <AssessmentIcon />;
    }
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box mb={4} display="flex" justifyContent="space-between" alignItems="center">
        <Box>
          <Typography variant="h4" fontWeight="bold" gutterBottom>
            Model Evaluation
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Analyze and compare your trained models across different tasks
          </Typography>
        </Box>

        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={handleRefresh}
          disabled={isLoading}
        >
          Refresh
        </Button>
      </Box>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Loading State */}
      {isLoading && (
        <Box display="flex" justifyContent="center" alignItems="center" py={8}>
          <CircularProgress />
          <Typography variant="body1" color="text.secondary" sx={{ ml: 2 }}>
            Loading model runs...
          </Typography>
        </Box>
      )}

      {/* Main Content */}
      {!isLoading && (
        <Paper elevation={2}>
          {/* Tab Navigation */}
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs
              value={currentTab}
              onChange={handleTabChange}
              aria-label="evaluation tabs"
              variant="fullWidth"
              sx={{ px: 2 }}
            >
              <Tab
                value={EvaluationTabEnum.CLASSIFICATION}
                label={
                  <Stack direction="row" spacing={1} alignItems="center">
                    {getTabIcon(EvaluationTabEnum.CLASSIFICATION)}
                    <span>Classification</span>
                    {classificationRuns.length > 0 && (
                      <Chip label={classificationRuns.length} size="small" color="primary" />
                    )}
                  </Stack>
                }
                id="evaluation-tab-classification"
                aria-controls="evaluation-tabpanel-classification"
              />
              <Tab
                value={EvaluationTabEnum.REGRESSION}
                label={
                  <Stack direction="row" spacing={1} alignItems="center">
                    {getTabIcon(EvaluationTabEnum.REGRESSION)}
                    <span>Regression</span>
                    {regressionRuns.length > 0 && (
                      <Chip label={regressionRuns.length} size="small" color="primary" />
                    )}
                  </Stack>
                }
                id="evaluation-tab-regression"
                aria-controls="evaluation-tabpanel-regression"
              />
              <Tab
                value={EvaluationTabEnum.CLUSTERING}
                label={
                  <Stack direction="row" spacing={1} alignItems="center">
                    {getTabIcon(EvaluationTabEnum.CLUSTERING)}
                    <span>Clustering</span>
                    {clusteringRuns.length > 0 && (
                      <Chip label={clusteringRuns.length} size="small" color="primary" />
                    )}
                  </Stack>
                }
                id="evaluation-tab-clustering"
                aria-controls="evaluation-tabpanel-clustering"
              />
              <Tab
                value={EvaluationTabEnum.COMPARISON}
                label={
                  <Stack direction="row" spacing={1} alignItems="center">
                    {getTabIcon(EvaluationTabEnum.COMPARISON)}
                    <span>Comparison</span>
                  </Stack>
                }
                id="evaluation-tab-comparison"
                aria-controls="evaluation-tabpanel-comparison"
              />
            </Tabs>
          </Box>

          {/* Tab Panels */}
          <Box sx={{ p: 3 }}>
            {/* Classification Tab */}
            <TabPanel value={EvaluationTabEnum.CLASSIFICATION} currentValue={currentTab}>
              {classificationRuns.length === 0 ? (
                <Alert severity="info">
                  <Typography variant="body2">
                    No classification model runs found. Train a classification model to see evaluation
                    metrics here.
                  </Typography>
                </Alert>
              ) : (
                <Box>
                  {/* Metrics Display */}
                  <MetricsDisplay
                    taskType={TaskTypeEnum.CLASSIFICATION}
                    metrics={classificationRuns[0]?.metrics || null}
                    isLoading={isLoading}
                    showDescription={true}
                    compact={false}
                  />

                  {/* Classification Charts */}
                  {classificationRuns[0]?.metrics && (
                    <Box mt={4}>
                      <ClassificationCharts
                        metrics={classificationRuns[0].metrics as any}
                        isLoading={isLoading}
                      />
                    </Box>
                  )}
                </Box>
              )}
            </TabPanel>

            {/* Regression Tab */}
            <TabPanel value={EvaluationTabEnum.REGRESSION} currentValue={currentTab}>
              {regressionRuns.length === 0 ? (
                <Alert severity="info">
                  <Typography variant="body2">
                    No regression model runs found. Train a regression model to see evaluation metrics
                    here.
                  </Typography>
                </Alert>
              ) : (
                <Box>
                  {/* Metrics Display */}
                  <MetricsDisplay
                    taskType={TaskTypeEnum.REGRESSION}
                    metrics={regressionRuns[0]?.metrics || null}
                    isLoading={isLoading}
                    showDescription={true}
                    compact={false}
                  />

                  {/* Regression Charts */}
                  {regressionRuns[0]?.metrics && (
                    <Box mt={4}>
                      <RegressionCharts
                        metrics={regressionRuns[0].metrics as any}
                        isLoading={isLoading}
                      />
                    </Box>
                  )}
                </Box>
              )}
            </TabPanel>

            {/* Clustering Tab */}
            <TabPanel value={EvaluationTabEnum.CLUSTERING} currentValue={currentTab}>
              {clusteringRuns.length === 0 ? (
                <Alert severity="info">
                  <Typography variant="body2">
                    No clustering model runs found. Train a clustering model to see evaluation metrics
                    here.
                  </Typography>
                </Alert>
              ) : (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Clustering Evaluation
                  </Typography>
                  <Alert severity="info" sx={{ mt: 2 }}>
                    <Typography variant="body2">
                      <strong>Available metrics:</strong> Silhouette Score, Inertia, Davies-Bouldin Index,
                      Calinski-Harabasz Score
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      <strong>Visualizations:</strong> Cluster Distribution, Silhouette Plot, 2D Projection
                    </Typography>
                  </Alert>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                    Found {clusteringRuns.length} clustering run(s). Detailed metrics and visualizations
                    will be displayed here.
                  </Typography>
                </Box>
              )}
            </TabPanel>

            {/* Comparison Tab */}
            <TabPanel value={EvaluationTabEnum.COMPARISON} currentValue={currentTab}>
              {runs.length === 0 ? (
                <Alert severity="info">
                  <Typography variant="body2">
                    No model runs available for comparison. Train at least two models to compare their
                    performance.
                  </Typography>
                </Alert>
              ) : (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Model Comparison
                  </Typography>
                  <Alert severity="info" sx={{ mt: 2 }}>
                    <Typography variant="body2">
                      Compare multiple models side-by-side to identify the best performing model for your
                      task.
                    </Typography>
                  </Alert>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                    {runs.length} model run(s) available for comparison across all task types.
                  </Typography>
                </Box>
              )}
            </TabPanel>
          </Box>
        </Paper>
      )}

      {/* Debug Info (Development only) */}
      {import.meta.env.DEV && runs.length > 0 && (
        <Paper sx={{ mt: 4, p: 3, bgcolor: 'grey.100' }}>
          <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
            Debug: Model Runs State
          </Typography>
          <Box
            component="pre"
            sx={{
              fontSize: '0.75rem',
              overflow: 'auto',
              maxHeight: 300,
            }}
          >
            {JSON.stringify({ currentTab, selectedRunId, totalRuns: runs.length, runs }, null, 2)}
          </Box>
        </Paper>
      )}
    </Container>
  );
};

// Mock data for development
const getMockRuns = (): ModelRun[] => {
  return [
    {
      id: 'run-1',
      name: 'Random Forest Classifier - Iris',
      modelId: 'random_forest_classifier',
      modelName: 'Random Forest Classifier',
      taskType: 'classification' as TaskType,
      datasetId: 'dataset-1',
      datasetName: 'Iris Dataset',
      status: 'completed',
      createdAt: '2025-12-27T10:00:00Z',
      completedAt: '2025-12-27T10:05:00Z',
      hyperparameters: {
        n_estimators: 100,
        max_depth: 10,
        min_samples_split: 2,
      },
      metrics: {
        accuracy: 0.96,
        precision: 0.95,
        recall: 0.96,
        f1Score: 0.95,
        auc: 0.98,
        confusionMatrix: [
          [45, 3, 2],
          [1, 48, 1],
          [2, 1, 47],
        ],
        classNames: ['Setosa', 'Versicolor', 'Virginica'],
        rocCurve: {
          fpr: Array.from({ length: 50 }, (_, i) => i / 49),
          tpr: Array.from({ length: 50 }, (_, i) => Math.min(1, (i / 49) * 1.1 + Math.random() * 0.05)),
          thresholds: Array.from({ length: 50 }, (_, i) => 1 - i / 49),
        },
        prCurve: {
          precision: Array.from({ length: 50 }, (_, i) => Math.max(0.8, 1 - (i / 49) * 0.2 + Math.random() * 0.05)),
          recall: Array.from({ length: 50 }, (_, i) => 1 - i / 49),
          thresholds: Array.from({ length: 50 }, (_, i) => 1 - i / 49),
        },
      },
    },
    {
      id: 'run-2',
      name: 'Linear Regression - Housing',
      modelId: 'linear_regression',
      modelName: 'Linear Regression',
      taskType: 'regression' as TaskType,
      datasetId: 'dataset-2',
      datasetName: 'Housing Prices',
      status: 'completed',
      createdAt: '2025-12-27T11:00:00Z',
      completedAt: '2025-12-27T11:03:00Z',
      hyperparameters: {
        fit_intercept: true,
        normalize: false,
      },
      metrics: {
        mae: 3421.5,
        mse: 15678234.2,
        rmse: 3959.5,
        r2: 0.82,
        mape: 12.3,
        // Generate realistic predicted and actual values for housing prices
        predicted: Array.from({ length: 100 }, (_, i) => {
          const base = 200000 + i * 5000;
          const noise = (Math.random() - 0.5) * 10000;
          return base + noise;
        }),
        actual: Array.from({ length: 100 }, (_, i) => {
          const base = 200000 + i * 5000;
          const noise = (Math.random() - 0.5) * 8000;
          const trend = i * 100; // Slight upward trend
          return base + noise + trend;
        }),
      },
    },
    {
      id: 'run-3',
      name: 'K-Means - Customer Segments',
      modelId: 'kmeans',
      modelName: 'K-Means Clustering',
      taskType: 'clustering' as TaskType,
      datasetId: 'dataset-3',
      datasetName: 'Customer Data',
      status: 'completed',
      createdAt: '2025-12-27T12:00:00Z',
      completedAt: '2025-12-27T12:04:00Z',
      hyperparameters: {
        n_clusters: 5,
        max_iter: 300,
        n_init: 10,
      },
      metrics: {
        silhouetteScore: 0.68,
        inertia: 1234.5,
        nClusters: 5,
      },
    },
  ];
};

export default EvaluationPage;
