/**
 * Evaluation Dashboard Component
 * 
 * Comprehensive dashboard for model evaluation with metrics, charts, and feature importance
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Tabs,
  Tab,
  Alert,
  CircularProgress,
  Chip,
  Button,
  IconButton,
  Tooltip,
  Divider,
  Skeleton,
  AlertTitle,
} from '@mui/material';
import {
  Assessment as AssessmentIcon,
  TrendingUp as TrendingUpIcon,
  Timeline as TimelineIcon,
  BarChart as BarChartIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Info as InfoIcon,
  ErrorOutline as ErrorOutlineIcon,
  HourglassEmpty as HourglassEmptyIcon,
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import { evaluationService } from '../../services/evaluationService';
import type {
  ModelMetricsResponse,
  FeatureImportanceResponse,
} from '../../services/evaluationService';

interface EvaluationDashboardProps {
  modelRunId: string;
  onRefresh?: () => void;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`evaluation-tabpanel-${index}`}
      aria-labelledby={`evaluation-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

const EvaluationDashboard: React.FC<EvaluationDashboardProps> = ({
  modelRunId,
  onRefresh,
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [metricsData, setMetricsData] = useState<ModelMetricsResponse | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportanceResponse | null>(null);

  useEffect(() => {
    fetchEvaluationData();
  }, [modelRunId]);

  const fetchEvaluationData = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await evaluationService.getEvaluationData(modelRunId, {
        includeFeatureImportance: true,
        includePlots: false,
        topFeatures: 20,
      });

      setMetricsData(data.metrics);
      setFeatureImportance(data.featureImportance || null);
    } catch (err: any) {
      console.error('Error fetching evaluation data:', err);
      setError(err.response?.data?.detail || 'Failed to load evaluation data');
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    fetchEvaluationData();
    onRefresh?.();
  };

  const handleExport = async (format: 'json' | 'csv') => {
    try {
      const blob = await evaluationService.exportEvaluationData(modelRunId, format);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `evaluation-${modelRunId}.${format}`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Error exporting data:', err);
    }
  };

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const formatMetricValue = (value: number, decimals: number = 4): string => {
    if (typeof value !== 'number' || isNaN(value)) return 'N/A';
    return value.toFixed(decimals);
  };

  const renderMetricsCards = () => {
    if (!metricsData) return null;

    const metrics = metricsData.metrics;
    const isClassification = metricsData.task_type === 'classification';
    const isRegression = metricsData.task_type === 'regression';

    return (
      <Box display="flex" flexWrap="wrap" gap={2}>
        {isClassification && (
          <>
            <Box sx={{ flex: '1 1 calc(25% - 16px)', minWidth: '200px' }}>
              <Card sx={{ bgcolor: '#e3f2fd', height: '100%' }}>
                <CardContent>
                  <Typography variant="subtitle2" color="text.secondary">
                    Accuracy
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {formatMetricValue(metrics.accuracy * 100, 2)}%
                  </Typography>
                </CardContent>
              </Card>
            </Box>
            <Box sx={{ flex: '1 1 calc(25% - 16px)', minWidth: '200px' }}>
              <Card sx={{ bgcolor: '#f3e5f5', height: '100%' }}>
                <CardContent>
                  <Typography variant="subtitle2" color="text.secondary">
                    Precision
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {formatMetricValue(metrics.precision * 100, 2)}%
                  </Typography>
                </CardContent>
              </Card>
            </Box>
            <Box sx={{ flex: '1 1 calc(25% - 16px)', minWidth: '200px' }}>
              <Card sx={{ bgcolor: '#e8f5e9', height: '100%' }}>
                <CardContent>
                  <Typography variant="subtitle2" color="text.secondary">
                    Recall
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {formatMetricValue(metrics.recall * 100, 2)}%
                  </Typography>
                </CardContent>
              </Card>
            </Box>
            <Box sx={{ flex: '1 1 calc(25% - 16px)', minWidth: '200px' }}>
              <Card sx={{ bgcolor: '#fff3e0', height: '100%' }}>
                <CardContent>
                  <Typography variant="subtitle2" color="text.secondary">
                    F1 Score
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {formatMetricValue(metrics.f1_score * 100, 2)}%
                  </Typography>
                </CardContent>
              </Card>
            </Box>
          </>
        )}

        {isRegression && (
          <>
            <Box sx={{ flex: '1 1 calc(25% - 16px)', minWidth: '200px' }}>
              <Card sx={{ bgcolor: '#e3f2fd', height: '100%' }}>
                <CardContent>
                  <Typography variant="subtitle2" color="text.secondary">
                    R² Score
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {formatMetricValue(metrics.r2_score, 4)}
                  </Typography>
                </CardContent>
              </Card>
            </Box>
            <Box sx={{ flex: '1 1 calc(25% - 16px)', minWidth: '200px' }}>
              <Card sx={{ bgcolor: '#f3e5f5', height: '100%' }}>
                <CardContent>
                  <Typography variant="subtitle2" color="text.secondary">
                    RMSE
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {formatMetricValue(metrics.rmse, 4)}
                  </Typography>
                </CardContent>
              </Card>
            </Box>
            <Box sx={{ flex: '1 1 calc(25% - 16px)', minWidth: '200px' }}>
              <Card sx={{ bgcolor: '#e8f5e9', height: '100%' }}>
                <CardContent>
                  <Typography variant="subtitle2" color="text.secondary">
                    MAE
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {formatMetricValue(metrics.mae, 4)}
                  </Typography>
                </CardContent>
              </Card>
            </Box>
            <Box sx={{ flex: '1 1 calc(25% - 16px)', minWidth: '200px' }}>
              <Card sx={{ bgcolor: '#fff3e0', height: '100%' }}>
                <CardContent>
                  <Typography variant="subtitle2" color="text.secondary">
                    MSE
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {formatMetricValue(metrics.mse, 4)}
                  </Typography>
                </CardContent>
              </Card>
            </Box>
          </>
        )}

        {/* Training Metadata */}
        <Box sx={{ flex: '1 1 calc(25% - 16px)', minWidth: '200px' }}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary">
                Training Time
              </Typography>
              <Typography variant="h5" fontWeight="bold">
                {metricsData.training_metadata.training_time
                  ? `${Math.round(metricsData.training_metadata.training_time)}s`
                  : 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Box>
        <Box sx={{ flex: '1 1 calc(25% - 16px)', minWidth: '200px' }}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary">
                Training Samples
              </Typography>
              <Typography variant="h5" fontWeight="bold">
                {metricsData.training_metadata.train_samples || 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Box>
        <Box sx={{ flex: '1 1 calc(25% - 16px)', minWidth: '200px' }}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary">
                Test Samples
              </Typography>
              <Typography variant="h5" fontWeight="bold">
                {metricsData.training_metadata.test_samples || 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Box>
        <Box sx={{ flex: '1 1 calc(25% - 16px)', minWidth: '200px' }}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary">
                Features
              </Typography>
              <Typography variant="h5" fontWeight="bold">
                {metricsData.training_metadata.n_features || 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Box>
      </Box>
    );
  };

  const renderFeatureImportancePlot = () => {
    if (!featureImportance?.has_feature_importance || !featureImportance.top_features) {
      return (
        <Alert severity="info">
          Feature importance not available for this model type.
        </Alert>
      );
    }

    const features = featureImportance.top_features.map(f => f.feature);
    const importances = featureImportance.top_features.map(f => f.importance);

    return (
      <Plot
        data={[{
          type: 'bar',
          x: importances,
          y: features,
          orientation: 'h',
          marker: {
            color: importances,
            colorscale: 'Viridis',
          },
        }]}
        layout={{
          title: 'Feature Importance',
          xaxis: { title: 'Importance Score' },
          yaxis: { title: 'Features', automargin: true },
          height: 400,
          margin: { l: 150 },
        }}
        config={{ responsive: true, displayModeBar: true }}
        style={{ width: '100%' }}
      />
    );
  };

  const renderMetricsTable = () => {
    if (!metricsData) return null;

    const metrics = metricsData.metrics;
    const entries = Object.entries(metrics);

    return (
      <Box>
        <Typography variant="h6" gutterBottom>
          All Metrics
        </Typography>
        <Box display="flex" flexWrap="wrap" gap={1}>
          {entries.map(([key, value]) => (
            <Box key={key} sx={{ flex: '1 1 calc(33.333% - 8px)', minWidth: '200px' }}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">
                    {key.replace(/_/g, ' ').toUpperCase()}
                  </Typography>
                  <Typography variant="body1" fontWeight="medium">
                    {typeof value === 'number' ? formatMetricValue(value) : String(value)}
                  </Typography>
                </CardContent>
              </Card>
            </Box>
          ))}
        </Box>
      </Box>
    );
  };

  if (loading) {
    return (
      <Box>
        {/* Header Skeleton */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Box display="flex" alignItems="center" gap={2}>
            <Skeleton variant="circular" width={32} height={32} />
            <Box>
              <Skeleton variant="text" width={200} height={32} />
              <Skeleton variant="text" width={150} height={24} />
            </Box>
          </Box>
          <Box display="flex" gap={1}>
            <Skeleton variant="circular" width={40} height={40} />
            <Skeleton variant="circular" width={40} height={40} />
          </Box>
        </Box>

        {/* Metrics Cards Skeleton */}
        <Box display="flex" flexWrap="wrap" gap={2} mb={3}>
          {[1, 2, 3, 4].map((i) => (
            <Box key={i} sx={{ flex: '1 1 calc(25% - 16px)', minWidth: '200px' }}>
              <Card sx={{ height: '120px' }}>
                <CardContent>
                  <Skeleton variant="text" width="60%" height={20} />
                  <Skeleton variant="text" width="80%" height={40} sx={{ mt: 1 }} />
                </CardContent>
              </Card>
            </Box>
          ))}
        </Box>

        <Divider sx={{ my: 3 }} />

        {/* Chart Skeleton */}
        <Card>
          <CardContent>
            <Skeleton variant="rectangular" height={400} />
          </CardContent>
        </Card>
      </Box>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" flexDirection="column" alignItems="center" py={4}>
            <ErrorOutlineIcon color="error" sx={{ fontSize: 64, mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              Failed to Load Evaluation Data
            </Typography>
            <Typography variant="body2" color="text.secondary" textAlign="center" mb={3}>
              {error}
            </Typography>
            <Box display="flex" gap={2}>
              <Button
                variant="contained"
                startIcon={<RefreshIcon />}
                onClick={handleRefresh}
              >
                Retry
              </Button>
              <Button
                variant="outlined"
                onClick={() => window.location.reload()}
              >
                Reload Page
              </Button>
            </Box>
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (!metricsData) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" flexDirection="column" alignItems="center" py={6}>
            <HourglassEmptyIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              No Evaluation Data Available
            </Typography>
            <Typography variant="body2" color="text.secondary" textAlign="center" mb={3}>
              This model run doesn't have evaluation metrics yet.
              <br />
              Please wait for the training to complete or check back later.
            </Typography>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={handleRefresh}
            >
              Check Again
            </Button>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box display="flex" alignItems="center" gap={2}>
          <AssessmentIcon color="primary" sx={{ fontSize: 32 }} />
          <Box>
            <Typography variant="h5" fontWeight="bold">
              Model Evaluation
            </Typography>
            <Box display="flex" alignItems="center" gap={1}>
              <Chip
                label={metricsData.task_type}
                size="small"
                color="primary"
                variant="outlined"
              />
              <Chip
                label={metricsData.model_type}
                size="small"
                variant="outlined"
              />
            </Box>
          </Box>
        </Box>
        <Box display="flex" gap={1}>
          <Tooltip title="Export as JSON">
            <IconButton onClick={() => handleExport('json')} size="small">
              <DownloadIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Refresh">
            <IconButton onClick={handleRefresh} size="small">
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Metrics Cards */}
      {renderMetricsCards()}

      <Divider sx={{ my: 3 }} />

      {/* Tabs */}
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={handleTabChange}>
            <Tab label="Feature Importance" icon={<TrendingUpIcon />} iconPosition="start" />
            <Tab label="All Metrics" icon={<BarChartIcon />} iconPosition="start" />
            <Tab label="Training Info" icon={<InfoIcon />} iconPosition="start" />
          </Tabs>
        </Box>

        <CardContent>
          <TabPanel value={activeTab} index={0}>
            {renderFeatureImportancePlot()}
            
            {featureImportance?.has_feature_importance && (
              <Box mt={2}>
                <Typography variant="body2" color="text.secondary">
                  <strong>Method:</strong> {featureImportance.importance_method || 'Unknown'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  <strong>Total Features:</strong> {featureImportance.total_features}
                </Typography>
              </Box>
            )}
          </TabPanel>

          <TabPanel value={activeTab} index={1}>
            {renderMetricsTable()}
          </TabPanel>

          <TabPanel value={activeTab} index={2}>
            <Box display="flex" flexWrap="wrap" gap={2}>
              <Box sx={{ flex: '1 1 calc(50% - 16px)', minWidth: '300px' }}>
                <Typography variant="subtitle2" gutterBottom>
                  Training Configuration
                </Typography>
                <Box component="pre" sx={{ fontSize: '0.875rem', overflow: 'auto' }}>
                  {JSON.stringify(metricsData.training_metadata.hyperparameters, null, 2)}
                </Box>
              </Box>
              <Box sx={{ flex: '1 1 calc(50% - 16px)', minWidth: '300px' }}>
                <Typography variant="subtitle2" gutterBottom>
                  Model Information
                </Typography>
                <Typography variant="body2" paragraph>
                  <strong>Model Type:</strong> {metricsData.model_type}
                </Typography>
                <Typography variant="body2" paragraph>
                  <strong>Task Type:</strong> {metricsData.task_type}
                </Typography>
                <Typography variant="body2" paragraph>
                  <strong>Created:</strong> {new Date(metricsData.training_metadata.created_at).toLocaleString()}
                </Typography>
              </Box>
            </Box>
          </TabPanel>
        </CardContent>
      </Card>

      {/* Metadata Footer */}
      <Box mt={2}>
        <Typography variant="caption" color="text.secondary">
          Model Run ID: {metricsData.model_run_id}
        </Typography>
      </Box>
    </Box>
  );
};

export default EvaluationDashboard;
