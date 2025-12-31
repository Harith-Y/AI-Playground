/**
 * Enhanced Model Comparison View Component
 * 
 * Comprehensive comparison UI with metrics tables, charts, statistics, and recommendations
 * Integrates with backend /api/v1/models/compare endpoint
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tab,
  IconButton,
  Tooltip,
  Alert as _Alert,
  Paper as _Paper,
  Divider,
  Button,
  CircularProgress as _CircularProgress,
  Skeleton,
  AlertTitle as _AlertTitle,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  EmojiEvents as TrophyIcon,
  BarChart as BarChartIcon,
  Timeline as TimelineIcon,
  Radar as RadarIcon,
  ScatterPlot as ScatterPlotIcon,
  Refresh as RefreshIcon,
  Download as _DownloadIcon,
  Lightbulb as LightbulbIcon,
  ErrorOutline as ErrorOutlineIcon,
  CompareArrows as CompareArrowsIcon,
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import type {
  ModelComparisonResponse,
  CompareModelsRequest,
  ModelComparisonItem,
  MetricStatistics,
} from '../../types/modelComparison';
import { modelComparisonService } from '../../services/modelComparisonService';

interface ModelComparisonViewEnhancedProps {
  modelRunIds: string[];
  comparisonMetrics?: string[];
  rankingCriteria?: string;
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
      id={`comparison-tabpanel-${index}`}
      aria-labelledby={`comparison-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const ModelComparisonViewEnhanced: React.FC<ModelComparisonViewEnhancedProps> = ({
  modelRunIds,
  comparisonMetrics,
  rankingCriteria,
  onRefresh,
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [comparisonData, setComparisonData] = useState<ModelComparisonResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (modelRunIds.length >= 2) {
      fetchComparison();
    }
  }, [modelRunIds, comparisonMetrics, rankingCriteria]);

  const fetchComparison = async () => {
    setLoading(true);
    setError(null);

    try {
      const request: CompareModelsRequest = {
        model_run_ids: modelRunIds,
        comparison_metrics: comparisonMetrics,
        ranking_criteria: rankingCriteria,
        include_statistical_tests: false,
      };

      const result = await modelComparisonService.compareModels(request);
      setComparisonData(result);
    } catch (err: any) {
      console.error('Error fetching comparison:', err);
      setError(err.response?.data?.detail || 'Failed to compare models');
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleRefresh = () => {
    fetchComparison();
    onRefresh?.();
  };

  const formatMetricValue = (value: number | undefined, decimals: number = 4) => {
    if (value === undefined || value === null) return 'N/A';
    return value.toFixed(decimals);
  };

  const _getModelTypeColor = (type: string): any => {
    const colors: Record<string, any> = {
      random_forest: 'primary',
      logistic_regression: 'secondary',
      svm: 'success',
      neural_network: 'warning',
      gradient_boosting: 'info',
      decision_tree: 'error',
      linear_regression: 'secondary',
    };
    
    // Match partial strings
    for (const [key, color] of Object.entries(colors)) {
      if (type.toLowerCase().includes(key)) {
        return color;
      }
    }
    
    return 'default';
  };

  const getBestWorstIndicator = (
    stats: MetricStatistics | undefined,
    modelId: string
  ) => {
    if (!stats) return null;
    
    if (stats.best_model_id === modelId) {
      return <TrendingUpIcon color="success" fontSize="small" />;
    } else if (stats.worst_model_id === modelId) {
      return <TrendingDownIcon color="error" fontSize="small" />;
    }
    
    return null;
  };

  // Create metrics comparison bar chart
  const createMetricsChart = () => {
    if (!comparisonData) return null;

    const metrics = comparisonData.metric_statistics.map(s => s.metric_name);
    const traces = comparisonData.compared_models.map((model, index) => ({
      x: metrics,
      y: metrics.map(metric => model.metrics[metric] || 0),
      name: model.model_type,
      type: 'bar' as const,
      marker: {
        color: `hsl(${index * 60}, 70%, 50%)`,
      },
    }));

    return (
      <Plot
        data={traces}
        layout={{
          title: 'Model Performance Comparison',
          xaxis: { title: 'Metrics' },
          yaxis: { title: 'Score' },
          barmode: 'group',
          height: 450,
          showlegend: true,
        }}
        config={{ responsive: true, displayModeBar: true }}
        style={{ width: '100%' }}
      />
    );
  };

  // Create radar chart for comprehensive comparison
  const createRadarChart = () => {
    if (!comparisonData) return null;

    const metrics = comparisonData.metric_statistics.map(s => s.metric_name);
    const traces = comparisonData.compared_models.map((model, index) => ({
      type: 'scatterpolar' as const,
      r: metrics.map(metric => model.metrics[metric] || 0),
      theta: metrics,
      fill: 'toself' as const,
      name: model.model_type,
      line: { color: `hsl(${index * 60}, 70%, 50%)` },
    }));

    return (
      <Plot
        data={traces}
        layout={{
          title: 'Model Performance Radar',
          polar: {
            radialaxis: {
              visible: true,
              range: [0, 1],
            },
          },
          height: 500,
          showlegend: true,
        }}
        config={{ responsive: true, displayModeBar: true }}
        style={{ width: '100%' }}
      />
    );
  };

  // Create training time vs performance scatter plot
  const createScatterChart = () => {
    if (!comparisonData) return null;

    // Use ranking score as performance metric
    const trace = {
      x: comparisonData.compared_models.map(m => m.training_time || 0),
      y: comparisonData.compared_models.map(m => m.ranking_score || 0),
      mode: 'markers+text' as const,
      type: 'scatter' as const,
      text: comparisonData.compared_models.map(m => m.model_type),
      textposition: 'top center' as const,
      marker: {
        size: 12,
        color: comparisonData.compared_models.map((_, i) => `hsl(${i * 60}, 70%, 50%)`),
      },
    };

    return (
      <Plot
        data={[trace]}
        layout={{
          title: 'Training Time vs Performance',
          xaxis: { title: 'Training Time (seconds)' },
          yaxis: { title: 'Ranking Score' },
          height: 450,
        }}
        config={{ responsive: true, displayModeBar: true }}
        style={{ width: '100%' }}
      />
    );
  };

  // Create statistical box plot
  const _createStatisticsChart = () => {
    if (!comparisonData) return null;

    const traces = comparisonData.metric_statistics.map(stat => ({
      y: [stat.min, stat.mean, stat.max],
      type: 'box' as const,
      name: stat.metric_name,
      boxmean: 'sd' as const,
    }));

    return (
      <Plot
        data={traces}
        layout={{
          title: 'Metric Statistics Distribution',
          yaxis: { title: 'Score' },
          height: 400,
          showlegend: true,
        }}
        config={{ responsive: true, displayModeBar: true }}
        style={{ width: '100%' }}
      />
    );
  };

  if (modelRunIds.length < 2) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" flexDirection="column" alignItems="center" py={6}>
            <CompareArrowsIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              Select Models to Compare
            </Typography>
            <Typography variant="body2" color="text.secondary" textAlign="center" mb={2}>
              Please select at least 2 models to see a detailed comparison
              <br />
              of their performance metrics and characteristics.
            </Typography>
            <Chip label={`${modelRunIds.length} model${modelRunIds.length === 1 ? '' : 's'} selected`} color="primary" />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (loading) {
    return (
      <Box>
        {/* Best Model Banner Skeleton */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box display="flex" alignItems="center" gap={2} mb={3}>
              <Skeleton variant="circular" width={40} height={40} />
              <Box flex={1}>
                <Skeleton variant="text" width="40%" height={32} />
                <Skeleton variant="text" width="30%" height={24} />
              </Box>
            </Box>
            <Box display="flex" flexWrap="wrap" gap={2}>
              {[1, 2, 3, 4].map((i) => (
                <Box key={i} sx={{ flex: '1 1 calc(25% - 16px)', minWidth: '150px' }}>
                  <Skeleton variant="text" width="60%" />
                  <Skeleton variant="text" width="80%" height={32} />
                </Box>
              ))}
            </Box>
          </CardContent>
        </Card>

        {/* Statistics Cards Skeleton */}
        <Skeleton variant="text" width="30%" height={32} sx={{ mb: 2 }} />
        <Box display="flex" flexWrap="wrap" gap={2} mb={3}>
          {[1, 2, 3].map((i) => (
            <Box key={i} sx={{ flex: '1 1 calc(33.333% - 16px)', minWidth: '250px' }}>
              <Card sx={{ height: '150px' }}>
                <CardContent>
                  <Skeleton variant="text" width="70%" />
                  <Skeleton variant="rectangular" height={80} sx={{ mt: 1 }} />
                </CardContent>
              </Card>
            </Box>
          ))}
        </Box>

        {/* Chart Skeleton */}
        <Card>
          <CardContent>
            <Skeleton variant="rectangular" height={450} />
          </CardContent>
        </Card>
      </Box>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" flexDirection="column" alignItems="center" py={6}>
            <ErrorOutlineIcon color="error" sx={{ fontSize: 64, mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              Comparison Failed
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
                Try Again
              </Button>
              <Button
                variant="outlined"
                onClick={() => window.location.reload()}
              >
                Reload Page
              </Button>
            </Box>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 2 }}>
              Comparing {modelRunIds.length} models
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (!comparisonData) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" flexDirection="column" alignItems="center" py={6}>
            <CompareArrowsIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              No Comparison Data
            </Typography>
            <Typography variant="body2" color="text.secondary" textAlign="center" mb={3}>
              Unable to retrieve comparison data for the selected models.
              <br />
              Please try refreshing or select different models.
            </Typography>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={handleRefresh}
            >
              Refresh
            </Button>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box>
      {/* Best Model Banner */}
      <Card
        sx={{
          mb: 3,
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
        }}
      >
        <CardContent>
          <Box display="flex" alignItems="center" gap={2} mb={2}>
            <TrophyIcon sx={{ fontSize: 40 }} />
            <Box>
              <Typography variant="h5" fontWeight="bold">
                Best Model
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                Based on {comparisonData.ranking_criteria}
              </Typography>
            </Box>
          </Box>
          
          <Box display="flex" flexWrap="wrap" gap={2}>
            <Box sx={{ flex: '1 1 calc(25% - 16px)', minWidth: '150px' }}>
              <Typography variant="subtitle2" sx={{ opacity: 0.8 }}>
                Model Type
              </Typography>
              <Typography variant="h6">
                {comparisonData.best_model.model_type}
              </Typography>
            </Box>
            <Box sx={{ flex: '1 1 calc(25% - 16px)', minWidth: '150px' }}>
              <Typography variant="subtitle2" sx={{ opacity: 0.8 }}>
                Ranking Score
              </Typography>
              <Typography variant="h6">
                {formatMetricValue(comparisonData.best_model.ranking_score)}
              </Typography>
            </Box>
            <Box sx={{ flex: '1 1 calc(25% - 16px)', minWidth: '150px' }}>
              <Typography variant="subtitle2" sx={{ opacity: 0.8 }}>
                Training Time
              </Typography>
              <Typography variant="h6">
                {comparisonData.best_model.training_time
                  ? `${Math.round(comparisonData.best_model.training_time)}s`
                  : 'N/A'}
              </Typography>
            </Box>
            <Box sx={{ flex: '1 1 calc(25% - 16px)', minWidth: '150px' }}>
              <Typography variant="subtitle2" sx={{ opacity: 0.8 }}>
                Rank
              </Typography>
              <Typography variant="h6">
                #{comparisonData.best_model.rank} of {comparisonData.total_models}
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* Recommendations */}
      {comparisonData.recommendations.length > 0 && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box display="flex" alignItems="center" gap={1} mb={2}>
              <LightbulbIcon color="warning" />
              <Typography variant="h6">
                Recommendations
              </Typography>
            </Box>
            <Box component="ul" sx={{ pl: 2, mb: 0 }}>
              {comparisonData.recommendations.map((rec, index) => (
                <Typography component="li" key={index} variant="body2" gutterBottom>
                  {rec}
                </Typography>
              ))}
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Statistics Summary Cards */}
      <Typography variant="h6" gutterBottom>
        Metric Statistics
      </Typography>
      <Box display="flex" flexWrap="wrap" gap={2} sx={{ mb: 3 }}>
        {comparisonData.metric_statistics.map((stat) => (
          <Box key={stat.metric_name} sx={{ flex: '1 1 calc(33.333% - 16px)', minWidth: '250px' }}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="subtitle2" color="primary" gutterBottom>
                  {stat.metric_name.replace(/_/g, ' ').toUpperCase()}
                </Typography>
                <Box display="flex" flexWrap="wrap" gap={1}>
                  <Box sx={{ flex: '1 1 calc(50% - 8px)', minWidth: '80px' }}>
                    <Typography variant="caption" color="text.secondary">
                      Mean
                    </Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {formatMetricValue(stat.mean)}
                    </Typography>
                  </Box>
                  <Box sx={{ flex: '1 1 calc(50% - 8px)', minWidth: '80px' }}>
                    <Typography variant="caption" color="text.secondary">
                      Std Dev
                    </Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {formatMetricValue(stat.std)}
                    </Typography>
                  </Box>
                  <Box sx={{ flex: '1 1 calc(50% - 8px)', minWidth: '80px' }}>
                    <Typography variant="caption" color="text.secondary">
                      Min
                    </Typography>
                    <Typography variant="body2">
                      {formatMetricValue(stat.min)}
                    </Typography>
                  </Box>
                  <Box sx={{ flex: '1 1 calc(50% - 8px)', minWidth: '80px' }}>
                    <Typography variant="caption" color="text.secondary">
                      Max
                    </Typography>
                    <Typography variant="body2">
                      {formatMetricValue(stat.max)}
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Box>
        ))}
      </Box>

      {/* Comparison Tabs */}
      <Card>
        <Box
          sx={{
            borderBottom: 1,
            borderColor: 'divider',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            pr: 2,
          }}
        >
          <Tabs value={activeTab} onChange={handleTabChange}>
            <Tab label="Metrics Table" icon={<BarChartIcon />} iconPosition="start" />
            <Tab label="Bar Chart" icon={<TimelineIcon />} iconPosition="start" />
            <Tab label="Radar Chart" icon={<RadarIcon />} iconPosition="start" />
            <Tab label="Time vs Performance" icon={<ScatterPlotIcon />} iconPosition="start" />
          </Tabs>
          <Tooltip title="Refresh comparison">
            <IconButton onClick={handleRefresh} size="small">
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Metrics Table Tab */}
        <TabPanel value={activeTab} index={0}>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Metric</strong></TableCell>
                  {comparisonData.compared_models.map(model => (
                    <TableCell key={model.model_run_id} align="center">
                      <Box>
                        <Typography variant="body2" fontWeight="bold">
                          {model.model_type}
                        </Typography>
                        <Chip
                          label={`Rank #${model.rank}`}
                          size="small"
                          color={model.rank === 1 ? 'success' : 'default'}
                          sx={{ mt: 0.5 }}
                        />
                      </Box>
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {comparisonData.metric_statistics.map((stat) => (
                  <TableRow key={stat.metric_name}>
                    <TableCell>
                      <Typography variant="body2" fontWeight="500">
                        {stat.metric_name.replace(/_/g, ' ')}
                      </Typography>
                    </TableCell>
                    {comparisonData.compared_models.map((model) => (
                      <TableCell key={model.model_run_id} align="center">
                        <Box display="flex" alignItems="center" justifyContent="center" gap={1}>
                          <Typography variant="body2">
                            {formatMetricValue(model.metrics[stat.metric_name])}
                          </Typography>
                          {getBestWorstIndicator(stat, model.model_run_id)}
                        </Box>
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
                
                {/* Training Information */}
                <TableRow>
                  <TableCell colSpan={comparisonData.compared_models.length + 1}>
                    <Divider sx={{ my: 1 }}>
                      <Chip label="Training Information" size="small" />
                    </Divider>
                  </TableCell>
                </TableRow>
                
                <TableRow>
                  <TableCell><strong>Training Time</strong></TableCell>
                  {comparisonData.compared_models.map(model => (
                    <TableCell key={model.model_run_id} align="center">
                      {model.training_time
                        ? `${Math.round(model.training_time)}s`
                        : 'N/A'}
                    </TableCell>
                  ))}
                </TableRow>
                
                <TableRow>
                  <TableCell><strong>Status</strong></TableCell>
                  {comparisonData.compared_models.map(model => (
                    <TableCell key={model.model_run_id} align="center">
                      <Chip
                        label={model.status}
                        color={model.status === 'completed' ? 'success' : 'default'}
                        size="small"
                      />
                    </TableCell>
                  ))}
                </TableRow>
                
                <TableRow>
                  <TableCell><strong>Created</strong></TableCell>
                  {comparisonData.compared_models.map(model => (
                    <TableCell key={model.model_run_id} align="center">
                      <Typography variant="caption">
                        {new Date(model.created_at).toLocaleDateString()}
                      </Typography>
                    </TableCell>
                  ))}
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>

        {/* Bar Chart Tab */}
        <TabPanel value={activeTab} index={1}>
          {createMetricsChart()}
        </TabPanel>

        {/* Radar Chart Tab */}
        <TabPanel value={activeTab} index={2}>
          {createRadarChart()}
        </TabPanel>

        {/* Scatter Chart Tab */}
        <TabPanel value={activeTab} index={3}>
          {createScatterChart()}
        </TabPanel>
      </Card>

      {/* Metadata */}
      <Box mt={2} display="flex" justifyContent="space-between" alignItems="center">
        <Typography variant="caption" color="text.secondary">
          Comparison ID: {comparisonData.comparison_id} | 
          Task Type: {comparisonData.task_type} | 
          Timestamp: {new Date(comparisonData.timestamp).toLocaleString()}
        </Typography>
      </Box>
    </Box>
  );
};

export default ModelComparisonViewEnhanced;
