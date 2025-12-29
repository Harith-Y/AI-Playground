import React, { useState } from 'react';
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
  Alert,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Remove as RemoveIcon,
  BarChart as BarChartIcon,
  Timeline as TimelineIcon,
  Radar as RadarIcon,
  ScatterPlot as ScatterPlotIcon,
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import type { ModelComparisonData } from '../../types/modelComparison';

interface ModelComparisonViewProps {
  models: ModelComparisonData[];
  onRemoveModel?: (modelId: string) => void;
  maxModels?: number;
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

const ModelComparisonView: React.FC<ModelComparisonViewProps> = ({
  models,
  onRemoveModel,
  maxModels = 4,
}) => {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const formatMetricValue = (value: number | undefined, isPercentage = false) => {
    if (value === undefined) return 'N/A';
    if (isPercentage) {
      return `${(value * 100).toFixed(2)}%`;
    }
    return value.toFixed(4);
  };

  const getModelTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      'random_forest': 'primary',
      'logistic_regression': 'secondary',
      'svm': 'success',
      'neural_network': 'warning',
      'gradient_boosting': 'info',
      'decision_tree': 'error',
    };
    return colors[type] || 'default';
  };

  const getBestWorstIndicator = (values: (number | undefined)[], index: number, higherIsBetter = true) => {
    const validValues = values.filter(v => v !== undefined) as number[];
    if (validValues.length === 0 || values[index] === undefined) return null;
    
    const currentValue = values[index] as number;
    const bestValue = higherIsBetter ? Math.max(...validValues) : Math.min(...validValues);
    const worstValue = higherIsBetter ? Math.min(...validValues) : Math.max(...validValues);
    
    if (currentValue === bestValue && validValues.length > 1) {
      return <TrendingUpIcon color="success" fontSize="small" />;
    } else if (currentValue === worstValue && validValues.length > 1) {
      return <TrendingDownIcon color="error" fontSize="small" />;
    }
    return null;
  };

  // Metrics comparison chart
  const createMetricsChart = () => {
    if (models.length === 0) return null;

    const metrics = ['accuracy', 'precision', 'recall', 'f1Score'];
    const traces = metrics.map(metric => ({
      x: models.map(m => m.name),
      y: models.map(m => m.metrics[metric] || 0),
      name: metric.charAt(0).toUpperCase() + metric.slice(1),
      type: 'bar' as const,
    }));

    return (
      <Plot
        data={traces}
        layout={{
          title: 'Model Performance Comparison',
          xaxis: { title: 'Models' },
          yaxis: { title: 'Score', range: [0, 1] },
          barmode: 'group',
          height: 400,
        }}
        config={{ responsive: true }}
        style={{ width: '100%' }}
      />
    );
  };

  // Radar chart for comprehensive comparison
  const createRadarChart = () => {
    if (models.length === 0) return null;

    const metrics = ['accuracy', 'precision', 'recall', 'f1Score'];
    const traces = models.map((model, index) => ({
      type: 'scatterpolar' as const,
      r: metrics.map(metric => model.metrics[metric] || 0),
      theta: metrics.map(m => m.charAt(0).toUpperCase() + m.slice(1)),
      fill: 'toself' as const,
      name: model.name,
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
        }}
        config={{ responsive: true }}
        style={{ width: '100%' }}
      />
    );
  };

  // Training time vs performance scatter plot
  const createScatterChart = () => {
    if (models.length === 0) return null;

    const trace = {
      x: models.map(m => m.trainingTime || 0),
      y: models.map(m => m.metrics.accuracy || m.metrics.r2Score || 0),
      mode: 'markers+text' as const,
      type: 'scatter' as const,
      text: models.map(m => m.name),
      textposition: 'top center' as const,
      marker: {
        size: 12,
        color: models.map((_, i) => `hsl(${i * 60}, 70%, 50%)`),
      },
    };

    return (
      <Plot
        data={[trace]}
        layout={{
          title: 'Training Time vs Performance',
          xaxis: { title: 'Training Time (seconds)' },
          yaxis: { title: 'Performance Score' },
          height: 400,
        }}
        config={{ responsive: true }}
        style={{ width: '100%' }}
      />
    );
  };

  if (models.length === 0) {
    return (
      <Alert severity="info">
        No models selected for comparison. Please select models from the list above.
      </Alert>
    );
  }

  if (models.length > maxModels) {
    return (
      <Alert severity="warning">
        Too many models selected. Please select at most {maxModels} models for comparison.
      </Alert>
    );
  }

  return (
    <Box>
      {/* Model Cards Overview */}
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: {
            xs: '1fr',
            sm: 'repeat(2, 1fr)',
            md: 'repeat(3, 1fr)',
            lg: 'repeat(4, 1fr)',
          },
          gap: 2,
          mb: 3,
        }}
      >
        {models.map((model) => (
          <Card key={model.id}>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={1}>
                <Typography variant="h6" noWrap>
                  {model.name}
                </Typography>
                {onRemoveModel && (
                  <Tooltip title="Remove from comparison">
                    <IconButton
                      size="small"
                      onClick={() => onRemoveModel(model.id)}
                    >
                      <RemoveIcon />
                    </IconButton>
                  </Tooltip>
                )}
              </Box>
              
              <Chip
                label={model.type}
                color={getModelTypeColor(model.type) as any}
                size="small"
                sx={{ mb: 1 }}
              />
              
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Dataset: {model.datasetName}
              </Typography>
              
              <Box mt={2}>
                <Typography variant="body2" color="text.secondary">
                  Best Metric
                </Typography>
                <Typography variant="h5">
                  {formatMetricValue(
                    model.metrics.accuracy || model.metrics.r2Score || model.metrics.f1Score,
                    true
                  )}
                </Typography>
              </Box>
              
              {model.trainingTime && (
                <Typography variant="body2" color="text.secondary" mt={1}>
                  Training: {Math.round(model.trainingTime)}s
                </Typography>
              )}
            </CardContent>
          </Card>
        ))}
      </Box>

      {/* Comparison Tabs */}
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={handleTabChange}>
            <Tab
              label="Metrics Table"
              icon={<BarChartIcon />}
              iconPosition="start"
            />
            <Tab
              label="Performance Charts"
              icon={<TimelineIcon />}
              iconPosition="start"
            />
            <Tab
              label="Radar Comparison"
              icon={<RadarIcon />}
              iconPosition="start"
            />
            <Tab
              label="Time vs Performance"
              icon={<ScatterPlotIcon />}
              iconPosition="start"
            />
          </Tabs>
        </Box>

        {/* Metrics Table Tab */}
        <TabPanel value={activeTab} index={0}>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Metric</strong></TableCell>
                  {models.map(model => (
                    <TableCell key={model.id} align="center">
                      <strong>{model.name}</strong>
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {/* Classification Metrics */}
                {models.some(m => m.metrics.accuracy !== undefined) && (
                  <>
                    <TableRow>
                      <TableCell colSpan={models.length + 1}>
                        <Typography variant="subtitle2" color="primary">
                          Classification Metrics
                        </Typography>
                      </TableCell>
                    </TableRow>
                    
                    <TableRow>
                      <TableCell>Accuracy</TableCell>
                      {models.map((model, index) => (
                        <TableCell key={model.id} align="center">
                          <Box display="flex" alignItems="center" justifyContent="center" gap={1}>
                            {formatMetricValue(model.metrics.accuracy, true)}
                            {getBestWorstIndicator(models.map(m => m.metrics.accuracy), index)}
                          </Box>
                        </TableCell>
                      ))}
                    </TableRow>
                    
                    <TableRow>
                      <TableCell>Precision</TableCell>
                      {models.map((model, index) => (
                        <TableCell key={model.id} align="center">
                          <Box display="flex" alignItems="center" justifyContent="center" gap={1}>
                            {formatMetricValue(model.metrics.precision, true)}
                            {getBestWorstIndicator(models.map(m => m.metrics.precision), index)}
                          </Box>
                        </TableCell>
                      ))}
                    </TableRow>
                    
                    <TableRow>
                      <TableCell>Recall</TableCell>
                      {models.map((model, index) => (
                        <TableCell key={model.id} align="center">
                          <Box display="flex" alignItems="center" justifyContent="center" gap={1}>
                            {formatMetricValue(model.metrics.recall, true)}
                            {getBestWorstIndicator(models.map(m => m.metrics.recall), index)}
                          </Box>
                        </TableCell>
                      ))}
                    </TableRow>
                    
                    <TableRow>
                      <TableCell>F1 Score</TableCell>
                      {models.map((model, index) => (
                        <TableCell key={model.id} align="center">
                          <Box display="flex" alignItems="center" justifyContent="center" gap={1}>
                            {formatMetricValue(model.metrics.f1Score, true)}
                            {getBestWorstIndicator(models.map(m => m.metrics.f1Score), index)}
                          </Box>
                        </TableCell>
                      ))}
                    </TableRow>
                  </>
                )}
                
                {/* Regression Metrics */}
                {models.some(m => m.metrics.rmse !== undefined) && (
                  <>
                    <TableRow>
                      <TableCell colSpan={models.length + 1}>
                        <Typography variant="subtitle2" color="primary">
                          Regression Metrics
                        </Typography>
                      </TableCell>
                    </TableRow>
                    
                    <TableRow>
                      <TableCell>R² Score</TableCell>
                      {models.map((model, index) => (
                        <TableCell key={model.id} align="center">
                          <Box display="flex" alignItems="center" justifyContent="center" gap={1}>
                            {formatMetricValue(model.metrics.r2Score)}
                            {getBestWorstIndicator(models.map(m => m.metrics.r2Score), index)}
                          </Box>
                        </TableCell>
                      ))}
                    </TableRow>
                    
                    <TableRow>
                      <TableCell>RMSE</TableCell>
                      {models.map((model, index) => (
                        <TableCell key={model.id} align="center">
                          <Box display="flex" alignItems="center" justifyContent="center" gap={1}>
                            {formatMetricValue(model.metrics.rmse)}
                            {getBestWorstIndicator(models.map(m => m.metrics.rmse), index, false)}
                          </Box>
                        </TableCell>
                      ))}
                    </TableRow>
                    
                    <TableRow>
                      <TableCell>MAE</TableCell>
                      {models.map((model, index) => (
                        <TableCell key={model.id} align="center">
                          <Box display="flex" alignItems="center" justifyContent="center" gap={1}>
                            {formatMetricValue(model.metrics.mae)}
                            {getBestWorstIndicator(models.map(m => m.metrics.mae), index, false)}
                          </Box>
                        </TableCell>
                      ))}
                    </TableRow>
                  </>
                )}
                
                {/* Training Metrics */}
                <TableRow>
                  <TableCell colSpan={models.length + 1}>
                    <Typography variant="subtitle2" color="primary">
                      Training Information
                    </Typography>
                  </TableCell>
                </TableRow>
                
                <TableRow>
                  <TableCell>Training Time</TableCell>
                  {models.map((model, index) => (
                    <TableCell key={model.id} align="center">
                      <Box display="flex" alignItems="center" justifyContent="center" gap={1}>
                        {model.trainingTime ? `${Math.round(model.trainingTime)}s` : 'N/A'}
                        {getBestWorstIndicator(models.map(m => m.trainingTime), index, false)}
                      </Box>
                    </TableCell>
                  ))}
                </TableRow>
                
                <TableRow>
                  <TableCell>Model Type</TableCell>
                  {models.map(model => (
                    <TableCell key={model.id} align="center">
                      <Chip
                        label={model.type}
                        color={getModelTypeColor(model.type) as any}
                        size="small"
                      />
                    </TableCell>
                  ))}
                </TableRow>
                
                <TableRow>
                  <TableCell>Status</TableCell>
                  {models.map(model => (
                    <TableCell key={model.id} align="center">
                      <Chip
                        label={model.status}
                        color={model.status === 'completed' ? 'success' : 'default'}
                        size="small"
                      />
                    </TableCell>
                  ))}
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>

        {/* Performance Charts Tab */}
        <TabPanel value={activeTab} index={1}>
          {createMetricsChart()}
        </TabPanel>

        {/* Radar Comparison Tab */}
        <TabPanel value={activeTab} index={2}>
          {createRadarChart()}
        </TabPanel>

        {/* Time vs Performance Tab */}
        <TabPanel value={activeTab} index={3}>
          {createScatterChart()}
        </TabPanel>
      </Card>
    </Box>
  );
};

export default ModelComparisonView;
