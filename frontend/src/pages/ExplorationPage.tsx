import React, { useEffect, useState } from 'react';
import {
  Typography,
  Box,
  Container,
  Grid,
  Card,
  CardContent,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  Chip,
  Paper,
  CircularProgress,
  Tabs,
  Tab,
} from '@mui/material';
import {
  Assessment,
  BarChart,
  ScatterPlot,
  ShowChart,
  GridOn,
  Refresh,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { LineChart, Line, BarChart as RechartsBar, Bar, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { api } from '../services/api';

interface VisualizationData {
  type: string;
  title: string;
  xLabel?: string;
  yLabel?: string;
  data: any;
}

const ExplorationPage: React.FC = () => {
  const { currentDataset, columns, stats } = useAppSelector((state) => state.dataset);
  
  const [visualizations, setVisualizations] = useState<VisualizationData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedColumn, setSelectedColumn] = useState<string>('');
  const [selectedXColumn, setSelectedXColumn] = useState<string>('');
  const [selectedYColumn, setSelectedYColumn] = useState<string>('');
  const [activeTab, setActiveTab] = useState(0);

  const numericColumns = columns.filter(col => 
    ['int64', 'float64', 'int32', 'float32', 'number'].includes(col.dataType.toLowerCase())
  );
  const categoricalColumns = columns.filter(col => 
    ['object', 'string', 'category'].includes(col.dataType.toLowerCase())
  );

  useEffect(() => {
    if (currentDataset && numericColumns.length > 0) {
      setSelectedColumn(numericColumns[0].name);
    }
    if (numericColumns.length >= 2) {
      setSelectedXColumn(numericColumns[0].name);
      setSelectedYColumn(numericColumns[1].name);
    }
  }, [currentDataset, columns]);

  const loadOverview = async () => {
    if (!currentDataset) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await api.get(`/api/v1/visualizations/${currentDataset.id}/overview`);
      setVisualizations(response.visualizations || []);
    } catch (err: any) {
      setError(err.message || 'Failed to load visualizations');
    } finally {
      setLoading(false);
    }
  };

  const loadHistogram = async () => {
    if (!currentDataset || !selectedColumn) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await api.get(`/api/v1/visualizations/${currentDataset.id}/histogram/${selectedColumn}`);
      console.log('Histogram response:', response);
      setVisualizations([response]);
    } catch (err: any) {
      setError(err.message || 'Failed to load histogram');
    } finally {
      setLoading(false);
    }
  };

  const loadCorrelation = async () => {
    if (!currentDataset) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await api.get(`/api/v1/visualizations/${currentDataset.id}/correlation`);
      console.log('Correlation response:', response);
      setVisualizations([response]);
    } catch (err: any) {
      setError(err.message || 'Failed to load correlation matrix');
    } finally {
      setLoading(false);
    }
  };

  const loadScatterPlot = async () => {
    if (!currentDataset || !selectedXColumn || !selectedYColumn) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await api.get(
        `/api/v1/visualizations/${currentDataset.id}/scatter/${selectedXColumn}/${selectedYColumn}`
      );
      console.log('Scatter response:', response);
      setVisualizations([response]);
    } catch (err: any) {
      setError(err.message || 'Failed to load scatter plot');
    } finally {
      setLoading(false);
    }
  };

  const loadBoxPlot = async () => {
    if (!currentDataset || !selectedColumn) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await api.get(`/api/v1/visualizations/${currentDataset.id}/boxplot/${selectedColumn}`);
      setVisualizations([response]);
    } catch (err: any) {
      setError(err.message || 'Failed to load box plot');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (currentDataset) {
      loadOverview();
    }
  }, [currentDataset]);

  if (!currentDataset) {
    return (
      <Container maxWidth="xl" sx={{ py: 4 }}>
        <Alert severity="info">
          Please upload a dataset first to explore visualizations.
        </Alert>
      </Container>
    );
  }

  const renderHistogram = (viz: VisualizationData) => {
    const histData = viz.data.bins.slice(0, -1).map((bin: number, i: number) => ({
      bin: `${bin.toFixed(2)}`,
      count: viz.data.counts[i]
    }));

    return (
      <Box>
        <ResponsiveContainer width="100%" height={300}>
          <RechartsBar data={histData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bin" label={{ value: viz.xLabel, position: 'insideBottom', offset: -5 }} />
            <YAxis label={{ value: viz.yLabel, angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" />
          </RechartsBar>
        </ResponsiveContainer>
        {viz.data.stats && (
          <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <Chip label={`Mean: ${viz.data.stats.mean?.toFixed(2)}`} size="small" />
            <Chip label={`Median: ${viz.data.stats.median?.toFixed(2)}`} size="small" />
            <Chip label={`Std Dev: ${viz.data.stats.std?.toFixed(2)}`} size="small" />
          </Box>
        )}
      </Box>
    );
  };

  const renderCorrelation = (viz: VisualizationData) => {
    const matrix = viz.data.matrix;
    const cols = viz.data.columns;
    
    return (
      <Box sx={{ overflowX: 'auto' }}>
        <Box sx={{ display: 'grid', gridTemplateColumns: `120px repeat(${cols.length}, 80px)`, gap: 0.5 }}>
          <Box />
          {cols.map((col: string) => (
            <Box key={col} sx={{ fontSize: '0.75rem', textAlign: 'center', fontWeight: 600 }}>
              {col}
            </Box>
          ))}
          {matrix.map((row: number[], i: number) => (
            <React.Fragment key={i}>
              <Box sx={{ fontSize: '0.75rem', fontWeight: 600, py: 1 }}>{cols[i]}</Box>
              {row.map((val: number, j: number) => (
                <Box
                  key={j}
                  sx={{
                    bgcolor: val > 0.7 ? '#dc2626' : val > 0.3 ? '#f97316' : val < -0.7 ? '#0ea5e9' : val < -0.3 ? '#38bdf8' : '#e5e7eb',
                    color: Math.abs(val) > 0.3 ? 'white' : 'black',
                    textAlign: 'center',
                    py: 1,
                    fontSize: '0.75rem',
                    fontWeight: 600
                  }}
                >
                  {val.toFixed(2)}
                </Box>
              ))}
            </React.Fragment>
          ))}
        </Box>
      </Box>
    );
  };

  const renderScatterPlot = (viz: VisualizationData) => {
    const scatterData = viz.data.x.map((x: number, i: number) => ({
      x,
      y: viz.data.y[i]
    }));

    return (
      <Box>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="x" name={viz.xLabel} label={{ value: viz.xLabel, position: 'insideBottom', offset: -5 }} />
            <YAxis dataKey="y" name={viz.yLabel} label={{ value: viz.yLabel, angle: -90, position: 'insideLeft' }} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Scatter data={scatterData} fill="#3b82f6" />
          </ScatterChart>
        </ResponsiveContainer>
        {viz.data.correlation !== undefined && (
          <Box sx={{ mt: 2 }}>
            <Chip label={`Correlation: ${viz.data.correlation.toFixed(3)}`} size="small" color="primary" />
          </Box>
        )}
      </Box>
    );
  };

  const renderBarChart = (viz: VisualizationData) => {
    const barData = viz.data.categories.map((cat: string, i: number) => ({
      category: cat,
      count: viz.data.counts[i]
    }));

    return (
      <ResponsiveContainer width="100%" height={300}>
        <RechartsBar data={barData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="category" label={{ value: viz.xLabel, position: 'insideBottom', offset: -5 }} />
          <YAxis label={{ value: viz.yLabel, angle: -90, position: 'insideLeft' }} />
          <Tooltip />
          <Bar dataKey="count" fill="#10b981" />
        </RechartsBar>
      </ResponsiveContainer>
    );
  };

  const renderVisualization = (viz: VisualizationData) => {
    switch (viz.type) {
      case 'histogram':
        return renderHistogram(viz);
      case 'correlation':
        return renderCorrelation(viz);
      case 'scatter':
        return renderScatterPlot(viz);
      case 'bar':
        return renderBarChart(viz);
      default:
        return <Typography>Unsupported visualization type: {viz.type}</Typography>;
    }
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Page Header */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <Assessment sx={{ fontSize: 40, color: 'primary.main' }} />
          <Box>
            <Typography variant="h4" component="h1" fontWeight={600}>
              Data Exploration (EDA)
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Visualize and understand your dataset before preprocessing
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* Dataset Info */}
      <Paper sx={{ p: 2, mb: 3, border: '1px solid #e2e8f0' }}>
        <Typography variant="h6" gutterBottom>{currentDataset.name}</Typography>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Chip label={`${stats?.rowCount || currentDataset.rowCount} rows`} size="small" />
          <Chip label={`${stats?.columnCount || currentDataset.columnCount} columns`} size="small" />
          <Chip label={`${numericColumns.length} numeric`} size="small" color="primary" />
          <Chip label={`${categoricalColumns.length} categorical`} size="small" color="secondary" />
        </Box>
      </Paper>

      {/* Visualization Controls */}
      <Paper sx={{ p: 3, mb: 3, border: '1px solid #e2e8f0' }}>
        <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} sx={{ mb: 2 }}>
          <Tab label="Overview" icon={<GridOn />} iconPosition="start" />
          <Tab label="Histogram" icon={<BarChart />} iconPosition="start" />
          <Tab label="Correlation" icon={<ShowChart />} iconPosition="start" />
          <Tab label="Scatter Plot" icon={<ScatterPlot />} iconPosition="start" />
        </Tabs>

        {activeTab === 0 && (
          <Button variant="contained" startIcon={<Refresh />} onClick={loadOverview} disabled={loading}>
            Load Overview
          </Button>
        )}

        {activeTab === 1 && (
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <FormControl sx={{ minWidth: 200 }}>
              <InputLabel>Column</InputLabel>
              <Select value={selectedColumn} onChange={(e) => setSelectedColumn(e.target.value)} label="Column">
                {numericColumns.map(col => (
                  <MenuItem key={col.name} value={col.name}>{col.name}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <Button variant="contained" onClick={loadHistogram} disabled={loading || !selectedColumn}>
              Generate Histogram
            </Button>
          </Box>
        )}

        {activeTab === 2 && (
          <Button variant="contained" onClick={loadCorrelation} disabled={loading || numericColumns.length < 2}>
            Generate Correlation Matrix
          </Button>
        )}

        {activeTab === 3 && (
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <FormControl sx={{ minWidth: 150 }}>
              <InputLabel>X Axis</InputLabel>
              <Select value={selectedXColumn} onChange={(e) => setSelectedXColumn(e.target.value)} label="X Axis">
                {numericColumns.map(col => (
                  <MenuItem key={col.name} value={col.name}>{col.name}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl sx={{ minWidth: 150 }}>
              <InputLabel>Y Axis</InputLabel>
              <Select value={selectedYColumn} onChange={(e) => setSelectedYColumn(e.target.value)} label="Y Axis">
                {numericColumns.map(col => (
                  <MenuItem key={col.name} value={col.name}>{col.name}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <Button variant="contained" onClick={loadScatterPlot} disabled={loading || !selectedXColumn || !selectedYColumn}>
              Generate Scatter Plot
            </Button>
          </Box>
        )}
      </Paper>

      {/* Loading State */}
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {/* Error State */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Visualizations Grid */}
      {!loading && visualizations.length > 0 && (
        <Grid container spacing={3}>
          {visualizations.map((viz, idx) => (
            <Grid size={{ xs: 12, md: viz.type === 'correlation' ? 12 : 6 }} key={idx}>
              <Card sx={{ border: '1px solid #e2e8f0' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>{viz.title}</Typography>
                  {renderVisualization(viz)}
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {!loading && visualizations.length === 0 && !error && (
        <Alert severity="info">
          No visualizations generated yet. Select a visualization type and click generate.
        </Alert>
      )}
    </Container>
  );
};

export default ExplorationPage;
