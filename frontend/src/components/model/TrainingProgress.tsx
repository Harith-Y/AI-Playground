import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Chip,
  Grid,
  Paper,
  List,
  ListItem,
  ListItemText,
  Divider,
  Alert,
  Button,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  HourglassEmpty as HourglassEmptyIcon,
  PlayArrow as PlayArrowIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  Timeline as TimelineIcon,
} from '@mui/icons-material';
import type { TrainingRun, TrainingLog } from '../../types/training';
import Plot from 'react-plotly.js';

interface TrainingProgressProps {
  trainingRun: TrainingRun;
  onCancel?: () => void;
  onRefresh?: () => void;
  showMetricsChart?: boolean;
}

const TrainingProgress: React.FC<TrainingProgressProps> = ({
  trainingRun,
  onCancel,
  onRefresh,
  showMetricsChart = true,
}) => {
  const [autoScroll, setAutoScroll] = useState(true);
  const logsEndRef = React.useRef<HTMLDivElement>(null);

  // Auto-scroll logs to bottom
  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [trainingRun.logs, autoScroll]);

  const getStatusIcon = (status: TrainingRun['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon color="success" />;
      case 'failed':
        return <ErrorIcon color="error" />;
      case 'running':
        return <PlayArrowIcon color="primary" />;
      case 'pending':
        return <HourglassEmptyIcon color="action" />;
      case 'cancelled':
        return <StopIcon color="warning" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status: TrainingRun['status']) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'running':
        return 'primary';
      case 'pending':
        return 'default';
      case 'cancelled':
        return 'warning';
      default:
        return 'default';
    }
  };

  const formatDuration = (seconds?: number) => {
    if (!seconds) return 'N/A';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  const getLogColor = (level: TrainingLog['level']) => {
    switch (level) {
      case 'error':
        return 'error.main';
      case 'warning':
        return 'warning.main';
      case 'info':
        return 'info.main';
      case 'debug':
        return 'text.secondary';
      default:
        return 'text.primary';
    }
  };

  const renderMetricsChart = () => {
    if (!showMetricsChart || !trainingRun.metrics.history) {
      return null;
    }

    const { history } = trainingRun.metrics;
    const traces: any[] = [];

    // Training loss
    if (history.trainLoss && history.trainLoss.length > 0) {
      traces.push({
        x: history.epoch,
        y: history.trainLoss,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Train Loss',
        line: { color: '#1976d2' },
      });
    }

    // Validation loss
    if (history.valLoss && history.valLoss.length > 0) {
      traces.push({
        x: history.epoch,
        y: history.valLoss,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Val Loss',
        line: { color: '#dc004e', dash: 'dash' },
      });
    }

    // Training accuracy
    if (history.trainAccuracy && history.trainAccuracy.length > 0) {
      traces.push({
        x: history.epoch,
        y: history.trainAccuracy,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Train Accuracy',
        line: { color: '#4caf50' },
        yaxis: 'y2',
      });
    }

    // Validation accuracy
    if (history.valAccuracy && history.valAccuracy.length > 0) {
      traces.push({
        x: history.epoch,
        y: history.valAccuracy,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Val Accuracy',
        line: { color: '#ff9800', dash: 'dash' },
        yaxis: 'y2',
      });
    }

    if (traces.length === 0) {
      return null;
    }

    return (
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Box display="flex" alignItems="center" mb={2}>
            <TimelineIcon sx={{ mr: 1 }} />
            <Typography variant="h6">Training Metrics</Typography>
          </Box>
          <Plot
            data={traces}
            layout={{
              autosize: true,
              height: 400,
              xaxis: {
                title: 'Epoch',
                gridcolor: '#e0e0e0',
              },
              yaxis: {
                title: 'Loss',
                gridcolor: '#e0e0e0',
              },
              yaxis2: {
                title: 'Accuracy',
                overlaying: 'y',
                side: 'right',
                gridcolor: '#e0e0e0',
              },
              legend: {
                x: 0,
                y: 1,
                orientation: 'h',
              },
              margin: { l: 50, r: 50, t: 20, b: 50 },
            }}
            config={{ responsive: true }}
            style={{ width: '100%' }}
          />
        </CardContent>
      </Card>
    );
  };

  return (
    <Box>
      {/* Status Header */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Box display="flex" alignItems="center" gap={2}>
              {getStatusIcon(trainingRun.status)}
              <Box>
                <Typography variant="h5">{trainingRun.modelName}</Typography>
                <Typography variant="body2" color="text.secondary">
                  {trainingRun.modelType} • Run ID: {trainingRun.id.slice(0, 8)}
                </Typography>
              </Box>
            </Box>
            <Box display="flex" gap={1}>
              <Chip
                label={trainingRun.status.toUpperCase()}
                color={getStatusColor(trainingRun.status) as any}
                size="small"
              />
              {onRefresh && (
                <Tooltip title="Refresh">
                  <IconButton size="small" onClick={onRefresh}>
                    <RefreshIcon />
                  </IconButton>
                </Tooltip>
              )}
              {onCancel && trainingRun.status === 'running' && (
                <Button
                  variant="outlined"
                  color="error"
                  size="small"
                  startIcon={<StopIcon />}
                  onClick={onCancel}
                >
                  Cancel
                </Button>
              )}
            </Box>
          </Box>

          {/* Progress Bar */}
          {(trainingRun.status === 'running' || trainingRun.status === 'pending') && (
            <Box mb={2}>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2" color="text.secondary">
                  Progress
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {trainingRun.progress}%
                  {trainingRun.currentEpoch && trainingRun.totalEpochs && (
                    <> • Epoch {trainingRun.currentEpoch}/{trainingRun.totalEpochs}</>
                  )}
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={trainingRun.progress}
                sx={{ height: 8, borderRadius: 1 }}
              />
            </Box>
          )}

          {/* Time Info */}
          <Grid container spacing={2}>
            <Grid size={{ xs: 12, sm: 4 }}>
              <Typography variant="body2" color="text.secondary">
                Started
              </Typography>
              <Typography variant="body1">
                {new Date(trainingRun.startTime).toLocaleString()}
              </Typography>
            </Grid>
            {trainingRun.endTime && (
              <Grid size={{ xs: 12, sm: 4 }}>
                <Typography variant="body2" color="text.secondary">
                  Ended
                </Typography>
                <Typography variant="body1">
                  {new Date(trainingRun.endTime).toLocaleString()}
                </Typography>
              </Grid>
            )}
            <Grid size={{ xs: 12, sm: 4 }}>
              <Typography variant="body2" color="text.secondary">
                Duration
              </Typography>
              <Typography variant="body1">
                {formatDuration(trainingRun.duration)}
              </Typography>
            </Grid>
          </Grid>

          {/* Error Message */}
          {trainingRun.error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              <Typography variant="body2">{trainingRun.error}</Typography>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Metrics Chart */}
      {renderMetricsChart()}

      {/* Metrics Summary */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Metrics Summary
          </Typography>
          <Grid container spacing={2}>
            {/* Training Metrics */}
            {trainingRun.metrics.trainScore !== undefined && (
              <Grid size={{ xs: 12, sm: 6, md: 4 }}>
                <Paper elevation={0} sx={{ p: 2, bgcolor: 'primary.light', color: 'primary.contrastText' }}>
                  <Typography variant="body2">Train Score</Typography>
                  <Typography variant="h4">
                    {(trainingRun.metrics.trainScore * 100).toFixed(2)}%
                  </Typography>
                </Paper>
              </Grid>
            )}

            {/* Validation Metrics */}
            {trainingRun.metrics.valScore !== undefined && (
              <Grid size={{ xs: 12, sm: 6, md: 4 }}>
                <Paper elevation={0} sx={{ p: 2, bgcolor: 'secondary.light', color: 'secondary.contrastText' }}>
                  <Typography variant="body2">Validation Score</Typography>
                  <Typography variant="h4">
                    {(trainingRun.metrics.valScore * 100).toFixed(2)}%
                  </Typography>
                </Paper>
              </Grid>
            )}

            {/* Test Metrics */}
            {trainingRun.metrics.testScore !== undefined && (
              <Grid size={{ xs: 12, sm: 6, md: 4 }}>
                <Paper elevation={0} sx={{ p: 2, bgcolor: 'success.light', color: 'success.contrastText' }}>
                  <Typography variant="body2">Test Score</Typography>
                  <Typography variant="h4">
                    {(trainingRun.metrics.testScore * 100).toFixed(2)}%
                  </Typography>
                </Paper>
              </Grid>
            )}

            {/* Additional Metrics */}
            {trainingRun.metrics.precision !== undefined && (
              <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.100' }}>
                  <Typography variant="body2" color="text.secondary">Precision</Typography>
                  <Typography variant="h5">
                    {(trainingRun.metrics.precision * 100).toFixed(2)}%
                  </Typography>
                </Paper>
              </Grid>
            )}

            {trainingRun.metrics.recall !== undefined && (
              <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.100' }}>
                  <Typography variant="body2" color="text.secondary">Recall</Typography>
                  <Typography variant="h5">
                    {(trainingRun.metrics.recall * 100).toFixed(2)}%
                  </Typography>
                </Paper>
              </Grid>
            )}

            {trainingRun.metrics.f1Score !== undefined && (
              <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.100' }}>
                  <Typography variant="body2" color="text.secondary">F1 Score</Typography>
                  <Typography variant="h5">
                    {(trainingRun.metrics.f1Score * 100).toFixed(2)}%
                  </Typography>
                </Paper>
              </Grid>
            )}

            {trainingRun.metrics.rmse !== undefined && (
              <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.100' }}>
                  <Typography variant="body2" color="text.secondary">RMSE</Typography>
                  <Typography variant="h5">
                    {trainingRun.metrics.rmse.toFixed(4)}
                  </Typography>
                </Paper>
              </Grid>
            )}

            {trainingRun.metrics.mae !== undefined && (
              <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.100' }}>
                  <Typography variant="body2" color="text.secondary">MAE</Typography>
                  <Typography variant="h5">
                    {trainingRun.metrics.mae.toFixed(4)}
                  </Typography>
                </Paper>
              </Grid>
            )}

            {trainingRun.metrics.r2Score !== undefined && (
              <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.100' }}>
                  <Typography variant="body2" color="text.secondary">R² Score</Typography>
                  <Typography variant="h5">
                    {trainingRun.metrics.r2Score.toFixed(4)}
                  </Typography>
                </Paper>
              </Grid>
            )}
          </Grid>
        </CardContent>
      </Card>

      {/* Training Logs */}
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">Training Logs</Typography>
            <Button
              size="small"
              onClick={() => setAutoScroll(!autoScroll)}
              variant={autoScroll ? 'contained' : 'outlined'}
            >
              Auto-scroll: {autoScroll ? 'ON' : 'OFF'}
            </Button>
          </Box>
          <Paper
            elevation={0}
            sx={{
              maxHeight: 400,
              overflow: 'auto',
              bgcolor: 'grey.900',
              color: 'grey.100',
              p: 2,
              fontFamily: 'monospace',
              fontSize: '0.875rem',
            }}
          >
            {trainingRun.logs.length === 0 ? (
              <Typography color="text.secondary">No logs available yet...</Typography>
            ) : (
              <List dense>
                {trainingRun.logs.map((log, index) => (
                  <React.Fragment key={log.id}>
                    <ListItem disableGutters>
                      <ListItemText
                        primary={
                          <Box display="flex" gap={1}>
                            <Typography
                              component="span"
                              sx={{ color: 'grey.500', minWidth: 80 }}
                            >
                              {new Date(log.timestamp).toLocaleTimeString()}
                            </Typography>
                            <Typography
                              component="span"
                              sx={{
                                color: getLogColor(log.level),
                                minWidth: 60,
                                fontWeight: 'bold',
                              }}
                            >
                              [{log.level.toUpperCase()}]
                            </Typography>
                            <Typography component="span" sx={{ color: 'grey.100' }}>
                              {log.message}
                            </Typography>
                          </Box>
                        }
                      />
                    </ListItem>
                    {index < trainingRun.logs.length - 1 && (
                      <Divider sx={{ bgcolor: 'grey.800' }} />
                    )}
                  </React.Fragment>
                ))}
                <div ref={logsEndRef} />
              </List>
            )}
          </Paper>
        </CardContent>
      </Card>
    </Box>
  );
};

export default TrainingProgress;
