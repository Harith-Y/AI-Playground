/**
 * Tuning Results Component
 * 
 * Displays hyperparameter tuning results including:
 * - Summary of best configuration
 * - Complete results table with all trials
 * - Top parameters analysis
 * - Export and apply controls
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  TableSortLabel,
  Chip,
  Button,
  IconButton,
  Tooltip,
  Alert,
  Divider,
  Stack,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
} from '@mui/material';
import {
  Download as DownloadIcon,
  CheckCircle as CheckCircleIcon,
  EmojiEvents as TrophyIcon,
  TrendingUp as TrendingUpIcon,
  Visibility as VisibilityIcon,
  ContentCopy as ContentCopyIcon,
  PlayArrow as PlayArrowIcon,
} from '@mui/icons-material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Cell } from 'recharts';

// Types
interface TrialResult {
  trial_id: number;
  parameters: Record<string, any>;
  score: number;
  cv_scores: number[];
  mean_score: number;
  std_score: number;
  fit_time: number;
  score_time: number;
  rank: number;
}

interface TuningResultsData {
  best_trial: TrialResult;
  all_trials: TrialResult[];
  best_parameters: Record<string, any>;
  best_score: number;
  total_trials: number;
  tuning_time: number;
  cv_folds: number;
  scoring_metric: string;
}

interface TuningResultsProps {
  resultsData?: TuningResultsData;
  onApplyConfig?: (parameters: Record<string, any>) => void;
  onExport?: () => void;
}

type Order = 'asc' | 'desc';

const TuningResults: React.FC<TuningResultsProps> = ({
  resultsData,
  onApplyConfig,
  onExport,
}) => {
  // Mock data for demonstration
  const mockData: TuningResultsData = {
    best_trial: {
      trial_id: 23,
      parameters: {
        n_estimators: 150,
        max_depth: 8,
        min_samples_split: 5,
        min_samples_leaf: 2,
      },
      score: 0.9245,
      cv_scores: [0.92, 0.93, 0.91, 0.92, 0.94],
      mean_score: 0.9245,
      std_score: 0.0112,
      fit_time: 45.2,
      score_time: 2.3,
      rank: 1,
    },
    all_trials: Array.from({ length: 50 }, (_, i) => ({
      trial_id: i + 1,
      parameters: {
        n_estimators: Math.floor(Math.random() * 190) + 10,
        max_depth: Math.floor(Math.random() * 18) + 3,
        min_samples_split: Math.floor(Math.random() * 19) + 2,
        min_samples_leaf: Math.floor(Math.random() * 10) + 1,
      },
      score: 0.7 + Math.random() * 0.25,
      cv_scores: Array.from({ length: 5 }, () => 0.7 + Math.random() * 0.25),
      mean_score: 0.7 + Math.random() * 0.25,
      std_score: Math.random() * 0.05,
      fit_time: Math.random() * 60 + 10,
      score_time: Math.random() * 5 + 1,
      rank: i + 1,
    })).sort((a, b) => b.score - a.score),
    best_parameters: {
      n_estimators: 150,
      max_depth: 8,
      min_samples_split: 5,
      min_samples_leaf: 2,
    },
    best_score: 0.9245,
    total_trials: 50,
    tuning_time: 1250,
    cv_folds: 5,
    scoring_metric: 'accuracy',
  };

  const data = resultsData || mockData;

  // State
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [orderBy, setOrderBy] = useState<keyof TrialResult>('score');
  const [order, setOrder] = useState<Order>('desc');
  const [detailsDialog, setDetailsDialog] = useState<TrialResult | null>(null);
  const [applyDialog, setApplyDialog] = useState(false);

  // Sorting
  const handleRequestSort = (property: keyof TrialResult) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
  };

  const sortedTrials = [...data.all_trials].sort((a, b) => {
    const aValue = a[orderBy];
    const bValue = b[orderBy];
    
    if (typeof aValue === 'number' && typeof bValue === 'number') {
      return order === 'asc' ? aValue - bValue : bValue - aValue;
    }
    return 0;
  });

  // Pagination
  const handleChangePage = (_event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  // Format time
  const formatTime = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return minutes > 0 ? `${minutes}m ${secs}s` : `${secs}s`;
  };

  // Get top N trials
  const topTrials = sortedTrials.slice(0, 5);

  // Analyze parameter importance (simplified)
  const analyzeParameters = () => {
    const paramStats: Record<string, { values: number[]; scores: number[] }> = {};
    
    data.all_trials.forEach(trial => {
      Object.entries(trial.parameters).forEach(([key, value]) => {
        if (!paramStats[key]) {
          paramStats[key] = { values: [], scores: [] };
        }
        if (typeof value === 'number') {
          paramStats[key].values.push(value);
          paramStats[key].scores.push(trial.score);
        }
      });
    });

    return Object.entries(paramStats).map(([param, stats]) => {
      const avgValue = stats.values.reduce((a, b) => a + b, 0) / stats.values.length;
      const avgScore = stats.scores.reduce((a, b) => a + b, 0) / stats.scores.length;
      return { param, avgValue, avgScore };
    });
  };

  const parameterAnalysis = analyzeParameters();

  // Handle apply configuration
  const handleApplyConfig = () => {
    onApplyConfig?.(data.best_parameters);
    setApplyDialog(false);
  };

  // Copy parameters to clipboard
  const handleCopyParameters = () => {
    const paramsText = JSON.stringify(data.best_parameters, null, 2);
    navigator.clipboard.writeText(paramsText);
  };

  return (
    <Box>
      {/* Best Configuration Summary */}
      <Card sx={{ mb: 3, bgcolor: 'success.lighter', borderLeft: '4px solid', borderColor: 'success.main' }}>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
            <Box display="flex" alignItems="center" gap={1}>
              <TrophyIcon sx={{ fontSize: 32, color: 'gold' }} />
              <Box>
                <Typography variant="h5" gutterBottom>
                  Best Configuration Found
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Trial #{data.best_trial.trial_id} • Score: {data.best_score.toFixed(4)}
                </Typography>
              </Box>
            </Box>
            <Box display="flex" gap={1}>
              <Tooltip title="Copy parameters">
                <IconButton size="small" onClick={handleCopyParameters}>
                  <ContentCopyIcon />
                </IconButton>
              </Tooltip>
              <Button
                variant="contained"
                color="primary"
                startIcon={<PlayArrowIcon />}
                onClick={() => setApplyDialog(true)}
              >
                Apply Configuration
              </Button>
            </Box>
          </Box>

          <Divider sx={{ mb: 2 }} />

          <Box display="flex" flexWrap="wrap" gap={3}>
            {/* Parameters */}
            <Box sx={{ flex: '1 1 300px' }}>
              <Typography variant="subtitle2" gutterBottom>
                Best Parameters:
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={1}>
                {Object.entries(data.best_parameters).map(([key, value]) => (
                  <Chip
                    key={key}
                    label={`${key}: ${value}`}
                    color="primary"
                    variant="outlined"
                  />
                ))}
              </Box>
            </Box>

            {/* Metrics */}
            <Box sx={{ flex: '1 1 300px' }}>
              <Typography variant="subtitle2" gutterBottom>
                Performance Metrics:
              </Typography>
              <Stack spacing={1}>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2" color="text.secondary">
                    Mean CV Score:
                  </Typography>
                  <Typography variant="body2" fontWeight="medium">
                    {data.best_trial.mean_score.toFixed(4)} ± {data.best_trial.std_score.toFixed(4)}
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2" color="text.secondary">
                    Fit Time:
                  </Typography>
                  <Typography variant="body2" fontWeight="medium">
                    {formatTime(data.best_trial.fit_time)}
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2" color="text.secondary">
                    Total Tuning Time:
                  </Typography>
                  <Typography variant="body2" fontWeight="medium">
                    {formatTime(data.tuning_time)}
                  </Typography>
                </Box>
              </Stack>
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* Top 5 Trials Comparison */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
            <TrendingUpIcon sx={{ mr: 1 }} />
            Top 5 Configurations
          </Typography>
          <Divider sx={{ mb: 2 }} />

          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={topTrials}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="trial_id" label={{ value: 'Trial ID', position: 'insideBottom', offset: -5 }} />
              <YAxis label={{ value: 'Score', angle: -90, position: 'insideLeft' }} domain={[0.7, 1.0]} />
              <RechartsTooltip />
              <Bar dataKey="score" fill="#8884d8">
                {topTrials.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={index === 0 ? '#82ca9d' : '#8884d8'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <Box mt={2}>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Rank</strong></TableCell>
                    <TableCell><strong>Trial</strong></TableCell>
                    <TableCell><strong>Score</strong></TableCell>
                    <TableCell><strong>Parameters</strong></TableCell>
                    <TableCell><strong>Actions</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {topTrials.map((trial, index) => (
                    <TableRow key={trial.trial_id}>
                      <TableCell>
                        {index === 0 ? (
                          <Chip icon={<TrophyIcon />} label="1st" color="success" size="small" />
                        ) : (
                          <Chip label={`${index + 1}${index === 1 ? 'nd' : index === 2 ? 'rd' : 'th'}`} size="small" />
                        )}
                      </TableCell>
                      <TableCell>#{trial.trial_id}</TableCell>
                      <TableCell>
                        <Typography variant="body2" fontWeight="medium">
                          {trial.score.toFixed(4)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Stack direction="row" spacing={0.5} flexWrap="wrap">
                          {Object.entries(trial.parameters).slice(0, 2).map(([key, value]) => (
                            <Chip key={key} label={`${key}: ${value}`} size="small" variant="outlined" />
                          ))}
                          {Object.keys(trial.parameters).length > 2 && (
                            <Chip label={`+${Object.keys(trial.parameters).length - 2}`} size="small" />
                          )}
                        </Stack>
                      </TableCell>
                      <TableCell>
                        <Tooltip title="View details">
                          <IconButton size="small" onClick={() => setDetailsDialog(trial)}>
                            <VisibilityIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        </CardContent>
      </Card>

      {/* Parameter Analysis */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Parameter Analysis
          </Typography>
          <Divider sx={{ mb: 2 }} />
          
          <Alert severity="info" sx={{ mb: 2 }}>
            Average parameter values across all trials and their correlation with performance.
          </Alert>

          <Box display="flex" flexWrap="wrap" gap={2}>
            {parameterAnalysis.map(({ param, avgValue, avgScore }) => (
              <Paper key={param} sx={{ p: 2, flex: '1 1 200px', minWidth: 200 }}>
                <Typography variant="subtitle2" gutterBottom>
                  {param}
                </Typography>
                <Typography variant="h6" color="primary">
                  {avgValue.toFixed(2)}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Avg score: {avgScore.toFixed(4)}
                </Typography>
              </Paper>
            ))}
          </Box>
        </CardContent>
      </Card>

      {/* All Trials Table */}
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">
              All Trials ({data.total_trials})
            </Typography>
            <Button
              startIcon={<DownloadIcon />}
              variant="outlined"
              size="small"
              onClick={onExport}
            >
              Export Results
            </Button>
          </Box>
          <Divider sx={{ mb: 2 }} />

          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>
                    <TableSortLabel
                      active={orderBy === 'rank'}
                      direction={orderBy === 'rank' ? order : 'asc'}
                      onClick={() => handleRequestSort('rank')}
                    >
                      <strong>Rank</strong>
                    </TableSortLabel>
                  </TableCell>
                  <TableCell>
                    <TableSortLabel
                      active={orderBy === 'trial_id'}
                      direction={orderBy === 'trial_id' ? order : 'asc'}
                      onClick={() => handleRequestSort('trial_id')}
                    >
                      <strong>Trial</strong>
                    </TableSortLabel>
                  </TableCell>
                  <TableCell>
                    <TableSortLabel
                      active={orderBy === 'score'}
                      direction={orderBy === 'score' ? order : 'asc'}
                      onClick={() => handleRequestSort('score')}
                    >
                      <strong>Score</strong>
                    </TableSortLabel>
                  </TableCell>
                  <TableCell>
                    <TableSortLabel
                      active={orderBy === 'mean_score'}
                      direction={orderBy === 'mean_score' ? order : 'asc'}
                      onClick={() => handleRequestSort('mean_score')}
                    >
                      <strong>Mean ± Std</strong>
                    </TableSortLabel>
                  </TableCell>
                  <TableCell><strong>Parameters</strong></TableCell>
                  <TableCell>
                    <TableSortLabel
                      active={orderBy === 'fit_time'}
                      direction={orderBy === 'fit_time' ? order : 'asc'}
                      onClick={() => handleRequestSort('fit_time')}
                    >
                      <strong>Fit Time</strong>
                    </TableSortLabel>
                  </TableCell>
                  <TableCell><strong>Actions</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {sortedTrials
                  .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                  .map((trial) => (
                    <TableRow
                      key={trial.trial_id}
                      sx={{
                        bgcolor: trial.rank === 1 ? 'success.lighter' : 'inherit',
                        '&:hover': { bgcolor: 'action.hover' },
                      }}
                    >
                      <TableCell>
                        {trial.rank === 1 ? (
                          <Chip icon={<CheckCircleIcon />} label="Best" color="success" size="small" />
                        ) : (
                          trial.rank
                        )}
                      </TableCell>
                      <TableCell>#{trial.trial_id}</TableCell>
                      <TableCell>
                        <Typography variant="body2" fontWeight={trial.rank === 1 ? 'bold' : 'normal'}>
                          {trial.score.toFixed(4)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {trial.mean_score.toFixed(4)} ± {trial.std_score.toFixed(4)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Stack direction="row" spacing={0.5} flexWrap="wrap">
                          {Object.entries(trial.parameters).slice(0, 2).map(([key, value]) => (
                            <Chip key={key} label={`${key}: ${value}`} size="small" variant="outlined" />
                          ))}
                          {Object.keys(trial.parameters).length > 2 && (
                            <Tooltip title="Click to view all parameters">
                              <Chip
                                label={`+${Object.keys(trial.parameters).length - 2}`}
                                size="small"
                                onClick={() => setDetailsDialog(trial)}
                                sx={{ cursor: 'pointer' }}
                              />
                            </Tooltip>
                          )}
                        </Stack>
                      </TableCell>
                      <TableCell>{formatTime(trial.fit_time)}</TableCell>
                      <TableCell>
                        <Tooltip title="View details">
                          <IconButton size="small" onClick={() => setDetailsDialog(trial)}>
                            <VisibilityIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          </TableContainer>

          <TablePagination
            rowsPerPageOptions={[5, 10, 25, 50]}
            component="div"
            count={sortedTrials.length}
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={handleChangePage}
            onRowsPerPageChange={handleChangeRowsPerPage}
          />
        </CardContent>
      </Card>

      {/* Trial Details Dialog */}
      <Dialog open={!!detailsDialog} onClose={() => setDetailsDialog(null)} maxWidth="md" fullWidth>
        <DialogTitle>
          Trial #{detailsDialog?.trial_id} Details
        </DialogTitle>
        <DialogContent>
          {detailsDialog && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Performance Metrics:
              </Typography>
              <Paper sx={{ p: 2, mb: 2, bgcolor: 'background.default' }}>
                <Stack spacing={1}>
                  <Box display="flex" justifyContent="space-between">
                    <Typography variant="body2">Score:</Typography>
                    <Typography variant="body2" fontWeight="medium">
                      {detailsDialog.score.toFixed(4)}
                    </Typography>
                  </Box>
                  <Box display="flex" justifyContent="space-between">
                    <Typography variant="body2">Mean CV Score:</Typography>
                    <Typography variant="body2" fontWeight="medium">
                      {detailsDialog.mean_score.toFixed(4)} ± {detailsDialog.std_score.toFixed(4)}
                    </Typography>
                  </Box>
                  <Box display="flex" justifyContent="space-between">
                    <Typography variant="body2">Fit Time:</Typography>
                    <Typography variant="body2" fontWeight="medium">
                      {formatTime(detailsDialog.fit_time)}
                    </Typography>
                  </Box>
                  <Box display="flex" justifyContent="space-between">
                    <Typography variant="body2">Score Time:</Typography>
                    <Typography variant="body2" fontWeight="medium">
                      {formatTime(detailsDialog.score_time)}
                    </Typography>
                  </Box>
                </Stack>
              </Paper>

              <Typography variant="subtitle2" gutterBottom>
                Parameters:
              </Typography>
              <TextField
                fullWidth
                multiline
                rows={6}
                value={JSON.stringify(detailsDialog.parameters, null, 2)}
                InputProps={{ readOnly: true }}
                sx={{ mb: 2 }}
              />

              <Typography variant="subtitle2" gutterBottom>
                Cross-Validation Scores:
              </Typography>
              <Box display="flex" gap={1} flexWrap="wrap">
                {detailsDialog.cv_scores.map((score, idx) => (
                  <Chip key={idx} label={`Fold ${idx + 1}: ${score.toFixed(4)}`} size="small" />
                ))}
              </Box>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsDialog(null)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Apply Configuration Dialog */}
      <Dialog open={applyDialog} onClose={() => setApplyDialog(false)}>
        <DialogTitle>Apply Best Configuration</DialogTitle>
        <DialogContent>
          <Typography variant="body2" paragraph>
            Are you sure you want to apply the best configuration to your model?
          </Typography>
          <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
            <Typography variant="subtitle2" gutterBottom>
              Parameters to apply:
            </Typography>
            <Box display="flex" flexWrap="wrap" gap={1}>
              {Object.entries(data.best_parameters).map(([key, value]) => (
                <Chip key={key} label={`${key}: ${value}`} color="primary" size="small" />
              ))}
            </Box>
          </Paper>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setApplyDialog(false)}>Cancel</Button>
          <Button onClick={handleApplyConfig} variant="contained" color="primary">
            Apply
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default TuningResults;
