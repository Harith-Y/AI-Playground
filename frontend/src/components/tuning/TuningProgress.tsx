/**
 * Tuning Progress Component
 * 
 * Real-time progress tracking for hyperparameter tuning including:
 * - Overall progress bar
 * - Current trial information
 * - Best-so-far metrics
 * - Trial history
 * - Performance charts
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Tooltip,
  Alert,
  Divider,
  CircularProgress,
  Button,
  Stack,
} from '@mui/material';
import {
  PlayArrow as PlayArrowIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  Timer as TimerIcon,
  Speed as SpeedIcon,
  EmojiEvents as TrophyIcon,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Legend } from 'recharts';

// Types
interface Trial {
  id: number;
  parameters: Record<string, any>;
  score: number;
  status: 'running' | 'completed' | 'failed';
  duration: number;
  timestamp: string;
}

interface TuningProgressData {
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed';
  current_trial: number;
  total_trials: number;
  best_score: number;
  best_parameters: Record<string, any>;
  current_parameters: Record<string, any>;
  elapsed_time: number;
  estimated_time_remaining: number;
  trials: Trial[];
  progress_history: Array<{ trial: number; score: number; best_score: number }>;
}

interface TuningProgressProps {
  tuningId?: string;
  onStatusChange?: (status: string) => void;
  refreshInterval?: number;
}

const TuningProgress: React.FC<TuningProgressProps> = ({
  tuningId: _tuningId,
  onStatusChange,
  refreshInterval = 2000, // 2 seconds
}) => {
  // State
  const [progressData, setProgressData] = useState<TuningProgressData>({
    status: 'running',
    current_trial: 15,
    total_trials: 50,
    best_score: 0.8945,
    best_parameters: {
      n_estimators: 150,
      max_depth: 8,
      min_samples_split: 5,
      min_samples_leaf: 2,
    },
    current_parameters: {
      n_estimators: 120,
      max_depth: 12,
      min_samples_split: 3,
      min_samples_leaf: 1,
    },
    elapsed_time: 450, // seconds
    estimated_time_remaining: 1050, // seconds
    trials: [],
    progress_history: [],
  });
  
  const [isPolling, setIsPolling] = useState(true);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Generate mock trial data
  useEffect(() => {
    const generateMockTrials = () => {
      const trials: Trial[] = [];
      const history: Array<{ trial: number; score: number; best_score: number }> = [];
      let bestScore = 0;
      
      for (let i = 1; i <= progressData.current_trial; i++) {
        const score = 0.7 + Math.random() * 0.25; // Random score between 0.7 and 0.95
        if (score > bestScore) bestScore = score;
        
        trials.push({
          id: i,
          parameters: {
            n_estimators: Math.floor(Math.random() * 190) + 10,
            max_depth: Math.floor(Math.random() * 18) + 3,
            min_samples_split: Math.floor(Math.random() * 19) + 2,
            min_samples_leaf: Math.floor(Math.random() * 10) + 1,
          },
          score: parseFloat(score.toFixed(4)),
          status: i === progressData.current_trial ? 'running' : 'completed',
          duration: Math.floor(Math.random() * 60) + 10,
          timestamp: new Date(Date.now() - (progressData.current_trial - i) * 30000).toISOString(),
        });
        
        history.push({
          trial: i,
          score: parseFloat(score.toFixed(4)),
          best_score: parseFloat(bestScore.toFixed(4)),
        });
      }
      
      setProgressData(prev => ({
        ...prev,
        trials: trials.reverse(), // Show latest first
        progress_history: history,
        best_score: parseFloat(bestScore.toFixed(4)),
      }));
    };
    
    generateMockTrials();
  }, [progressData.current_trial]);
  
  // Polling for updates
  useEffect(() => {
    if (isPolling && progressData.status === 'running') {
      intervalRef.current = setInterval(() => {
        // Simulate progress
        setProgressData(prev => {
          if (prev.current_trial >= prev.total_trials) {
            setIsPolling(false);
            onStatusChange?.('completed');
            return { ...prev, status: 'completed' };
          }
          
          return {
            ...prev,
            current_trial: Math.min(prev.current_trial + 1, prev.total_trials),
            elapsed_time: prev.elapsed_time + refreshInterval / 1000,
            estimated_time_remaining: Math.max(0, prev.estimated_time_remaining - refreshInterval / 1000),
          };
        });
      }, refreshInterval);
    }
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPolling, progressData.status, refreshInterval, onStatusChange]);
  
  // Format time
  const formatTime = (seconds: number): string => {
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
  
  // Calculate progress percentage
  const progressPercentage = (progressData.current_trial / progressData.total_trials) * 100;
  
  // Handle control actions
  const handlePause = () => {
    setIsPolling(false);
    setProgressData(prev => ({ ...prev, status: 'paused' }));
    onStatusChange?.('paused');
  };
  
  const handleResume = () => {
    setIsPolling(true);
    setProgressData(prev => ({ ...prev, status: 'running' }));
    onStatusChange?.('running');
  };
  
  const handleStop = () => {
    setIsPolling(false);
    setProgressData(prev => ({ ...prev, status: 'completed' }));
    onStatusChange?.('completed');
  };
  
  return (
    <Box>
      {/* Progress Overview */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">
              Tuning Progress
            </Typography>
            <Box display="flex" gap={1}>
              {progressData.status === 'running' ? (
                <>
                  <Tooltip title="Pause tuning">
                    <IconButton onClick={handlePause} color="warning" size="small">
                      <PauseIcon />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Stop tuning">
                    <IconButton onClick={handleStop} color="error" size="small">
                      <StopIcon />
                    </IconButton>
                  </Tooltip>
                </>
              ) : progressData.status === 'paused' ? (
                <>
                  <Tooltip title="Resume tuning">
                    <IconButton onClick={handleResume} color="success" size="small">
                      <PlayArrowIcon />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Stop tuning">
                    <IconButton onClick={handleStop} color="error" size="small">
                      <StopIcon />
                    </IconButton>
                  </Tooltip>
                </>
              ) : (
                <Tooltip title="Refresh">
                  <IconButton color="primary" size="small">
                    <RefreshIcon />
                  </IconButton>
                </Tooltip>
              )}
            </Box>
          </Box>
          
          {/* Progress Bar */}
          <Box mb={2}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
              <Typography variant="body2" color="text.secondary">
                Trial {progressData.current_trial} of {progressData.total_trials}
              </Typography>
              <Typography variant="body2" fontWeight="medium">
                {progressPercentage.toFixed(1)}%
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={progressPercentage} 
              sx={{ height: 8, borderRadius: 4 }}
            />
          </Box>
          
          {/* Status and Time Info */}
          <Box display="flex" flexWrap="wrap" gap={2}>
            <Box sx={{ flex: '1 1 auto', minWidth: 120, textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Status
              </Typography>
              <Box display="flex" justifyContent="center" mt={0.5}>
                <Chip 
                  label={progressData.status.charAt(0).toUpperCase() + progressData.status.slice(1)}
                  color={progressData.status === 'running' ? 'primary' : 
                         progressData.status === 'completed' ? 'success' : 
                         progressData.status === 'paused' ? 'warning' : 'error'}
                  size="small"
                />
              </Box>
            </Box>
            
            <Box sx={{ flex: '1 1 auto', minWidth: 120, textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Elapsed Time
              </Typography>
              <Typography variant="body2" fontWeight="medium" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mt: 0.5 }}>
                <TimerIcon sx={{ fontSize: 16, mr: 0.5 }} />
                {formatTime(progressData.elapsed_time)}
              </Typography>
            </Box>
            
            <Box sx={{ flex: '1 1 auto', minWidth: 120, textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Estimated Remaining
              </Typography>
              <Typography variant="body2" fontWeight="medium" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mt: 0.5 }}>
                <SpeedIcon sx={{ fontSize: 16, mr: 0.5 }} />
                {formatTime(progressData.estimated_time_remaining)}
              </Typography>
            </Box>
            
            <Box sx={{ flex: '1 1 auto', minWidth: 120, textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Best Score
              </Typography>
              <Typography variant="body2" fontWeight="medium" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mt: 0.5 }}>
                <TrophyIcon sx={{ fontSize: 16, mr: 0.5, color: 'gold' }} />
                {progressData.best_score.toFixed(4)}
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>
      
      {/* Current Trial and Best Parameters */}
      <Box display="flex" flexDirection={{ xs: 'column', md: 'row' }} gap={3} mb={3}>
        {/* Current Trial */}
        <Box sx={{ flex: 1 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <CircularProgress size={20} sx={{ mr: 1 }} />
                Current Trial #{progressData.current_trial}
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Typography variant="subtitle2" gutterBottom>
                Parameters:
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={1} mb={2}>
                {Object.entries(progressData.current_parameters).map(([key, value]) => (
                  <Chip 
                    key={key} 
                    label={`${key}: ${value}`} 
                    size="small" 
                    variant="outlined"
                  />
                ))}
              </Box>
              <Typography variant="body2" color="text.secondary">
                Running for {formatTime(30)} • Estimated completion in {formatTime(45)}
              </Typography>
            </CardContent>
          </Card>
        </Box>
        
        {/* Best Parameters */}
        <Box sx={{ flex: 1 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <TrophyIcon sx={{ mr: 1, color: 'gold' }} />
                Best Configuration
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Typography variant="subtitle2" gutterBottom>
                Score: {progressData.best_score.toFixed(4)}
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={1} mb={2}>
                {Object.entries(progressData.best_parameters).map(([key, value]) => (
                  <Chip 
                    key={key} 
                    label={`${key}: ${value}`} 
                    size="small" 
                    color="primary"
                    variant="outlined"
                  />
                ))}
              </Box>
              <Button 
                size="small" 
                variant="outlined" 
                color="primary"
                disabled={progressData.status === 'running'}
              >
                Apply Configuration
              </Button>
            </CardContent>
          </Card>
        </Box>
      </Box>
      
      {/* Progress Chart */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
            <TrendingUpIcon sx={{ mr: 1 }} />
            Performance Over Time
          </Typography>
          <Divider sx={{ mb: 2 }} />
          
          {progressData.progress_history.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={progressData.progress_history}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="trial" 
                  label={{ value: 'Trial Number', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  label={{ value: 'Score', angle: -90, position: 'insideLeft' }}
                  domain={[0.6, 1.0]}
                />
                <RechartsTooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="score" 
                  stroke="#8884d8" 
                  name="Trial Score"
                  dot={{ r: 3 }}
                  strokeWidth={2}
                />
                <Line 
                  type="stepAfter" 
                  dataKey="best_score" 
                  stroke="#82ca9d" 
                  name="Best Score"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <Alert severity="info">
              No trial data available yet. Progress will appear as trials complete.
            </Alert>
          )}
        </CardContent>
      </Card>
      
      {/* Trial History Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Trial History
          </Typography>
          <Divider sx={{ mb: 2 }} />
          
          {progressData.trials.length > 0 ? (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Trial</strong></TableCell>
                    <TableCell><strong>Status</strong></TableCell>
                    <TableCell><strong>Score</strong></TableCell>
                    <TableCell><strong>Parameters</strong></TableCell>
                    <TableCell><strong>Duration</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {progressData.trials.slice(0, 10).map((trial) => (
                    <TableRow 
                      key={trial.id}
                      sx={{ 
                        bgcolor: trial.id === progressData.current_trial ? 'action.hover' : 'inherit',
                        '&:hover': { bgcolor: 'action.selected' }
                      }}
                    >
                      <TableCell>#{trial.id}</TableCell>
                      <TableCell>
                        <Chip 
                          label={trial.status}
                          size="small"
                          color={trial.status === 'completed' ? 'success' : 
                                 trial.status === 'running' ? 'primary' : 'error'}
                        />
                      </TableCell>
                      <TableCell>
                        <Typography 
                          variant="body2" 
                          fontWeight={trial.score === progressData.best_score ? 'bold' : 'normal'}
                          color={trial.score === progressData.best_score ? 'success.main' : 'inherit'}
                        >
                          {trial.score.toFixed(4)}
                          {trial.score === progressData.best_score && ' 🏆'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Stack direction="row" spacing={0.5} flexWrap="wrap">
                          {Object.entries(trial.parameters).slice(0, 2).map(([key, value]) => (
                            <Chip 
                              key={key}
                              label={`${key}: ${value}`}
                              size="small"
                              variant="outlined"
                            />
                          ))}
                          {Object.keys(trial.parameters).length > 2 && (
                            <Chip 
                              label={`+${Object.keys(trial.parameters).length - 2} more`}
                              size="small"
                              variant="outlined"
                            />
                          )}
                        </Stack>
                      </TableCell>
                      <TableCell>{trial.duration}s</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Alert severity="info">
              No trials completed yet. Trial history will appear here as tuning progresses.
            </Alert>
          )}
          
          {progressData.trials.length > 10 && (
            <Box mt={2} textAlign="center">
              <Typography variant="body2" color="text.secondary">
                Showing 10 of {progressData.trials.length} trials
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default TuningProgress;
