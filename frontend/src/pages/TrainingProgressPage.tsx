import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Container,
  Box,
  Typography,
  Button,
  Alert,
  CircularProgress,
  Breadcrumbs,
  Link,
} from '@mui/material';
import {
  ArrowBack as ArrowBackIcon,
  Home as HomeIcon,
} from '@mui/icons-material';
import TrainingProgress from '../components/model/TrainingProgress';
import { TrainingRun, TrainingLog } from '../types/training';
import { useWebSocket } from '../hooks/useWebSocket';

const TrainingProgressPage: React.FC = () => {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const [trainingRun, setTrainingRun] = useState<TrainingRun | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // WebSocket connection for real-time updates
  const { isConnected, lastMessage } = useWebSocket(
    `ws://localhost:8000/ws/training/${runId}`
  );

  // Fetch initial training run data
  useEffect(() => {
    const fetchTrainingRun = async () => {
      try {
        setLoading(true);
        setError(null);

        // TODO: Replace with actual API call
        const response = await fetch(`/api/training/runs/${runId}`);
        
        if (!response.ok) {
          throw new Error('Failed to fetch training run');
        }

        const data = await response.json();
        setTrainingRun(data);
      } catch (err) {
        console.error('Error fetching training run:', err);
        setError(err instanceof Error ? err.message : 'Failed to load training run');
        
        // For demo purposes, create mock data
        setTrainingRun(createMockTrainingRun(runId || ''));
      } finally {
        setLoading(false);
      }
    };

    if (runId) {
      fetchTrainingRun();
    }
  }, [runId]);

  // Handle WebSocket updates
  useEffect(() => {
    if (lastMessage && trainingRun) {
      try {
        const update = JSON.parse(lastMessage);
        
        setTrainingRun((prev) => {
          if (!prev) return prev;

          return {
            ...prev,
            status: update.status || prev.status,
            progress: update.progress !== undefined ? update.progress : prev.progress,
            currentEpoch: update.currentEpoch || prev.currentEpoch,
            metrics: {
              ...prev.metrics,
              ...update.metrics,
            },
            logs: update.log
              ? [...prev.logs, update.log]
              : prev.logs,
            endTime: update.status === 'completed' || update.status === 'failed'
              ? new Date().toISOString()
              : prev.endTime,
            duration: update.duration || prev.duration,
            error: update.error || prev.error,
          };
        });
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    }
  }, [lastMessage, trainingRun]);

  const handleCancel = async () => {
    try {
      // TODO: Replace with actual API call
      await fetch(`/api/training/runs/${runId}/cancel`, {
        method: 'POST',
      });

      setTrainingRun((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          status: 'cancelled',
          endTime: new Date().toISOString(),
        };
      });
    } catch (err) {
      console.error('Error cancelling training:', err);
      setError('Failed to cancel training');
    }
  };

  const handleRefresh = () => {
    window.location.reload();
  };

  const handleBack = () => {
    navigate('/modeling');
  };

  if (loading) {
    return (
      <Container maxWidth="lg">
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          minHeight="60vh"
        >
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  if (error && !trainingRun) {
    return (
      <Container maxWidth="lg">
        <Box py={4}>
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
          <Button
            variant="contained"
            startIcon={<ArrowBackIcon />}
            onClick={handleBack}
          >
            Back to Modeling
          </Button>
        </Box>
      </Container>
    );
  }

  if (!trainingRun) {
    return (
      <Container maxWidth="lg">
        <Box py={4}>
          <Alert severity="warning">Training run not found</Alert>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box py={4}>
        {/* Breadcrumbs */}
        <Breadcrumbs sx={{ mb: 2 }}>
          <Link
            component="button"
            variant="body2"
            onClick={() => navigate('/')}
            sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}
          >
            <HomeIcon fontSize="small" />
            Home
          </Link>
          <Link
            component="button"
            variant="body2"
            onClick={() => navigate('/modeling')}
          >
            Modeling
          </Link>
          <Typography color="text.primary">Training Progress</Typography>
        </Breadcrumbs>

        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h4" component="h1">
            Training Progress
          </Typography>
          <Button
            variant="outlined"
            startIcon={<ArrowBackIcon />}
            onClick={handleBack}
          >
            Back to Modeling
          </Button>
        </Box>

        {/* WebSocket Connection Status */}
        {!isConnected && trainingRun.status === 'running' && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            Real-time updates disconnected. Refresh to see latest status.
          </Alert>
        )}

        {/* Training Progress Component */}
        <TrainingProgress
          trainingRun={trainingRun}
          onCancel={trainingRun.status === 'running' ? handleCancel : undefined}
          onRefresh={handleRefresh}
          showMetricsChart={true}
        />

        {/* Action Buttons */}
        {trainingRun.status === 'completed' && (
          <Box mt={3} display="flex" gap={2}>
            <Button
              variant="contained"
              onClick={() => navigate(`/evaluation/${trainingRun.modelId}`)}
            >
              View Evaluation
            </Button>
            <Button
              variant="outlined"
              onClick={() => navigate(`/models/${trainingRun.modelId}`)}
            >
              View Model Details
            </Button>
          </Box>
        )}
      </Box>
    </Container>
  );
};

// Mock data generator for demo purposes
function createMockTrainingRun(runId: string): TrainingRun {
  const status: TrainingRun['status'] = 'running';
  const progress = 65;
  const currentEpoch = 13;
  const totalEpochs = 20;

  const logs: TrainingLog[] = [
    {
      id: '1',
      timestamp: new Date(Date.now() - 60000).toISOString(),
      level: 'info',
      message: 'Starting training process...',
    },
    {
      id: '2',
      timestamp: new Date(Date.now() - 55000).toISOString(),
      level: 'info',
      message: 'Loading dataset...',
    },
    {
      id: '3',
      timestamp: new Date(Date.now() - 50000).toISOString(),
      level: 'info',
      message: 'Dataset loaded: 1000 samples, 20 features',
    },
    {
      id: '4',
      timestamp: new Date(Date.now() - 45000).toISOString(),
      level: 'info',
      message: 'Splitting data: 70% train, 15% val, 15% test',
    },
    {
      id: '5',
      timestamp: new Date(Date.now() - 40000).toISOString(),
      level: 'info',
      message: 'Initializing Random Forest Classifier...',
    },
    {
      id: '6',
      timestamp: new Date(Date.now() - 35000).toISOString(),
      level: 'info',
      message: 'Training started with 100 estimators',
    },
    {
      id: '7',
      timestamp: new Date(Date.now() - 30000).toISOString(),
      level: 'info',
      message: 'Epoch 5/20 - Train Score: 0.8523, Val Score: 0.8234',
    },
    {
      id: '8',
      timestamp: new Date(Date.now() - 20000).toISOString(),
      level: 'info',
      message: 'Epoch 10/20 - Train Score: 0.8945, Val Score: 0.8567',
    },
    {
      id: '9',
      timestamp: new Date(Date.now() - 10000).toISOString(),
      level: 'info',
      message: `Epoch ${currentEpoch}/${totalEpochs} - Train Score: 0.9123, Val Score: 0.8789`,
    },
  ];

  return {
    id: runId,
    modelId: 'model-123',
    modelName: 'Random Forest Classifier',
    modelType: 'random_forest_classifier',
    status,
    progress,
    currentEpoch,
    totalEpochs,
    startTime: new Date(Date.now() - 60000).toISOString(),
    duration: 60,
    metrics: {
      trainScore: 0.9123,
      valScore: 0.8789,
      trainAccuracy: 0.9123,
      valAccuracy: 0.8789,
      history: {
        epoch: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        trainLoss: [0.45, 0.38, 0.32, 0.28, 0.25, 0.22, 0.20, 0.18, 0.17, 0.15, 0.14, 0.13, 0.12],
        valLoss: [0.48, 0.42, 0.37, 0.34, 0.31, 0.29, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21],
        trainAccuracy: [0.75, 0.80, 0.83, 0.85, 0.87, 0.88, 0.89, 0.90, 0.905, 0.908, 0.910, 0.911, 0.9123],
        valAccuracy: [0.72, 0.77, 0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.873, 0.875, 0.877, 0.878, 0.8789],
      },
    },
    logs,
  };
}

export default TrainingProgressPage;
