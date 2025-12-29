import React, { useState } from 'react';
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
  Snackbar,
} from '@mui/material';
import {
  ArrowBack as ArrowBackIcon,
  Home as HomeIcon,
} from '@mui/icons-material';
import TrainingProgress from '../components/model/TrainingProgress';
import ConfirmDialog from '../components/common/ConfirmDialog';
import { useTrainingProgress } from '../hooks/useTrainingProgress';

const TrainingProgressPage: React.FC = () => {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  
  const [showStopDialog, setShowStopDialog] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'info';
  }>({
    open: false,
    message: '',
    severity: 'info',
  });

  // Use the training progress hook
  const {
    trainingRun,
    loading,
    error,
    isConnected,
    connectionState,
    refresh,
    stopTraining,
  } = useTrainingProgress({
    runId: runId || '',
    enableWebSocket: true,
    pollingInterval: 5000,
    onStatusChange: (status) => {
      console.log('[TrainingProgressPage] Status changed:', status);
      
      if (status === 'completed') {
        setSnackbar({
          open: true,
          message: 'Training completed successfully!',
          severity: 'success',
        });
      } else if (status === 'failed') {
        setSnackbar({
          open: true,
          message: 'Training failed. Check logs for details.',
          severity: 'error',
        });
      } else if (status === 'cancelled') {
        setSnackbar({
          open: true,
          message: 'Training was cancelled.',
          severity: 'info',
        });
      }
    },
    onComplete: (run) => {
      console.log('[TrainingProgressPage] Training completed:', run);
    },
    onError: (errorMsg) => {
      console.error('[TrainingProgressPage] Training error:', errorMsg);
    },
  });

  const handleStopClick = () => {
    setShowStopDialog(true);
  };

  const handleStopConfirm = async () => {
    try {
      setStopping(true);
      await stopTraining();
      
      setSnackbar({
        open: true,
        message: 'Training stopped successfully',
        severity: 'success',
      });
      
      setShowStopDialog(false);
    } catch (err) {
      console.error('Error stopping training:', err);
      
      setSnackbar({
        open: true,
        message: 'Failed to stop training. Please try again.',
        severity: 'error',
      });
    } finally {
      setStopping(false);
    }
  };

  const handleStopCancel = () => {
    setShowStopDialog(false);
  };

  const handleRefresh = () => {
    refresh();
  };

  const handleBack = () => {
    navigate('/modeling');
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
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
            {connectionState === 'connecting' && ' Reconnecting...'}
          </Alert>
        )}

        {connectionState === 'connected' && trainingRun.status === 'running' && (
          <Alert severity="success" sx={{ mb: 2 }}>
            Connected - Receiving real-time updates
          </Alert>
        )}

        {/* Training Progress Component */}
        <TrainingProgress
          trainingRun={trainingRun}
          onCancel={trainingRun.status === 'running' ? handleStopClick : undefined}
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

        {/* Stop Training Confirmation Dialog */}
        <ConfirmDialog
          open={showStopDialog}
          title="Stop Training?"
          message="Are you sure you want to stop this training run? This action cannot be undone. The model will be saved in its current state."
          confirmText="Stop Training"
          cancelText="Continue Training"
          severity="warning"
          onConfirm={handleStopConfirm}
          onCancel={handleStopCancel}
          loading={stopping}
        />

        {/* Snackbar for notifications */}
        <Snackbar
          open={snackbar.open}
          autoHideDuration={6000}
          onClose={handleCloseSnackbar}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert
            onClose={handleCloseSnackbar}
            severity={snackbar.severity}
            sx={{ width: '100%' }}
          >
            {snackbar.message}
          </Alert>
        </Snackbar>
      </Box>
    </Container>
  );
};

export default TrainingProgressPage;
