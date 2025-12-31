/**
 * Tuning Status Message
 * 
 * Displays contextual status messages for different tuning states
 * with appropriate icons and actions
 */

import React from 'react';
import {
  Alert,
  AlertTitle,
  Button,
  Box,
  LinearProgress,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Refresh as RefreshIcon,
  PlayArrow as PlayArrowIcon,
} from '@mui/icons-material';

interface TuningStatusMessageProps {
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
  message?: string;
  details?: string;
  onRetry?: () => void;
  onResume?: () => void;
  showProgress?: boolean;
}

const TuningStatusMessage: React.FC<TuningStatusMessageProps> = ({
  status,
  message,
  details,
  onRetry,
  onResume,
  showProgress = false,
}) => {
  const getAlertConfig = () => {
    switch (status) {
      case 'completed':
        return {
          severity: 'success' as const,
          icon: <CheckCircleIcon />,
          title: 'Tuning Completed',
          defaultMessage: 'Hyperparameter tuning completed successfully!',
        };
      case 'failed':
        return {
          severity: 'error' as const,
          icon: <ErrorIcon />,
          title: 'Tuning Failed',
          defaultMessage: 'An error occurred during hyperparameter tuning.',
        };
      case 'cancelled':
        return {
          severity: 'warning' as const,
          icon: <WarningIcon />,
          title: 'Tuning Cancelled',
          defaultMessage: 'Hyperparameter tuning was cancelled.',
        };
      case 'paused':
        return {
          severity: 'warning' as const,
          icon: <WarningIcon />,
          title: 'Tuning Paused',
          defaultMessage: 'Hyperparameter tuning is paused.',
        };
      case 'running':
        return {
          severity: 'info' as const,
          icon: <InfoIcon />,
          title: 'Tuning in Progress',
          defaultMessage: 'Hyperparameter tuning is currently running...',
        };
      default:
        return {
          severity: 'info' as const,
          icon: <InfoIcon />,
          title: 'Ready to Start',
          defaultMessage: 'Configure your tuning parameters and start the optimization.',
        };
    }
  };

  const config = getAlertConfig();

  return (
    <Alert
      severity={config.severity}
      icon={config.icon}
      sx={{ mb: 3 }}
    >
      <AlertTitle>{config.title}</AlertTitle>
      {message || config.defaultMessage}
      
      {details && (
        <Box sx={{ mt: 1, fontSize: '0.875rem', opacity: 0.9 }}>
          {details}
        </Box>
      )}

      {showProgress && status === 'running' && (
        <LinearProgress sx={{ mt: 2 }} />
      )}

      {status === 'failed' && onRetry && (
        <Button
          size="small"
          startIcon={<RefreshIcon />}
          onClick={onRetry}
          sx={{ mt: 2 }}
        >
          Retry
        </Button>
      )}

      {status === 'paused' && onResume && (
        <Button
          size="small"
          startIcon={<PlayArrowIcon />}
          onClick={onResume}
          sx={{ mt: 2 }}
        >
          Resume
        </Button>
      )}
    </Alert>
  );
};

export default TuningStatusMessage;
