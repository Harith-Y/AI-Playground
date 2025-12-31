/**
 * Tuning Timeout Warning
 * 
 * Warning component for long-running tuning operations
 * Provides options to continue waiting or cancel
 */

import React, { useState, useEffect } from 'react';
import {
  Alert,
  AlertTitle,
  Button,
  Box,
  LinearProgress,
  Typography,
} from '@mui/material';
import {
  AccessTime as AccessTimeIcon,
  Cancel as CancelIcon,
} from '@mui/icons-material';

interface TuningTimeoutWarningProps {
  elapsedTime: number;
  estimatedTime: number;
  warningThreshold?: number; // seconds
  onCancel?: () => void;
}

const TuningTimeoutWarning: React.FC<TuningTimeoutWarningProps> = ({
  elapsedTime,
  estimatedTime,
  warningThreshold = 1800, // 30 minutes default
  onCancel,
}) => {
  const [showWarning, setShowWarning] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    if (elapsedTime > warningThreshold && !dismissed) {
      setShowWarning(true);
    }
  }, [elapsedTime, warningThreshold, dismissed]);

  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  const handleDismiss = () => {
    setDismissed(true);
    setShowWarning(false);
  };

  if (!showWarning) {
    return null;
  }

  const isOverEstimate = elapsedTime > estimatedTime;

  return (
    <Alert
      severity="warning"
      icon={<AccessTimeIcon />}
      sx={{ mb: 3 }}
      onClose={handleDismiss}
    >
      <AlertTitle>Long Running Tuning Process</AlertTitle>
      
      <Typography variant="body2" paragraph>
        This tuning process has been running for {formatTime(elapsedTime)}.
        {isOverEstimate && ' It has exceeded the estimated completion time.'}
      </Typography>

      <Box sx={{ mb: 2 }}>
        <Typography variant="caption" color="text.secondary">
          Elapsed: {formatTime(elapsedTime)} / Estimated: {formatTime(estimatedTime)}
        </Typography>
        <LinearProgress
          variant="determinate"
          value={Math.min((elapsedTime / estimatedTime) * 100, 100)}
          sx={{ mt: 1, height: 6, borderRadius: 3 }}
          color={isOverEstimate ? 'warning' : 'primary'}
        />
      </Box>

      <Typography variant="body2" color="text.secondary" paragraph>
        You can continue waiting for the process to complete, or cancel it to save partial results.
      </Typography>

      {onCancel && (
        <Button
          size="small"
          color="error"
          startIcon={<CancelIcon />}
          onClick={onCancel}
          variant="outlined"
        >
          Cancel Tuning
        </Button>
      )}
    </Alert>
  );
};

export default TuningTimeoutWarning;
