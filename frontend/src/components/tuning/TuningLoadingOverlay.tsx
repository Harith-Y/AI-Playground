/**
 * Tuning Loading Overlay
 * 
 * Loading overlay for tuning operations with progress indication
 * and cancellation support
 */

import React from 'react';
import {
  Box,
  CircularProgress,
  Typography,
  Button,
  Paper,
  LinearProgress,
} from '@mui/material';
import {
  Cancel as CancelIcon,
} from '@mui/icons-material';

interface TuningLoadingOverlayProps {
  message?: string;
  progress?: number;
  onCancel?: () => void;
  cancelable?: boolean;
}

const TuningLoadingOverlay: React.FC<TuningLoadingOverlayProps> = ({
  message = 'Processing...',
  progress,
  onCancel,
  cancelable = false,
}) => {
  return (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        bgcolor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999,
      }}
    >
      <Paper
        sx={{
          p: 4,
          minWidth: 300,
          maxWidth: 500,
          textAlign: 'center',
        }}
      >
        <CircularProgress size={60} sx={{ mb: 2 }} />
        
        <Typography variant="h6" gutterBottom>
          {message}
        </Typography>

        {progress !== undefined && (
          <Box sx={{ mt: 2, mb: 2 }}>
            <LinearProgress variant="determinate" value={progress} sx={{ height: 8, borderRadius: 4 }} />
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              {progress.toFixed(0)}% complete
            </Typography>
          </Box>
        )}

        {cancelable && onCancel && (
          <Button
            variant="outlined"
            color="error"
            startIcon={<CancelIcon />}
            onClick={onCancel}
            sx={{ mt: 2 }}
          >
            Cancel
          </Button>
        )}
      </Paper>
    </Box>
  );
};

export default TuningLoadingOverlay;
