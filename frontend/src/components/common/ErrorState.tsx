import React from 'react';
import { Box, Typography, Button, Alert, AlertTitle } from '@mui/material';
import { ErrorOutline, Refresh } from '@mui/icons-material';

interface ErrorStateProps {
  message?: string;
  title?: string;
  onRetry?: () => void;
  fullscreen?: boolean;
  variant?: 'alert' | 'banner';
}

const ErrorState: React.FC<ErrorStateProps> = ({
  message = 'An unexpected error occurred',
  title = 'Error',
  onRetry,
  fullscreen = false,
  variant = 'banner',
}) => {
  if (variant === 'alert') {
    return (
      <Box sx={{ width: '100%', mt: 2 }}>
        <Alert
          severity="error"
          action={
            onRetry && (
              <Button
                color="inherit"
                size="small"
                onClick={onRetry}
                startIcon={<Refresh />}
              >
                Retry
              </Button>
            )
          }
        >
          {title !== 'Error' && <AlertTitle>{title}</AlertTitle>}
          {message}
        </Alert>
      </Box>
    );
  }

  const content = (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        p: 4,
      }}
    >
      <ErrorOutline
        sx={{
          fontSize: 64,
          color: 'error.main',
          mb: 2,
        }}
      />
      <Typography variant="h6" fontWeight={600} gutterBottom>
        {title}
      </Typography>
      <Typography
        variant="body2"
        color="text.secondary"
        sx={{ mb: 3, maxWidth: 500 }}
      >
        {message}
      </Typography>
      {onRetry && (
        <Button
          variant="contained"
          startIcon={<Refresh />}
          onClick={onRetry}
          sx={{ mt: 1 }}
        >
          Try Again
        </Button>
      )}
    </Box>
  );

  if (fullscreen) {
    return (
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '100vh',
          width: '100%',
          backgroundColor: 'background.default',
        }}
      >
        {content}
      </Box>
    );
  }

  return content;
};

export default ErrorState;
