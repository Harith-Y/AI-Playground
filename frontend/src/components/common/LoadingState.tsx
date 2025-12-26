import React from 'react';
import { Box, CircularProgress, LinearProgress, Typography } from '@mui/material';

interface LoadingStateProps {
  message?: string;
  variant?: 'circular' | 'linear';
  fullscreen?: boolean;
  size?: number;
}

const LoadingState: React.FC<LoadingStateProps> = ({
  message = 'Loading...',
  variant = 'circular',
  fullscreen = false,
  size = 40,
}) => {
  const content = (
    <>
      {variant === 'circular' ? (
        <CircularProgress size={size} sx={{ color: 'primary.main' }} />
      ) : (
        <Box sx={{ width: '100%', maxWidth: 400 }}>
          <LinearProgress />
        </Box>
      )}
      {message && (
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ mt: 2 }}
          align="center"
        >
          {message}
        </Typography>
      )}
    </>
  );

  if (fullscreen) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
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

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        p: 4,
        width: '100%',
      }}
    >
      {content}
    </Box>
  );
};

export default LoadingState;
