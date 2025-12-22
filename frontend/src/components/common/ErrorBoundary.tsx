import React, { Component } from 'react';
import type { ReactNode } from 'react';
import { Box, Button, Typography, Paper } from '@mui/material';
import type { SxProps, Theme } from '@mui/material';
import { ErrorOutline, Refresh } from '@mui/icons-material';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onReset?: () => void;
  fullscreen?: boolean;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({
      error,
      errorInfo,
    });
  }

  handleReset = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
    
    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  render(): ReactNode {
    const { hasError, error, errorInfo } = this.state;
    const { children, fallback, fullscreen = true } = this.props;

    if (hasError) {
      if (fallback) {
        return fallback;
      }

      const containerSx: SxProps<Theme> = fullscreen
        ? {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '100vh',
            width: '100%',
            backgroundColor: 'background.default',
            p: 3,
          }
        : {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            p: 3,
          };

      return (
        <Box sx={containerSx}>
          <ErrorOutline
            sx={{ fontSize: 80, color: 'error.main', mb: 3 }}
          />
          <Typography variant="h4" component="h1" fontWeight={600} gutterBottom>
            Oops! Something went wrong
          </Typography>
          <Typography
            variant="body1"
            color="text.secondary"
            sx={{ mb: 4, textAlign: 'center', maxWidth: 600 }}
          >
            We apologize for the inconvenience. An unexpected error has occurred.
            Please try refreshing the page or contact support if the problem persists.
          </Typography>

          <Paper
            sx={{
              p: 3,
              maxWidth: 800,
              width: '100%',
              backgroundColor: 'background.paper',
              border: '1px solid #334155',
              mb: 3,
            }}
          >
            <Typography
              variant="subtitle2"
              color="error"
              sx={{ mb: 2, fontWeight: 600 }}
            >
              Error Details:
            </Typography>
            <Typography
              variant="body2"
              sx={{
                fontFamily: 'monospace',
                color: 'text.secondary',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}
            >
              {error?.toString()}
            </Typography>
            
            {import.meta.env.DEV && errorInfo && (
              <>
                <Typography
                  variant="subtitle2"
                  color="error"
                  sx={{ mt: 3, mb: 2, fontWeight: 600 }}
                >
                  Component Stack:
                </Typography>
                <Typography
                  variant="body2"
                  sx={{
                    fontFamily: 'monospace',
                    color: 'text.secondary',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    maxHeight: 200,
                    overflow: 'auto',
                  }}
                >
                  {errorInfo.componentStack}
                </Typography>
              </>
            )}
          </Paper>

          <Button
            variant="contained"
            startIcon={<Refresh />}
            onClick={this.handleReset}
            sx={{
              background: 'linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%)',
              '&:hover': {
                background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
              },
            }}
          >
            Try Again
          </Button>
        </Box>
      );
    }

    return children;
  }
}

export default ErrorBoundary;
