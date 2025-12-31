/**
 * Tuning Error Boundary
 * 
 * Error boundary component specifically for tuning operations
 * Provides graceful error handling and recovery options
 */

import React, { Component } from 'react';
import type { ReactNode, ErrorInfo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Alert,
  AlertTitle,
  Stack,
} from '@mui/material';
import {
  Error as ErrorIcon,
  Refresh as RefreshIcon,
  Home as HomeIcon,
} from '@mui/icons-material';

interface Props {
  children: ReactNode;
  onReset?: () => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

class TuningErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
      errorInfo: null,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Tuning Error Boundary caught an error:', error, errorInfo);
    this.setState({
      error,
      errorInfo,
    });
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
    this.props.onReset?.();
  };

  render() {
    if (this.state.hasError) {
      return (
        <Box sx={{ p: 3 }}>
          <Paper sx={{ p: 4, textAlign: 'center' }}>
            <ErrorIcon sx={{ fontSize: 64, color: 'error.main', mb: 2 }} />
            
            <Typography variant="h5" gutterBottom>
              Something went wrong
            </Typography>
            
            <Typography variant="body1" color="text.secondary" paragraph>
              An error occurred while processing your tuning request.
            </Typography>

            {this.state.error && (
              <Alert severity="error" sx={{ mb: 3, textAlign: 'left' }}>
                <AlertTitle>Error Details</AlertTitle>
                <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                  {this.state.error.message}
                </Typography>
              </Alert>
            )}

            <Stack direction="row" spacing={2} justifyContent="center">
              <Button
                variant="contained"
                startIcon={<RefreshIcon />}
                onClick={this.handleReset}
              >
                Try Again
              </Button>
              <Button
                variant="outlined"
                startIcon={<HomeIcon />}
                onClick={() => window.location.href = '/'}
              >
                Go Home
              </Button>
            </Stack>
          </Paper>
        </Box>
      );
    }

    return this.props.children;
  }
}

export default TuningErrorBoundary;
