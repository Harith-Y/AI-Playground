import React, { useState } from 'react';
import {
  Alert,
  AlertTitle,
  Box,
  Button,
  Collapse,
  IconButton,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Typography,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Lightbulb as LightbulbIcon,
  Refresh as RefreshIcon,
  Close as CloseIcon,
} from '@mui/icons-material';
import type { ErrorHandlerResult } from '../../utils/trainingErrorHandler';

export interface TrainingErrorDisplayProps {
  error: ErrorHandlerResult;
  onRetry?: () => void;
  onDismiss?: () => void;
  showTechnicalDetails?: boolean;
}

const TrainingErrorDisplay: React.FC<TrainingErrorDisplayProps> = ({
  error,
  onRetry,
  onDismiss,
  showTechnicalDetails = false,
}) => {
  const [showDetails, setShowDetails] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);

  return (
    <Alert
      severity={error.severity}
      action={
        <Box display="flex" gap={0.5}>
          {error.retryable && onRetry && (
            <Button
              color="inherit"
              size="small"
              startIcon={<RefreshIcon />}
              onClick={onRetry}
            >
              Retry
            </Button>
          )}
          {onDismiss && (
            <IconButton
              size="small"
              color="inherit"
              onClick={onDismiss}
            >
              <CloseIcon fontSize="small" />
            </IconButton>
          )}
        </Box>
      }
    >
      <AlertTitle>{error.userMessage}</AlertTitle>

      {/* Technical Details */}
      {showTechnicalDetails && error.technicalMessage && (
        <Box mt={1}>
          <Button
            size="small"
            onClick={() => setShowDetails(!showDetails)}
            endIcon={showDetails ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            sx={{ textTransform: 'none', p: 0, minWidth: 0 }}
          >
            Technical Details
          </Button>
          <Collapse in={showDetails}>
            <Box
              mt={1}
              p={1}
              sx={{
                bgcolor: 'rgba(0, 0, 0, 0.05)',
                borderRadius: 1,
                fontFamily: 'monospace',
                fontSize: '0.875rem',
                wordBreak: 'break-word',
              }}
            >
              {error.technicalMessage}
            </Box>
          </Collapse>
        </Box>
      )}

      {/* Suggestions */}
      {error.suggestions.length > 0 && (
        <Box mt={2}>
          <Button
            size="small"
            onClick={() => setShowSuggestions(!showSuggestions)}
            startIcon={<LightbulbIcon />}
            endIcon={showSuggestions ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            sx={{ textTransform: 'none', p: 0, minWidth: 0 }}
          >
            Suggestions ({error.suggestions.length})
          </Button>
          <Collapse in={showSuggestions}>
            <List dense sx={{ mt: 1 }}>
              {error.suggestions.map((suggestion, index) => (
                <ListItem key={index} sx={{ py: 0.5, px: 0 }}>
                  <ListItemIcon sx={{ minWidth: 32 }}>
                    <LightbulbIcon fontSize="small" color="action" />
                  </ListItemIcon>
                  <ListItemText
                    primary={suggestion}
                    primaryTypographyProps={{ variant: 'body2' }}
                  />
                </ListItem>
              ))}
            </List>
          </Collapse>
        </Box>
      )}

      {/* Recovery Info */}
      {!error.recoverable && (
        <Box mt={2}>
          <Typography variant="body2" color="text.secondary">
            This error cannot be automatically recovered. Please contact support.
          </Typography>
        </Box>
      )}
    </Alert>
  );
};

export default TrainingErrorDisplay;
