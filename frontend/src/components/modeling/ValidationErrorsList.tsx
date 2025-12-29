import React from 'react';
import {
  Alert,
  AlertTitle,
  Box,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
} from '@mui/material';
import {
  Error as ErrorIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import type { ValidationError } from '../../utils/trainingErrorHandler';

export interface ValidationErrorsListProps {
  errors: ValidationError[];
  onDismiss?: () => void;
}

const ValidationErrorsList: React.FC<ValidationErrorsListProps> = ({
  errors,
  onDismiss,
}) => {
  if (errors.length === 0) return null;

  const errorCount = errors.filter(e => e.severity === 'error').length;
  const warningCount = errors.filter(e => e.severity === 'warning').length;

  const severity = errorCount > 0 ? 'error' : 'warning';
  const title = errorCount > 0
    ? `${errorCount} Configuration Error${errorCount > 1 ? 's' : ''}`
    : `${warningCount} Configuration Warning${warningCount > 1 ? 's' : ''}`;

  return (
    <Alert severity={severity} onClose={onDismiss}>
      <AlertTitle>
        <Box display="flex" alignItems="center" gap={1}>
          {title}
          {errorCount > 0 && warningCount > 0 && (
            <Chip
              label={`${warningCount} warning${warningCount > 1 ? 's' : ''}`}
              size="small"
              color="warning"
            />
          )}
        </Box>
      </AlertTitle>

      <List dense>
        {errors.map((error, index) => (
          <ListItem key={index} sx={{ py: 0.5, px: 0 }}>
            <ListItemIcon sx={{ minWidth: 32 }}>
              {error.severity === 'error' ? (
                <ErrorIcon fontSize="small" color="error" />
              ) : (
                <WarningIcon fontSize="small" color="warning" />
              )}
            </ListItemIcon>
            <ListItemText
              primary={
                <Box>
                  <strong>{error.field}:</strong> {error.message}
                </Box>
              }
              primaryTypographyProps={{ variant: 'body2' }}
            />
          </ListItem>
        ))}
      </List>

      {errorCount > 0 && (
        <Box mt={1}>
          <Alert severity="info" sx={{ py: 0.5 }}>
            Please fix all errors before starting training.
          </Alert>
        </Box>
      )}
    </Alert>
  );
};

export default ValidationErrorsList;
