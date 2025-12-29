import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
  Box,
  Typography,
} from '@mui/material';
import {
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Help as HelpIcon,
} from '@mui/icons-material';

export interface ConfirmDialogProps {
  open: boolean;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  severity?: 'warning' | 'error' | 'info' | 'question';
  onConfirm: () => void;
  onCancel: () => void;
  loading?: boolean;
}

const ConfirmDialog: React.FC<ConfirmDialogProps> = ({
  open,
  title,
  message,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  severity = 'warning',
  onConfirm,
  onCancel,
  loading = false,
}) => {
  const getIcon = () => {
    switch (severity) {
      case 'error':
        return <ErrorIcon color="error" sx={{ fontSize: 48 }} />;
      case 'warning':
        return <WarningIcon color="warning" sx={{ fontSize: 48 }} />;
      case 'info':
        return <InfoIcon color="info" sx={{ fontSize: 48 }} />;
      case 'question':
        return <HelpIcon color="primary" sx={{ fontSize: 48 }} />;
      default:
        return <WarningIcon color="warning" sx={{ fontSize: 48 }} />;
    }
  };

  const getColor = () => {
    switch (severity) {
      case 'error':
        return 'error';
      case 'warning':
        return 'warning';
      case 'info':
        return 'info';
      case 'question':
        return 'primary';
      default:
        return 'warning';
    }
  };

  return (
    <Dialog
      open={open}
      onClose={loading ? undefined : onCancel}
      maxWidth="sm"
      fullWidth
    >
      <DialogTitle>
        <Box display="flex" alignItems="center" gap={2}>
          {getIcon()}
          <Typography variant="h6">{title}</Typography>
        </Box>
      </DialogTitle>
      <DialogContent>
        <DialogContentText>{message}</DialogContentText>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button
          onClick={onCancel}
          disabled={loading}
          variant="outlined"
        >
          {cancelText}
        </Button>
        <Button
          onClick={onConfirm}
          disabled={loading}
          variant="contained"
          color={getColor()}
          autoFocus
        >
          {loading ? 'Processing...' : confirmText}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ConfirmDialog;
