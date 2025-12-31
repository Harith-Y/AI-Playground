/**
 * Tuning Cancel Dialog
 * 
 * Confirmation dialog for cancelling tuning operations
 * with options to save partial results
 */

import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Alert,
  FormControlLabel,
  Checkbox,
  Box,
} from '@mui/material';
import {
  Warning as WarningIcon,
} from '@mui/icons-material';

interface TuningCancelDialogProps {
  open: boolean;
  onClose: () => void;
  onConfirm: (saveResults: boolean) => void;
  currentTrial?: number;
  totalTrials?: number;
}

const TuningCancelDialog: React.FC<TuningCancelDialogProps> = ({
  open,
  onClose,
  onConfirm,
  currentTrial = 0,
  totalTrials = 0,
}) => {
  const [saveResults, setSaveResults] = React.useState(true);

  const handleConfirm = () => {
    onConfirm(saveResults);
    onClose();
  };

  const progress = totalTrials > 0 ? (currentTrial / totalTrials) * 100 : 0;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <WarningIcon color="warning" />
        Cancel Tuning?
      </DialogTitle>
      
      <DialogContent>
        <Alert severity="warning" sx={{ mb: 2 }}>
          Are you sure you want to cancel the hyperparameter tuning process?
        </Alert>

        <Typography variant="body2" paragraph>
          Current progress: {currentTrial} of {totalTrials} trials completed ({progress.toFixed(1)}%)
        </Typography>

        <Typography variant="body2" color="text.secondary" paragraph>
          Cancelling will stop the tuning process immediately. You can choose to save the
          results from completed trials or discard them.
        </Typography>

        <Box sx={{ mt: 2 }}>
          <FormControlLabel
            control={
              <Checkbox
                checked={saveResults}
                onChange={(e) => setSaveResults(e.target.checked)}
              />
            }
            label="Save results from completed trials"
          />
        </Box>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>
          Continue Tuning
        </Button>
        <Button onClick={handleConfirm} color="error" variant="contained">
          Cancel Tuning
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default TuningCancelDialog;
