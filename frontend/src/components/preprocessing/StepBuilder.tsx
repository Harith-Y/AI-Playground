import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Box,
  Typography,
  Chip,
  Grid,
  Switch,
  FormControlLabel,
  Tooltip,
  IconButton,
  Alert,
  AlertTitle,
} from '@mui/material';
import { Info, Close, Warning, Error as ErrorIcon } from '@mui/icons-material';
import { useAppSelector } from '../../hooks/redux';
import type { PreprocessingStepCreate, StepType, ParameterField } from '../../types/preprocessing';
import { STEP_TYPE_CONFIGS } from '../../types/preprocessing';
import { validatePreprocessingStep, validateColumnForStep } from '../../utils/preprocessingValidation';

interface StepBuilderProps {
  open: boolean;
  onClose: () => void;
  onSave: (step: PreprocessingStepCreate) => void;
  editStep?: PreprocessingStepCreate | null;
}

const StepBuilder: React.FC<StepBuilderProps> = ({ open, onClose, onSave, editStep }) => {
  const { currentDataset, columns } = useAppSelector((state) => state.dataset);

  const [stepType, setStepType] = useState<StepType | ''>(editStep?.step_type || '');
  const [selectedColumn, setSelectedColumn] = useState<string>(editStep?.column_name || '');
  const [parameters, setParameters] = useState<Record<string, any>>(editStep?.parameters || {});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [columnWarning, setColumnWarning] = useState<string | null>(null);
  const [generalError, setGeneralError] = useState<string | null>(null);

  const selectedConfig = stepType ? STEP_TYPE_CONFIGS[stepType] : null;

  // Validate column compatibility when column or step type changes
  useEffect(() => {
    if (stepType && selectedColumn && selectedConfig?.requiresColumn) {
      const column = columns.find(col => col.name === selectedColumn);
      if (column) {
        const validation = validateColumnForStep(selectedColumn, column.dtype, stepType);
        setColumnWarning(validation.warning || null);
      }
    } else {
      setColumnWarning(null);
    }
  }, [stepType, selectedColumn, columns, selectedConfig]);

  const handleStepTypeChange = (type: StepType) => {
    setStepType(type);
    
    // Set default parameters for the selected step type
    const config = STEP_TYPE_CONFIGS[type];
    const defaultParams: Record<string, any> = {};
    
    if (config && config.parameterSchema) {
      config.parameterSchema.forEach((field: ParameterField) => {
        if (field.defaultValue !== undefined) {
          defaultParams[field.name] = field.defaultValue;
        }
      });
    }
    
    setParameters(defaultParams);
    setSelectedColumn('');
    setErrors({});
  };

  const handleParameterChange = (fieldName: string, value: any) => {
    setParameters(prev => ({
      ...prev,
      [fieldName]: value,
    }));

    // Clear error for this field
    if (errors[fieldName]) {
      setErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[fieldName];
        return newErrors;
      });
    }
  };

  const validate = (): boolean => {
    if (!currentDataset || !stepType) {
      setGeneralError('Invalid configuration. Please select a step type.');
      return false;
    }

    // Check if dataset is empty
    if (columns.length === 0) {
      setGeneralError('Cannot create preprocessing step. Dataset has no columns.');
      return false;
    }

    // Use validation utility
    const validationResult = validatePreprocessingStep(
      {
        dataset_id: currentDataset.id,
        step_type: stepType,
        parameters,
        column_name: selectedColumn,
      },
      columns
    );

    setErrors(validationResult.errors);
    setGeneralError(null);

    // Show general error if column warning exists and it's invalid
    if (columnWarning && !columnWarning.toLowerCase().includes('may')) {
      setGeneralError(columnWarning);
    }

    return validationResult.isValid && !columnWarning?.toLowerCase().includes('requires');
  };

  const handleSave = () => {
    if (!validate() || !currentDataset || !stepType) return;

    const step: PreprocessingStepCreate = {
      dataset_id: currentDataset.id,
      step_type: stepType,
      parameters,
      ...(selectedConfig?.requiresColumn && { column_name: selectedColumn }),
    };

    onSave(step);
    handleClose();
  };

  const handleClose = () => {
    setStepType('');
    setSelectedColumn('');
    setParameters({});
    setErrors({});
    setColumnWarning(null);
    setGeneralError(null);
    onClose();
  };

  const renderParameterField = (field: ParameterField) => {
    const value = parameters[field.name] ?? field.defaultValue ?? '';

    switch (field.type) {
      case 'select':
        return (
          <FormControl fullWidth error={!!errors[field.name]} key={field.name}>
            <InputLabel>{field.label}</InputLabel>
            <Select
              value={value}
              label={field.label}
              onChange={(e) => handleParameterChange(field.name, e.target.value)}
            >
              {field.options?.map((option) => (
                <MenuItem key={option.value} value={option.value}>
                  {option.label}
                </MenuItem>
              ))}
            </Select>
            {errors[field.name] && (
              <Typography variant="caption" color="error" sx={{ mt: 0.5 }}>
                {errors[field.name]}
              </Typography>
            )}
          </FormControl>
        );

      case 'number':
        return (
          <TextField
            key={field.name}
            fullWidth
            type="number"
            label={field.label}
            value={value}
            onChange={(e) => handleParameterChange(field.name, e.target.value)}
            error={!!errors[field.name]}
            helperText={errors[field.name] || field.tooltip}
            inputProps={{
              min: field.min,
              max: field.max,
              step: field.step || 1,
            }}
          />
        );

      case 'boolean':
        return (
          <FormControlLabel
            key={field.name}
            control={
              <Switch
                checked={value === true}
                onChange={(e) => handleParameterChange(field.name, e.target.checked)}
              />
            }
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {field.label}
                {field.tooltip && (
                  <Tooltip title={field.tooltip}>
                    <Info fontSize="small" color="action" />
                  </Tooltip>
                )}
              </Box>
            }
          />
        );

      case 'text':
      default:
        return (
          <TextField
            key={field.name}
            fullWidth
            label={field.label}
            value={value}
            onChange={(e) => handleParameterChange(field.name, e.target.value)}
            error={!!errors[field.name]}
            helperText={errors[field.name] || field.tooltip}
          />
        );
    }
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6">
            {editStep ? 'Edit Preprocessing Step' : 'Add Preprocessing Step'}
          </Typography>
          <IconButton onClick={handleClose} size="small">
            <Close />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent dividers>
        <Grid container spacing={3}>
          {/* General Error Alert */}
          {generalError && (
            <Grid size={{ xs: 12 }}>
              <Alert severity="error" icon={<ErrorIcon />}>
                <AlertTitle>Configuration Error</AlertTitle>
                {generalError}
              </Alert>
            </Grid>
          )}

          {/* Empty Dataset Warning */}
          {columns.length === 0 && (
            <Grid size={{ xs: 12 }}>
              <Alert severity="warning" icon={<Warning />}>
                <AlertTitle>No Columns Available</AlertTitle>
                The selected dataset has no columns. Please upload a valid dataset before creating preprocessing steps.
              </Alert>
            </Grid>
          )}

          {/* Step Type Selection */}
          <Grid size={{ xs: 12 }}>
            <Typography variant="subtitle2" gutterBottom fontWeight={600}>
              Step Type *
            </Typography>
            <Grid container spacing={1}>
              {(Object.keys(STEP_TYPE_CONFIGS) as StepType[]).map((type) => {
                const config = STEP_TYPE_CONFIGS[type];
                const isSelected = stepType === type;

                return (
                  <Grid size={{ xs: 12, sm: 6 }} key={type}>
                    <Box
                      onClick={() => handleStepTypeChange(type)}
                      sx={{
                        p: 2,
                        border: '1px solid',
                        borderColor: isSelected ? 'primary.main' : 'divider',
                        borderRadius: 1,
                        cursor: 'pointer',
                        bgcolor: isSelected ? 'primary.50' : 'background.paper',
                        transition: 'all 0.2s',
                        '&:hover': {
                          borderColor: 'primary.main',
                          bgcolor: 'primary.50',
                        },
                      }}
                    >
                      <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                        {config.label}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {config.description}
                      </Typography>
                    </Box>
                  </Grid>
                );
              })}
            </Grid>
            {errors.stepType && (
              <Typography variant="caption" color="error" sx={{ mt: 1, display: 'block' }}>
                {errors.stepType}
              </Typography>
            )}
          </Grid>

          {/* Column Selection */}
          {selectedConfig?.requiresColumn && (
            <Grid size={{ xs: 12 }}>
              <FormControl fullWidth error={!!errors.column}>
                <InputLabel>Target Column *</InputLabel>
                <Select
                  value={selectedColumn}
                  label="Target Column *"
                  onChange={(e) => setSelectedColumn(e.target.value)}
                  disabled={columns.length === 0}
                >
                  {columns.map((col) => (
                    <MenuItem key={col.name} value={col.name}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                        <span>{col.name}</span>
                        <Chip label={col.dtype} size="small" sx={{ ml: 2 }} />
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
                {errors.column && (
                  <Typography variant="caption" color="error" sx={{ mt: 0.5 }}>
                    {errors.column}
                  </Typography>
                )}
              </FormControl>

              {/* Column Type Warning */}
              {columnWarning && (
                <Alert
                  severity={columnWarning.toLowerCase().includes('requires') ? 'error' : 'warning'}
                  sx={{ mt: 2 }}
                  icon={<Warning />}
                >
                  {columnWarning}
                </Alert>
              )}
            </Grid>
          )}

          {/* Parameters */}
          {selectedConfig && selectedConfig.parameterSchema.length > 0 && (
            <Grid size={{ xs: 12 }}>
              <Typography variant="subtitle2" gutterBottom fontWeight={600}>
                Parameters
              </Typography>
              <Grid container spacing={2}>
                {selectedConfig.parameterSchema
                  .filter((field) => {
                    // Conditionally show fill_value only when strategy is "constant"
                    if (field.name === 'fill_value') {
                      return parameters['strategy'] === 'constant';
                    }
                    return true;
                  })
                  .map((field) => (
                    <Grid size={{ xs: 12, sm: 6 }} key={field.name}>
                      {renderParameterField(field)}
                    </Grid>
                  ))}
              </Grid>
            </Grid>
          )}

          {/* Info Alert */}
          {selectedConfig && (
            <Grid size={{ xs: 12 }}>
              <Alert severity="info" icon={<Info />}>
                <Typography variant="body2">
                  <strong>{selectedConfig.label}:</strong> {selectedConfig.description}
                </Typography>
              </Alert>
            </Grid>
          )}
        </Grid>
      </DialogContent>

      <DialogActions sx={{ px: 3, py: 2 }}>
        <Button onClick={handleClose} color="inherit">
          Cancel
        </Button>
        <Button onClick={handleSave} variant="contained" disabled={!stepType}>
          {editStep ? 'Update Step' : 'Add Step'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default StepBuilder;
