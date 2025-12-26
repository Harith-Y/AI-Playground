import React, { useState } from 'react';
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
} from '@mui/material';
import { Info, Close } from '@mui/icons-material';
import { useAppSelector } from '../../hooks/redux';
import type { PreprocessingStepCreate, StepType, ParameterField } from '../../types/preprocessing';
import { STEP_TYPE_CONFIGS } from '../../types/preprocessing';

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

  const selectedConfig = stepType ? STEP_TYPE_CONFIGS[stepType] : null;

  const handleStepTypeChange = (type: StepType) => {
    setStepType(type);
    setParameters({});
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
    const newErrors: Record<string, string> = {};

    if (!stepType) {
      newErrors.stepType = 'Please select a step type';
    }

    if (selectedConfig?.requiresColumn && !selectedColumn) {
      newErrors.column = 'Please select a column';
    }

    // Validate required parameters
    selectedConfig?.parameterSchema.forEach((field: ParameterField) => {
      if (field.required && !parameters[field.name]) {
        newErrors[field.name] = `${field.label} is required`;
      }

      // Type-specific validation
      if (parameters[field.name] !== undefined && parameters[field.name] !== '') {
        if (field.type === 'number') {
          const value = Number(parameters[field.name]);
          if (isNaN(value)) {
            newErrors[field.name] = 'Must be a valid number';
          } else if (field.min !== undefined && value < field.min) {
            newErrors[field.name] = `Must be at least ${field.min}`;
          } else if (field.max !== undefined && value > field.max) {
            newErrors[field.name] = `Must be at most ${field.max}`;
          }
        }
      }
    });

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
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
          {/* Step Type Selection */}
          <Grid item xs={12}>
            <Typography variant="subtitle2" gutterBottom fontWeight={600}>
              Step Type *
            </Typography>
            <Grid container spacing={1}>
              {(Object.keys(STEP_TYPE_CONFIGS) as StepType[]).map((type) => {
                const config = STEP_TYPE_CONFIGS[type];
                const isSelected = stepType === type;

                return (
                  <Grid item xs={12} sm={6} key={type}>
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
            <Grid item xs={12}>
              <FormControl fullWidth error={!!errors.column}>
                <InputLabel>Target Column *</InputLabel>
                <Select
                  value={selectedColumn}
                  label="Target Column *"
                  onChange={(e) => setSelectedColumn(e.target.value)}
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
            </Grid>
          )}

          {/* Parameters */}
          {selectedConfig && selectedConfig.parameterSchema.length > 0 && (
            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom fontWeight={600}>
                Parameters
              </Typography>
              <Grid container spacing={2}>
                {selectedConfig.parameterSchema.map((field) => (
                  <Grid item xs={12} sm={6} key={field.name}>
                    {renderParameterField(field)}
                  </Grid>
                ))}
              </Grid>
            </Grid>
          )}

          {/* Info Alert */}
          {selectedConfig && (
            <Grid item xs={12}>
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
