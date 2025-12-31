/**
 * HyperparameterEditor Component
 *
 * Edits hyperparameters for selected model with preset management
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Slider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Switch,
  FormControlLabel,
  Stack,
  Chip,
  // Button, // Unused
  Divider,
  Tooltip,
  IconButton,
  Alert,
} from '@mui/material';
import {
  Info as InfoIcon,
  Restore as RestoreIcon,
  Tune as TuneIcon,
} from '@mui/icons-material';
import type {
  HyperparameterEditorProps,
  HyperparameterDef,
} from '../../types/modelSelection';
import { HyperparamType } from '../../types/modelSelection';

const HyperparameterEditor: React.FC<HyperparameterEditorProps> = ({
  model,
  values,
  onChange,
  onPresetSelect,
  disabled = false,
}) => {
  const [localValues, setLocalValues] = useState(values);

  // Sync local values with prop changes
  useEffect(() => {
    setLocalValues(values);
  }, [values]);

  // Handle value change
  const handleValueChange = (paramName: string, value: any) => {
    const newValues = { ...localValues, [paramName]: value };
    setLocalValues(newValues);
    onChange(newValues);
  };

  // Handle preset selection
  const handlePresetClick = (presetId: string) => {
    const preset = model.presets.find((p) => p.id === presetId);
    if (preset) {
      setLocalValues(preset.values);
      onChange(preset.values);
      if (onPresetSelect) {
        onPresetSelect(presetId);
      }
    }
  };

  // Reset to default preset
  const handleReset = () => {
    const defaultPreset = model.presets.find((p) => p.isDefault) || model.presets[0];
    if (defaultPreset) {
      handlePresetClick(defaultPreset.id);
    }
  };

  // Render input for a hyperparameter
  const renderInput = (param: HyperparameterDef) => {
    const value = localValues[param.name] ?? param.default;

    switch (param.type) {
      case HyperparamType.INTEGER:
      case HyperparamType.FLOAT:
        const isFloat = param.type === HyperparamType.FLOAT;
        const step = param.step || (isFloat ? 0.1 : 1);

        return (
          <Box key={param.name}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
              <Typography variant="body2" fontWeight="500">
                {param.displayName}
              </Typography>
              <Chip label={value} size="small" color="primary" />
            </Box>

            <Slider
              value={value}
              onChange={(_, newValue) => handleValueChange(param.name, newValue)}
              min={param.min}
              max={param.max}
              step={step}
              marks={[
                { value: param.min!, label: String(param.min) },
                { value: param.max!, label: String(param.max) },
              ]}
              valueLabelDisplay="auto"
              disabled={disabled}
            />

            <TextField
              type="number"
              value={value}
              onChange={(e) =>
                handleValueChange(
                  param.name,
                  isFloat ? parseFloat(e.target.value) : parseInt(e.target.value, 10)
                )
              }
              size="small"
              fullWidth
              disabled={disabled}
              inputProps={{
                min: param.min,
                max: param.max,
                step,
              }}
              sx={{ mt: 1 }}
            />

            <Typography variant="caption" color="text.secondary" display="block" mt={0.5}>
              {param.description}
            </Typography>
          </Box>
        );

      case HyperparamType.CATEGORICAL:
        return (
          <FormControl key={param.name} fullWidth size="small">
            <InputLabel>{param.displayName}</InputLabel>
            <Select
              value={value}
              onChange={(e) => handleValueChange(param.name, e.target.value)}
              label={param.displayName}
              disabled={disabled}
            >
              {param.options?.map((option) => (
                <MenuItem key={String(option.value)} value={option.value}>
                  {option.label}
                </MenuItem>
              ))}
            </Select>
            <Typography variant="caption" color="text.secondary" display="block" mt={0.5}>
              {param.description}
            </Typography>
          </FormControl>
        );

      case HyperparamType.BOOLEAN:
        return (
          <Box key={param.name}>
            <FormControlLabel
              control={
                <Switch
                  checked={value}
                  onChange={(e) => handleValueChange(param.name, e.target.checked)}
                  disabled={disabled}
                />
              }
              label={param.displayName}
            />
            <Typography variant="caption" color="text.secondary" display="block">
              {param.description}
            </Typography>
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Card elevation={2}>
      <CardContent>
        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Box display="flex" alignItems="center" gap={1}>
            <TuneIcon color="primary" />
            <Typography variant="h6" fontWeight="bold">
              Hyperparameters
            </Typography>
          </Box>
          <Tooltip title="Reset to default preset">
            <IconButton onClick={handleReset} disabled={disabled} size="small">
              <RestoreIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Model Info */}
        <Alert severity="info" icon={<InfoIcon />} sx={{ mb: 2 }}>
          <Typography variant="body2">
            <strong>{model.displayName}</strong>
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {model.description}
          </Typography>
        </Alert>

        {/* Presets */}
        {model.presets.length > 0 && (
          <Box mb={3}>
            <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
              Quick Presets
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap">
              {model.presets.map((preset) => (
                <Chip
                  key={preset.id}
                  label={preset.name}
                  onClick={() => handlePresetClick(preset.id)}
                  variant={preset.isDefault ? 'filled' : 'outlined'}
                  color="primary"
                  disabled={disabled}
                  sx={{ cursor: 'pointer' }}
                />
              ))}
            </Stack>
            <Typography variant="caption" color="text.secondary" display="block" mt={0.5}>
              Click a preset to quickly apply recommended hyperparameter values
            </Typography>
          </Box>
        )}

        <Divider sx={{ mb: 3 }} />

        {/* Hyperparameter Inputs */}
        <Stack spacing={3}>
          {model.hyperparameters.length === 0 ? (
            <Alert severity="info">
              This model has no tunable hyperparameters.
            </Alert>
          ) : (
            model.hyperparameters.map((param) => renderInput(param))
          )}
        </Stack>

        {/* Current Configuration Summary */}
        {model.hyperparameters.length > 0 && (
          <Box mt={3}>
            <Divider sx={{ mb: 2 }} />
            <Typography variant="caption" color="text.secondary" fontWeight="bold">
              Current Configuration:
            </Typography>
            <Box
              component="pre"
              sx={{
                fontSize: '0.75rem',
                bgcolor: 'grey.100',
                p: 1.5,
                borderRadius: 1,
                overflow: 'auto',
                mt: 1,
              }}
            >
              {JSON.stringify(localValues, null, 2)}
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default HyperparameterEditor;
