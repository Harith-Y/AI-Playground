/**
 * DataSplitConfig Component
 *
 * Comprehensive data split configuration with train/val/test sliders,
 * random seed, cross-validation, and preset management.
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Slider,
  TextField,
  FormControlLabel,
  Switch,
  Stack,
  Alert,
  AlertTitle,
  Divider,
  Chip,
  Tooltip,
  IconButton,
  ToggleButtonGroup,
  ToggleButton,
  Paper,
} from '@mui/material';
import {
  Info as InfoIcon,
  Shuffle as ShuffleIcon,
  Check as CheckIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';
import type {
  DataSplitConfigProps,
  SplitRatios,
} from '../../types/dataSplit';
import {
  SplitStrategy,
  validateDataSplitConfig,
} from '../../types/dataSplit';

const DataSplitConfigComponent: React.FC<DataSplitConfigProps> = ({
  config,
  onChange,
  onValidationChange,
  disabled = false,
  showStratify = true,
}) => {
  // Local state for slider values (to prevent rounding issues)
  const [sliderValues, setSliderValues] = useState<SplitRatios>(config.splitRatios);
  const [isDragging, setIsDragging] = useState(false);

  // Validation
  const validation = useMemo(() => {
    return validateDataSplitConfig(config);
  }, [config]);

  // Update parent when validation changes
  useEffect(() => {
    if (onValidationChange) {
      onValidationChange(validation);
    }
  }, [validation, onValidationChange]);

  // Sync slider values with config when not dragging
  useEffect(() => {
    if (!isDragging) {
      setSliderValues(config.splitRatios);
    }
  }, [config.splitRatios, isDragging]);

  // Handle strategy change
  const handleStrategyChange = (
    _event: React.MouseEvent<HTMLElement>,
    newStrategy: SplitStrategy | null
  ) => {
    if (newStrategy === null || disabled) return;

    const newConfig = { ...config, strategy: newStrategy };

    // Adjust split ratios for cross-validation
    if (newStrategy === SplitStrategy.CROSS_VALIDATION) {
      newConfig.splitRatios = { train: 80, validation: 0, test: 20 };
      newConfig.crossValidation.enabled = true;
    } else {
      newConfig.crossValidation.enabled = false;
      if (config.splitRatios.validation === 0) {
        newConfig.splitRatios = { train: 70, validation: 15, test: 15 };
      }
    }

    onChange(newConfig);
  };

  // Handle split ratio changes
  const handleSplitChange = useCallback((
    field: keyof SplitRatios,
    value: number
  ) => {
    if (disabled) return;

    const newRatios = { ...sliderValues, [field]: value };

    // Auto-adjust other values to sum to 100
    const others = Object.keys(newRatios).filter((k) => k !== field) as Array<keyof SplitRatios>;
    const remaining = 100 - value;

    if (config.strategy === SplitStrategy.CROSS_VALIDATION) {
      // For CV, validation is always 0, only adjust test
      if (field === 'train') {
        newRatios.test = remaining;
        newRatios.validation = 0;
      } else if (field === 'test') {
        newRatios.train = remaining;
        newRatios.validation = 0;
      }
    } else {
      // For holdout, distribute remaining proportionally
      const othersSum = others.reduce((sum, k) => sum + sliderValues[k], 0);
      if (othersSum > 0) {
        others.forEach((k) => {
          newRatios[k] = (sliderValues[k] / othersSum) * remaining;
        });
      } else {
        // Equal distribution
        others.forEach((k) => {
          newRatios[k] = remaining / others.length;
        });
      }
    }

    setSliderValues(newRatios);
  }, [sliderValues, disabled, config.strategy]);

  // Commit slider changes
  const handleSliderCommit = useCallback(() => {
    setIsDragging(false);
    onChange({ ...config, splitRatios: sliderValues });
  }, [config, sliderValues, onChange]);

  // Handle random seed change
  const handleSeedChange = (value: string) => {
    if (disabled) return;

    const seed = value === '' ? null : parseInt(value, 10);
    onChange({ ...config, randomSeed: seed });
  };

  // Handle CV folds change
  const handleFoldsChange = (value: string) => {
    if (disabled) return;

    const folds = parseInt(value, 10);
    if (!isNaN(folds) && folds > 0) {
      onChange({
        ...config,
        crossValidation: { ...config.crossValidation, folds },
      });
    }
  };

  // Generate random seed
  const generateRandomSeed = () => {
    if (disabled) return;
    const seed = Math.floor(Math.random() * 10000);
    onChange({ ...config, randomSeed: seed });
  };

  // Get all errors and warnings
  const allErrors = Object.values(validation.errors).filter(Boolean);
  const allWarnings = Object.values(validation.warnings).filter(Boolean);

  return (
    <Card elevation={2}>
      <CardContent>
        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" fontWeight="bold">
            Data Split Configuration
          </Typography>
          <Tooltip title="Configure how your data is split for training and evaluation">
            <IconButton size="small">
              <InfoIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Validation Summary */}
        {(allErrors.length > 0 || allWarnings.length > 0) && (
          <Box mb={2}>
            {allErrors.length > 0 && (
              <Alert severity="error" icon={<ErrorIcon />} sx={{ mb: 1 }}>
                <AlertTitle>Configuration Errors</AlertTitle>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {allErrors.map((error, idx) => (
                    <li key={idx}>
                      <Typography variant="body2">{error}</Typography>
                    </li>
                  ))}
                </ul>
              </Alert>
            )}
            {allWarnings.length > 0 && (
              <Alert severity="warning" icon={<WarningIcon />}>
                <AlertTitle>Warnings</AlertTitle>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {allWarnings.map((warning, idx) => (
                    <li key={idx}>
                      <Typography variant="body2">{warning}</Typography>
                    </li>
                  ))}
                </ul>
              </Alert>
            )}
          </Box>
        )}

        {validation.isValid && (
          <Alert severity="success" icon={<CheckIcon />} sx={{ mb: 2 }}>
            Configuration is valid and ready for training
          </Alert>
        )}

        <Stack spacing={3}>
          {/* Strategy Selection */}
          <Box>
            <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
              Split Strategy
            </Typography>
            <ToggleButtonGroup
              value={config.strategy}
              exclusive
              onChange={handleStrategyChange}
              disabled={disabled}
              fullWidth
              color="primary"
            >
              <ToggleButton value={SplitStrategy.HOLDOUT}>
                <Box textAlign="center">
                  <Typography variant="body2" fontWeight="bold">
                    Holdout
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Single split
                  </Typography>
                </Box>
              </ToggleButton>
              <ToggleButton value={SplitStrategy.CROSS_VALIDATION}>
                <Box textAlign="center">
                  <Typography variant="body2" fontWeight="bold">
                    Cross-Validation
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    K-Fold CV
                  </Typography>
                </Box>
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>

          <Divider />

          {/* Split Ratios */}
          <Box>
            <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
              Data Split Ratios
            </Typography>

            {/* Visual representation */}
            <Paper
              elevation={0}
              sx={{
                p: 2,
                mb: 2,
                bgcolor: 'background.default',
                border: '1px solid',
                borderColor: 'divider',
              }}
            >
              <Box display="flex" height={40} borderRadius={1} overflow="hidden">
                <Tooltip title={`Training: ${sliderValues.train.toFixed(1)}%`}>
                  <Box
                    flex={sliderValues.train}
                    bgcolor="primary.main"
                    display="flex"
                    alignItems="center"
                    justifyContent="center"
                    sx={{ cursor: 'pointer', transition: 'all 0.2s' }}
                  >
                    {sliderValues.train > 15 && (
                      <Typography variant="caption" fontWeight="bold" color="white">
                        Train {sliderValues.train.toFixed(1)}%
                      </Typography>
                    )}
                  </Box>
                </Tooltip>
                {config.strategy === SplitStrategy.HOLDOUT && sliderValues.validation > 0 && (
                  <Tooltip title={`Validation: ${sliderValues.validation.toFixed(1)}%`}>
                    <Box
                      flex={sliderValues.validation}
                      bgcolor="warning.main"
                      display="flex"
                      alignItems="center"
                      justifyContent="center"
                      sx={{ cursor: 'pointer', transition: 'all 0.2s' }}
                    >
                      {sliderValues.validation > 10 && (
                        <Typography variant="caption" fontWeight="bold" color="white">
                          Val {sliderValues.validation.toFixed(1)}%
                        </Typography>
                      )}
                    </Box>
                  </Tooltip>
                )}
                <Tooltip title={`Test: ${sliderValues.test.toFixed(1)}%`}>
                  <Box
                    flex={sliderValues.test}
                    bgcolor="error.main"
                    display="flex"
                    alignItems="center"
                    justifyContent="center"
                    sx={{ cursor: 'pointer', transition: 'all 0.2s' }}
                  >
                    {sliderValues.test > 10 && (
                      <Typography variant="caption" fontWeight="bold" color="white">
                        Test {sliderValues.test.toFixed(1)}%
                      </Typography>
                    )}
                  </Box>
                </Tooltip>
              </Box>
            </Paper>

            {/* Train Slider */}
            <Box mb={2}>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2" color="primary">
                  Training Set
                </Typography>
                <Chip label={`${sliderValues.train.toFixed(1)}%`} size="small" color="primary" />
              </Box>
              <Slider
                value={sliderValues.train}
                onChange={(_, value) => {
                  setIsDragging(true);
                  handleSplitChange('train', value as number);
                }}
                onChangeCommitted={handleSliderCommit}
                disabled={disabled}
                min={10}
                max={90}
                step={1}
                marks={[
                  { value: 50, label: '50%' },
                  { value: 70, label: '70%' },
                  { value: 80, label: '80%' },
                ]}
                valueLabelDisplay="auto"
                color="primary"
              />
            </Box>

            {/* Validation Slider (only for holdout) */}
            {config.strategy === SplitStrategy.HOLDOUT && (
              <Box mb={2}>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2" color="warning.main">
                    Validation Set
                  </Typography>
                  <Chip
                    label={`${sliderValues.validation.toFixed(1)}%`}
                    size="small"
                    sx={{ bgcolor: 'warning.main', color: 'white' }}
                  />
                </Box>
                <Slider
                  value={sliderValues.validation}
                  onChange={(_, value) => {
                    setIsDragging(true);
                    handleSplitChange('validation', value as number);
                  }}
                  onChangeCommitted={handleSliderCommit}
                  disabled={disabled}
                  min={0}
                  max={40}
                  step={1}
                  marks={[
                    { value: 0, label: '0%' },
                    { value: 10, label: '10%' },
                    { value: 20, label: '20%' },
                  ]}
                  valueLabelDisplay="auto"
                  sx={{
                    color: 'warning.main',
                    '& .MuiSlider-thumb': { bgcolor: 'warning.main' },
                    '& .MuiSlider-track': { bgcolor: 'warning.main' },
                  }}
                />
              </Box>
            )}

            {/* Test Slider */}
            <Box>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2" color="error.main">
                  Test Set
                </Typography>
                <Chip
                  label={`${sliderValues.test.toFixed(1)}%`}
                  size="small"
                  sx={{ bgcolor: 'error.main', color: 'white' }}
                />
              </Box>
              <Slider
                value={sliderValues.test}
                onChange={(_, value) => {
                  setIsDragging(true);
                  handleSplitChange('test', value as number);
                }}
                onChangeCommitted={handleSliderCommit}
                disabled={disabled}
                min={5}
                max={50}
                step={1}
                marks={[
                  { value: 10, label: '10%' },
                  { value: 20, label: '20%' },
                  { value: 30, label: '30%' },
                ]}
                valueLabelDisplay="auto"
                color="error"
              />
            </Box>
          </Box>

          <Divider />

          {/* Cross-Validation Settings */}
          {config.strategy === SplitStrategy.CROSS_VALIDATION && (
            <Box>
              <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                Cross-Validation Settings
              </Typography>

              <Stack spacing={2}>
                <TextField
                  label="Number of Folds"
                  type="number"
                  value={config.crossValidation.folds}
                  onChange={(e) => handleFoldsChange(e.target.value)}
                  disabled={disabled}
                  size="small"
                  fullWidth
                  inputProps={{ min: 2, max: 20 }}
                  helperText="Number of folds for cross-validation (typically 5 or 10)"
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={config.crossValidation.shuffle}
                      onChange={(e) =>
                        onChange({
                          ...config,
                          crossValidation: {
                            ...config.crossValidation,
                            shuffle: e.target.checked,
                          },
                        })
                      }
                      disabled={disabled}
                    />
                  }
                  label="Shuffle data before splitting"
                />
              </Stack>
            </Box>
          )}

          <Divider />

          {/* Random Seed */}
          <Box>
            <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
              Random Seed
            </Typography>

            <Stack direction="row" spacing={1}>
              <TextField
                label="Seed Value"
                type="number"
                value={config.randomSeed ?? ''}
                onChange={(e) => handleSeedChange(e.target.value)}
                disabled={disabled}
                size="small"
                fullWidth
                placeholder="Leave empty for random"
                inputProps={{ min: 0 }}
                helperText="Set a seed for reproducible splits (recommended: 42)"
              />
              <Tooltip title="Generate random seed">
                <IconButton
                  onClick={generateRandomSeed}
                  disabled={disabled}
                  color="primary"
                  sx={{ height: 40 }}
                >
                  <ShuffleIcon />
                </IconButton>
              </Tooltip>
            </Stack>
          </Box>

          {/* Stratify Option */}
          {showStratify && (
            <>
              <Divider />
              <Box>
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.stratify}
                      onChange={(e) =>
                        onChange({ ...config, stratify: e.target.checked })
                      }
                      disabled={disabled}
                    />
                  }
                  label={
                    <Box>
                      <Typography variant="body2">
                        Stratified Split
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Maintain class distribution across splits (for classification)
                      </Typography>
                    </Box>
                  }
                />
              </Box>
            </>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
};

export default DataSplitConfigComponent;
