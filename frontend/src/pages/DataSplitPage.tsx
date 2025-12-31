/**
 * DataSplitPage
 *
 * Page for configuring data split and run configuration with preset management
 */

import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Button,
  Stack,
  Card,
  CardContent,
  Alert as _Alert,
  Divider,
  Paper,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import {
  ArrowBack,
  ArrowForward,
  Save as SaveIcon,
} from '@mui/icons-material';
import DataSplitConfig from '../components/config/DataSplitConfig';
import PresetManager from '../components/config/PresetManager';
import type {
  DataSplitConfig as DataSplitConfigType,
  DataSplitValidation,
  DataSplitPreset,
} from '../types/dataSplit';
import {
  DEFAULT_PRESETS,
  getDefaultDataSplitConfig,
} from '../types/dataSplit';

const DataSplitPage: React.FC = () => {
  const navigate = useNavigate();

  // State
  const [config, setConfig] = useState<DataSplitConfigType>(getDefaultDataSplitConfig());
  const [validation, setValidation] = useState<DataSplitValidation | null>(null);
  const [presets, setPresets] = useState<DataSplitPreset[]>([...DEFAULT_PRESETS]);
  const [selectedPresetId, setSelectedPresetId] = useState<string | null>(DEFAULT_PRESETS[0].id);

  // Load saved configuration from sessionStorage
  useEffect(() => {
    const savedConfig = sessionStorage.getItem('dataSplitConfig');
    if (savedConfig) {
      try {
        const parsed = JSON.parse(savedConfig);
        setConfig(parsed);
      } catch (e) {
        console.error('Failed to load saved config:', e);
      }
    }

    // Load custom presets from localStorage
    const savedPresets = localStorage.getItem('dataSplitPresets');
    if (savedPresets) {
      try {
        const customPresets = JSON.parse(savedPresets);
        setPresets([...DEFAULT_PRESETS, ...customPresets]);
      } catch (e) {
        console.error('Failed to load custom presets:', e);
      }
    }
  }, []);

  // Save configuration to sessionStorage whenever it changes
  useEffect(() => {
    sessionStorage.setItem('dataSplitConfig', JSON.stringify(config));
  }, [config]);

  // Handle preset selection
  const handlePresetSelect = (presetId: string) => {
    const preset = presets.find((p) => p.id === presetId);
    if (preset) {
      setConfig(preset.config);
      setSelectedPresetId(presetId);
    }
  };

  // Handle preset save
  const handlePresetSave = (preset: Omit<DataSplitPreset, 'id' | 'createdAt'>) => {
    const newPreset: DataSplitPreset = {
      ...preset,
      id: `custom-${Date.now()}`,
      createdAt: new Date().toISOString(),
      config: config, // Save current config
    };

    const customPresets = presets.filter((p) => !p.isDefault);
    const updatedCustomPresets = [...customPresets, newPreset];

    // Save to localStorage
    localStorage.setItem('dataSplitPresets', JSON.stringify(updatedCustomPresets));

    // Update state
    setPresets([...DEFAULT_PRESETS, ...updatedCustomPresets]);
    setSelectedPresetId(newPreset.id);
  };

  // Handle preset delete
  const handlePresetDelete = (presetId: string) => {
    const customPresets = presets.filter((p) => !p.isDefault && p.id !== presetId);

    // Save to localStorage
    localStorage.setItem('dataSplitPresets', JSON.stringify(customPresets));

    // Update state
    setPresets([...DEFAULT_PRESETS, ...customPresets]);

    // If deleted preset was selected, select default
    if (selectedPresetId === presetId) {
      setSelectedPresetId(DEFAULT_PRESETS[0].id);
      setConfig(DEFAULT_PRESETS[0].config);
    }
  };

  // Handle navigation
  const handleBack = () => {
    navigate('/feature-selection');
  };

  const handleContinue = () => {
    if (validation?.isValid) {
      sessionStorage.setItem('dataSplitConfig', JSON.stringify(config));
      navigate('/model-selection');
    }
  };

  const handleSaveAndExit = () => {
    sessionStorage.setItem('dataSplitConfig', JSON.stringify(config));
    navigate('/');
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box mb={4}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Data Split Configuration
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Configure how your dataset will be split for training, validation, and testing.
          Choose between holdout split or cross-validation strategies.
        </Typography>
      </Box>

      {/* Main Content */}
      <Stack direction={{ xs: 'column', lg: 'row' }} spacing={3}>
        {/* Left Column - Configuration */}
        <Box flex={2}>
          <DataSplitConfig
            config={config}
            onChange={setConfig}
            onValidationChange={setValidation}
            showStratify={true}
          />

          {/* Configuration Summary */}
          <Card elevation={2} sx={{ mt: 3 }}>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                Configuration Summary
              </Typography>
              <Divider sx={{ mb: 2 }} />

              <Stack spacing={2}>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Split Strategy
                  </Typography>
                  <Typography variant="body1" fontWeight="bold">
                    {config.strategy === 'holdout' ? 'Holdout Split' : 'Cross-Validation'}
                  </Typography>
                </Box>

                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Data Distribution
                  </Typography>
                  <Stack direction="row" spacing={2} mt={0.5}>
                    <Paper
                      elevation={0}
                      sx={{
                        p: 1.5,
                        bgcolor: 'primary.main',
                        color: 'white',
                        borderRadius: 1,
                        flex: 1,
                      }}
                    >
                      <Typography variant="caption">Training</Typography>
                      <Typography variant="h6" fontWeight="bold">
                        {config.splitRatios.train.toFixed(1)}%
                      </Typography>
                    </Paper>
                    {config.strategy === 'holdout' && config.splitRatios.validation > 0 && (
                      <Paper
                        elevation={0}
                        sx={{
                          p: 1.5,
                          bgcolor: 'warning.main',
                          color: 'white',
                          borderRadius: 1,
                          flex: 1,
                        }}
                      >
                        <Typography variant="caption">Validation</Typography>
                        <Typography variant="h6" fontWeight="bold">
                          {config.splitRatios.validation.toFixed(1)}%
                        </Typography>
                      </Paper>
                    )}
                    <Paper
                      elevation={0}
                      sx={{
                        p: 1.5,
                        bgcolor: 'error.main',
                        color: 'white',
                        borderRadius: 1,
                        flex: 1,
                      }}
                    >
                      <Typography variant="caption">Test</Typography>
                      <Typography variant="h6" fontWeight="bold">
                        {config.splitRatios.test.toFixed(1)}%
                      </Typography>
                    </Paper>
                  </Stack>
                </Box>

                {config.crossValidation.enabled && (
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Cross-Validation
                    </Typography>
                    <Typography variant="body1">
                      {config.crossValidation.folds}-Fold CV
                      {config.crossValidation.shuffle && ' (Shuffled)'}
                    </Typography>
                  </Box>
                )}

                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Random Seed
                  </Typography>
                  <Typography variant="body1">
                    {config.randomSeed !== null ? config.randomSeed : 'Not set (random)'}
                  </Typography>
                </Box>

                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Stratified Split
                  </Typography>
                  <Typography variant="body1">
                    {config.stratify ? 'Yes (maintains class distribution)' : 'No'}
                  </Typography>
                </Box>
              </Stack>
            </CardContent>
          </Card>
        </Box>

        {/* Right Column - Presets */}
        <Box flex={1}>
          <PresetManager
            presets={presets}
            selectedPresetId={selectedPresetId}
            onPresetSelect={handlePresetSelect}
            onPresetSave={handlePresetSave}
            onPresetDelete={handlePresetDelete}
          />

          {/* Info Card */}
          <Card elevation={1} sx={{ mt: 3, bgcolor: 'info.lighter' }}>
            <CardContent>
              <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                💡 Split Strategy Tips
              </Typography>
              <Stack spacing={1}>
                <Typography variant="body2">
                  • <strong>Holdout:</strong> Faster, suitable for large datasets
                </Typography>
                <Typography variant="body2">
                  • <strong>Cross-Validation:</strong> More robust, better for smaller datasets
                </Typography>
                <Typography variant="body2">
                  • <strong>Training set:</strong> 70-80% is typical
                </Typography>
                <Typography variant="body2">
                  • <strong>Random seed:</strong> Set for reproducibility
                </Typography>
                <Typography variant="body2">
                  • <strong>Stratify:</strong> Enable for classification tasks
                </Typography>
              </Stack>
            </CardContent>
          </Card>
        </Box>
      </Stack>

      {/* Action Buttons */}
      <Box mt={4} display="flex" justifyContent="space-between" alignItems="center">
        <Button startIcon={<ArrowBack />} onClick={handleBack} variant="outlined">
          Back to Feature Selection
        </Button>

        <Stack direction="row" spacing={2}>
          <Button
            startIcon={<SaveIcon />}
            onClick={handleSaveAndExit}
            variant="outlined"
          >
            Save & Exit
          </Button>
          <Button
            endIcon={<ArrowForward />}
            onClick={handleContinue}
            variant="contained"
            disabled={!validation?.isValid}
          >
            Continue to Model Selection
          </Button>
        </Stack>
      </Box>

      {/* Debug Info (Development only) */}
      {import.meta.env.DEV && (
        <Card sx={{ mt: 4, bgcolor: 'grey.100' }}>
          <CardContent>
            <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
              Debug: Current Configuration
            </Typography>
            <Box
              component="pre"
              sx={{
                fontSize: '0.75rem',
                overflow: 'auto',
                maxHeight: 300,
                bgcolor: 'grey.900',
                color: 'grey.100',
                p: 2,
                borderRadius: 1,
              }}
            >
              {JSON.stringify({ config, validation }, null, 2)}
            </Box>
          </CardContent>
        </Card>
      )}
    </Container>
  );
};

export default DataSplitPage;
