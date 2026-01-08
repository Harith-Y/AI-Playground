/**
 * FeatureSelection Component
 *
 * Combined component for feature and target selection with validation.
 * Orchestrates FeatureSelector and TargetColumnSelector components.
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Alert,
  Stack,
  Typography,
  Paper,
  AlertTitle,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import FeatureSelector from './FeatureSelector';
import TargetColumnSelector from './TargetColumnSelector';
import type {
  FeatureSelectionProps,
  FeatureSelectionConfig,
} from '../../types/featureSelection';
import { validateFeatureSelection } from '../../types/featureSelection';

const FeatureSelection: React.FC<FeatureSelectionProps> = ({
  datasetId,
  columns,
  initialConfig,
  onConfigChange,
  onValidationChange,
  disabled = false,
}) => {
  // Local state for configuration
  const [config, setConfig] = useState<FeatureSelectionConfig>({
    inputFeatures: initialConfig?.inputFeatures || [],
    targetColumn: initialConfig?.targetColumn || null,
    taskType: initialConfig?.taskType || null,
    excludedColumns: initialConfig?.excludedColumns || [],
  });

  // Update local config when initialConfig changes (e.g. restoration from Redux)
  useEffect(() => {
    if (initialConfig) {
      setConfig(prev => {
        // Only update if actually different to avoid infinite loops if onConfigChange updates parent state that feeds back here
        if (
          JSON.stringify(prev.inputFeatures) !== JSON.stringify(initialConfig.inputFeatures) ||
          prev.targetColumn !== initialConfig.targetColumn ||
          prev.taskType !== initialConfig.taskType
        ) {
          return {
            inputFeatures: initialConfig.inputFeatures || [],
            targetColumn: initialConfig.targetColumn || null,
            taskType: initialConfig.taskType || null,
            excludedColumns: initialConfig.excludedColumns || [],
          };
        }
        return prev;
      });
    }
  }, [initialConfig]);

  // Validation state
  const validation = useMemo(() => {
    return validateFeatureSelection(config);
  }, [config]);

  // Update parent when config changes
  useEffect(() => {
    onConfigChange(config);
  }, [config, onConfigChange]);

  // Update parent when validation changes
  useEffect(() => {
    if (onValidationChange) {
      onValidationChange(validation);
    }
  }, [validation, onValidationChange]);

  // Handle input features change
  const handleInputFeaturesChange = (features: string[]) => {
    setConfig((prev) => ({
      ...prev,
      inputFeatures: features,
    }));
  };

  // Handle target column change
  const handleTargetChange = (target: string | null) => {
    setConfig((prev) => {
      // If target was in input features, remove it
      const newInputFeatures = target
        ? prev.inputFeatures.filter((f) => f !== target)
        : prev.inputFeatures;

      return {
        ...prev,
        targetColumn: target,
        inputFeatures: newInputFeatures,
        // Update excluded columns
        excludedColumns: target ? [target] : [],
      };
    });
  };

  // Handle task type change
  const handleTaskTypeChange = (taskType: any) => {
    setConfig((prev) => ({
      ...prev,
      taskType,
    }));
  };

  // Get all errors and warnings
  const allErrors = Object.values(validation.errors).filter(Boolean);
  const allWarnings = Object.values(validation.warnings).filter(Boolean);

  return (
    <Box>
      {/* Validation Summary */}
      <Box mb={3}>
        {validation.isValid ? (
          <Alert
            severity="success"
            icon={<CheckCircleIcon />}
            sx={{ borderRadius: 2 }}
          >
            <AlertTitle>Configuration Valid</AlertTitle>
            Your feature selection is ready for model training.
          </Alert>
        ) : (
          <Stack spacing={1}>
            {allErrors.length > 0 && (
              <Alert severity="error" icon={<ErrorIcon />} sx={{ borderRadius: 2 }}>
                <AlertTitle>Validation Errors</AlertTitle>
                <ul style={{ margin: '8px 0 0 20px', padding: 0 }}>
                  {allErrors.map((error, idx) => (
                    <li key={idx}>
                      <Typography variant="body2">{error}</Typography>
                    </li>
                  ))}
                </ul>
              </Alert>
            )}
            {allWarnings.length > 0 && (
              <Alert severity="warning" icon={<WarningIcon />} sx={{ borderRadius: 2 }}>
                <AlertTitle>Warnings</AlertTitle>
                <ul style={{ margin: '8px 0 0 20px', padding: 0 }}>
                  {allWarnings.map((warning, idx) => (
                    <li key={idx}>
                      <Typography variant="body2">{warning}</Typography>
                    </li>
                  ))}
                </ul>
              </Alert>
            )}
          </Stack>
        )}
      </Box>

      {/* Configuration Summary */}
      <Paper elevation={1} sx={{ p: 2, mb: 3, bgcolor: 'background.default' }}>
        <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
          Configuration Summary
        </Typography>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
          <Box flex={1}>
            <Typography variant="caption" color="text.secondary">
              Input Features
            </Typography>
            <Typography variant="body2" fontWeight="bold">
              {config.inputFeatures.length} selected
            </Typography>
          </Box>
          <Box flex={1}>
            <Typography variant="caption" color="text.secondary">
              Target Column
            </Typography>
            <Typography variant="body2" fontWeight="bold">
              {config.targetColumn || 'None'}
            </Typography>
          </Box>
          <Box flex={1}>
            <Typography variant="caption" color="text.secondary">
              Task Type
            </Typography>
            <Typography variant="body2" fontWeight="bold">
              {config.taskType ? config.taskType.replace('_', ' ').toUpperCase() : 'Not selected'}
            </Typography>
          </Box>
        </Stack>
      </Paper>

      {/* Selectors */}
      <Stack direction={{ xs: 'column', lg: 'row' }} spacing={3}>
        {/* Feature Selector */}
        <Box flex={1}>
          <FeatureSelector
            datasetId={datasetId}
            columns={columns}
            selectedFeatures={config.inputFeatures}
            excludedColumns={config.excludedColumns}
            onChange={handleInputFeaturesChange}
            disabled={disabled}
          />
        </Box>

        {/* Target Column Selector */}
        <Box flex={1}>
          <TargetColumnSelector
            datasetId={datasetId}
            columns={columns}
            selectedTarget={config.targetColumn}
            taskType={config.taskType}
            onChange={handleTargetChange}
            onTaskTypeChange={handleTaskTypeChange}
            disabled={disabled}
            excludedColumns={config.inputFeatures}
          />
        </Box>
      </Stack>

      {/* Selected Features Preview */}
      {config.inputFeatures.length > 0 && (
        <Box mt={3}>
          <Paper elevation={1} sx={{ p: 2 }}>
            <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
              Selected Features ({config.inputFeatures.length})
            </Typography>
            <Box display="flex" flexWrap="wrap" gap={1}>
              {config.inputFeatures.map((feature) => {
                const column = columns.find((col) => col.name === feature);
                return (
                  <Box
                    key={feature}
                    sx={{
                      px: 2,
                      py: 0.5,
                      border: '1px solid',
                      borderColor: 'divider',
                      borderRadius: 1,
                      bgcolor: 'background.paper',
                    }}
                  >
                    <Typography variant="body2">
                      {feature}
                      {column && (
                        <Typography
                          component="span"
                          variant="caption"
                          color="text.secondary"
                          ml={0.5}
                        >
                          ({column.dtype})
                        </Typography>
                      )}
                    </Typography>
                  </Box>
                );
              })}
            </Box>
          </Paper>
        </Box>
      )}
    </Box>
  );
};

export default FeatureSelection;