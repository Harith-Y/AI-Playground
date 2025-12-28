/**
 * ModelSelectionPage
 *
 * Page for selecting ML model and configuring hyperparameters
 */

import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Button,
  Stack,
  Alert,
  Paper,
  Divider,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { ArrowBack, ArrowForward } from '@mui/icons-material';
import { ModelSelector, HyperparameterEditor } from '../components/model';
import type { ModelSelection, TaskType } from '../types/modelSelection';
import { TaskType as TaskTypeEnum, MODEL_REGISTRY } from '../types/modelSelection';

const ModelSelectionPage: React.FC = () => {
  const navigate = useNavigate();

  // Get task type from sessionStorage (from previous step)
  const [taskType, setTaskType] = useState<TaskType>(TaskTypeEnum.CLASSIFICATION);
  const [modelSelection, setModelSelection] = useState<ModelSelection | null>(null);

  // Load from sessionStorage on mount
  useEffect(() => {
    const savedTaskType = sessionStorage.getItem('taskType');
    if (savedTaskType) {
      setTaskType(savedTaskType as TaskType);
    }

    const savedModelSelection = sessionStorage.getItem('modelSelection');
    if (savedModelSelection) {
      try {
        setModelSelection(JSON.parse(savedModelSelection));
      } catch (e) {
        console.error('Failed to load saved model selection:', e);
      }
    }
  }, []);

  // Save to sessionStorage when changes
  useEffect(() => {
    if (modelSelection) {
      sessionStorage.setItem('modelSelection', JSON.stringify(modelSelection));
    }
  }, [modelSelection]);

  // Handle model selection
  const handleModelSelect = (selection: ModelSelection) => {
    setModelSelection(selection);
  };

  // Handle hyperparameter changes
  const handleHyperparametersChange = (values: Record<string, any>) => {
    if (modelSelection) {
      setModelSelection({
        ...modelSelection,
        hyperparameters: values,
      });
    }
  };

  // Handle preset selection
  const handlePresetSelect = (presetId: string) => {
    if (modelSelection) {
      setModelSelection({
        ...modelSelection,
        presetId,
      });
    }
  };

  // Navigation handlers
  const handleBack = () => {
    navigate('/data-split');
  };

  const handleContinue = () => {
    if (modelSelection) {
      sessionStorage.setItem('modelSelection', JSON.stringify(modelSelection));
      navigate('/training');
    }
  };

  const selectedModelDef = modelSelection
    ? MODEL_REGISTRY[modelSelection.modelId]
    : null;

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box mb={4}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Model Selection
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Choose a machine learning model and configure its hyperparameters for your task.
        </Typography>
      </Box>

      {/* Task Type Info */}
      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2">
          <strong>Task Type:</strong> {taskType}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Models are filtered to show only those compatible with your task type.
        </Typography>
      </Alert>

      {/* Main Content */}
      <Stack direction={{ xs: 'column', lg: 'row' }} spacing={3}>
        {/* Left Column - Model Selector */}
        <Box flex={2}>
          <ModelSelector
            taskType={taskType}
            selectedModel={modelSelection}
            onModelSelect={handleModelSelect}
          />
        </Box>

        {/* Right Column - Hyperparameter Editor */}
        <Box flex={1}>
          {selectedModelDef ? (
            <HyperparameterEditor
              model={selectedModelDef}
              values={modelSelection?.hyperparameters || {}}
              onChange={handleHyperparametersChange}
              onPresetSelect={handlePresetSelect}
            />
          ) : (
            <Paper elevation={2} sx={{ p: 3, textAlign: 'center' }}>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                No Model Selected
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Select a model from the list to configure its hyperparameters
              </Typography>
            </Paper>
          )}
        </Box>
      </Stack>

      {/* Configuration Summary */}
      {modelSelection && selectedModelDef && (
        <Paper elevation={2} sx={{ mt: 3, p: 3 }}>
          <Typography variant="h6" fontWeight="bold" gutterBottom>
            Configuration Summary
          </Typography>
          <Divider sx={{ mb: 2 }} />

          <Stack spacing={2}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Model
              </Typography>
              <Typography variant="body1" fontWeight="bold">
                {selectedModelDef.displayName}
              </Typography>
            </Box>

            <Box>
              <Typography variant="caption" color="text.secondary">
                Category
              </Typography>
              <Typography variant="body1">
                {selectedModelDef.category}
              </Typography>
            </Box>

            <Box>
              <Typography variant="caption" color="text.secondary">
                Characteristics
              </Typography>
              <Stack direction="row" spacing={2} mt={0.5}>
                <Typography variant="body2">
                  Speed: <strong>{selectedModelDef.trainingSpeed}</strong>
                </Typography>
                <Typography variant="body2">
                  Complexity: <strong>{selectedModelDef.complexity}</strong>
                </Typography>
                <Typography variant="body2">
                  Interpretability: <strong>{selectedModelDef.interpretability}</strong>
                </Typography>
              </Stack>
            </Box>

            <Box>
              <Typography variant="caption" color="text.secondary">
                Hyperparameters
              </Typography>
              <Box
                component="pre"
                sx={{
                  fontSize: '0.75rem',
                  bgcolor: 'grey.100',
                  p: 1.5,
                  borderRadius: 1,
                  overflow: 'auto',
                  mt: 0.5,
                }}
              >
                {JSON.stringify(modelSelection.hyperparameters, null, 2)}
              </Box>
            </Box>
          </Stack>
        </Paper>
      )}

      {/* Action Buttons */}
      <Box mt={4} display="flex" justifyContent="space-between" alignItems="center">
        <Button startIcon={<ArrowBack />} onClick={handleBack} variant="outlined">
          Back to Data Split
        </Button>

        <Button
          endIcon={<ArrowForward />}
          onClick={handleContinue}
          variant="contained"
          disabled={!modelSelection}
        >
          Continue to Training
        </Button>
      </Box>

      {/* Debug Info (Development only) */}
      {import.meta.env.DEV && modelSelection && (
        <Paper sx={{ mt: 4, p: 3, bgcolor: 'grey.100' }}>
          <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
            Debug: Model Selection State
          </Typography>
          <Box
            component="pre"
            sx={{
              fontSize: '0.75rem',
                overflow: 'auto',
              maxHeight: 300,
            }}
          >
            {JSON.stringify({ taskType, modelSelection }, null, 2)}
          </Box>
        </Paper>
      )}
    </Container>
  );
};

export default ModelSelectionPage;
