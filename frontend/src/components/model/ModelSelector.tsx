/**
 * ModelSelector Component
 *
 * Displays available ML models grouped by category with descriptions,
 * pros/cons, and quick selection interface.
 */

import React, { useState, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Stack,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Alert,
  Divider,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  CheckCircle as CheckCircleIcon,
  Info as InfoIcon,
  Speed as SpeedIcon,
  Psychology as ComplexityIcon,
  Visibility as InterpretabilityIcon,
} from '@mui/icons-material';
import type {
  ModelSelectorProps,
  ModelDefinition,
  ModelCategory,
} from '../../types/modelSelection';
import {
  getModelsGroupedByCategory,
  getCategoryDisplayName,
  createDefaultModelSelection,
} from '../../types/modelSelection';

const ModelSelector: React.FC<ModelSelectorProps> = ({
  taskType,
  selectedModel,
  onModelSelect,
  disabled = false,
}) => {
  const [expandedCategory, setExpandedCategory] = useState<ModelCategory | null>(null);

  // Group models by category
  const groupedModels = useMemo(() => {
    return getModelsGroupedByCategory(taskType);
  }, [taskType]);

  // Handle category expansion
  const handleCategoryChange = (category: ModelCategory) => (
    _event: React.SyntheticEvent,
    isExpanded: boolean
  ) => {
    setExpandedCategory(isExpanded ? category : null);
  };

  // Handle model selection
  const handleModelClick = (model: ModelDefinition) => {
    if (disabled) return;

    const selection = createDefaultModelSelection(model.id, taskType);
    onModelSelect(selection);
  };

  // Get icon for complexity level
  const getComplexityIcon = (level: string) => {
    const colors: Record<string, string> = {
      low: 'success.main',
      medium: 'warning.main',
      high: 'error.main',
    };
    return <ComplexityIcon sx={{ color: colors[level] || 'text.secondary', fontSize: 16 }} />;
  };

  // Get icon for speed
  const getSpeedIcon = (speed: string) => {
    const colors: Record<string, string> = {
      fast: 'success.main',
      medium: 'warning.main',
      slow: 'error.main',
    };
    return <SpeedIcon sx={{ color: colors[speed] || 'text.secondary', fontSize: 16 }} />;
  };

  // Get icon for interpretability
  const getInterpretabilityIcon = (level: string) => {
    const colors: Record<string, string> = {
      high: 'success.main',
      medium: 'warning.main',
      low: 'error.main',
    };
    return <InterpretabilityIcon sx={{ color: colors[level] || 'text.secondary', fontSize: 16 }} />;
  };

  const categories = Object.keys(groupedModels) as ModelCategory[];

  if (categories.length === 0) {
    return (
      <Alert severity="info">
        No models available for the selected task type: {taskType}
      </Alert>
    );
  }

  return (
    <Card elevation={2}>
      <CardContent>
        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" fontWeight="bold">
            Select Model
          </Typography>
          <Tooltip title="Choose a model based on your data characteristics and requirements">
            <IconButton size="small">
              <InfoIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>

        {selectedModel && (
          <Alert severity="success" icon={<CheckCircleIcon />} sx={{ mb: 2 }}>
            Selected: <strong>{selectedModel.modelId}</strong>
          </Alert>
        )}

        {/* Model Categories */}
        <Stack spacing={1}>
          {categories.map((category) => {
            const models = groupedModels[category] || [];
            if (models.length === 0) return null;

            return (
              <Accordion
                key={category}
                expanded={expandedCategory === category}
                onChange={handleCategoryChange(category)}
                elevation={1}
              >
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box display="flex" alignItems="center" gap={1} width="100%">
                    <Typography variant="subtitle1" fontWeight="bold">
                      {getCategoryDisplayName(category)}
                    </Typography>
                    <Chip label={`${models.length} model${models.length > 1 ? 's' : ''}`} size="small" />
                  </Box>
                </AccordionSummary>
                <AccordionDetails sx={{ p: 0 }}>
                  <List disablePadding>
                    {models.map((model, index) => {
                      const isSelected = selectedModel?.modelId === model.id;

                      return (
                        <React.Fragment key={model.id}>
                          {index > 0 && <Divider />}
                          <ListItem
                            disablePadding
                            sx={{
                              bgcolor: isSelected ? 'action.selected' : 'transparent',
                            }}
                          >
                            <ListItemButton
                              onClick={() => handleModelClick(model)}
                              disabled={disabled}
                              sx={{ py: 2 }}
                            >
                              <ListItemText
                                primary={
                                  <Box display="flex" alignItems="center" gap={1} mb={0.5}>
                                    <Typography variant="body1" fontWeight={isSelected ? 600 : 500}>
                                      {model.displayName}
                                    </Typography>
                                    {isSelected && (
                                      <CheckCircleIcon fontSize="small" color="primary" />
                                    )}
                                  </Box>
                                }
                                secondary={
                                  <Stack spacing={1} mt={1}>
                                    {/* Description */}
                                    <Typography variant="body2" color="text.secondary">
                                      {model.description}
                                    </Typography>

                                    {/* Characteristics */}
                                    <Stack direction="row" spacing={2} flexWrap="wrap">
                                      <Tooltip title="Training Speed">
                                        <Box display="flex" alignItems="center" gap={0.5}>
                                          {getSpeedIcon(model.trainingSpeed)}
                                          <Typography variant="caption" color="text.secondary">
                                            {model.trainingSpeed}
                                          </Typography>
                                        </Box>
                                      </Tooltip>
                                      <Tooltip title="Complexity">
                                        <Box display="flex" alignItems="center" gap={0.5}>
                                          {getComplexityIcon(model.complexity)}
                                          <Typography variant="caption" color="text.secondary">
                                            {model.complexity}
                                          </Typography>
                                        </Box>
                                      </Tooltip>
                                      <Tooltip title="Interpretability">
                                        <Box display="flex" alignItems="center" gap={0.5}>
                                          {getInterpretabilityIcon(model.interpretability)}
                                          <Typography variant="caption" color="text.secondary">
                                            {model.interpretability}
                                          </Typography>
                                        </Box>
                                      </Tooltip>
                                    </Stack>

                                    {/* Pros/Cons */}
                                    <Box display="flex" gap={2} mt={1}>
                                      <Box flex={1}>
                                        <Typography variant="caption" fontWeight="bold" color="success.main">
                                          Pros:
                                        </Typography>
                                        <ul style={{ margin: '4px 0', paddingLeft: 20 }}>
                                          {model.pros.slice(0, 2).map((pro, idx) => (
                                            <li key={idx}>
                                              <Typography variant="caption">{pro}</Typography>
                                            </li>
                                          ))}
                                        </ul>
                                      </Box>
                                      <Box flex={1}>
                                        <Typography variant="caption" fontWeight="bold" color="error.main">
                                          Cons:
                                        </Typography>
                                        <ul style={{ margin: '4px 0', paddingLeft: 20 }}>
                                          {model.cons.slice(0, 2).map((con, idx) => (
                                            <li key={idx}>
                                              <Typography variant="caption">{con}</Typography>
                                            </li>
                                          ))}
                                        </ul>
                                      </Box>
                                    </Box>

                                    {/* Use Cases */}
                                    <Box mt={1}>
                                      <Typography variant="caption" fontWeight="bold" color="primary">
                                        Use Cases:
                                      </Typography>
                                      <Typography variant="caption" display="block" color="text.secondary">
                                        {model.useCases.slice(0, 2).join(' • ')}
                                      </Typography>
                                    </Box>

                                    {/* Presets */}
                                    <Box mt={1}>
                                      <Stack direction="row" spacing={0.5} flexWrap="wrap">
                                        <Typography variant="caption" color="text.secondary">
                                          Presets:
                                        </Typography>
                                        {model.presets.map((preset) => (
                                          <Chip
                                            key={preset.id}
                                            label={preset.name}
                                            size="small"
                                            variant={preset.isDefault ? 'filled' : 'outlined'}
                                            color={preset.isDefault ? 'primary' : 'default'}
                                          />
                                        ))}
                                      </Stack>
                                    </Box>
                                  </Stack>
                                }
                              />
                            </ListItemButton>
                          </ListItem>
                        </React.Fragment>
                      );
                    })}
                  </List>
                </AccordionDetails>
              </Accordion>
            );
          })}
        </Stack>

        {/* Help Text */}
        <Alert severity="info" sx={{ mt: 2 }}>
          <Typography variant="caption">
            💡 <strong>Tip:</strong> Start with simple models (Linear, Logistic Regression) as baselines,
            then move to more complex models (Random Forest, Gradient Boosting) if needed.
          </Typography>
        </Alert>
      </CardContent>
    </Card>
  );
};

export default ModelSelector;
