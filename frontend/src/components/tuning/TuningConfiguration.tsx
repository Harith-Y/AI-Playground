/**
 * Tuning Configuration Component
 * 
 * Provides UI for configuring hyperparameter tuning including:
 * - Tuning method selection (Grid/Random/Bayesian)
 * - Parameter ranges and search space configuration
 * - Cross-validation settings
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Button,
  Chip,
  IconButton,
  Tooltip,
  Alert,
  Divider,
  Paper,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Tune as TuneIcon,
  Settings as SettingsIcon,
  Info as InfoIcon,
} from '@mui/icons-material';

// Types
interface ParameterRange {
  name: string;
  type: 'int' | 'float' | 'categorical' | 'boolean';
  min?: number;
  max?: number;
  step?: number;
  values?: string[];
  default?: any;
}

interface TuningConfig {
  method: 'grid_search' | 'random_search' | 'bayesian_optimization';
  n_iter: number;
  cv_folds: number;
  scoring: string;
  random_state: number;
  n_jobs: number;
  parameters: ParameterRange[];
}

interface TuningConfigurationProps {
  modelType?: string;
  onConfigChange?: (config: TuningConfig) => void;
  disabled?: boolean;
}

const TuningConfiguration: React.FC<TuningConfigurationProps> = ({
  modelType = 'random_forest_classifier',
  onConfigChange,
  disabled = false,
}) => {
  // State
  const [config, setConfig] = useState<TuningConfig>({
    method: 'random_search',
    n_iter: 50,
    cv_folds: 5,
    scoring: 'accuracy',
    random_state: 42,
    n_jobs: -1,
    parameters: [],
  });
  
  const [expandedSections, setExpandedSections] = useState<string[]>(['method', 'parameters']);
  
  // Default parameters for different model types
  const getDefaultParameters = (modelType: string): ParameterRange[] => {
    switch (modelType) {
      case 'random_forest_classifier':
      case 'random_forest_regressor':
        return [
          {
            name: 'n_estimators',
            type: 'int',
            min: 10,
            max: 200,
            step: 10,
            default: 100,
          },
          {
            name: 'max_depth',
            type: 'int',
            min: 3,
            max: 20,
            step: 1,
            default: 10,
          },
          {
            name: 'min_samples_split',
            type: 'int',
            min: 2,
            max: 20,
            step: 1,
            default: 2,
          },
          {
            name: 'min_samples_leaf',
            type: 'int',
            min: 1,
            max: 10,
            step: 1,
            default: 1,
          },
        ];
      
      case 'xgboost_classifier':
      case 'xgboost_regressor':
        return [
          {
            name: 'n_estimators',
            type: 'int',
            min: 50,
            max: 300,
            step: 25,
            default: 100,
          },
          {
            name: 'max_depth',
            type: 'int',
            min: 3,
            max: 10,
            step: 1,
            default: 6,
          },
          {
            name: 'learning_rate',
            type: 'float',
            min: 0.01,
            max: 0.3,
            step: 0.01,
            default: 0.1,
          },
          {
            name: 'subsample',
            type: 'float',
            min: 0.6,
            max: 1.0,
            step: 0.1,
            default: 1.0,
          },
        ];
      
      case 'svm_classifier':
      case 'svm_regressor':
        return [
          {
            name: 'C',
            type: 'float',
            min: 0.1,
            max: 100,
            step: 0.1,
            default: 1.0,
          },
          {
            name: 'kernel',
            type: 'categorical',
            values: ['linear', 'poly', 'rbf', 'sigmoid'],
            default: 'rbf',
          },
          {
            name: 'gamma',
            type: 'categorical',
            values: ['scale', 'auto'],
            default: 'scale',
          },
        ];
      
      default:
        return [];
    }
  };
  
  // Initialize parameters when model type changes
  useEffect(() => {
    const defaultParams = getDefaultParameters(modelType);
    setConfig(prev => ({
      ...prev,
      parameters: defaultParams,
    }));
  }, [modelType]);
  
  // Notify parent of config changes
  useEffect(() => {
    onConfigChange?.(config);
  }, [config, onConfigChange]);
  
  // Handle config updates
  const updateConfig = (updates: Partial<TuningConfig>) => {
    setConfig(prev => ({ ...prev, ...updates }));
  };
  
  // Handle parameter updates
  const updateParameter = (index: number, updates: Partial<ParameterRange>) => {
    const newParameters = [...config.parameters];
    newParameters[index] = { ...newParameters[index], ...updates };
    updateConfig({ parameters: newParameters });
  };
  
  // Add new parameter
  const addParameter = () => {
    const newParam: ParameterRange = {
      name: 'new_parameter',
      type: 'float',
      min: 0,
      max: 1,
      step: 0.1,
      default: 0.5,
    };
    updateConfig({ parameters: [...config.parameters, newParam] });
  };
  
  // Remove parameter
  const removeParameter = (index: number) => {
    const newParameters = config.parameters.filter((_, i) => i !== index);
    updateConfig({ parameters: newParameters });
  };
  
  // Handle section expansion
  const handleSectionToggle = (section: string) => {
    setExpandedSections(prev => 
      prev.includes(section)
        ? prev.filter(s => s !== section)
        : [...prev, section]
    );
  };
  
  // Get scoring options based on model type
  const getScoringOptions = () => {
    if (modelType.includes('classifier')) {
      return [
        { value: 'accuracy', label: 'Accuracy' },
        { value: 'f1', label: 'F1 Score' },
        { value: 'precision', label: 'Precision' },
        { value: 'recall', label: 'Recall' },
        { value: 'roc_auc', label: 'ROC AUC' },
      ];
    } else {
      return [
        { value: 'neg_mean_squared_error', label: 'MSE (negative)' },
        { value: 'neg_mean_absolute_error', label: 'MAE (negative)' },
        { value: 'r2', label: 'R² Score' },
        { value: 'neg_root_mean_squared_error', label: 'RMSE (negative)' },
      ];
    }
  };
  
  return (
    <Box>
      {/* Method Selection */}
      <Accordion 
        expanded={expandedSections.includes('method')}
        onChange={() => handleSectionToggle('method')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Box display="flex" alignItems="center" gap={1}>
            <TuneIcon color="primary" />
            <Typography variant="h6">Tuning Method</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Box display="flex" flexDirection="column" gap={2}>
            <FormControl fullWidth disabled={disabled}>
              <InputLabel>Optimization Method</InputLabel>
              <Select
                value={config.method}
                label="Optimization Method"
                onChange={(e) => updateConfig({ method: e.target.value as any })}
              >
                <MenuItem value="grid_search">
                  <Box>
                    <Typography variant="body1">Grid Search</Typography>
                    <Typography variant="caption" color="text.secondary">
                      Exhaustive search over parameter grid
                    </Typography>
                  </Box>
                </MenuItem>
                <MenuItem value="random_search">
                  <Box>
                    <Typography variant="body1">Random Search</Typography>
                    <Typography variant="caption" color="text.secondary">
                      Random sampling from parameter space
                    </Typography>
                  </Box>
                </MenuItem>
                <MenuItem value="bayesian_optimization">
                  <Box>
                    <Typography variant="body1">Bayesian Optimization</Typography>
                    <Typography variant="caption" color="text.secondary">
                      Smart search using Gaussian processes
                    </Typography>
                  </Box>
                </MenuItem>
              </Select>
            </FormControl>
            
            <Box display="flex" gap={2}>
              <TextField
                fullWidth
                label="Number of Iterations"
                type="number"
                value={config.n_iter}
                onChange={(e) => updateConfig({ n_iter: parseInt(e.target.value) || 50 })}
                disabled={disabled || config.method === 'grid_search'}
                helperText={config.method === 'grid_search' ? 'Determined by parameter grid' : 'Number of parameter combinations to try'}
              />
              
              <TextField
                fullWidth
                label="Random State"
                type="number"
                value={config.random_state}
                onChange={(e) => updateConfig({ random_state: parseInt(e.target.value) || 42 })}
                disabled={disabled}
                helperText="For reproducible results"
              />
            </Box>
          </Box>
        </AccordionDetails>
      </Accordion>
      
      {/* Parameters Configuration */}
      <Accordion 
        expanded={expandedSections.includes('parameters')}
        onChange={() => handleSectionToggle('parameters')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Box display="flex" alignItems="center" gap={1}>
            <SettingsIcon color="primary" />
            <Typography variant="h6">Search Space</Typography>
            <Chip 
              label={`${config.parameters.length} parameters`} 
              size="small" 
              color="primary" 
              variant="outlined"
            />
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Box>
            {/* Add Parameter Button */}
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="body2" color="text.secondary">
                Configure the hyperparameter search space for {modelType}
              </Typography>
              <Button
                startIcon={<AddIcon />}
                onClick={addParameter}
                disabled={disabled}
                size="small"
              >
                Add Parameter
              </Button>
            </Box>
            
            {/* Parameters List */}
            {config.parameters.length === 0 ? (
              <Alert severity="info">
                No parameters configured. Click "Add Parameter" to define the search space.
              </Alert>
            ) : (
              <Box display="flex" flexDirection="column" gap={2}>
                {config.parameters.map((param, index) => (
                  <Paper key={index} sx={{ p: 2, border: '1px solid', borderColor: 'divider' }}>
                    <Box display="flex" flexDirection="column" gap={2}>
                      {/* Parameter Name and Type */}
                      <Box display="flex" gap={2} alignItems="center">
                        <TextField
                          label="Parameter Name"
                          value={param.name}
                          onChange={(e) => updateParameter(index, { name: e.target.value })}
                          disabled={disabled}
                          size="small"
                          sx={{ flex: 2 }}
                        />
                        
                        <FormControl size="small" disabled={disabled} sx={{ flex: 1 }}>
                          <InputLabel>Type</InputLabel>
                          <Select
                            value={param.type}
                            label="Type"
                            onChange={(e) => updateParameter(index, { type: e.target.value as any })}
                          >
                            <MenuItem value="int">Integer</MenuItem>
                            <MenuItem value="float">Float</MenuItem>
                            <MenuItem value="categorical">Categorical</MenuItem>
                            <MenuItem value="boolean">Boolean</MenuItem>
                          </Select>
                        </FormControl>
                        
                        <Tooltip title="Remove parameter">
                          <IconButton
                            onClick={() => removeParameter(index)}
                            disabled={disabled}
                            color="error"
                            size="small"
                          >
                            <DeleteIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                      
                      {/* Parameter Range/Values */}
                      <Box>
                        {param.type === 'categorical' ? (
                          <TextField
                            fullWidth
                            label="Values (comma-separated)"
                            value={param.values?.join(', ') || ''}
                            onChange={(e) => updateParameter(index, { 
                              values: e.target.value.split(',').map(v => v.trim()).filter(v => v)
                            })}
                            disabled={disabled}
                            size="small"
                            placeholder="value1, value2, value3"
                          />
                        ) : param.type === 'boolean' ? (
                          <Typography variant="body2" color="text.secondary">
                            Boolean parameter (True/False)
                          </Typography>
                        ) : (
                          <Box display="flex" gap={1}>
                            <TextField
                              label="Min"
                              type="number"
                              value={param.min || 0}
                              onChange={(e) => updateParameter(index, { min: parseFloat(e.target.value) || 0 })}
                              disabled={disabled}
                              size="small"
                              sx={{ flex: 1 }}
                            />
                            <TextField
                              label="Max"
                              type="number"
                              value={param.max || 1}
                              onChange={(e) => updateParameter(index, { max: parseFloat(e.target.value) || 1 })}
                              disabled={disabled}
                              size="small"
                              sx={{ flex: 1 }}
                            />
                            {param.type === 'int' && (
                              <TextField
                                label="Step"
                                type="number"
                                value={param.step || 1}
                                onChange={(e) => updateParameter(index, { step: parseFloat(e.target.value) || 1 })}
                                disabled={disabled}
                                size="small"
                                sx={{ flex: 1 }}
                              />
                            )}
                          </Box>
                        )}
                      </Box>
                    </Box>
                  </Paper>
                ))}
              </Box>
            )}
          </Box>
        </AccordionDetails>
      </Accordion>
      
      {/* Cross-Validation Settings */}
      <Accordion 
        expanded={expandedSections.includes('cv')}
        onChange={() => handleSectionToggle('cv')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Box display="flex" alignItems="center" gap={1}>
            <InfoIcon color="primary" />
            <Typography variant="h6">Cross-Validation</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Box display="flex" flexDirection="column" gap={2}>
            <Box display="flex" gap={2}>
              <TextField
                fullWidth
                label="CV Folds"
                type="number"
                value={config.cv_folds}
                onChange={(e) => updateConfig({ cv_folds: parseInt(e.target.value) || 5 })}
                disabled={disabled}
                helperText="Number of cross-validation folds"
                inputProps={{ min: 2, max: 10 }}
              />
              
              <FormControl fullWidth disabled={disabled}>
                <InputLabel>Scoring Metric</InputLabel>
                <Select
                  value={config.scoring}
                  label="Scoring Metric"
                  onChange={(e) => updateConfig({ scoring: e.target.value })}
                >
                  {getScoringOptions().map(option => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>
            
            <TextField
              fullWidth
              label="Parallel Jobs"
              type="number"
              value={config.n_jobs}
              onChange={(e) => updateConfig({ n_jobs: parseInt(e.target.value) || -1 })}
              disabled={disabled}
              helperText="Number of parallel jobs (-1 for all cores)"
              inputProps={{ min: -1, max: 16 }}
            />
          </Box>
        </AccordionDetails>
      </Accordion>
      
      {/* Configuration Summary */}
      <Card sx={{ mt: 2, bgcolor: 'background.default' }}>
        <CardContent>
          <Typography variant="subtitle2" gutterBottom>
            Configuration Summary
          </Typography>
          <Divider sx={{ mb: 1 }} />
          <Box display="flex" flexWrap="wrap" gap={1}>
            <Chip label={`Method: ${config.method.replace(/_/g, ' ')}`} size="small" />
            <Chip label={`Iterations: ${config.n_iter}`} size="small" />
            <Chip label={`CV: ${config.cv_folds}-fold`} size="small" />
            <Chip label={`Scoring: ${config.scoring}`} size="small" />
            <Chip label={`Parameters: ${config.parameters.length}`} size="small" />
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default TuningConfiguration;
