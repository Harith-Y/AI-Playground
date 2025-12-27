/**
 * TargetColumnSelector Component
 *
 * Allows users to select the target column and task type for model training
 * with intelligent recommendations based on column data types.
 */

import React, { useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Chip,
  Stack,
  RadioGroup,
  FormControlLabel,
  Radio,
  Paper,
  Tooltip,
  IconButton,
  Divider,
} from '@mui/material';
import {
  Info as InfoIcon,
  Category as CategoryIcon,
  TrendingUp as TrendingUpIcon,
  ScatterPlot as ScatterPlotIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
} from '@mui/icons-material';
import type {
  TargetColumnSelectorProps,
} from '../../types/featureSelection';
import {
  TaskType,
  ColumnDataType,
  TASK_TYPE_CONFIGS,
  getColumnDataType,
} from '../../types/featureSelection';

const TargetColumnSelector: React.FC<TargetColumnSelectorProps> = ({
  datasetId,
  columns,
  selectedTarget,
  taskType,
  onChange,
  onTaskTypeChange,
  disabled = false,
  excludedColumns = [],
}) => {
  // Filter available columns (exclude input features)
  const availableColumns = useMemo(() => {
    return columns.filter((col) => !excludedColumns.includes(col.name));
  }, [columns, excludedColumns]);

  // Get recommended task types for selected target
  const recommendedTaskTypes = useMemo(() => {
    if (!selectedTarget) return [];

    const targetColumn = columns.find((col) => col.name === selectedTarget);
    if (!targetColumn) return [];

    const dataType = getColumnDataType(targetColumn.dtype);

    return Object.entries(TASK_TYPE_CONFIGS).filter(([_, config]) =>
      config.recommendedTargetTypes.includes(dataType)
    );
  }, [selectedTarget, columns]);

  // Get task icon
  const getTaskIcon = (task: TaskType) => {
    switch (task) {
      case TaskType.CLASSIFICATION:
        return <CategoryIcon />;
      case TaskType.REGRESSION:
        return <TrendingUpIcon />;
      case TaskType.CLUSTERING:
        return <ScatterPlotIcon />;
      case TaskType.ANOMALY_DETECTION:
        return <WarningIcon />;
      default:
        return null;
    }
  };

  // Check if task type is recommended for current target
  const isTaskTypeRecommended = (task: TaskType): boolean => {
    return recommendedTaskTypes.some(([taskKey]) => taskKey === task);
  };

  // Handle target column change
  const handleTargetChange = (target: string | null) => {
    onChange(target);

    // Auto-select recommended task type if only one recommendation
    if (target && recommendedTaskTypes.length === 1) {
      onTaskTypeChange(recommendedTaskTypes[0][0] as TaskType);
    }
  };

  // Get column type color
  const getTypeColor = (dataType: ColumnDataType): string => {
    switch (dataType) {
      case ColumnDataType.NUMERIC:
        return 'primary';
      case ColumnDataType.CATEGORICAL:
        return 'secondary';
      case ColumnDataType.BOOLEAN:
        return 'success';
      case ColumnDataType.DATETIME:
        return 'warning';
      case ColumnDataType.TEXT:
        return 'info';
      default:
        return 'default';
    }
  };

  return (
    <Card elevation={2}>
      <CardContent>
        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" fontWeight="bold">
            Target Column & Task Type
          </Typography>
          <Tooltip title="Select what you want to predict and the type of ML task">
            <IconButton size="small">
              <InfoIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>

        <Stack spacing={3}>
          {/* Target Column Selection */}
          <Box>
            <FormControl fullWidth size="medium">
              <InputLabel>Target Column</InputLabel>
              <Select
                value={selectedTarget || ''}
                label="Target Column"
                onChange={(e) => handleTargetChange(e.target.value || null)}
                disabled={disabled}
              >
                <MenuItem value="">
                  <em>None (for unsupervised learning)</em>
                </MenuItem>
                {availableColumns.map((column) => {
                  const dataType = getColumnDataType(column.dtype);
                  return (
                    <MenuItem key={column.name} value={column.name}>
                      <Box display="flex" justifyContent="space-between" width="100%" alignItems="center">
                        <Typography>{column.name}</Typography>
                        <Stack direction="row" spacing={0.5}>
                          <Chip
                            label={dataType}
                            size="small"
                            color={getTypeColor(dataType) as any}
                            sx={{ textTransform: 'capitalize' }}
                          />
                          {column.unique_count !== undefined && (
                            <Chip
                              label={`${column.unique_count} unique`}
                              size="small"
                              variant="outlined"
                            />
                          )}
                        </Stack>
                      </Box>
                    </MenuItem>
                  );
                })}
              </Select>
            </FormControl>

            {/* Target column info */}
            {selectedTarget && (
              <Box mt={1}>
                {(() => {
                  const targetCol = columns.find((col) => col.name === selectedTarget);
                  if (!targetCol) return null;

                  return (
                    <Stack direction="row" spacing={1} flexWrap="wrap">
                      <Chip
                        size="small"
                        label={`Type: ${getColumnDataType(targetCol.dtype)}`}
                        color={getTypeColor(getColumnDataType(targetCol.dtype)) as any}
                      />
                      {targetCol.unique_count !== undefined && (
                        <Chip
                          size="small"
                          label={`Unique values: ${targetCol.unique_count}`}
                          variant="outlined"
                        />
                      )}
                      {targetCol.missing_count !== undefined && targetCol.missing_count > 0 && (
                        <Chip
                          size="small"
                          label={`Missing: ${targetCol.missing_percentage?.toFixed(1)}%`}
                          color="warning"
                          variant="outlined"
                        />
                      )}
                    </Stack>
                  );
                })()}
              </Box>
            )}
          </Box>

          <Divider />

          {/* Task Type Selection */}
          <Box>
            <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
              Select Task Type
            </Typography>

            {!selectedTarget && (
              <Alert severity="info" sx={{ mb: 2 }}>
                Select a target column first, or choose an unsupervised task type below.
              </Alert>
            )}

            <RadioGroup
              value={taskType || ''}
              onChange={(e) => onTaskTypeChange(e.target.value as TaskType)}
            >
              <Stack spacing={1.5}>
                {Object.entries(TASK_TYPE_CONFIGS).map(([key, config]) => {
                  const task = key as TaskType;
                  const isRecommended = selectedTarget ? isTaskTypeRecommended(task) : !config.requiresTarget;
                  const isDisabled = disabled || (config.requiresTarget && !selectedTarget);

                  return (
                    <Paper
                      key={task}
                      elevation={taskType === task ? 3 : 1}
                      sx={{
                        p: 2,
                        border: '2px solid',
                        borderColor:
                          taskType === task
                            ? 'primary.main'
                            : isRecommended
                            ? 'success.light'
                            : 'divider',
                        cursor: isDisabled ? 'not-allowed' : 'pointer',
                        opacity: isDisabled ? 0.6 : 1,
                        bgcolor:
                          taskType === task
                            ? 'primary.50'
                            : isRecommended
                            ? 'success.50'
                            : 'background.paper',
                        transition: 'all 0.2s',
                        '&:hover': {
                          borderColor: isDisabled
                            ? undefined
                            : taskType === task
                            ? 'primary.dark'
                            : 'primary.light',
                        },
                      }}
                      onClick={() => !isDisabled && onTaskTypeChange(task)}
                    >
                      <FormControlLabel
                        value={task}
                        control={<Radio disabled={isDisabled} />}
                        label={
                          <Box width="100%">
                            <Box display="flex" alignItems="center" gap={1} mb={0.5}>
                              {getTaskIcon(task)}
                              <Typography variant="subtitle2" fontWeight="bold">
                                {config.label}
                              </Typography>
                              {isRecommended && (
                                <Chip
                                  icon={<CheckCircleIcon />}
                                  label="Recommended"
                                  size="small"
                                  color="success"
                                  variant="outlined"
                                />
                              )}
                              {config.requiresTarget && (
                                <Chip
                                  label="Supervised"
                                  size="small"
                                  variant="outlined"
                                  color="info"
                                />
                              )}
                            </Box>
                            <Typography variant="body2" color="text.secondary">
                              {config.description}
                            </Typography>
                          </Box>
                        }
                        sx={{ m: 0, width: '100%' }}
                      />
                    </Paper>
                  );
                })}
              </Stack>
            </RadioGroup>
          </Box>

          {/* Validation warnings */}
          {selectedTarget && taskType && !isTaskTypeRecommended(taskType) && (
            <Alert severity="warning">
              <Typography variant="body2">
                <strong>{TASK_TYPE_CONFIGS[taskType].label}</strong> is not typically recommended
                for{' '}
                <strong>
                  {(() => {
                    const targetCol = columns.find((col) => col.name === selectedTarget);
                    return targetCol ? getColumnDataType(targetCol.dtype) : 'this';
                  })()}
                </strong>{' '}
                target columns. Consider selecting a recommended task type for better results.
              </Typography>
            </Alert>
          )}

          {taskType && TASK_TYPE_CONFIGS[taskType].requiresTarget && !selectedTarget && (
            <Alert severity="error">
              <Typography variant="body2">
                <strong>{TASK_TYPE_CONFIGS[taskType].label}</strong> requires a target column.
                Please select a target column above.
              </Typography>
            </Alert>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
};

export default TargetColumnSelector;
