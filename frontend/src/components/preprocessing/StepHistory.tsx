import React from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Paper,
  Tooltip,
  Divider,
  Button,
} from '@mui/material';
import {
  Delete,
  Edit,
  DragIndicator,
  PlayArrow,
  Refresh,
  KeyboardArrowUp,
  KeyboardArrowDown,
  DeleteOutline,
} from '@mui/icons-material';
import { DragDropContext, Droppable, Draggable, DropResult } from '@hello-pangea/dnd';
import type { PreprocessingStep } from '../../types/preprocessing';
import { STEP_TYPE_CONFIGS } from '../../types/preprocessing';

interface StepHistoryProps {
  steps: PreprocessingStep[];
  onEdit: (step: PreprocessingStep) => void;
  onDelete: (stepId: string) => void;
  onReorder: (steps: PreprocessingStep[]) => void;
  onMoveUp?: (index: number) => void;
  onMoveDown?: (index: number) => void;
  onRemove?: (stepId: string) => void;
  onExecute?: () => void;
  onRefresh?: () => void;
  isProcessing?: boolean;
}

const StepHistory: React.FC<StepHistoryProps> = ({
  steps,
  onEdit,
  onDelete,
  onReorder,
  onMoveUp,
  onMoveDown,
  onRemove,
  onExecute,
  onRefresh,
  isProcessing = false,
}) => {
  const handleDragEnd = (result: DropResult) => {
    if (!result.destination) return;

    const items = Array.from(steps);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    // Update order property
    const updatedSteps = items.map((step, index) => ({
      ...step,
      order: index,
    }));

    onReorder(updatedSteps);
  };

  const getStepLabel = (stepType: string): string => {
    const config = STEP_TYPE_CONFIGS[stepType as keyof typeof STEP_TYPE_CONFIGS];
    return config?.label || stepType;
  };

  const getStepChipColor = (stepType: string): "default" | "primary" | "secondary" | "error" | "info" | "success" | "warning" => {
    const colorMap: Record<string, "default" | "primary" | "secondary" | "error" | "info" | "success" | "warning"> = {
      missing_value_imputation: 'warning',
      scaling: 'primary',
      encoding: 'secondary',
      outlier_detection: 'error',
      feature_selection: 'info',
      transformation: 'success',
    };
    return colorMap[stepType] || 'default';
  };

  const formatParameters = (parameters: Record<string, any>): string => {
    return Object.entries(parameters)
      .map(([key, value]) => `${key}: ${value}`)
      .join(', ');
  };

  return (
    <Paper
      elevation={0}
      sx={{
        height: '100%',
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 2,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 2,
          borderBottom: '1px solid',
          borderColor: 'divider',
          bgcolor: 'action.hover',
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6" fontWeight={600}>
            Pipeline Steps
          </Typography>
          {onRefresh && (
            <Tooltip title="Refresh steps">
              <IconButton onClick={onRefresh} size="small" disabled={isProcessing}>
                <Refresh />
              </IconButton>
            </Tooltip>
          )}
        </Box>
        <Typography variant="caption" color="text.secondary">
          {steps.length} {steps.length === 1 ? 'step' : 'steps'} configured
        </Typography>
      </Box>

      {/* Steps List */}
      <Box sx={{ flex: 1, overflow: 'auto', p: 1 }}>
        {steps.length === 0 ? (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              p: 3,
              textAlign: 'center',
            }}
          >
            <Typography variant="body2" color="text.secondary" gutterBottom>
              No preprocessing steps yet
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Add steps to build your preprocessing pipeline
            </Typography>
          </Box>
        ) : (
          <DragDropContext onDragEnd={handleDragEnd}>
            <Droppable droppableId="steps">
              {(provided, snapshot) => (
                <List
                  {...provided.droppableProps}
                  ref={provided.innerRef}
                  sx={{
                    bgcolor: snapshot.isDraggingOver ? 'action.hover' : 'transparent',
                    borderRadius: 1,
                    transition: 'background-color 0.2s',
                  }}
                >
                  {steps.map((step, index) => (
                    <Draggable key={step.id} draggableId={step.id} index={index}>
                      {(provided, snapshot) => (
                        <ListItem
                          ref={provided.innerRef}
                          {...provided.draggableProps}
                          sx={{
                            mb: 1,
                            bgcolor: 'background.paper',
                            border: '1px solid',
                            borderColor: snapshot.isDragging ? 'primary.main' : 'divider',
                            borderRadius: 1,
                            boxShadow: snapshot.isDragging ? 2 : 0,
                            transition: 'all 0.2s',
                            '&:hover': {
                              borderColor: 'primary.light',
                              bgcolor: 'action.hover',
                            },
                          }}
                        >
                          {/* Drag Handle */}
                          <Box
                            {...provided.dragHandleProps}
                            sx={{
                              mr: 1,
                              display: 'flex',
                              alignItems: 'center',
                              cursor: 'grab',
                              color: 'text.secondary',
                              '&:active': {
                                cursor: 'grabbing',
                              },
                            }}
                          >
                            <DragIndicator fontSize="small" />
                          </Box>

                          {/* Order Badge */}
                          <Chip
                            label={index + 1}
                            size="small"
                            sx={{
                              mr: 1,
                              minWidth: 32,
                              height: 24,
                              fontWeight: 600,
                            }}
                          />

                          {/* Step Info */}
                          <ListItemText
                            primary={
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                                <Typography variant="body2" fontWeight={600}>
                                  {getStepLabel(step.step_type)}
                                </Typography>
                                <Chip
                                  label={step.step_type}
                                  size="small"
                                  color={getStepChipColor(step.step_type)}
                                  sx={{ height: 20, fontSize: '0.7rem' }}
                                />
                              </Box>
                            }
                            secondary={
                              <Box sx={{ mt: 0.5 }}>
                                {step.column_name && (
                                  <Typography variant="caption" display="block" color="text.secondary">
                                    Column: <strong>{step.column_name}</strong>
                                  </Typography>
                                )}
                                {Object.keys(step.parameters).length > 0 && (
                                  <Typography
                                    variant="caption"
                                    display="block"
                                    color="text.secondary"
                                    sx={{
                                      mt: 0.5,
                                      wordBreak: 'break-word',
                                    }}
                                  >
                                    {formatParameters(step.parameters)}
                                  </Typography>
                                )}
                              </Box>
                            }
                          />

                          {/* Actions */}
                          <ListItemSecondaryAction>
                            <Box sx={{ display: 'flex', gap: 0.5 }}>
                              {/* Move Up */}
                              {onMoveUp && (
                                <Tooltip title="Move up">
                                  <span>
                                    <IconButton
                                      edge="end"
                                      size="small"
                                      onClick={() => onMoveUp(index)}
                                      disabled={isProcessing || index === 0}
                                    >
                                      <KeyboardArrowUp fontSize="small" />
                                    </IconButton>
                                  </span>
                                </Tooltip>
                              )}

                              {/* Move Down */}
                              {onMoveDown && (
                                <Tooltip title="Move down">
                                  <span>
                                    <IconButton
                                      edge="end"
                                      size="small"
                                      onClick={() => onMoveDown(index)}
                                      disabled={isProcessing || index === steps.length - 1}
                                    >
                                      <KeyboardArrowDown fontSize="small" />
                                    </IconButton>
                                  </span>
                                </Tooltip>
                              )}

                              {/* Edit */}
                              <Tooltip title="Edit step">
                                <IconButton
                                  edge="end"
                                  size="small"
                                  onClick={() => onEdit(step)}
                                  disabled={isProcessing}
                                >
                                  <Edit fontSize="small" />
                                </IconButton>
                              </Tooltip>

                              {/* Remove (Local) or Delete (with API) */}
                              {onRemove ? (
                                <Tooltip title="Remove step">
                                  <IconButton
                                    edge="end"
                                    size="small"
                                    onClick={() => onRemove(step.id)}
                                    disabled={isProcessing}
                                  >
                                    <DeleteOutline fontSize="small" />
                                  </IconButton>
                                </Tooltip>
                              ) : (
                                <Tooltip title="Delete step">
                                  <IconButton
                                    edge="end"
                                    size="small"
                                    onClick={() => onDelete(step.id)}
                                    disabled={isProcessing}
                                  >
                                    <Delete fontSize="small" />
                                  </IconButton>
                                </Tooltip>
                              )}
                            </Box>
                          </ListItemSecondaryAction>
                        </ListItem>
                      )}
                    </Draggable>
                  ))}
                  {provided.placeholder}
                </List>
              )}
            </Droppable>
          </DragDropContext>
        )}
      </Box>

      {/* Footer - Execute Button */}
      {steps.length > 0 && onExecute && (
        <>
          <Divider />
          <Box sx={{ p: 2 }}>
            <Button
              fullWidth
              variant="contained"
              startIcon={<PlayArrow />}
              onClick={onExecute}
              disabled={isProcessing}
            >
              {isProcessing ? 'Processing...' : 'Execute Pipeline'}
            </Button>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block', textAlign: 'center' }}>
              Apply all steps in order
            </Typography>
          </Box>
        </>
      )}
    </Paper>
  );
};

export default StepHistory;
