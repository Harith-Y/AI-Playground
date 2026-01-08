/**
 * ConfusionMatrixHeatmap Component
 *
 * Visualizes confusion matrix as an interactive heatmap
 */

import React, { useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Chip,
  Alert,
} from '@mui/material';
import { Info as InfoIcon } from '@mui/icons-material';

/**
 * Props for ConfusionMatrixHeatmap component
 */
export interface ConfusionMatrixHeatmapProps {
  confusionMatrix: number[][];
  classNames?: string[];
  title?: string;
  showPercentages?: boolean;
  showCounts?: boolean;
  colorScheme?: 'blue' | 'green' | 'purple' | 'red';
  normalize?: boolean;
}

const ConfusionMatrixHeatmap: React.FC<ConfusionMatrixHeatmapProps> = ({
  confusionMatrix,
  classNames,
  title = 'Confusion Matrix',
  showPercentages = true,
  showCounts = true,
  colorScheme = 'blue',
  normalize = false,
}) => {
  // Validate confusion matrix
  if (!confusionMatrix || confusionMatrix.length === 0) {
    return (
      <Alert severity="warning">
        <Typography variant="body2">
          No confusion matrix data available
        </Typography>
      </Alert>
    );
  }

  const numClasses = confusionMatrix.length;

  // Generate default class names if not provided
  const labels = useMemo(() => {
    if (classNames && classNames.length === numClasses) {
      return classNames;
    }
    return Array.from({ length: numClasses }, (_, i) => `Class ${i}`);
  }, [classNames, numClasses]);

  // Calculate normalized matrix (if requested)
  const normalizedMatrix = useMemo(() => {
    if (!normalize) return confusionMatrix;

    return confusionMatrix.map((row) => {
      const rowSum = row.reduce((sum, val) => sum + val, 0);
      return rowSum > 0 ? row.map((val) => val / rowSum) : row;
    });
  }, [confusionMatrix, normalize]);

  // Calculate total samples
  const totalSamples = useMemo(() => {
    return confusionMatrix.reduce(
      (total, row) => total + row.reduce((sum, val) => sum + val, 0),
      0
    );
  }, [confusionMatrix]);

  // Calculate max value for color scaling
  const maxValue = useMemo(() => {
    return Math.max(...normalizedMatrix.flat());
  }, [normalizedMatrix]);

  // Get color intensity based on value
  const getColorIntensity = (value: number): string => {
    const intensity = maxValue > 0 ? value / maxValue : 0;
    const alpha = 0.1 + intensity * 0.9; // Range from 0.1 to 1.0

    const colors = {
      blue: `rgba(25, 118, 210, ${alpha})`,
      green: `rgba(46, 125, 50, ${alpha})`,
      purple: `rgba(123, 31, 162, ${alpha})`,
      red: `rgba(211, 47, 47, ${alpha})`,
    };

    return colors[colorScheme];
  };

  // Get text color based on background intensity
  const getTextColor = (value: number): string => {
    const intensity = maxValue > 0 ? value / maxValue : 0;
    return intensity > 0.5 ? '#ffffff' : '#000000';
  };

  // Calculate metrics for each cell
  const getCellMetrics = (actual: number, predicted: number) => {
    const count = confusionMatrix[actual][predicted];
    const normalizedValue = normalizedMatrix[actual][predicted];
    const percentage = normalize
      ? normalizedValue * 100
      : (count / totalSamples) * 100;

    return { count, percentage, normalizedValue };
  };

  // Calculate overall accuracy
  const accuracy = useMemo(() => {
    const correctPredictions = confusionMatrix.reduce(
      (sum, row, i) => sum + row[i],
      0
    );
    return totalSamples > 0 ? (correctPredictions / totalSamples) * 100 : 0;
  }, [confusionMatrix, totalSamples]);

  return (
    <Paper elevation={2} sx={{ p: 3 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Box>
          <Typography variant="h6" fontWeight="bold" gutterBottom>
            {title}
          </Typography>
          <Box display="flex" gap={1} alignItems="center">
            <Chip
              label={`${numClasses} Classes`}
              size="small"
              color="primary"
              variant="outlined"
            />
            <Chip
              label={`${totalSamples.toLocaleString()} Samples`}
              size="small"
              variant="outlined"
            />
            <Chip
              label={`Accuracy: ${accuracy.toFixed(2)}%`}
              size="small"
              color="success"
            />
          </Box>
        </Box>

        <Tooltip
          title="Rows represent actual classes, columns represent predicted classes. Diagonal cells show correct predictions."
          arrow
        >
          <InfoIcon color="action" sx={{ cursor: 'help' }} />
        </Tooltip>
      </Box>

      {/* Confusion Matrix Table */}
      <TableContainer sx={{ maxHeight: 600, overflowX: 'auto' }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              <TableCell
                sx={{
                  fontWeight: 'bold',
                  bgcolor: 'action.hover',
                  border: '2px solid',
                  borderColor: 'grey.300',
                }}
              >
                <Box display="flex" flexDirection="column">
                  <Typography variant="caption" fontWeight="bold">
                    Actual ↓
                  </Typography>
                  <Typography variant="caption" fontWeight="bold">
                    Predicted →
                  </Typography>
                </Box>
              </TableCell>
              {labels.map((label, index) => (
                <TableCell
                  key={index}
                  align="center"
                  sx={{
                    fontWeight: 'bold',
                    bgcolor: 'action.hover',
                    border: '2px solid',
                    borderColor: 'grey.300',
                    minWidth: 100,
                  }}
                >
                  <Typography variant="body2" fontWeight="bold" noWrap>
                    {label}
                  </Typography>
                </TableCell>
              ))}
              <TableCell
                align="center"
                sx={{
                  fontWeight: 'bold',
                  bgcolor: 'grey.200',
                  border: '2px solid',
                  borderColor: 'grey.300',
                }}
              >
                <Typography variant="body2" fontWeight="bold">
                  Total
                </Typography>
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {confusionMatrix.map((row, actualIndex) => {
              const rowSum = row.reduce((sum, val) => sum + val, 0);

              return (
                <TableRow key={actualIndex}>
                  {/* Row Label */}
                  <TableCell
                    component="th"
                    scope="row"
                    sx={{
                      fontWeight: 'bold',
                      bgcolor: 'action.hover',
                      border: '2px solid',
                      borderColor: 'grey.300',
                      position: 'sticky',
                      left: 0,
                      zIndex: 1,
                    }}
                  >
                    <Typography variant="body2" fontWeight="bold" noWrap>
                      {labels[actualIndex]}
                    </Typography>
                  </TableCell>

                  {/* Matrix Cells */}
                  {row.map((value, predictedIndex) => {
                    const { count, percentage, normalizedValue } = getCellMetrics(
                      actualIndex,
                      predictedIndex
                    );
                    const isDiagonal = actualIndex === predictedIndex;

                    return (
                      <Tooltip
                        key={predictedIndex}
                        title={
                          <Box>
                            <Typography variant="caption" display="block">
                              Actual: {labels[actualIndex]}
                            </Typography>
                            <Typography variant="caption" display="block">
                              Predicted: {labels[predictedIndex]}
                            </Typography>
                            <Typography variant="caption" display="block" fontWeight="bold">
                              Count: {count}
                            </Typography>
                            <Typography variant="caption" display="block">
                              Percentage: {percentage.toFixed(2)}%
                            </Typography>
                            {isDiagonal && (
                              <Typography variant="caption" display="block" color="success.light">
                                ✓ Correct Prediction
                              </Typography>
                            )}
                          </Box>
                        }
                        arrow
                      >
                        <TableCell
                          align="center"
                          sx={{
                            bgcolor: getColorIntensity(normalizedValue),
                            color: getTextColor(normalizedValue),
                            border: isDiagonal ? '3px solid' : '1px solid',
                            borderColor: isDiagonal ? 'success.main' : 'grey.300',
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            '&:hover': {
                              transform: 'scale(1.05)',
                              boxShadow: 2,
                              zIndex: 2,
                            },
                          }}
                        >
                          <Box display="flex" flexDirection="column" alignItems="center" gap={0.5}>
                            {showCounts && (
                              <Typography
                                variant="body1"
                                fontWeight="bold"
                                sx={{ lineHeight: 1 }}
                              >
                                {count}
                              </Typography>
                            )}
                            {showPercentages && (
                              <Typography
                                variant="caption"
                                sx={{ lineHeight: 1, opacity: 0.9 }}
                              >
                                {percentage.toFixed(1)}%
                              </Typography>
                            )}
                          </Box>
                        </TableCell>
                      </Tooltip>
                    );
                  })}

                  {/* Row Total */}
                  <TableCell
                    align="center"
                    sx={{
                      fontWeight: 'bold',
                      bgcolor: 'action.hover',
                      border: '2px solid',
                      borderColor: 'grey.300',
                    }}
                  >
                    <Typography variant="body2" fontWeight="bold">
                      {rowSum.toLocaleString()}
                    </Typography>
                  </TableCell>
                </TableRow>
              );
            })}

            {/* Column Totals Row */}
            <TableRow>
              <TableCell
                sx={{
                  fontWeight: 'bold',
                  bgcolor: 'grey.200',
                  border: '2px solid',
                  borderColor: 'grey.300',
                }}
              >
                <Typography variant="body2" fontWeight="bold">
                  Total
                </Typography>
              </TableCell>
              {Array.from({ length: numClasses }).map((_, colIndex) => {
                const colSum = confusionMatrix.reduce((sum, row) => sum + row[colIndex], 0);
                return (
                  <TableCell
                    key={colIndex}
                    align="center"
                    sx={{
                      fontWeight: 'bold',
                      bgcolor: 'action.hover',
                      border: '2px solid',
                      borderColor: 'grey.300',
                    }}
                  >
                    <Typography variant="body2" fontWeight="bold">
                      {colSum.toLocaleString()}
                    </Typography>
                  </TableCell>
                );
              })}
              <TableCell
                align="center"
                sx={{
                  fontWeight: 'bold',
                  bgcolor: 'grey.200',
                  border: '2px solid',
                  borderColor: 'grey.300',
                }}
              >
                <Typography variant="body2" fontWeight="bold">
                  {totalSamples.toLocaleString()}
                </Typography>
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </TableContainer>

      {/* Legend */}
      <Box mt={2} p={2} bgcolor="grey.50" borderRadius={1}>
        <Typography variant="caption" fontWeight="bold" display="block" gutterBottom>
          How to Read:
        </Typography>
        <Box display="flex" flexDirection="column" gap={0.5}>
          <Typography variant="caption" color="text.secondary">
            • <strong>Rows</strong>: Actual (true) class labels
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Columns</strong>: Predicted class labels
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Diagonal cells</strong> (outlined in green): Correct predictions
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Off-diagonal cells</strong>: Misclassifications
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Darker colors</strong>: Higher frequency of predictions
          </Typography>
        </Box>
      </Box>
    </Paper>
  );
};

export default ConfusionMatrixHeatmap;
