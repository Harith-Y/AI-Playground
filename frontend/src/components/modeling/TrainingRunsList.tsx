import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  IconButton,
  Chip,
  Tooltip,
  LinearProgress,
  Alert,
  CircularProgress,
  Button,
  Checkbox,
} from '@mui/material';
import {
  Delete as DeleteIcon,
  Visibility as ViewIcon,
  Refresh as RefreshIcon,
  Cancel as CancelIcon,
  Replay as RetryIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import type { TrainingRunListItem } from '../../services/modelingService';

export interface TrainingRunsListProps {
  runs: TrainingRunListItem[];
  total: number;
  page: number;
  pageSize: number;
  loading: boolean;
  error: string | null;
  onPageChange: (page: number) => void;
  onPageSizeChange: (pageSize: number) => void;
  onRefresh: () => void;
  onDelete: (runId: string) => Promise<void>;
  onDeleteMultiple?: (runIds: string[]) => Promise<void>;
  onCancel: (runId: string) => Promise<void>;
  onRetry: (runId: string) => Promise<void>;
  onExport?: (runId: string) => void;
}

const TrainingRunsList: React.FC<TrainingRunsListProps> = ({
  runs,
  total,
  page,
  pageSize,
  loading,
  error,
  onPageChange,
  onPageSizeChange,
  onRefresh,
  onDelete,
  onDeleteMultiple,
  onCancel,
  onRetry,
  onExport,
}) => {
  const navigate = useNavigate();
  const [selectedRuns, setSelectedRuns] = useState<string[]>([]);
  const [deletingRuns, setDeletingRuns] = useState<Set<string>>(new Set());
  const [cancellingRuns, setCancellingRuns] = useState<Set<string>>(new Set());

  const handleSelectAll = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.checked) {
      setSelectedRuns(runs.map(run => run.id));
    } else {
      setSelectedRuns([]);
    }
  };

  const handleSelectRun = (runId: string) => {
    setSelectedRuns(prev => {
      if (prev.includes(runId)) {
        return prev.filter(id => id !== runId);
      } else {
        return [...prev, runId];
      }
    });
  };

  const handleDeleteMultiple = async () => {
    if (!onDeleteMultiple || selectedRuns.length === 0) return;

    try {
      await onDeleteMultiple(selectedRuns);
      setSelectedRuns([]);
    } catch (err) {
      console.error('Error deleting runs:', err);
    }
  };

  const handleDelete = async (runId: string) => {
    setDeletingRuns(prev => new Set(prev).add(runId));
    try {
      await onDelete(runId);
    } catch (err) {
      console.error('Error deleting run:', err);
    } finally {
      setDeletingRuns(prev => {
        const newSet = new Set(prev);
        newSet.delete(runId);
        return newSet;
      });
    }
  };

  const handleCancel = async (runId: string) => {
    setCancellingRuns(prev => new Set(prev).add(runId));
    try {
      await onCancel(runId);
    } catch (err) {
      console.error('Error cancelling run:', err);
    } finally {
      setCancellingRuns(prev => {
        const newSet = new Set(prev);
        newSet.delete(runId);
        return newSet;
      });
    }
  };

  const handleView = (runId: string) => {
    navigate(`/training/${runId}`);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'running':
        return 'primary';
      case 'failed':
        return 'error';
      case 'cancelled':
        return 'default';
      case 'pending':
        return 'warning';
      default:
        return 'default';
    }
  };

  const formatDuration = (startTime: string, endTime?: string) => {
    const start = new Date(startTime).getTime();
    const end = endTime ? new Date(endTime).getTime() : Date.now();
    const duration = Math.floor((end - start) / 1000);

    if (duration < 60) return `${duration}s`;
    if (duration < 3600) return `${Math.floor(duration / 60)}m`;
    return `${Math.floor(duration / 3600)}h ${Math.floor((duration % 3600) / 60)}m`;
  };

  const formatMetrics = (metrics?: Record<string, number>) => {
    if (!metrics) return 'N/A';
    const entries = Object.entries(metrics);
    if (entries.length === 0) return 'N/A';
    
    const [key, value] = entries[0];
    return `${key}: ${(value * 100).toFixed(2)}%`;
  };

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">
            Training Runs ({total})
          </Typography>
          <Box display="flex" gap={1}>
            {selectedRuns.length > 0 && onDeleteMultiple && (
              <Button
                size="small"
                color="error"
                startIcon={<DeleteIcon />}
                onClick={handleDeleteMultiple}
              >
                Delete Selected ({selectedRuns.length})
              </Button>
            )}
            <Button
              size="small"
              startIcon={<RefreshIcon />}
              onClick={onRefresh}
              disabled={loading}
            >
              Refresh
            </Button>
          </Box>
        </Box>

        {loading && <LinearProgress sx={{ mb: 2 }} />}

        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                {onDeleteMultiple && (
                  <TableCell padding="checkbox">
                    <Checkbox
                      checked={selectedRuns.length === runs.length && runs.length > 0}
                      indeterminate={selectedRuns.length > 0 && selectedRuns.length < runs.length}
                      onChange={handleSelectAll}
                    />
                  </TableCell>
                )}
                <TableCell>Model</TableCell>
                <TableCell>Dataset</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Progress</TableCell>
                <TableCell>Metrics</TableCell>
                <TableCell>Duration</TableCell>
                <TableCell>Started</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {runs.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={onDeleteMultiple ? 9 : 8} align="center">
                    <Typography variant="body2" color="text.secondary" py={4}>
                      No training runs found
                    </Typography>
                  </TableCell>
                </TableRow>
              ) : (
                runs.map((run) => (
                  <TableRow key={run.id} hover>
                    {onDeleteMultiple && (
                      <TableCell padding="checkbox">
                        <Checkbox
                          checked={selectedRuns.includes(run.id)}
                          onChange={() => handleSelectRun(run.id)}
                        />
                      </TableCell>
                    )}
                    <TableCell>
                      <Typography variant="body2" fontWeight="medium">
                        {run.modelName}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {run.modelType}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {run.datasetName}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={run.status}
                        color={getStatusColor(run.status) as any}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Box display="flex" alignItems="center" gap={1}>
                        <Box sx={{ width: 60 }}>
                          <LinearProgress
                            variant="determinate"
                            value={run.progress}
                            color={run.status === 'failed' ? 'error' : 'primary'}
                          />
                        </Box>
                        <Typography variant="caption">
                          {run.progress}%
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {formatMetrics(run.metrics)}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {formatDuration(run.startTime, run.endTime)}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {new Date(run.startTime).toLocaleString()}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Box display="flex" justifyContent="flex-end" gap={0.5}>
                        <Tooltip title="View Details">
                          <IconButton
                            size="small"
                            onClick={() => handleView(run.id)}
                          >
                            <ViewIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>

                        {run.status === 'running' && (
                          <Tooltip title="Cancel">
                            <IconButton
                              size="small"
                              onClick={() => handleCancel(run.id)}
                              disabled={cancellingRuns.has(run.id)}
                            >
                              {cancellingRuns.has(run.id) ? (
                                <CircularProgress size={16} />
                              ) : (
                                <CancelIcon fontSize="small" />
                              )}
                            </IconButton>
                          </Tooltip>
                        )}

                        {run.status === 'failed' && (
                          <Tooltip title="Retry">
                            <IconButton
                              size="small"
                              onClick={() => onRetry(run.id)}
                            >
                              <RetryIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}

                        {run.status === 'completed' && onExport && (
                          <Tooltip title="Export Model">
                            <IconButton
                              size="small"
                              onClick={() => onExport(run.id)}
                            >
                              <DownloadIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}

                        <Tooltip title="Delete">
                          <IconButton
                            size="small"
                            color="error"
                            onClick={() => handleDelete(run.id)}
                            disabled={deletingRuns.has(run.id)}
                          >
                            {deletingRuns.has(run.id) ? (
                              <CircularProgress size={16} />
                            ) : (
                              <DeleteIcon fontSize="small" />
                            )}
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>

        <TablePagination
          component="div"
          count={total}
          page={page - 1}
          onPageChange={(_, newPage) => onPageChange(newPage + 1)}
          rowsPerPage={pageSize}
          onRowsPerPageChange={(e) => onPageSizeChange(parseInt(e.target.value, 10))}
          rowsPerPageOptions={[5, 10, 25, 50]}
        />
      </CardContent>
    </Card>
  );
};

export default TrainingRunsList;
