import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Alert,
  Snackbar,
  LinearProgress,
  Card,
  CardContent,
  Divider,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Collapse,
} from '@mui/material';
import {
  CloudUpload,
  CheckCircle,
  Error as _ErrorIcon,
  Info as InfoIcon,
  ExpandMore,
  ExpandLess,
  Description,
  DataArray,
  Storage,
  ErrorOutline,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  uploadDataset,
  fetchDatasetStats,
  fetchDatasetPreview,
  clearDatasetError,
} from '../store/slices/datasetSlice';
import { formatFileSize, formatDate } from '../utils/helpers';
import MissingValuesAnalysis from '../components/dataset/MissingValuesAnalysis';

const DatasetUploadPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { currentDataset, stats, columns, preview, isLoading, error, uploadProgress } =
    useAppSelector((state) => state.dataset);

  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [showStats, setShowStats] = useState(true);
  const [showPreview, setShowPreview] = useState(true);
  const [showMissingValues, setShowMissingValues] = useState(true);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState<'success' | 'error'>('success');

  // File validation
  const validateFile = (file: File): string | null => {
    const maxSize = 100 * 1024 * 1024; // 100MB
    const allowedExtensions = ['.csv', '.xlsx', '.xls', '.json'];

    const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();

    if (!allowedExtensions.includes(fileExtension)) {
      return `Invalid file type. Please upload ${allowedExtensions.join(', ')} files.`;
    }

    if (file.size > maxSize) {
      return `File size exceeds ${formatFileSize(maxSize)}. Please upload a smaller file.`;
    }

    return null;
  };

  // Handle file selection
  const handleFileSelect = useCallback((file: File) => {
    const error = validateFile(file);
    if (error) {
      setValidationError(error);
      setSelectedFile(null);
      return;
    }

    setValidationError(null);
    setSelectedFile(file);
  }, []);

  // Handle drag and drop
  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      const files = Array.from(e.dataTransfer.files);
      if (files.length > 0) {
        handleFileSelect(files[0]);
      }
    },
    [handleFileSelect]
  );

  // Handle file input change
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  // Handle upload
  const handleUpload = async () => {
    if (!selectedFile) return;

    try {
      const result = await dispatch(uploadDataset(selectedFile)).unwrap();
      setSnackbarMessage('Dataset uploaded successfully!');
      setSnackbarSeverity('success');
      setSnackbarOpen(true);
      setSelectedFile(null);

      // Fetch stats and preview independently (don't block on errors)
      if (result.id) {
        try {
          await dispatch(fetchDatasetStats(result.id)).unwrap();
        } catch (statsError: any) {
          console.error('Failed to fetch dataset stats:', statsError);
          // Don't show error to user if stats fail - it's non-critical
        }

        try {
          await dispatch(fetchDatasetPreview(result.id)).unwrap();
        } catch (previewError: any) {
          console.error('Failed to fetch dataset preview:', previewError);
          // Don't show error to user if preview fails - it's non-critical
        }
      }
    } catch (err: any) {
      let errorMessage = 'Failed to upload dataset';
      if (typeof err === 'string') {
        errorMessage = err;
      } else if (typeof err?.message === 'string') {
        errorMessage = err.message;
      }
      setSnackbarMessage(errorMessage);
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
    }
  };

  // Clear errors when component unmounts
  useEffect(() => {
    return () => {
      dispatch(clearDatasetError());
    };
  }, [dispatch]);

  const handleSnackbarClose = () => {
    setSnackbarOpen(false);
  };

  return (
    <Box
      sx={{
        width: '100%',
        minHeight: 'calc(100vh - 64px)',
        background: 'linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%)',
        p: 4,
      }}
    >
      <Box sx={{ maxWidth: 1400, mx: 'auto' }}>
        {/* Header */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h4" fontWeight={700} gutterBottom>
            Dataset Upload
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Upload your dataset and explore its structure and statistics
          </Typography>
        </Box>

        {/* Upload Section */}
        <Paper
          sx={{
            p: 4,
            mb: 3,
            border: '1px solid #e2e8f0',
            background: '#FFFFFF',
          }}
        >
          <Typography variant="h6" fontWeight={600} gutterBottom>
            Select Dataset
          </Typography>

          {/* Drag and Drop Zone */}
          <Box
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            sx={{
              mt: 2,
              p: 6,
              border: `2px dashed ${isDragging ? '#2563eb' : '#cbd5e1'}`,
              borderRadius: 2,
              background: isDragging
                ? 'linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%)'
                : '#F8FAFC',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              '&:hover': {
                borderColor: '#2563eb',
                background: 'linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%)',
              },
            }}
            onClick={() => document.getElementById('file-input')?.click()}
          >
            <CloudUpload
              sx={{
                fontSize: 64,
                color: isDragging ? 'primary.main' : 'text.secondary',
                mb: 2,
              }}
            />
            <Typography variant="h6" gutterBottom>
              {selectedFile ? selectedFile.name : 'Drag and drop your dataset here'}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              or click to browse
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Supported formats: CSV, XLSX, XLS, JSON • Maximum size: 100MB
            </Typography>
            <input
              id="file-input"
              type="file"
              accept=".csv,.xlsx,.xls,.json"
              onChange={handleFileInputChange}
              style={{ display: 'none' }}
            />
          </Box>

          {/* Validation Error */}
          {validationError && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {validationError}
            </Alert>
          )}

          {/* Selected File Info */}
          {selectedFile && !validationError && (
            <Box sx={{ mt: 3 }}>
              <Card variant="outlined" sx={{ background: '#F8FAFC' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                    <Description sx={{ fontSize: 40, color: 'primary.main' }} />
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="subtitle1" fontWeight={600}>
                        {selectedFile.name}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Size: {formatFileSize(selectedFile.size)}
                      </Typography>
                    </Box>
                    <Button
                      variant="contained"
                      onClick={handleUpload}
                      disabled={isLoading}
                      sx={{
                        background: 'linear-gradient(135deg, #2563eb 0%, #3b82f6 100%)',
                        '&:hover': {
                          background: 'linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%)',
                        },
                      }}
                    >
                      Upload Dataset
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Box>
          )}

          {/* Upload Progress */}
          {isLoading && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Uploading... {uploadProgress}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={uploadProgress}
                sx={{ height: 8, borderRadius: 4 }}
              />
            </Box>
          )}
        </Paper>

        {/* Dataset Information */}
        {currentDataset && (
          <>
            {/* Dataset Overview */}
            <Paper
              sx={{
                p: 4,
                mb: 3,
                border: '1px solid #e2e8f0',
                background: '#FFFFFF',
              }}
            >
              <Box sx={{ display: 'flex', gap: 3, alignItems: 'center' }}>
                <Box
                  sx={{
                    width: 60,
                    height: 60,
                    borderRadius: 2,
                    background: 'linear-gradient(135deg, #2563eb 0%, #3b82f6 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                  }}
                >
                  <CheckCircle sx={{ fontSize: 32, color: 'white' }} />
                </Box>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="h5" fontWeight={600} gutterBottom>
                    {currentDataset.name}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip
                      label={`${currentDataset.rowCount.toLocaleString()} rows`}
                      size="small"
                      color="primary"
                      variant="outlined"
                    />
                    <Chip
                      label={`${currentDataset.columnCount} columns`}
                      size="small"
                      color="primary"
                      variant="outlined"
                    />
                    <Chip
                      label={formatFileSize(currentDataset.size)}
                      size="small"
                      variant="outlined"
                    />
                    <Chip
                      label={`Uploaded ${formatDate(currentDataset.createdAt)}`}
                      size="small"
                      variant="outlined"
                    />
                  </Box>
                </Box>
              </Box>
            </Paper>

            {/* Loading State for Stats */}
            {currentDataset && !stats && isLoading && (
              <Paper
                sx={{
                  mb: 3,
                  border: '1px solid #e2e8f0',
                  background: '#FFFFFF',
                  p: 3,
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <DataArray color="primary" />
                  <Typography variant="h6" fontWeight={600}>
                    Dataset Statistics
                  </Typography>
                </Box>
                <LinearProgress sx={{ height: 8, borderRadius: 4 }} />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                  Loading dataset statistics and metadata...
                </Typography>
              </Paper>
            )}

            {/* Statistics */}
            {stats && (
              <Paper
                sx={{
                  mb: 3,
                  border: '1px solid #e2e8f0',
                  background: '#FFFFFF',
                }}
              >
                <Box
                  sx={{
                    p: 2,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    cursor: 'pointer',
                  }}
                  onClick={() => setShowStats(!showStats)}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <DataArray color="primary" />
                    <Typography variant="h6" fontWeight={600}>
                      Dataset Statistics
                    </Typography>
                  </Box>
                  <IconButton size="small">
                    {showStats ? <ExpandLess /> : <ExpandMore />}
                  </IconButton>
                </Box>
                <Collapse in={showStats}>
                  <Divider />
                  <Box sx={{ p: 3 }}>
                    <Box
                      sx={{
                        display: 'grid',
                        gridTemplateColumns: {
                          xs: '1fr',
                          sm: 'repeat(2, 1fr)',
                          md: 'repeat(3, 1fr)',
                        },
                        gap: 3,
                      }}
                    >
                      <Card variant="outlined" sx={{ background: '#F8FAFC' }}>
                        <CardContent>
                          <Typography variant="body2" color="text.secondary">
                            Total Rows
                          </Typography>
                          <Typography variant="h5" fontWeight={600}>
                            {stats.rowCount.toLocaleString()}
                          </Typography>
                        </CardContent>
                      </Card>
                      <Card variant="outlined" sx={{ background: '#F8FAFC' }}>
                        <CardContent>
                          <Typography variant="body2" color="text.secondary">
                            Numeric Columns
                          </Typography>
                          <Typography variant="h5" fontWeight={600}>
                            {stats.numericColumns}
                          </Typography>
                        </CardContent>
                      </Card>
                      <Card variant="outlined" sx={{ background: '#F8FAFC' }}>
                        <CardContent>
                          <Typography variant="body2" color="text.secondary">
                            Categorical Columns
                          </Typography>
                          <Typography variant="h5" fontWeight={600}>
                            {stats.categoricalColumns}
                          </Typography>
                        </CardContent>
                      </Card>
                      <Card variant="outlined" sx={{ background: '#F8FAFC' }}>
                        <CardContent>
                          <Typography variant="body2" color="text.secondary">
                            Missing Values
                          </Typography>
                          <Typography variant="h5" fontWeight={600}>
                            {stats.missingValues.toLocaleString()}
                          </Typography>
                        </CardContent>
                      </Card>
                      <Card variant="outlined" sx={{ background: '#F8FAFC' }}>
                        <CardContent>
                          <Typography variant="body2" color="text.secondary">
                            Duplicate Rows
                          </Typography>
                          <Typography variant="h5" fontWeight={600}>
                            {stats.duplicateRows.toLocaleString()}
                          </Typography>
                        </CardContent>
                      </Card>
                      <Card variant="outlined" sx={{ background: '#F8FAFC' }}>
                        <CardContent>
                          <Typography variant="body2" color="text.secondary">
                            Memory Usage
                          </Typography>
                          <Typography variant="h5" fontWeight={600}>
                            {formatFileSize(stats.memoryUsage)}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Box>
                  </Box>
                </Collapse>
              </Paper>
            )}

            {/* Missing Values Analysis */}
            {columns && columns.length > 0 && stats && (
              <Paper
                sx={{
                  mb: 3,
                  border: '1px solid #e2e8f0',
                  background: '#FFFFFF',
                }}
              >
                <Box
                  sx={{
                    p: 2,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    cursor: 'pointer',
                  }}
                  onClick={() => setShowMissingValues(!showMissingValues)}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <ErrorOutline color="primary" />
                    <Typography variant="h6" fontWeight={600}>
                      Missing Values Analysis
                    </Typography>
                  </Box>
                  <IconButton size="small">
                    {showMissingValues ? <ExpandLess /> : <ExpandMore />}
                  </IconButton>
                </Box>
                <Collapse in={showMissingValues}>
                  <Divider />
                  <Box sx={{ p: 3 }}>
                    <MissingValuesAnalysis
                      columns={columns}
                      totalRows={stats.rowCount}
                      isLoading={false}
                    />
                  </Box>
                </Collapse>
              </Paper>
            )}

            {/* Preview */}
            {preview && preview.length > 0 && (
              <Paper
                sx={{
                  mb: 3,
                  border: '1px solid #e2e8f0',
                  background: '#FFFFFF',
                }}
              >
                <Box
                  sx={{
                    p: 2,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    cursor: 'pointer',
                  }}
                  onClick={() => setShowPreview(!showPreview)}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Storage color="primary" />
                    <Typography variant="h6" fontWeight={600}>
                      Data Preview (First 10 rows)
                    </Typography>
                  </Box>
                  <IconButton size="small">
                    {showPreview ? <ExpandLess /> : <ExpandMore />}
                  </IconButton>
                </Box>
                <Collapse in={showPreview}>
                  <Divider />
                  <TableContainer sx={{ maxHeight: 500 }}>
                    <Table stickyHeader size="small">
                      <TableHead>
                        <TableRow>
                          {columns.map((col) => (
                            <TableCell
                              key={col.name}
                              sx={{
                                fontWeight: 600,
                                background: '#F8FAFC',
                              }}
                            >
                              <Box>
                                <Typography variant="body2" fontWeight={600}>
                                  {col.name}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                  {col.dataType}
                                </Typography>
                              </Box>
                            </TableCell>
                          ))}
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {preview.map((row, idx) => (
                          <TableRow key={idx} hover>
                            {columns.map((col) => (
                              <TableCell key={col.name}>
                                {row[col.name]?.toString() || '-'}
                              </TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Collapse>
              </Paper>
            )}
          </>
        )}

        {/* Info Box */}
        {!currentDataset && !selectedFile && (
          <Paper
            sx={{
              p: 3,
              border: '1px solid #e2e8f0',
              background: '#F8FAFC',
            }}
          >
            <Box sx={{ display: 'flex', gap: 2 }}>
              <InfoIcon color="info" />
              <Box>
                <Typography variant="body1" fontWeight={600} gutterBottom>
                  Getting Started
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Upload your dataset to get started with data exploration and machine learning. We support CSV, Excel,
                  and JSON formats up to 100MB.
                </Typography>
              </Box>
            </Box>
          </Paper>
        )}
      </Box>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={handleSnackbarClose} severity={snackbarSeverity} sx={{ width: '100%' }}>
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default DatasetUploadPage;
