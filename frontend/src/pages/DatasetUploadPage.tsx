import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Typography,
  Button,
  LinearProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
} from '@mui/material';
import {
  CloudUpload,
  InsertDriveFile,
  CheckCircle,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { datasetService } from '../services/datasetService';
import type { Dataset } from '../types/dataset';

const DatasetUploadPage: React.FC = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedDataset, setUploadedDataset] = useState<Dataset | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const ALLOWED_EXTENSIONS = ['.csv', '.xlsx', '.xls', '.json', '.parquet'];
  const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB

  const validateFile = (file: File): string | null => {
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();

    if (!ALLOWED_EXTENSIONS.includes(fileExtension)) {
      return `Invalid file type. Allowed types: ${ALLOWED_EXTENSIONS.join(', ')}`;
    }

    if (file.size > MAX_FILE_SIZE) {
      return `File size exceeds maximum limit of ${MAX_FILE_SIZE / (1024 * 1024)}MB`;
    }

    return null;
  };

  const handleFileSelect = (file: File) => {
    setError(null);
    setUploadedDataset(null);

    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
  };

  const handleFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setError(null);
    setUploadProgress(0);

    try {
      const dataset = await datasetService.uploadDataset(selectedFile);
      setUploadedDataset(dataset);
      setUploadProgress(100);

      // Redirect to dataset details page after 2 seconds
      setTimeout(() => {
        navigate(`/datasets/${dataset.id}`);
      }, 2000);
    } catch (err: any) {
      setError(err.message || 'Failed to upload dataset. Please try again.');
      setUploadProgress(0);
    } finally {
      setUploading(false);
    }
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: 'calc(100vh - 64px)',
        width: '100%',
        textAlign: 'center',
        p: 3,
      }}
    >
      <CloudUpload sx={{ fontSize: 80, color: 'primary.main', mb: 3 }} />
      <Typography variant="h3" component="h1" fontWeight={600} gutterBottom>
        Dataset Upload
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Upload your dataset to start exploring and building models
      </Typography>

      <Paper
        sx={{
          p: 4,
          width: '100%',
          maxWidth: 600,
          border: '1px solid #334155',
        }}
      >
        {/* Drag and Drop Area */}
        <Box
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          sx={{
            border: `2px dashed ${dragActive ? '#3b82f6' : '#475569'}`,
            borderRadius: 2,
            p: 4,
            mb: 3,
            backgroundColor: dragActive ? 'rgba(59, 130, 246, 0.05)' : 'transparent',
            transition: 'all 0.2s ease',
            cursor: 'pointer',
          }}
          onClick={handleBrowseClick}
        >
          <CloudUpload sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            Drag and drop your file here
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            or click to browse
          </Typography>
          <Chip
            label={`Supported: ${ALLOWED_EXTENSIONS.join(', ')}`}
            size="small"
            sx={{ mb: 1 }}
          />
          <Typography variant="caption" display="block" color="text.secondary">
            Maximum file size: {MAX_FILE_SIZE / (1024 * 1024)}MB
          </Typography>
        </Box>

        <input
          ref={fileInputRef}
          type="file"
          accept={ALLOWED_EXTENSIONS.join(',')}
          onChange={handleFileInputChange}
          style={{ display: 'none' }}
        />

        {/* Selected File Info */}
        {selectedFile && !uploadedDataset && (
          <Box sx={{ mb: 3 }}>
            <List>
              <ListItem
                sx={{
                  border: '1px solid #334155',
                  borderRadius: 1,
                  mb: 2,
                }}
              >
                <ListItemIcon>
                  <InsertDriveFile color="primary" />
                </ListItemIcon>
                <ListItemText
                  primary={selectedFile.name}
                  secondary={formatFileSize(selectedFile.size)}
                />
              </ListItem>
            </List>

            <Button
              variant="contained"
              fullWidth
              size="large"
              startIcon={<CloudUpload />}
              onClick={handleUpload}
              disabled={uploading}
            >
              {uploading ? 'Uploading...' : 'Upload Dataset'}
            </Button>
          </Box>
        )}

        {/* Upload Progress */}
        {uploading && (
          <Box sx={{ mb: 3 }}>
            <LinearProgress variant="determinate" value={uploadProgress} />
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
              Uploading: {uploadProgress}%
            </Typography>
          </Box>
        )}

        {/* Success Message */}
        {uploadedDataset && (
          <Alert
            severity="success"
            icon={<CheckCircle />}
            sx={{ mb: 2 }}
          >
            <Typography variant="body2">
              Dataset uploaded successfully! Redirecting to dataset details...
            </Typography>
          </Alert>
        )}

        {/* Error Message */}
        {error && (
          <Alert
            severity="error"
            icon={<ErrorIcon />}
            onClose={() => setError(null)}
            sx={{ mb: 2 }}
          >
            {error}
          </Alert>
        )}

        {/* Help Text */}
        <Box sx={{ mt: 3, textAlign: 'left' }}>
          <Typography variant="subtitle2" gutterBottom>
            Supported File Formats:
          </Typography>
          <List dense>
            <ListItem>
              <Typography variant="body2" color="text.secondary">
                • CSV (.csv) - Comma-separated values
              </Typography>
            </ListItem>
            <ListItem>
              <Typography variant="body2" color="text.secondary">
                • Excel (.xlsx, .xls) - Microsoft Excel files
              </Typography>
            </ListItem>
            <ListItem>
              <Typography variant="body2" color="text.secondary">
                • JSON (.json) - JavaScript Object Notation
              </Typography>
            </ListItem>
            <ListItem>
              <Typography variant="body2" color="text.secondary">
                • Parquet (.parquet) - Apache Parquet format
              </Typography>
            </ListItem>
          </List>
        </Box>
      </Paper>
    </Box>
  );
};

export default DatasetUploadPage;
