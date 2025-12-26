import React, { useState, useCallback } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Alert,
  LinearProgress,
} from '@mui/material';
import { CloudUpload, CheckCircle } from '@mui/icons-material';
import { formatFileSize } from '../../utils/helpers';

interface DatasetUploadProps {
  onFileSelect: (file: File) => void;
  onUpload: () => void;
  selectedFile: File | null;
  isLoading?: boolean;
  uploadProgress?: number;
  error?: string | null;
}

const DatasetUpload: React.FC<DatasetUploadProps> = ({
  onFileSelect,
  onUpload,
  selectedFile,
  isLoading = false,
  uploadProgress = 0,
  error = null,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

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
  const handleFileSelect = useCallback(
    (file: File) => {
      const validationErr = validateFile(file);
      if (validationErr) {
        setValidationError(validationErr);
        return;
      }

      setValidationError(null);
      onFileSelect(file);
    },
    [onFileSelect]
  );

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

  return (
    <Paper
      sx={{
        p: 4,
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
        onClick={() => document.getElementById('dataset-file-input')?.click()}
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
          id="dataset-file-input"
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

      {/* API Error */}
      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      {/* Selected File Info */}
      {selectedFile && !validationError && (
        <Box
          sx={{
            mt: 3,
            p: 2,
            borderRadius: 2,
            background: '#F0FDF4',
            border: '1px solid #86EFAC',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <CheckCircle sx={{ color: '#16A34A' }} />
            <Box>
              <Typography variant="body1" fontWeight={600}>
                {selectedFile.name}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {formatFileSize(selectedFile.size)}
              </Typography>
            </Box>
          </Box>
          <Button
            variant="contained"
            onClick={onUpload}
            disabled={isLoading}
            startIcon={<CloudUpload />}
            sx={{
              background: 'linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)',
              '&:hover': {
                background: 'linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%)',
              },
            }}
          >
            {isLoading ? 'Uploading...' : 'Upload Dataset'}
          </Button>
        </Box>
      )}

      {/* Upload Progress */}
      {isLoading && (
        <Box sx={{ mt: 2 }}>
          <LinearProgress variant="determinate" value={uploadProgress} />
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
            Uploading... {uploadProgress}%
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default DatasetUpload;
