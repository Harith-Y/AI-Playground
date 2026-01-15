import React, { useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Button,
  IconButton,
  Chip,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  Delete as DeleteIcon,
  PlayArrow as PlayArrowIcon,
  CloudUpload as CloudUploadIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { fetchDatasets, deleteDataset, setCurrentDataset } from '../store/slices/datasetSlice';
import type { Dataset } from '../types/dataset';

const DatasetsPage: React.FC = () => {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();
  const { datasets, isLoading, error } = useAppSelector((state) => state.dataset);

  useEffect(() => {
    dispatch(fetchDatasets());
  }, [dispatch]);

  const handleSelectDataset = (dataset: Dataset) => {
    dispatch(setCurrentDataset(dataset));
    navigate('/dataset-upload');
  };

  const handleDeleteDataset = async (datasetId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    if (window.confirm('Are you sure you want to delete this dataset?')) {
      await dispatch(deleteDataset(datasetId));
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString() + ' ' + 
           new Date(dateString).toLocaleTimeString();
  };

  if (isLoading && datasets.length === 0) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
        <Typography variant="h4" component="h1" fontWeight={600} color="primary">
          Datasets
        </Typography>
        <Button
          variant="contained"
          startIcon={<CloudUploadIcon />}
          onClick={() => navigate('/dataset-upload')}
        >
          Upload New
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {datasets.length === 0 ? (
        <Card sx={{ textAlign: 'center', py: 8 }}>
          <CardContent>
            <Typography variant="h6" color="text.secondary" gutterBottom>
              No datasets found
            </Typography>
            <Typography variant="body2" color="text.secondary" mb={3}>
              Upload a dataset to get started with your analysis.
            </Typography>
            <Button
              variant="outlined"
              onClick={() => navigate('/dataset-upload')}
            >
              Go to Upload
            </Button>
          </CardContent>
        </Card>
      ) : (
        <TableContainer component={Paper} sx={{ borderRadius: 2, boxShadow: 3 }}>
          <Table>
            <TableHead sx={{ bgcolor: 'action.hover' }}>
              <TableRow>
                <TableCell sx={{ fontWeight: 600 }}>Name</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Dimensions</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Size</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Uploaded At</TableCell>
                <TableCell sx={{ fontWeight: 600 }} align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {datasets.map((dataset) => (
                <TableRow
                  key={dataset.id}
                  hover
                  sx={{ cursor: 'pointer' }}
                  onClick={() => handleSelectDataset(dataset)}
                >
                  <TableCell>
                    <Typography variant="subtitle2" fontWeight={600}>
                      {dataset.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {dataset.filename}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={`${dataset.rowCount} x ${dataset.columnCount}`}
                      size="small"
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell>
                    {dataset.size ? (dataset.size / 1024 / 1024).toFixed(2) + ' MB' : 'N/A'}
                  </TableCell>
                  <TableCell>{formatDate(dataset.createdAt)}</TableCell>
                  <TableCell align="right">
                    <Button
                      variant="text"
                      color="primary"
                      size="small"
                      startIcon={<PlayArrowIcon />}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleSelectDataset(dataset);
                      }}
                      sx={{ mr: 1 }}
                    >
                      Load
                    </Button>
                    <IconButton
                      size="small"
                      color="error"
                      onClick={(e) => handleDeleteDataset(dataset.id, e)}
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Container>
  );
};

export default DatasetsPage;
