/**
 * Feature Selection Page
 *
 * Example page demonstrating how to use the FeatureSelection component
 * with dataset integration.
 */

import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Button,
  Stack,
  Card,
  CardContent,
  Alert,
  CircularProgress,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { ArrowBack, ArrowForward } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { fetchDatasetStats } from '../store/slices/datasetSlice';
import FeatureSelection from '../components/features/FeatureSelection';
import type {
  FeatureSelectionConfig,
  FeatureSelectionValidation,
  ColumnInfo,
} from '../types/featureSelection';
import { getColumnDataType } from '../types/featureSelection';

const FeatureSelectionPage: React.FC = () => {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();

  // Get current dataset from Redux
  const { currentDataset, stats, columns: datasetColumns, isLoading, error } = useAppSelector((state) => state.dataset);

  // Local state for feature selection
  const [config, setConfig] = useState<FeatureSelectionConfig | null>(null);
  const [validation, setValidation] = useState<FeatureSelectionValidation | null>(null);

  // Fetch dataset stats when component mounts
  useEffect(() => {
    if (currentDataset?.id && !stats) {
      dispatch(fetchDatasetStats(currentDataset.id));
    }
  }, [currentDataset, stats, dispatch]);

  // Convert stats columns to ColumnInfo format
  const columns: ColumnInfo[] = React.useMemo(() => {
    if (!datasetColumns || datasetColumns.length === 0) return [];

    return datasetColumns.map((col) => ({
      name: col.name,
      dataType: getColumnDataType(col.dataType),
      dtype: col.dataType,
      missing_count: col.nullCount,
      missing_percentage: col.nullCount > 0 && currentDataset ? (col.nullCount / currentDataset.rowCount) * 100 : 0,
      unique_count: col.uniqueCount,
      sample_values: col.sampleValues,
    }));
  }, [datasetColumns, currentDataset]);

  // Handle configuration change
  const handleConfigChange = (newConfig: FeatureSelectionConfig) => {
    setConfig(newConfig);
    console.log('Feature selection config updated:', newConfig);
  };

  // Handle validation change
  const handleValidationChange = (newValidation: FeatureSelectionValidation) => {
    setValidation(newValidation);
  };

  // Handle continue to next step
  const handleContinue = () => {
    if (!validation?.isValid) {
      return;
    }

    // Save config to sessionStorage or Redux
    if (config) {
      sessionStorage.setItem('featureSelectionConfig', JSON.stringify(config));
      console.log('Feature selection saved:', config);
    }

    // Navigate to next step (e.g., model training)
    navigate('/modeling');
  };

  // Handle go back
  const handleBack = () => {
    navigate('/preprocessing');
  };

  // Loading state
  if (isLoading) {
    return (
      <Container maxWidth="xl" sx={{ py: 4 }}>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  // Error state
  if (error) {
    return (
      <Container maxWidth="xl" sx={{ py: 4 }}>
        <Alert severity="error">
          <Typography variant="h6">Error Loading Dataset</Typography>
          <Typography variant="body2">{error}</Typography>
        </Alert>
      </Container>
    );
  }

  // No dataset selected
  if (!currentDataset) {
    return (
      <Container maxWidth="xl" sx={{ py: 4 }}>
        <Alert severity="warning">
          <Typography variant="h6">No Dataset Selected</Typography>
          <Typography variant="body2">
            Please upload and select a dataset first.
          </Typography>
          <Button
            variant="contained"
            sx={{ mt: 2 }}
            onClick={() => navigate('/dataset-upload')}
          >
            Upload Dataset
          </Button>
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box mb={4}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Feature Selection
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Select input features, target column, and task type for model training
        </Typography>
      </Box>

      {/* Dataset Info Card */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Current Dataset: {currentDataset.name}
          </Typography>
          <Stack direction="row" spacing={4}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Rows
              </Typography>
              <Typography variant="body1" fontWeight="bold">
                {stats?.rowCount?.toLocaleString() || currentDataset.rowCount?.toLocaleString()}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Columns
              </Typography>
              <Typography variant="body1" fontWeight="bold">
                {stats?.columnCount || currentDataset.columnCount}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Available Columns
              </Typography>
              <Typography variant="body1" fontWeight="bold">
                {columns.length}
              </Typography>
            </Box>
          </Stack>
        </CardContent>
      </Card>

      {/* Feature Selection Component */}
      {columns.length > 0 ? (
        <FeatureSelection
          datasetId={currentDataset.id}
          columns={columns}
          onConfigChange={handleConfigChange}
          onValidationChange={handleValidationChange}
        />
      ) : (
        <Alert severity="info">
          Loading dataset columns...
        </Alert>
      )}

      {/* Navigation Buttons */}
      <Box display="flex" justifyContent="space-between" mt={4}>
        <Button
          variant="outlined"
          startIcon={<ArrowBack />}
          onClick={handleBack}
          size="large"
        >
          Back to Preprocessing
        </Button>
        <Button
          variant="contained"
          endIcon={<ArrowForward />}
          onClick={handleContinue}
          disabled={!validation?.isValid}
          size="large"
        >
          Continue to Model Training
        </Button>
      </Box>

      {/* Debug Info (remove in production) */}
      {import.meta.env.DEV && config && (
        <Card sx={{ mt: 4, bgcolor: 'action.hover' }}>
          <CardContent>
            <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
              Debug: Current Configuration
            </Typography>
            <pre style={{ fontSize: '12px', overflow: 'auto' }}>
              {JSON.stringify(config, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}
    </Container>
  );
};

export default FeatureSelectionPage;
