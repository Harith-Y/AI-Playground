import React, { useEffect, useState, useCallback } from 'react';
import { Typography, Box, Container, Grid, Card, CardContent, Button, Alert } from '@mui/material';
import { TrendingUp, CheckCircle } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import {
  fetchCorrelationMatrix,
  fetchFeatureImportance,
  clearFeatureError,
  setSelectedFeatures,
  setAvailableFeatures,
} from '../store/slices/featureSlice';
import LoadingState from '../components/common/LoadingState';
import ErrorState from '../components/common/ErrorState';
import EmptyState from '../components/common/EmptyState';
import CorrelationMatrix from '../components/features/CorrelationMatrix';
import FeatureImportance from '../components/features/FeatureImportance';
import FeatureSelection from '../components/features/FeatureSelection';
import type { FeatureSelectionConfig, FeatureSelectionValidation } from '../types/featureSelection';
import { getColumnDataType } from '../types/featureSelection';

const FeatureEngineeringPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { correlationMatrix, featureImportance, isLoading, error, selectedFeatures } = useAppSelector(
    (state) => state.feature
  );
  const { currentDataset, columns } = useAppSelector((state) => state.dataset);
  
  // Convert dataset columns to feature selection format
  const featureColumns = React.useMemo(() => {
    if (!columns) return [];
    return columns.map(col => ({
      name: col.name,
      dataType: getColumnDataType(col.dtype),
      dtype: col.dtype,
      missing_count: col.nullCount,
      unique_count: col.uniqueCount,
      sample_values: col.sampleValues,
    }));
  }, [columns]);
  
  // Feature selection state
  const [featureConfig, setFeatureConfig] = useState<FeatureSelectionConfig>({
    inputFeatures: [],
    targetColumn: null,
    taskType: null,
    excludedColumns: [],
  });
  const [validation, setValidation] = useState<FeatureSelectionValidation | null>(null);
  const [showSuccess, setShowSuccess] = useState(false);

  useEffect(() => {
    // Set available features from dataset columns
    if (columns && columns.length > 0) {
      dispatch(setAvailableFeatures(columns.map(col => col.name)));
    }
  }, [columns, dispatch]);

  useEffect(() => {
    // Restore selected features if they exist
    if (selectedFeatures && selectedFeatures.length > 0) {
      setFeatureConfig(prev => ({
        ...prev,
        inputFeatures: selectedFeatures,
      }));
    }
  }, [selectedFeatures]);
const handleConfigChange = useCallback((config: FeatureSelectionConfig) => {
    setFeatureConfig(config);
  }, []);

  const handleValidationChange = useCallback((newValidation: FeatureSelectionValidation) => {
    setValidation(newValidation);
  }, []);

  const handleContinue = () => {
    if (validation?.isValid) {
      // Save selected features to Redux store
      dispatch(setSelectedFeatures(featureConfig.inputFeatures));
      setShowSuccess(true);
      setTimeout(() => setShowSuccess(false), 3000);
    }
  };

  
  useEffect(() => {
    return () => {
      dispatch(clearFeatureError());
    };
  }, [dispatch]);

  const handleRetry = () => {
    dispatch(clearFeatureError());
    if (currentDataset?.id) {
      dispatch(fetchCorrelationMatrix(currentDataset.id));
      dispatch(fetchFeatureImportance(currentDataset.id));
    }
  };

  const handleRefreshCorrelation = () => {
    if (currentDataset?.id) {
      dispatch(fetchCorrelationMatrix(currentDataset.id));
    }
  };

  const _handleRefreshImportance = () => {
    if (currentDataset?.id) {
      dispatch(fetchFeatureImportance(currentDataset.id));
    }
  };

  if (isLoading && !correlationMatrix && !featureImportance.length) {
    return <LoadingState message="Analyzing features..." />;
  }

  if (error && !correlationMatrix && !featureImportance.length) {
    return (
      <ErrorState
        title="Feature Analysis Failed"
        message={error}
        onRetry={handleRetry}
      />
    );
  }

  if (!currentDataset) {
    return (
      <EmptyState
        title="No Dataset Selected"
        message="Please upload a dataset first to analyze features"
      />
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Page Header */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <TrendingUp sx={{ fontSize: 40, color: 'primary.main' }} />
          <Box>
            <Typography variant="h4" component="h1" fontWeight={600}>
              Feature Engineering
      {/* Success Alert */}
      {showSuccess && (
        <Alert severity="success" icon={<CheckCircle />} sx={{ mb: 3 }}>
          Feature selection saved! You can now proceed to model training with {featureConfig.inputFeatures.length} selected features.
        </AfeatureColumns && featureColumns.length > 0 ? (
            <>
              <FeatureSelection
                datasetId={currentDataset!.id}
                columns={featureCorder: '1px solid #e2e8f0' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom fontWeight={600} color="primary.main">
            Feature Selection
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Select input features and target column for model training
          </Typography>
          
          {columns && columns.length > 0 ? (
            <>
              <FeatureSelection
                datasetId={currentDataset!.id}
                columns={columns}
                initialConfig={featureConfig}
                onConfigChange={handleConfigChange}
                onValidationChange={handleValidationChange}
              />
              <Box mt={3} display="flex" justifyContent="flex-end">
                <Button
                  variant="contained"
                  size="large"
                  onClick={handleContinue}
                  disabled={!validation?.isValid}
                  startIcon={<CheckCircle />}
                >
                  Continue to Modeling
                </Button>
              </Box>
            </>
          ) : (
            <Alert severity="info">
              No columns available. Please ensure your dataset is loaded properly.
            </Alert>
          )}
        </CardContent>
      </Card>

            </Typography>
            <Typography variant="body2" color="text.secondary">
              Analyze and select the most important features
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* Error Alert (non-blocking) */}
      {error && (correlationMatrix || featureImportance.length > 0) && (
        <ErrorState
          message={error}
          onRetry={handleRetry}
          variant="alert"
        />
      )}

      <Grid container spacing={3}>
        {/* Correlation Matrix */}
        <Grid size={{ xs: 12, lg: 6 }}>
          <Card sx={{ border: '1px solid #e2e8f0' }}>
            <CardContent>
              <CorrelationMatrix
                matrix={correlationMatrix}
                isLoading={isLoading}
                onRefresh={handleRefreshCorrelation}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Feature Importance */}
        <Grid size={{ xs: 12, lg: 6 }}>
          <Card sx={{ border: '1px solid #e2e8f0' }}>
            <CardContent>
              <FeatureImportance
                features={featureImportance}
                isLoading={isLoading}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default FeatureEngineeringPage;
