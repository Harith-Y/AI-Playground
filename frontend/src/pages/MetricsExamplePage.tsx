/**
 * MetricsExamplePage
 *
 * Example page demonstrating MetricsDisplay and MetricCards
 * Shows all three task types with mock data
 */

import React, { useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Tabs,
  Tab,
  Paper,
  Button,
  Stack,
  Divider,
} from '@mui/material';
import { Refresh as RefreshIcon } from '@mui/icons-material';
import { MetricsDisplay, MetricCard } from '../components/evaluation/metrics';
import { ClusteringCharts } from '../components/evaluation/charts';
import type {
  ClassificationMetrics,
  RegressionMetrics,
  ClusteringMetrics,
} from '../types/evaluation';

type MetricTab = 'classification' | 'regression' | 'clustering' | 'custom';

const MetricsExamplePage: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<MetricTab>('classification');
  const [isLoading, setIsLoading] = useState(false);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: MetricTab) => {
    setCurrentTab(newValue);
  };

  const handleRefresh = () => {
    setIsLoading(true);
    setTimeout(() => setIsLoading(false), 1500);
  };

  // Mock Classification Metrics
  const classificationMetrics: ClassificationMetrics = {
    accuracy: 0.956,
    precision: 0.934,
    recall: 0.967,
    f1Score: 0.950,
    auc: 0.982,
    confusionMatrix: [
      [45, 3],
      [2, 50],
    ],
    classNames: ['Class A', 'Class B'],
  };

  // Mock Regression Metrics
  const regressionMetrics: RegressionMetrics = {
    mae: 3421.5,
    mse: 15678234.2,
    rmse: 3959.5,
    r2: 0.8245,
    mape: 12.34,
  };

  // Mock Clustering Metrics
  const clusteringMetrics: ClusteringMetrics = {
    silhouetteScore: 0.683,
    inertia: 12345.67,
    daviesBouldinScore: 0.45,
    calinskiHarabaszScore: 234.56,
    nClusters: 5,
    clusterSizes: [120, 98, 135, 87, 110],
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box mb={4} display="flex" justifyContent="space-between" alignItems="center">
        <Box>
          <Typography variant="h4" fontWeight="bold" gutterBottom>
            Metrics Display Examples
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Demonstration of MetricsDisplay and MetricCards components
          </Typography>
        </Box>

        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={handleRefresh}
          disabled={isLoading}
        >
          Reload
        </Button>
      </Box>

      {/* Tabs */}
      <Paper elevation={2}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={currentTab} onChange={handleTabChange} variant="fullWidth">
            <Tab value="classification" label="Classification" />
            <Tab value="regression" label="Regression" />
            <Tab value="clustering" label="Clustering" />
            <Tab value="custom" label="Custom Cards" />
          </Tabs>
        </Box>

        <Box sx={{ p: 3 }}>
          {/* Classification Tab */}
          {currentTab === 'classification' && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Classification Metrics Example
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Example metrics from a Random Forest Classifier on the Iris dataset
              </Typography>
              <Divider sx={{ mb: 3 }} />

              <MetricsDisplay
                taskType="classification"
                metrics={classificationMetrics}
                isLoading={isLoading}
                showDescription={true}
                compact={false}
              />
            </Box>
          )}

          {/* Regression Tab */}
          {currentTab === 'regression' && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Regression Metrics Example
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Example metrics from a Linear Regression model on housing price data
              </Typography>
              <Divider sx={{ mb: 3 }} />

              <MetricsDisplay
                taskType="regression"
                metrics={regressionMetrics}
                isLoading={isLoading}
                showDescription={true}
                compact={false}
              />
            </Box>
          )}

          {/* Clustering Tab */}
          {currentTab === 'clustering' && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Clustering Metrics Example
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Example metrics from a K-Means model on customer segmentation data
              </Typography>
              <Divider sx={{ mb: 3 }} />

              <MetricsDisplay
                taskType="clustering"
                metrics={clusteringMetrics}
                isLoading={isLoading}
                showDescription={true}
                compact={false}
              />

              {/* Clustering Charts */}
              <Box mt={4}>
                <Typography variant="h6" gutterBottom>
                  Clustering Visualizations
                </Typography>
                <ClusteringCharts
                  metrics={clusteringMetrics}
                  isLoading={isLoading}
                  showSilhouette={true}
                  showInertia={true}
                  showDistribution={true}
                  showProjection={false}
                />
              </Box>
            </Box>
          )}

          {/* Custom Cards Tab */}
          {currentTab === 'custom' && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Custom MetricCard Examples
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Examples of individual MetricCard components with different configurations
              </Typography>
              <Divider sx={{ mb: 3 }} />

              {/* Success Cards */}
              <Typography variant="subtitle2" fontWeight="bold" gutterBottom sx={{ mt: 3 }}>
                Success Variant (Green)
              </Typography>
              <Box
                display="grid"
                gridTemplateColumns={{ xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }}
                gap={2}
                mb={4}
              >
                <MetricCard
                  label="Accuracy"
                  value={0.95}
                  format="percentage"
                  variant="success"
                  tooltip="Overall model accuracy"
                />
                <MetricCard
                  label="F1 Score"
                  value={0.92}
                  format="percentage"
                  variant="success"
                  showProgress
                  min={0}
                  max={1}
                />
                <MetricCard
                  label="Revenue"
                  value={125000}
                  format="number"
                  precision={0}
                  unit="$"
                  variant="success"
                  trend="up"
                  trendValue={15.5}
                />
                <MetricCard
                  label="R² Score"
                  value={0.89}
                  format="percentage"
                  variant="success"
                  description="Variance explained"
                />
              </Box>

              {/* Warning Cards */}
              <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                Warning Variant (Orange)
              </Typography>
              <Box
                display="grid"
                gridTemplateColumns={{ xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }}
                gap={2}
                mb={4}
              >
                <MetricCard
                  label="Recall"
                  value={0.68}
                  format="percentage"
                  variant="warning"
                  tooltip="True positive rate"
                />
                <MetricCard
                  label="Precision"
                  value={0.72}
                  format="percentage"
                  variant="warning"
                  showProgress
                  min={0}
                  max={1}
                />
                <MetricCard
                  label="Error Rate"
                  value={0.15}
                  format="percentage"
                  variant="warning"
                  trend="down"
                  trendValue={-2.3}
                />
                <MetricCard
                  label="MAPE"
                  value={22.5}
                  format="decimal"
                  precision={1}
                  unit="%"
                  variant="warning"
                />
              </Box>

              {/* Error Cards */}
              <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                Error Variant (Red)
              </Typography>
              <Box
                display="grid"
                gridTemplateColumns={{ xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }}
                gap={2}
                mb={4}
              >
                <MetricCard
                  label="Accuracy"
                  value={0.42}
                  format="percentage"
                  variant="error"
                  tooltip="Below acceptable threshold"
                />
                <MetricCard
                  label="Silhouette"
                  value={0.15}
                  format="decimal"
                  precision={2}
                  variant="error"
                  showProgress
                  min={-1}
                  max={1}
                />
                <MetricCard
                  label="Loss"
                  value={0.85}
                  format="decimal"
                  precision={2}
                  variant="error"
                  trend="up"
                  trendValue={12.8}
                />
                <MetricCard
                  label="R² Score"
                  value={0.25}
                  format="percentage"
                  variant="error"
                  description="Poor fit"
                />
              </Box>

              {/* Default Cards */}
              <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                Default Variant (Blue)
              </Typography>
              <Box
                display="grid"
                gridTemplateColumns={{ xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }}
                gap={2}
                mb={4}
              >
                <MetricCard label="Samples" value={1250} format="integer" />
                <MetricCard
                  label="Features"
                  value={42}
                  format="integer"
                  description="Input dimensions"
                />
                <MetricCard
                  label="Epochs"
                  value={150}
                  format="integer"
                  trend="neutral"
                />
                <MetricCard
                  label="Batch Size"
                  value={32}
                  format="integer"
                  tooltip="Training batch size"
                />
              </Box>

              {/* Large Numbers */}
              <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                Large Number Formatting
              </Typography>
              <Box
                display="grid"
                gridTemplateColumns={{ xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }}
                gap={2}
              >
                <MetricCard
                  label="MAE"
                  value={3421.5}
                  format="number"
                  precision={1}
                  description="Mean absolute error"
                />
                <MetricCard
                  label="RMSE"
                  value={15678.23}
                  format="number"
                  precision={2}
                />
                <MetricCard
                  label="Inertia"
                  value={1234567.89}
                  format="number"
                  precision={0}
                />
                <MetricCard
                  label="Total Cost"
                  value={5500000}
                  format="number"
                  precision={0}
                  unit="$"
                />
              </Box>
            </Box>
          )}
        </Box>
      </Paper>

      {/* Compact Mode Example */}
      <Paper elevation={2} sx={{ mt: 4, p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Compact Mode Example
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          Metrics displayed in compact mode (suitable for dashboards)
        </Typography>
        <Divider sx={{ mb: 3 }} />

        <MetricsDisplay
          taskType="classification"
          metrics={classificationMetrics}
          showDescription={false}
          compact={true}
        />
      </Paper>

      {/* States Demo */}
      <Paper elevation={2} sx={{ mt: 4, p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Component States
        </Typography>
        <Divider sx={{ mb: 3 }} />

        <Stack spacing={3}>
          {/* Loading State */}
          <Box>
            <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
              Loading State
            </Typography>
            <MetricsDisplay
              taskType="classification"
              metrics={null}
              isLoading={true}
            />
          </Box>

          {/* Empty State */}
          <Box>
            <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
              Empty State
            </Typography>
            <MetricsDisplay
              taskType="regression"
              metrics={null}
              isLoading={false}
              emptyMessage="No metrics available yet. Train a model to see results."
            />
          </Box>

          {/* Error State */}
          <Box>
            <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
              Error State
            </Typography>
            <MetricsDisplay
              taskType="clustering"
              metrics={null}
              isLoading={false}
              error="Failed to load metrics from the API. Please try again."
            />
          </Box>
        </Stack>
      </Paper>
    </Container>
  );
};

export default MetricsExamplePage;
