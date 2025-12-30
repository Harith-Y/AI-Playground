import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  Container,
  Box,
  Typography,
  Button,
  Alert,
  CircularProgress,
  Breadcrumbs,
  Link,
  Divider,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Paper,
} from '@mui/material';
import {
  ArrowBack as ArrowBackIcon,
  Home as HomeIcon,
  Compare as CompareIcon,
} from '@mui/icons-material';
import ModelListSelector from '../components/model/ModelListSelector';
import ModelComparisonViewEnhanced from '../components/model/ModelComparisonViewEnhanced';
import type { ModelComparisonData } from '../types/modelComparison';
import { modelComparisonService } from '../services/modelComparisonService';

const ModelComparisonPage: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  
  const [models, setModels] = useState<ModelComparisonData[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Comparison options
  const [comparisonMetrics, setComparisonMetrics] = useState<string>('');
  const [rankingCriteria, setRankingCriteria] = useState<string>('');

  // Fetch models on component mount
  useEffect(() => {
    fetchModels();
    
    // Check for pre-selected models from URL params
    const preselected = searchParams.get('models');
    if (preselected) {
      setSelectedModels(preselected.split(','));
    }
  }, []);

  const fetchModels = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch model runs from backend
      const modelRuns = await modelComparisonService.listModelRuns(
        undefined, // experiment_id
        'completed', // only completed models
        undefined, // model_type
        100, // limit
        0 // offset
      );

      // Transform backend data to frontend format
      const transformedModels: ModelComparisonData[] = modelRuns.map((run: any) => ({
        id: run.id,
        name: run.model_type || `Model ${run.id.slice(0, 8)}`,
        type: run.model_type,
        status: run.status,
        createdAt: run.created_at,
        trainingTime: run.training_time,
        hyperparameters: run.hyperparameters || {},
        datasetId: run.dataset_id || '',
        datasetName: run.dataset_name || 'Unknown Dataset',
        metrics: run.metrics || {},
        featureImportance: run.feature_importance,
      }));

      setModels(transformedModels);
    } catch (err: any) {
      console.error('Error fetching models:', err);
      setError(err.response?.data?.detail || 'Failed to load models');
      
      // Fallback: Load mock data for development
      if (process.env.NODE_ENV === 'development') {
        console.warn('Loading mock data for development');
        setModels(createMockModels());
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSelectionChange = (selectedIds: string[]) => {
    setSelectedModels(selectedIds);
  };

  const handleBack = () => {
    navigate('/modeling');
  };

  const handleCompare = () => {
    // Trigger comparison by updating URL params
    const params = new URLSearchParams();
    params.set('models', selectedModels.join(','));
    navigate(`?${params.toString()}`, { replace: true });
  };

  const getMetricsArray = (): string[] | undefined => {
    if (!comparisonMetrics.trim()) return undefined;
    return comparisonMetrics.split(',').map(m => m.trim()).filter(Boolean);
  };

  if (loading) {
    return (
      <Container maxWidth="xl">
        <Box
          display="flex"
          flexDirection="column"
          justifyContent="center"
          alignItems="center"
          minHeight="60vh"
          gap={2}
        >
          <CircularProgress size={60} />
          <Typography variant="h6" color="text.secondary">
            Loading models...
          </Typography>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl">
      <Box py={4}>
        {/* Breadcrumbs */}
        <Breadcrumbs sx={{ mb: 2 }}>
          <Link
            component="button"
            variant="body2"
            onClick={() => navigate('/')}
            sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}
          >
            <HomeIcon fontSize="small" />
            Home
          </Link>
          <Link
            component="button"
            variant="body2"
            onClick={() => navigate('/modeling')}
          >
            Modeling
          </Link>
          <Typography color="text.primary">Model Comparison</Typography>
        </Breadcrumbs>

        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Box display="flex" alignItems="center" gap={2}>
            <CompareIcon color="primary" sx={{ fontSize: 40 }} />
            <Box>
              <Typography variant="h4" component="h1">
                Model Comparison
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Compare multiple models side-by-side to analyze performance
              </Typography>
            </Box>
          </Box>
          <Button
            variant="outlined"
            startIcon={<ArrowBackIcon />}
            onClick={handleBack}
          >
            Back to Modeling
          </Button>
        </Box>

        {/* Error Alert */}
        {error && (
          <Alert severity="warning" sx={{ mb: 3 }} onClose={() => setError(null)}>
            {error}
            {models.length > 0 && (
              <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                Showing mock data for demonstration
              </Typography>
            )}
          </Alert>
        )}

        {/* Comparison Options */}
        <Paper sx={{ p: 2, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Comparison Options
          </Typography>
          <Box display="flex" gap={2} flexWrap="wrap">
            <TextField
              label="Comparison Metrics (comma-separated)"
              placeholder="e.g., accuracy, f1_score, precision"
              value={comparisonMetrics}
              onChange={(e) => setComparisonMetrics(e.target.value)}
              size="small"
              sx={{ flexGrow: 1, minWidth: 300 }}
              helperText="Leave empty for auto-detection"
            />
            <FormControl size="small" sx={{ minWidth: 200 }}>
              <InputLabel>Ranking Criteria</InputLabel>
              <Select
                value={rankingCriteria}
                onChange={(e) => setRankingCriteria(e.target.value)}
                label="Ranking Criteria"
              >
                <MenuItem value="">
                  <em>Auto-detect</em>
                </MenuItem>
                <MenuItem value="accuracy">Accuracy</MenuItem>
                <MenuItem value="f1_score">F1 Score</MenuItem>
                <MenuItem value="precision">Precision</MenuItem>
                <MenuItem value="recall">Recall</MenuItem>
                <MenuItem value="r2_score">R² Score</MenuItem>
                <MenuItem value="rmse">RMSE</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </Paper>

        {/* Model Selection */}
        <Box mb={4}>
          <ModelListSelector
            models={models}
            selectedModels={selectedModels}
            onSelectionChange={handleSelectionChange}
            maxSelection={10}
            loading={loading}
          />
        </Box>

        {/* Compare Button */}
        {selectedModels.length >= 2 && (
          <Box mb={3} display="flex" justifyContent="center">
            <Button
              variant="contained"
              size="large"
              startIcon={<CompareIcon />}
              onClick={handleCompare}
              disabled={selectedModels.length < 2 || selectedModels.length > 10}
            >
              Compare {selectedModels.length} Models
            </Button>
          </Box>
        )}

        {/* Divider */}
        {selectedModels.length >= 2 && (
          <Divider sx={{ mb: 4 }}>
            <Typography variant="h6" color="primary">
              Comparison Results
            </Typography>
          </Divider>
        )}

        {/* Model Comparison View */}
        {selectedModels.length >= 2 ? (
          <ModelComparisonViewEnhanced
            modelRunIds={selectedModels}
            comparisonMetrics={getMetricsArray()}
            rankingCriteria={rankingCriteria || undefined}
            onRefresh={fetchModels}
          />
        ) : (
          <Alert severity="info" sx={{ mt: 4 }}>
            <Typography variant="body1" gutterBottom>
              <strong>Get Started:</strong>
            </Typography>
            <Typography variant="body2" component="div">
              1. Select at least 2 models from the list above
              <br />
              2. (Optional) Configure comparison metrics and ranking criteria
              <br />
              3. Click "Compare" to see detailed analysis
            </Typography>
          </Alert>
        )}
      </Box>
    </Container>
  );
};

// Mock data generator for demo/development purposes
function createMockModels(): ModelComparisonData[] {
  return [
    {
      id: 'model-1',
      name: 'Random Forest Classifier v1',
      type: 'random_forest',
      status: 'completed',
      createdAt: new Date(Date.now() - 86400000).toISOString(), // 1 day ago
      trainingTime: 45,
      hyperparameters: {
        n_estimators: 100,
        max_depth: 10,
        min_samples_split: 2,
      },
      datasetId: 'dataset-1',
      datasetName: 'Customer Churn Dataset',
      metrics: {
        accuracy: 0.8945,
        precision: 0.8723,
        recall: 0.8567,
        f1Score: 0.8644,
        rocAuc: 0.9234,
        trainScore: 0.9123,
        valScore: 0.8945,
        testScore: 0.8789,
      },
      featureImportance: [
        { feature: 'tenure', importance: 0.234 },
        { feature: 'monthly_charges', importance: 0.189 },
        { feature: 'total_charges', importance: 0.156 },
        { feature: 'contract_type', importance: 0.123 },
        { feature: 'payment_method', importance: 0.098 },
      ],
    },
    {
      id: 'model-2',
      name: 'Logistic Regression v2',
      type: 'logistic_regression',
      status: 'completed',
      createdAt: new Date(Date.now() - 172800000).toISOString(), // 2 days ago
      trainingTime: 12,
      hyperparameters: {
        C: 1.0,
        penalty: 'l2',
        max_iter: 1000,
      },
      datasetId: 'dataset-1',
      datasetName: 'Customer Churn Dataset',
      metrics: {
        accuracy: 0.8234,
        precision: 0.8156,
        recall: 0.7989,
        f1Score: 0.8071,
        rocAuc: 0.8756,
        trainScore: 0.8345,
        valScore: 0.8234,
        testScore: 0.8123,
      },
    },
    {
      id: 'model-3',
      name: 'SVM Classifier',
      type: 'svm',
      status: 'completed',
      createdAt: new Date(Date.now() - 259200000).toISOString(), // 3 days ago
      trainingTime: 89,
      hyperparameters: {
        C: 1.0,
        kernel: 'rbf',
        gamma: 'scale',
      },
      datasetId: 'dataset-1',
      datasetName: 'Customer Churn Dataset',
      metrics: {
        accuracy: 0.8567,
        precision: 0.8445,
        recall: 0.8234,
        f1Score: 0.8338,
        rocAuc: 0.8923,
        trainScore: 0.8678,
        valScore: 0.8567,
        testScore: 0.8456,
      },
    },
    {
      id: 'model-4',
      name: 'Gradient Boosting v1',
      type: 'gradient_boosting',
      status: 'completed',
      createdAt: new Date(Date.now() - 345600000).toISOString(), // 4 days ago
      trainingTime: 156,
      hyperparameters: {
        n_estimators: 100,
        learning_rate: 0.1,
        max_depth: 6,
      },
      datasetId: 'dataset-1',
      datasetName: 'Customer Churn Dataset',
      metrics: {
        accuracy: 0.9123,
        precision: 0.8967,
        recall: 0.8834,
        f1Score: 0.8900,
        rocAuc: 0.9456,
        trainScore: 0.9234,
        valScore: 0.9123,
        testScore: 0.9012,
      },
    },
    {
      id: 'model-5',
      name: 'Linear Regression v1',
      type: 'linear_regression',
      status: 'completed',
      createdAt: new Date(Date.now() - 432000000).toISOString(), // 5 days ago
      trainingTime: 8,
      hyperparameters: {
        fit_intercept: true,
        normalize: false,
      },
      datasetId: 'dataset-2',
      datasetName: 'House Prices Dataset',
      metrics: {
        rmse: 0.1234,
        mae: 0.0987,
        r2Score: 0.8456,
        mse: 0.0152,
        trainScore: 0.8567,
        valScore: 0.8456,
        testScore: 0.8345,
      },
    },
    {
      id: 'model-6',
      name: 'Random Forest Regressor',
      type: 'random_forest',
      status: 'completed',
      createdAt: new Date(Date.now() - 518400000).toISOString(), // 6 days ago
      trainingTime: 67,
      hyperparameters: {
        n_estimators: 200,
        max_depth: 15,
        min_samples_split: 5,
      },
      datasetId: 'dataset-2',
      datasetName: 'House Prices Dataset',
      metrics: {
        rmse: 0.0987,
        mae: 0.0756,
        r2Score: 0.8923,
        mse: 0.0097,
        trainScore: 0.9012,
        valScore: 0.8923,
        testScore: 0.8834,
      },
      featureImportance: [
        { feature: 'square_footage', importance: 0.345 },
        { feature: 'location', importance: 0.234 },
        { feature: 'bedrooms', importance: 0.156 },
        { feature: 'bathrooms', importance: 0.123 },
        { feature: 'age', importance: 0.098 },
      ],
    },
    {
      id: 'model-7',
      name: 'Neural Network v1',
      type: 'neural_network',
      status: 'training',
      createdAt: new Date(Date.now() - 3600000).toISOString(), // 1 hour ago
      trainingTime: 234,
      hyperparameters: {
        hidden_layers: [100, 50, 25],
        activation: 'relu',
        learning_rate: 0.001,
      },
      datasetId: 'dataset-1',
      datasetName: 'Customer Churn Dataset',
      metrics: {
        accuracy: 0.8756, // Partial results
        trainScore: 0.8834,
        valScore: 0.8756,
      },
    },
    {
      id: 'model-8',
      name: 'Decision Tree v1',
      type: 'decision_tree',
      status: 'failed',
      createdAt: new Date(Date.now() - 7200000).toISOString(), // 2 hours ago
      trainingTime: 15,
      hyperparameters: {
        max_depth: 20,
        min_samples_split: 2,
        criterion: 'gini',
      },
      datasetId: 'dataset-1',
      datasetName: 'Customer Churn Dataset',
      metrics: {},
    },
  ];
}

export default ModelComparisonPage;
