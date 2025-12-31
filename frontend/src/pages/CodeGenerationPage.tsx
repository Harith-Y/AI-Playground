/**
 * Code Generation Page
 * 
 * Main code generation interface with:
 * - Dataset/experiment selection
 * - Format choices (Python/Notebook/FastAPI)
 * - Code preview and download
 */

import React, { useState } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Breadcrumbs,
  Link,
  Card,
  CardContent,
  Button,
  Chip,
  Divider,
  Alert,
  Stack,
  ToggleButtonGroup,
  ToggleButton,
  TextField,
  Autocomplete,
} from '@mui/material';
import {
  NavigateNext as NavigateNextIcon,
  Home as HomeIcon,
  Code as CodeIcon,
  Description as DescriptionIcon,
  Api as ApiIcon,
  Share as ShareIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import CodePreview from '../components/codegen/CodePreview';
import DownloadOptions from '../components/codegen/DownloadOptions';
import ExperimentSummary from '../components/codegen/ExperimentSummary';
import GenerationHistory from '../components/codegen/GenerationHistory';
import ShareExperiment from '../components/codegen/ShareExperiment';
import type { ZipOptions } from '../components/codegen/DownloadOptions';
import {
  generatePythonScript,
  generateJupyterNotebook,
  generateFastAPIService,
} from '../utils/codeGenerators';

// Types
interface Dataset {
  id: string;
  name: string;
  rows: number;
  columns: number;
  created_at: string;
}

interface Experiment {
  id: string;
  name: string;
  model_type: string;
  score: number;
  created_at: string;
}

type CodeFormat = 'python' | 'notebook' | 'fastapi';

const CodeGenerationPage: React.FC = () => {
  const navigate = useNavigate();

  // State
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [selectedExperiment, setSelectedExperiment] = useState<Experiment | null>(null);
  const [codeFormat, setCodeFormat] = useState<CodeFormat>('python');
  const [includePreprocessing, setIncludePreprocessing] = useState(true);
  const [includeEvaluation, setIncludeEvaluation] = useState(true);
  const [includeVisualization, setIncludeVisualization] = useState(false);
  const [generatedCode, setGeneratedCode] = useState<Array<{ filename: string; content: string; language: string }>>([]);
  const [shareDialogOpen, setShareDialogOpen] = useState(false);
  const [generationHistory, setGenerationHistory] = useState<Array<{
    id: string;
    experimentName: string;
    datasetName: string;
    format: CodeFormat;
    timestamp: string;
  }>>([]);

  // Mock data
  const mockDatasets: Dataset[] = [
    { id: '1', name: 'Customer Churn Dataset', rows: 10000, columns: 15, created_at: '2024-01-15' },
    { id: '2', name: 'Sales Forecast Data', rows: 5000, columns: 12, created_at: '2024-01-20' },
    { id: '3', name: 'Product Recommendations', rows: 25000, columns: 20, created_at: '2024-02-01' },
  ];

  const mockExperiments: Experiment[] = [
    { id: '1', name: 'Random Forest Classifier', model_type: 'classification', score: 0.89, created_at: '2024-01-16' },
    { id: '2', name: 'XGBoost Regressor', model_type: 'regression', score: 0.92, created_at: '2024-01-22' },
    { id: '3', name: 'Neural Network', model_type: 'classification', score: 0.91, created_at: '2024-02-02' },
  ];

  // Handlers
  const handleBreadcrumbClick = (path: string) => {
    navigate(path);
  };

  const handleFormatChange = (_event: React.MouseEvent<HTMLElement>, newFormat: CodeFormat | null) => {
    if (newFormat !== null) {
      setCodeFormat(newFormat);
    }
  };

  const handleGenerate = () => {
    if (!selectedDataset || !selectedExperiment) return;

    const options = {
      datasetName: selectedDataset.name,
      modelName: selectedExperiment.name,
      modelType: selectedExperiment.model_type,
      includePreprocessing,
      includeEvaluation,
      includeVisualization,
    };

    let code: string;
    let filename: string;
    let language: string;

    switch (codeFormat) {
      case 'python':
        code = generatePythonScript(options);
        filename = 'model.py';
        language = 'python';
        break;
      case 'notebook':
        code = generateJupyterNotebook(options);
        filename = 'model.ipynb';
        language = 'json';
        break;
      case 'fastapi':
        code = generateFastAPIService(options);
        filename = 'main.py';
        language = 'python';
        break;
      default:
        return;
    }

    setGeneratedCode([{ filename, content: code, language }]);

    // Add to history
    const historyItem = {
      id: Date.now().toString(),
      experimentName: selectedExperiment.name,
      datasetName: selectedDataset.name,
      format: codeFormat,
      timestamp: new Date().toISOString(),
    };
    setGenerationHistory(prev => [historyItem, ...prev].slice(0, 10)); // Keep last 10
  };

  const handleDownloadFile = (_filename: string) => {
    if (generatedCode.length === 0) return;

    const file = generatedCode[0];
    const blob = new Blob([file.content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = file.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleDownloadZip = (options: ZipOptions) => {
    if (!selectedDataset || !selectedExperiment || generatedCode.length === 0) return;

    // In a real implementation, you would use JSZip or similar
    console.log('Downloading ZIP with options:', options);
    
    // Mock ZIP download
    alert(`ZIP package "${options.projectName}.zip" would be downloaded with:\n` +
          `- Main code file\n` +
          (options.includeRequirements ? '- requirements.txt\n' : '') +
          (options.includeReadme ? '- README.md\n' : '') +
          (options.includeDockerfile ? '- Dockerfile\n' : '') +
          (options.includeTests ? '- test_model.py\n' : ''));
  };

  const handleHistorySelect = (item: typeof generationHistory[0]) => {
    // Find and select the dataset and experiment
    const dataset = mockDatasets.find(d => d.name === item.datasetName);
    const experiment = mockExperiments.find(e => e.name === item.experimentName);
    
    if (dataset && experiment) {
      setSelectedDataset(dataset);
      setSelectedExperiment(experiment);
      setCodeFormat(item.format);
      // Regenerate code
      setTimeout(() => handleGenerate(), 100);
    }
  };

  const handleHistoryDelete = (id: string) => {
    setGenerationHistory(prev => prev.filter(item => item.id !== id));
  };

  const handleHistoryClear = () => {
    setGenerationHistory([]);
  };

  const canGenerate = selectedDataset && selectedExperiment;

  return (
    <Box sx={{ flexGrow: 1, bgcolor: 'background.default', minHeight: '100vh', py: 3 }}>
      <Container maxWidth="xl">
        {/* Header Section */}
        <Box sx={{ mb: 3 }}>
          {/* Breadcrumbs */}
          <Breadcrumbs
            separator={<NavigateNextIcon fontSize="small" />}
            aria-label="breadcrumb"
            sx={{ mb: 2 }}
          >
            <Link
              color="inherit"
              href="#"
              onClick={(e) => {
                e.preventDefault();
                handleBreadcrumbClick('/');
              }}
              sx={{ display: 'flex', alignItems: 'center', textDecoration: 'none' }}
            >
              <HomeIcon sx={{ mr: 0.5 }} fontSize="small" />
              Home
            </Link>
            <Typography color="text.primary" sx={{ display: 'flex', alignItems: 'center' }}>
              <CodeIcon sx={{ mr: 0.5 }} fontSize="small" />
              Code Generation
            </Typography>
          </Breadcrumbs>

          {/* Page Title */}
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Box>
              <Typography variant="h4" component="h1" gutterBottom>
                Code Generation
              </Typography>
              <Typography variant="body1" color="text.secondary">
                Generate production-ready code from your ML experiments
              </Typography>
            </Box>
            <Chip
              icon={<CodeIcon />}
              label="Export Code"
              color="primary"
              variant="outlined"
            />
          </Box>
        </Box>

        {/* Info Alert */}
        <Alert severity="info" sx={{ mb: 3 }}>
          Select a dataset and experiment to generate production-ready code. Choose your preferred format
          and customize the output options.
        </Alert>

        {/* Main Content */}
        <Box display="flex" flexDirection={{ xs: 'column', lg: 'row' }} gap={3}>
          {/* Left Column - Configuration */}
          <Box sx={{ flex: { xs: '1 1 100%', lg: '0 0 40%' } }}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                  <CodeIcon sx={{ mr: 1 }} />
                  Configuration
                </Typography>
                <Divider sx={{ mb: 3 }} />

                {/* Dataset Selection */}
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Select Dataset
                  </Typography>
                  <Autocomplete
                    options={mockDatasets}
                    getOptionLabel={(option) => option.name}
                    value={selectedDataset}
                    onChange={(_event, newValue) => setSelectedDataset(newValue)}
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        placeholder="Choose a dataset"
                        helperText="Select the dataset used in your experiment"
                      />
                    )}
                    renderOption={(props, option) => (
                      <li {...props}>
                        <Box>
                          <Typography variant="body2">{option.name}</Typography>
                          <Typography variant="caption" color="text.secondary">
                            {option.rows.toLocaleString()} rows × {option.columns} columns
                          </Typography>
                        </Box>
                      </li>
                    )}
                  />
                </Box>

                {/* Experiment Selection */}
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Select Experiment
                  </Typography>
                  <Autocomplete
                    options={mockExperiments}
                    getOptionLabel={(option) => option.name}
                    value={selectedExperiment}
                    onChange={(_event, newValue) => setSelectedExperiment(newValue)}
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        placeholder="Choose an experiment"
                        helperText="Select the trained model to export"
                      />
                    )}
                    renderOption={(props, option) => (
                      <li {...props}>
                        <Box>
                          <Typography variant="body2">{option.name}</Typography>
                          <Typography variant="caption" color="text.secondary">
                            {option.model_type} • Score: {option.score.toFixed(3)}
                          </Typography>
                        </Box>
                      </li>
                    )}
                  />
                </Box>

                {/* Format Selection */}
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Output Format
                  </Typography>
                  <ToggleButtonGroup
                    value={codeFormat}
                    exclusive
                    onChange={handleFormatChange}
                    fullWidth
                    sx={{ mb: 1 }}
                  >
                    <ToggleButton value="python">
                      <Box display="flex" flexDirection="column" alignItems="center" py={1}>
                        <CodeIcon sx={{ mb: 0.5 }} />
                        <Typography variant="caption">Python Script</Typography>
                      </Box>
                    </ToggleButton>
                    <ToggleButton value="notebook">
                      <Box display="flex" flexDirection="column" alignItems="center" py={1}>
                        <DescriptionIcon sx={{ mb: 0.5 }} />
                        <Typography variant="caption">Jupyter Notebook</Typography>
                      </Box>
                    </ToggleButton>
                    <ToggleButton value="fastapi">
                      <Box display="flex" flexDirection="column" alignItems="center" py={1}>
                        <ApiIcon sx={{ mb: 0.5 }} />
                        <Typography variant="caption">FastAPI Service</Typography>
                      </Box>
                    </ToggleButton>
                  </ToggleButtonGroup>
                  <Typography variant="caption" color="text.secondary">
                    {codeFormat === 'python' && 'Standalone Python script with all dependencies'}
                    {codeFormat === 'notebook' && 'Interactive Jupyter notebook with explanations'}
                    {codeFormat === 'fastapi' && 'REST API service with prediction endpoints'}
                  </Typography>
                </Box>

                {/* Options */}
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Include in Generated Code
                  </Typography>
                  <Stack spacing={1}>
                    <Box display="flex" alignItems="center" justifyContent="space-between">
                      <Typography variant="body2">Preprocessing Pipeline</Typography>
                      <Button
                        size="small"
                        variant={includePreprocessing ? 'contained' : 'outlined'}
                        onClick={() => setIncludePreprocessing(!includePreprocessing)}
                      >
                        {includePreprocessing ? 'Included' : 'Excluded'}
                      </Button>
                    </Box>
                    <Box display="flex" alignItems="center" justifyContent="space-between">
                      <Typography variant="body2">Model Evaluation</Typography>
                      <Button
                        size="small"
                        variant={includeEvaluation ? 'contained' : 'outlined'}
                        onClick={() => setIncludeEvaluation(!includeEvaluation)}
                      >
                        {includeEvaluation ? 'Included' : 'Excluded'}
                      </Button>
                    </Box>
                    <Box display="flex" alignItems="center" justifyContent="space-between">
                      <Typography variant="body2">Visualization Code</Typography>
                      <Button
                        size="small"
                        variant={includeVisualization ? 'contained' : 'outlined'}
                        onClick={() => setIncludeVisualization(!includeVisualization)}
                      >
                        {includeVisualization ? 'Included' : 'Excluded'}
                      </Button>
                    </Box>
                  </Stack>
                </Box>

                {/* Generate Button */}
                <Button
                  variant="contained"
                  color="primary"
                  size="large"
                  fullWidth
                  startIcon={<CodeIcon />}
                  onClick={handleGenerate}
                  disabled={!canGenerate}
                >
                  Generate Code
                </Button>

                {!canGenerate && (
                  <Typography variant="caption" color="error" sx={{ mt: 1, display: 'block', textAlign: 'center' }}>
                    Please select both dataset and experiment
                  </Typography>
                )}
              </CardContent>
            </Card>

            {/* Selection Summary */}
            {canGenerate && (
              <>
                <ExperimentSummary
                  experimentName={selectedExperiment.name}
                  datasetName={selectedDataset.name}
                  modelType={selectedExperiment.model_type}
                  score={selectedExperiment.score}
                  createdAt={selectedExperiment.created_at}
                />

                <Box mt={2}>
                  <Button
                    variant="outlined"
                    startIcon={<ShareIcon />}
                    onClick={() => setShareDialogOpen(true)}
                    fullWidth
                  >
                    Share Experiment
                  </Button>
                </Box>
              </>
            )}

            {/* Generation History */}
            {generationHistory.length > 0 && (
              <Box mt={2}>
                <GenerationHistory
                  history={generationHistory}
                  onSelect={handleHistorySelect}
                  onDownload={(item) => console.log('Download:', item)}
                  onDelete={handleHistoryDelete}
                  onClear={handleHistoryClear}
                />
              </Box>
            )}
          </Box>

          {/* Right Column - Preview/Info */}
          <Box sx={{ flex: { xs: '1 1 100%', lg: '1 1 60%' } }}>
            {!canGenerate ? (
              <Paper
                sx={{
                  p: 6,
                  textAlign: 'center',
                  bgcolor: 'background.default',
                  border: '2px dashed',
                  borderColor: 'divider',
                  minHeight: 400,
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                }}
              >
                <CodeIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2, mx: 'auto' }} />
                <Typography variant="h6" gutterBottom>
                  Ready to Generate Code
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  Select a dataset and experiment from the left panel to generate production-ready code.
                  You can choose between Python scripts, Jupyter notebooks, or FastAPI services.
                </Typography>
                <Box display="flex" gap={2} justifyContent="center" flexWrap="wrap">
                  <Chip icon={<CodeIcon />} label="Python Scripts" />
                  <Chip icon={<DescriptionIcon />} label="Jupyter Notebooks" />
                  <Chip icon={<ApiIcon />} label="FastAPI Services" />
                </Box>
              </Paper>
            ) : generatedCode.length > 0 ? (
              <Card>
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                    <Typography variant="h6">
                      Generated Code
                    </Typography>
                    <DownloadOptions
                      onDownloadFile={handleDownloadFile}
                      onDownloadZip={handleDownloadZip}
                      disabled={generatedCode.length === 0}
                    />
                  </Box>
                  <Divider sx={{ mb: 2 }} />

                  <CodePreview files={generatedCode} />

                  <Alert severity="success" sx={{ mt: 2 }}>
                    Code generated successfully! Use the download button above to save your files.
                  </Alert>
                </CardContent>
              </Card>
            ) : (
              <Paper
                sx={{
                  p: 6,
                  textAlign: 'center',
                  bgcolor: 'background.default',
                  border: '2px dashed',
                  borderColor: 'divider',
                  minHeight: 400,
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                }}
              >
                <CodeIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2, mx: 'auto' }} />
                <Typography variant="h6" gutterBottom>
                  Click "Generate Code" to Start
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  Your generated code will appear here. You can then download it as a single file
                  or as a complete project package.
                </Typography>
              </Paper>
            )}
          </Box>
        </Box>

        {/* Help Section */}
        <Paper sx={{ p: 3, mt: 3, bgcolor: 'info.lighter' }}>
          <Typography variant="h6" gutterBottom>
            💡 Code Generation Tips
          </Typography>
          <Stack spacing={2} direction={{ xs: 'column', md: 'row' }}>
            <Box sx={{ flex: 1 }}>
              <Typography variant="subtitle2" gutterBottom>
                Python Scripts
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Best for production deployments and batch processing. Includes all necessary imports
                and can be run directly with Python.
              </Typography>
            </Box>
            <Box sx={{ flex: 1 }}>
              <Typography variant="subtitle2" gutterBottom>
                Jupyter Notebooks
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Ideal for exploration and documentation. Includes markdown cells with explanations
                and visualizations for better understanding.
              </Typography>
            </Box>
            <Box sx={{ flex: 1 }}>
              <Typography variant="subtitle2" gutterBottom>
                FastAPI Services
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Perfect for deploying models as REST APIs. Includes prediction endpoints, request
                validation, and API documentation.
              </Typography>
            </Box>
          </Stack>
        </Paper>

        {/* Share Dialog */}
        {selectedExperiment && (
          <ShareExperiment
            open={shareDialogOpen}
            onClose={() => setShareDialogOpen(false)}
            experimentName={selectedExperiment.name}
            experimentId={selectedExperiment.id}
          />
        )}
      </Container>
    </Box>
  );
};

export default CodeGenerationPage;
