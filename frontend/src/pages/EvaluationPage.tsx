/**
 * EvaluationPage
 *
 * Standalone evaluation screen with tabs for classification, regression, clustering, and comparison
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Container,
  Typography,
  Box,
  Tabs,
  Tab,
  Paper,
  Alert,
  Chip,
  Stack,
  Button,
  CircularProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  SelectChangeEvent,
  TextField,
  InputAdornment,
  Autocomplete,
  IconButton,
  Tooltip,
  Collapse,
  Divider,
} from '@mui/material';
import {
  Assessment as AssessmentIcon,
  BarChart as BarChartIcon,
  ScatterPlot as ScatterPlotIcon,
  BubbleChart as BubbleChartIcon,
  CompareArrows as CompareArrowsIcon,
  Refresh as RefreshIcon,
  ScienceOutlined as ScienceIcon,
  TrendingUp as TrendingUpIcon,
  FilterList as FilterListIcon,
  Search as SearchIcon,
  Clear as ClearIcon,
  Sort as SortIcon,
} from '@mui/icons-material';
import type {
  EvaluationTab,
  ModelRun,
  TaskType,
  EvaluationPageProps,
} from '../types/evaluation';
import { EvaluationTab as EvaluationTabEnum, TaskType as TaskTypeEnum } from '../types/evaluation';
import { MetricsDisplay } from '../components/evaluation/metrics';
import { ClassificationCharts, RegressionCharts, ClusteringCharts } from '../components/evaluation/charts';
import EvaluationDashboard from '../components/evaluation/EvaluationDashboard';
import PlotViewer from '../components/evaluation/PlotViewer';
import { modelService } from '../services/modelService';

// Tab panel component
interface TabPanelProps {
  children?: React.ReactNode;
  value: EvaluationTab;
  currentValue: EvaluationTab;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, currentValue }) => {
  return (
    <Box
      role="tabpanel"
      hidden={value !== currentValue}
      id={`evaluation-tabpanel-${value}`}
      aria-labelledby={`evaluation-tab-${value}`}
    >
      {value === currentValue && <Box sx={{ py: 3 }}>{children}</Box>}
    </Box>
  );
};

const EvaluationPage: React.FC<EvaluationPageProps> = ({
  initialTab = EvaluationTabEnum.CLASSIFICATION,
  runId,
}) => {
  const [currentTab, setCurrentTab] = useState<EvaluationTab>(initialTab);
  const [runs, setRuns] = useState<ModelRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | undefined>(runId);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Filter states
  const [showFilters, setShowFilters] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [modelTypeFilter, setModelTypeFilter] = useState<string>('all');
  const [sortBy, setSortBy] = useState<string>('date-desc');

  // Load runs from sessionStorage or API
  useEffect(() => {
    loadRuns();
  }, []);

  // Update tab when task type changes
  useEffect(() => {
    if (selectedRunId) {
      const selectedRun = runs.find((r) => r.id === selectedRunId);
      if (selectedRun) {
        // Auto-switch to appropriate tab based on task type
        const tabMap: Record<TaskType, EvaluationTab> = {
          [TaskTypeEnum.CLASSIFICATION]: EvaluationTabEnum.CLASSIFICATION,
          [TaskTypeEnum.REGRESSION]: EvaluationTabEnum.REGRESSION,
          [TaskTypeEnum.CLUSTERING]: EvaluationTabEnum.CLUSTERING,
        };
        const newTab = tabMap[selectedRun.taskType];
        if (newTab && newTab !== currentTab) {
          setCurrentTab(newTab);
        }
      }
    }
  }, [selectedRunId, runs]);

  const loadRuns = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Fetch from API
      const response = await modelService.listModels();
      const apiRuns: ModelRun[] = (Array.isArray(response) ? response : (response as any).items || []).map((model: any) => ({
        id: model.id,
        name: model.name || `Model ${model.id}`,
        taskType: (model.task_type || 'classification') as TaskType,
        modelType: model.model_type || 'unknown',
        createdAt: model.created_at,
        status: model.status,
        metrics: model.metrics || {},
      }));
      setRuns(apiRuns);

      // If runId was provided and not in selectedRunId, set it
      if (runId && !selectedRunId) {
        setSelectedRunId(runId);
      }
    } catch (err) {
      setError('Failed to load model runs. Please try again.');
      console.error('Error loading runs:', err);
      
      // Fallback to mock data
      setRuns(getMockRuns());
    } finally {
      setIsLoading(false);
    }
  };

  const handleTabChange = (_event: React.SyntheticEvent, newValue: EvaluationTab) => {
    setCurrentTab(newValue);
  };

  const handleRefresh = () => {
    loadRuns();
  };

  const handleClearFilters = () => {
    setSearchQuery('');
    setStatusFilter('all');
    setModelTypeFilter('all');
    setSortBy('date-desc');
  };

  // Filter and sort runs
  const filteredRuns = useMemo(() => {
    let filtered = [...runs];

    // Search filter
    if (searchQuery) {
      filtered = filtered.filter(
        (run) =>
          run.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          run.modelType?.toLowerCase().includes(searchQuery.toLowerCase()) ||
          run.id.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    // Status filter
    if (statusFilter !== 'all') {
      filtered = filtered.filter((run) => run.status === statusFilter);
    }

    // Model type filter
    if (modelTypeFilter !== 'all') {
      filtered = filtered.filter((run) => run.modelType === modelTypeFilter);
    }

    // Sort
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'date-desc':
          return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
        case 'date-asc':
          return new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
        case 'name-asc':
          return a.name.localeCompare(b.name);
        case 'name-desc':
          return b.name.localeCompare(a.name);
        default:
          return 0;
      }
    });

    return filtered;
  }, [runs, searchQuery, statusFilter, modelTypeFilter, sortBy]);

  // Get unique model types and statuses for filters
  const uniqueModelTypes = useMemo(() => {
    return Array.from(new Set(runs.map((r) => r.modelType).filter(Boolean)));
  }, [runs]);

  const uniqueStatuses = useMemo(() => {
    return Array.from(new Set(runs.map((r) => r.status).filter(Boolean)));
  }, [runs]);

  // Filter runs by task type for each tab
  const classificationRuns = filteredRuns.filter((r) => r.taskType === TaskTypeEnum.CLASSIFICATION);
  const regressionRuns = filteredRuns.filter((r) => r.taskType === TaskTypeEnum.REGRESSION);
  const clusteringRuns = filteredRuns.filter((r) => r.taskType === TaskTypeEnum.CLUSTERING);

  // Get icon for tab
  const getTabIcon = (tab: EvaluationTab) => {
    switch (tab) {
      case EvaluationTabEnum.CLASSIFICATION:
        return <BarChartIcon />;
      case EvaluationTabEnum.REGRESSION:
        return <ScatterPlotIcon />;
      case EvaluationTabEnum.CLUSTERING:
        return <BubbleChartIcon />;
      case EvaluationTabEnum.COMPARISON:
        return <CompareArrowsIcon />;
      default:
        return <AssessmentIcon />;
    }
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box mb={4} display="flex" justifyContent="space-between" alignItems="center">
        <Box>
          <Typography variant="h4" fontWeight="bold" gutterBottom>
            Model Evaluation
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Analyze and compare your trained models across different tasks
          </Typography>
        </Box>

        <Box display="flex" gap={1}>
          <Tooltip title={showFilters ? 'Hide Filters' : 'Show Filters'}>
            <IconButton
              onClick={() => setShowFilters(!showFilters)}
              color={showFilters ? 'primary' : 'default'}
            >
              <FilterListIcon />
            </IconButton>
          </Tooltip>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={handleRefresh}
            disabled={isLoading}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {/* Filters Panel */}
      <Collapse in={showFilters}>
        <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Typography variant="h6" display="flex" alignItems="center" gap={1}>
              <FilterListIcon />
              Filters & Search
            </Typography>
            <Button
              size="small"
              startIcon={<ClearIcon />}
              onClick={handleClearFilters}
            >
              Clear All
            </Button>
          </Box>

          <Box display="flex" flexWrap="wrap" gap={2}>
            {/* Search */}
            <Box sx={{ flex: '1 1 300px', minWidth: '250px' }}>
              <TextField
                fullWidth
                size="small"
                placeholder="Search models by name, type, or ID..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon fontSize="small" />
                    </InputAdornment>
                  ),
                  endAdornment: searchQuery && (
                    <InputAdornment position="end">
                      <IconButton size="small" onClick={() => setSearchQuery('')}>
                        <ClearIcon fontSize="small" />
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
              />
            </Box>

            {/* Model Type Filter */}
            <FormControl size="small" sx={{ minWidth: 180 }}>
              <InputLabel>Model Type</InputLabel>
              <Select
                value={modelTypeFilter}
                onChange={(e) => setModelTypeFilter(e.target.value)}
                label="Model Type"
              >
                <MenuItem value="all">All Types</MenuItem>
                {uniqueModelTypes.map((type) => (
                  <MenuItem key={type} value={type}>
                    {type}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Status Filter */}
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Status</InputLabel>
              <Select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                label="Status"
              >
                <MenuItem value="all">All Status</MenuItem>
                {uniqueStatuses.map((status) => (
                  <MenuItem key={status} value={status}>
                    <Chip
                      label={status}
                      size="small"
                      color={status === 'completed' ? 'success' : status === 'failed' ? 'error' : 'default'}
                    />
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Sort By */}
            <FormControl size="small" sx={{ minWidth: 180 }}>
              <InputLabel>Sort By</InputLabel>
              <Select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                label="Sort By"
                startAdornment={
                  <InputAdornment position="start">
                    <SortIcon fontSize="small" />
                  </InputAdornment>
                }
              >
                <MenuItem value="date-desc">Newest First</MenuItem>
                <MenuItem value="date-asc">Oldest First</MenuItem>
                <MenuItem value="name-asc">Name (A-Z)</MenuItem>
                <MenuItem value="name-desc">Name (Z-A)</MenuItem>
              </Select>
            </FormControl>
          </Box>

          {/* Active Filters Summary */}
          {(searchQuery || statusFilter !== 'all' || modelTypeFilter !== 'all') && (
            <Box mt={2} display="flex" alignItems="center" gap={1} flexWrap="wrap">
              <Typography variant="caption" color="text.secondary">
                Active filters:
              </Typography>
              {searchQuery && (
                <Chip
                  size="small"
                  label={`Search: "${searchQuery}"`}
                  onDelete={() => setSearchQuery('')}
                />
              )}
              {statusFilter !== 'all' && (
                <Chip
                  size="small"
                  label={`Status: ${statusFilter}`}
                  onDelete={() => setStatusFilter('all')}
                />
              )}
              {modelTypeFilter !== 'all' && (
                <Chip
                  size="small"
                  label={`Type: ${modelTypeFilter}`}
                  onDelete={() => setModelTypeFilter('all')}
                />
              )}
              <Typography variant="caption" color="primary" fontWeight="bold">
                ({filteredRuns.length} of {runs.length} models)
              </Typography>
            </Box>
          )}
        </Paper>
      </Collapse>

      {/* Error Alert */}
      {error && (
        <Alert 
          severity="error" 
          sx={{ mb: 3 }} 
          onClose={() => setError(null)}
          action={
            <Button color="inherit" size="small" onClick={handleRefresh}>
              Retry
            </Button>
          }
        >
          {error}
        </Alert>
      )}

      {/* Results Summary Banner */}
      {!isLoading && filteredRuns.length > 0 && (
        <Paper elevation={0} sx={{ p: 2, mb: 3, bgcolor: 'primary.50', borderLeft: 4, borderColor: 'primary.main' }}>
          <Box display="flex" alignItems="center" justifyContent="space-between" flexWrap="wrap" gap={2}>
            <Box display="flex" alignItems="center" gap={3}>
              <Box>
                <Typography variant="h4" fontWeight="bold" color="primary">
                  {filteredRuns.length}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Total Models
                </Typography>
              </Box>
              <Divider orientation="vertical" flexItem />
              <Box>
                <Typography variant="h6" fontWeight="bold" color="success.main">
                  {classificationRuns.length}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Classification
                </Typography>
              </Box>
              <Box>
                <Typography variant="h6" fontWeight="bold" color="info.main">
                  {regressionRuns.length}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Regression
                </Typography>
              </Box>
              <Box>
                <Typography variant="h6" fontWeight="bold" color="warning.main">
                  {clusteringRuns.length}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Clustering
                </Typography>
              </Box>
            </Box>
            
            {filteredRuns.length !== runs.length && (
              <Chip
                label={`Filtered: ${filteredRuns.length} of ${runs.length}`}
                color="primary"
                size="small"
                variant="outlined"
              />
            )}
          </Box>
        </Paper>
      )}

      {/* Loading State */}
      {isLoading && (
        <Box display="flex" justifyContent="center" alignItems="center" py={8}>
          <CircularProgress />
          <Typography variant="body1" color="text.secondary" sx={{ ml: 2 }}>
            Loading model runs...
          </Typography>
        </Box>
      )}

      {/* Main Content */}
      {!isLoading && (
        <Paper elevation={2}>
          {/* Tab Navigation */}
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs
              value={currentTab}
              onChange={handleTabChange}
              aria-label="evaluation tabs"
              variant="fullWidth"
              sx={{ px: 2 }}
            >
              <Tab
                value={EvaluationTabEnum.CLASSIFICATION}
                label={
                  <Stack direction="row" spacing={1} alignItems="center">
                    {getTabIcon(EvaluationTabEnum.CLASSIFICATION)}
                    <span>Classification</span>
                    {classificationRuns.length > 0 && (
                      <Chip label={classificationRuns.length} size="small" color="primary" />
                    )}
                  </Stack>
                }
                id="evaluation-tab-classification"
                aria-controls="evaluation-tabpanel-classification"
              />
              <Tab
                value={EvaluationTabEnum.REGRESSION}
                label={
                  <Stack direction="row" spacing={1} alignItems="center">
                    {getTabIcon(EvaluationTabEnum.REGRESSION)}
                    <span>Regression</span>
                    {regressionRuns.length > 0 && (
                      <Chip label={regressionRuns.length} size="small" color="primary" />
                    )}
                  </Stack>
                }
                id="evaluation-tab-regression"
                aria-controls="evaluation-tabpanel-regression"
              />
              <Tab
                value={EvaluationTabEnum.CLUSTERING}
                label={
                  <Stack direction="row" spacing={1} alignItems="center">
                    {getTabIcon(EvaluationTabEnum.CLUSTERING)}
                    <span>Clustering</span>
                    {clusteringRuns.length > 0 && (
                      <Chip label={clusteringRuns.length} size="small" color="primary" />
                    )}
                  </Stack>
                }
                id="evaluation-tab-clustering"
                aria-controls="evaluation-tabpanel-clustering"
              />
              <Tab
                value={EvaluationTabEnum.COMPARISON}
                label={
                  <Stack direction="row" spacing={1} alignItems="center">
                    {getTabIcon(EvaluationTabEnum.COMPARISON)}
                    <span>Comparison</span>
                  </Stack>
                }
                id="evaluation-tab-comparison"
                aria-controls="evaluation-tabpanel-comparison"
              />
            </Tabs>
          </Box>

          {/* Tab Panels */}
          <Box sx={{ p: 3 }}>
            {/* Model Run Selector */}
            {selectedRunId && filteredRuns.length > 0 && (
              <Box mb={3}>
                <Autocomplete
                  value={filteredRuns.find((run) => run.id === selectedRunId) || null}
                  onChange={(_event, newValue) => {
                    setSelectedRunId(newValue?.id);
                  }}
                  options={filteredRuns}
                  getOptionLabel={(option) => option.name}
                  renderInput={(params) => (
                    <TextField
                      {...params}
                      label="Select Model Run"
                      placeholder="Choose a model to evaluate"
                    />
                  )}
                  renderOption={(props, option) => (
                    <li {...props}>
                      <Box display="flex" flexDirection="column" width="100%">
                        <Box display="flex" alignItems="center" justifyContent="space-between">
                          <Typography variant="body2" fontWeight="medium">
                            {option.name}
                          </Typography>
                          <Chip
                            label={option.status}
                            size="small"
                            color={option.status === 'completed' ? 'success' : 'default'}
                          />
                        </Box>
                        <Typography variant="caption" color="text.secondary">
                          {option.modelType} • {option.taskType} • {new Date(option.createdAt).toLocaleDateString()}
                        </Typography>
                      </Box>
                    </li>
                  )}
                  fullWidth
                  size="small"
                />
              </Box>
            )}

            {/* Classification Tab */}
            <TabPanel value={EvaluationTabEnum.CLASSIFICATION} currentValue={currentTab}>
              {classificationRuns.length === 0 ? (
                <Box display="flex" flexDirection="column" alignItems="center" py={8}>
                  <ScienceIcon sx={{ fontSize: 80, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h5" gutterBottom>
                    No Classification Models Yet
                  </Typography>
                  <Typography variant="body1" color="text.secondary" textAlign="center" mb={3}>
                    Train a classification model to see evaluation metrics, confusion matrices,
                    <br />
                    ROC curves, and performance insights here.
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={<TrendingUpIcon />}
                    onClick={() => window.location.href = '/training'}
                  >
                    Start Training
                  </Button>
                </Box>
              ) : selectedRunId ? (
                <>
                  <EvaluationDashboard
                    modelRunId={selectedRunId}
                    onRefresh={handleRefresh}
                  />
                  <Box mt={4}>
                    <PlotViewer
                      modelRunId={selectedRunId}
                      availablePlots={['roc_curve', 'confusion_matrix', 'precision_recall_curve', 'learning_curve']}
                    />
                  </Box>
                </>
              ) : (
                <Box>
                  {/* Metrics Display */}
                  <MetricsDisplay
                    taskType={TaskTypeEnum.CLASSIFICATION}
                    metrics={classificationRuns[0]?.metrics || null}
                    isLoading={isLoading}
                    showDescription={true}
                    compact={false}
                  />

                  {/* Classification Charts */}
                  {classificationRuns[0]?.metrics && (
                    <Box mt={4}>
                      <ClassificationCharts
                        metrics={classificationRuns[0].metrics as any}
                        isLoading={isLoading}
                      />
                    </Box>
                  )}
                </Box>
              )}
            </TabPanel>

            {/* Regression Tab */}
            <TabPanel value={EvaluationTabEnum.REGRESSION} currentValue={currentTab}>
              {regressionRuns.length === 0 ? (
                <Box display="flex" flexDirection="column" alignItems="center" py={8}>
                  <ScatterPlotIcon sx={{ fontSize: 80, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h5" gutterBottom>
                    No Regression Models Yet
                  </Typography>
                  <Typography variant="body1" color="text.secondary" textAlign="center" mb={3}>
                    Train a regression model to see evaluation metrics, residual plots,
                    <br />
                    and prediction accuracy analysis here.
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={<TrendingUpIcon />}
                    onClick={() => window.location.href = '/training'}
                  >
                    Start Training
                  </Button>
                </Box>
              ) : selectedRunId ? (
                <>
                  <EvaluationDashboard
                    modelRunId={selectedRunId}
                    onRefresh={handleRefresh}
                  />
                  <Box mt={4}>
                    <PlotViewer
                      modelRunId={selectedRunId}
                      availablePlots={['residuals', 'learning_curve']}
                    />
                  </Box>
                </>
              ) : (
                <Box>
                  {/* Metrics Display */}
                  <MetricsDisplay
                    taskType={TaskTypeEnum.REGRESSION}
                    metrics={regressionRuns[0]?.metrics || null}
                    isLoading={isLoading}
                    showDescription={true}
                    compact={false}
                  />

                  {/* Regression Charts */}
                  {regressionRuns[0]?.metrics && (
                    <Box mt={4}>
                      <RegressionCharts
                        metrics={regressionRuns[0].metrics as any}
                        isLoading={isLoading}
                      />
                    </Box>
                  )}
                </Box>
              )}
            </TabPanel>

            {/* Clustering Tab */}
            <TabPanel value={EvaluationTabEnum.CLUSTERING} currentValue={currentTab}>
              {clusteringRuns.length === 0 ? (
                <Box display="flex" flexDirection="column" alignItems="center" py={8}>
                  <BubbleChartIcon sx={{ fontSize: 80, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h5" gutterBottom>
                    No Clustering Models Yet
                  </Typography>
                  <Typography variant="body1" color="text.secondary" textAlign="center" mb={3}>
                    Train a clustering model to see silhouette scores, cluster distributions,
                    <br />
                    and pattern analysis here.
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={<TrendingUpIcon />}
                    onClick={() => window.location.href = '/training'}
                  >
                    Start Training
                  </Button>
                </Box>
              ) : selectedRunId ? (
                <>
                  <EvaluationDashboard
                    modelRunId={selectedRunId}
                    onRefresh={handleRefresh}
                  />
                  <Box mt={4}>
                    <PlotViewer
                      modelRunId={selectedRunId}
                      availablePlots={['feature_importance', 'learning_curve']}
                    />
                  </Box>
                </>
              ) : (
                <Box>
                  {/* Metrics Display */}
                  <MetricsDisplay
                    taskType={TaskTypeEnum.CLUSTERING}
                    metrics={clusteringRuns[0]?.metrics || null}
                    isLoading={isLoading}
                    showDescription={true}
                    compact={false}
                  />

                  {/* Clustering Charts */}
                  {clusteringRuns[0]?.metrics && (
                    <Box mt={4}>
                      <ClusteringCharts
                        metrics={clusteringRuns[0].metrics as any}
                        isLoading={isLoading}
                        showSilhouette={true}
                        showInertia={true}
                        showDistribution={true}
                        showProjection={false}
                      />
                    </Box>
                  )}
                </Box>
              )}
            </TabPanel>

            {/* Comparison Tab */}
            <TabPanel value={EvaluationTabEnum.COMPARISON} currentValue={currentTab}>
              {filteredRuns.length === 0 ? (
                <Box display="flex" flexDirection="column" alignItems="center" py={8}>
                  <CompareArrowsIcon sx={{ fontSize: 80, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h5" gutterBottom>
                    No Models to Compare
                  </Typography>
                  <Typography variant="body1" color="text.secondary" textAlign="center" mb={3}>
                    Train at least two models to compare their performance side-by-side,
                    <br />
                    analyze metrics, and identify the best model for your task.
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={<TrendingUpIcon />}
                    onClick={() => window.location.href = '/training'}
                  >
                    Start Training
                  </Button>
                </Box>
              ) : filteredRuns.length < 2 ? (
                <Box display="flex" flexDirection="column" alignItems="center" py={8}>
                  <CompareArrowsIcon sx={{ fontSize: 80, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h5" gutterBottom>
                    Need More Models
                  </Typography>
                  <Typography variant="body1" color="text.secondary" textAlign="center" mb={3}>
                    You have {filteredRuns.length} model. Train at least one more model
                    <br />
                    to enable model comparison features.
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={<TrendingUpIcon />}
                    onClick={() => window.location.href = '/training'}
                  >
                    Train Another Model
                  </Button>
                </Box>
              ) : (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Model Comparison
                  </Typography>
                  <Alert severity="info" sx={{ mt: 2 }}>
                    <Typography variant="body2">
                      Compare multiple models side-by-side to identify the best performing model for your
                      task. Select models from the dropdown above to start comparing.
                    </Typography>
                  </Alert>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                    {filteredRuns.length} model run(s) available for comparison across all task types.
                  </Typography>
                </Box>
              )}
            </TabPanel>
          </Box>
        </Paper>
      )}

      {/* Debug Info (Development only) */}
      {import.meta.env.DEV && runs.length > 0 && (
        <Paper sx={{ mt: 4, p: 3, bgcolor: 'action.hover' }}>
          <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
            Debug: Model Runs State
          </Typography>
          <Box
            component="pre"
            sx={{
              fontSize: '0.75rem',
              overflow: 'auto',
              maxHeight: 300,
            }}
          >
            {JSON.stringify({ currentTab, selectedRunId, totalRuns: runs.length, runs }, null, 2)}
          </Box>
        </Paper>
      )}
    </Container>
  );
};

// Mock data for development
const getMockRuns = (): ModelRun[] => {
  return [
    {
      id: 'run-1',
      name: 'Random Forest Classifier - Iris',
      modelId: 'random_forest_classifier',
      modelName: 'Random Forest Classifier',
      taskType: 'classification' as TaskType,
      datasetId: 'dataset-1',
      datasetName: 'Iris Dataset',
      status: 'completed',
      createdAt: '2025-12-27T10:00:00Z',
      completedAt: '2025-12-27T10:05:00Z',
      hyperparameters: {
        n_estimators: 100,
        max_depth: 10,
        min_samples_split: 2,
      },
      metrics: {
        accuracy: 0.96,
        precision: 0.95,
        recall: 0.96,
        f1Score: 0.95,
        auc: 0.98,
        confusionMatrix: [
          [45, 3, 2],
          [1, 48, 1],
          [2, 1, 47],
        ],
        classNames: ['Setosa', 'Versicolor', 'Virginica'],
        rocCurve: {
          fpr: Array.from({ length: 50 }, (_, i) => i / 49),
          tpr: Array.from({ length: 50 }, (_, i) => Math.min(1, (i / 49) * 1.1 + Math.random() * 0.05)),
          thresholds: Array.from({ length: 50 }, (_, i) => 1 - i / 49),
        },
        prCurve: {
          precision: Array.from({ length: 50 }, (_, i) => Math.max(0.8, 1 - (i / 49) * 0.2 + Math.random() * 0.05)),
          recall: Array.from({ length: 50 }, (_, i) => 1 - i / 49),
          thresholds: Array.from({ length: 50 }, (_, i) => 1 - i / 49),
        },
      },
    },
    {
      id: 'run-2',
      name: 'Linear Regression - Housing',
      modelId: 'linear_regression',
      modelName: 'Linear Regression',
      taskType: 'regression' as TaskType,
      datasetId: 'dataset-2',
      datasetName: 'Housing Prices',
      status: 'completed',
      createdAt: '2025-12-27T11:00:00Z',
      completedAt: '2025-12-27T11:03:00Z',
      hyperparameters: {
        fit_intercept: true,
        normalize: false,
      },
      metrics: {
        mae: 3421.5,
        mse: 15678234.2,
        rmse: 3959.5,
        r2: 0.82,
        mape: 12.3,
        // Generate realistic predicted and actual values for housing prices
        predicted: Array.from({ length: 100 }, (_, i) => {
          const base = 200000 + i * 5000;
          const noise = (Math.random() - 0.5) * 10000;
          return base + noise;
        }),
        actual: Array.from({ length: 100 }, (_, i) => {
          const base = 200000 + i * 5000;
          const noise = (Math.random() - 0.5) * 8000;
          const trend = i * 100; // Slight upward trend
          return base + noise + trend;
        }),
      },
    },
    {
      id: 'run-3',
      name: 'K-Means - Customer Segments',
      modelId: 'kmeans',
      modelName: 'K-Means Clustering',
      taskType: 'clustering' as TaskType,
      datasetId: 'dataset-3',
      datasetName: 'Customer Data',
      status: 'completed',
      createdAt: '2025-12-27T12:00:00Z',
      completedAt: '2025-12-27T12:04:00Z',
      hyperparameters: {
        n_clusters: 5,
        max_iter: 300,
        n_init: 10,
      },
      metrics: {
        silhouetteScore: 0.68,
        inertia: 1234.5,
        nClusters: 5,
      },
    },
  ];
};

export default EvaluationPage;
