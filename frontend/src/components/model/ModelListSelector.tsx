import React, { useState, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Checkbox,
  FormControlLabel,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Button,
  IconButton,
  Alert,
  Collapse,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
} from '@mui/material';
import {
  FilterList as FilterListIcon,
  Clear as ClearIcon,
  Search as SearchIcon,
  Sort as SortIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Compare as CompareIcon,
} from '@mui/icons-material';
import type { ModelComparisonData, ModelComparisonFilter, ModelComparisonSort } from '../../types/modelComparison';

interface ModelListSelectorProps {
  models: ModelComparisonData[];
  selectedModels: string[];
  onSelectionChange: (selectedIds: string[]) => void;
  maxSelection?: number;
  loading?: boolean;
}

const ModelListSelector: React.FC<ModelListSelectorProps> = ({
  models,
  selectedModels,
  onSelectionChange,
  maxSelection = 4,
  loading = false,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState<ModelComparisonFilter>({
    modelType: '',
    status: '',
    datasetId: '',
  });
  const [sort, setSort] = useState<ModelComparisonSort>({
    field: 'createdAt',
    direction: 'desc',
  });

  // Get unique values for filter options
  const filterOptions = useMemo(() => {
    const modelTypes = [...new Set(models.map(m => m.type))];
    const statuses = [...new Set(models.map(m => m.status))];
    const datasets = [...new Set(models.map(m => ({ id: m.datasetId, name: m.datasetName })))];
    
    return {
      modelTypes,
      statuses,
      datasets,
    };
  }, [models]);

  // Filter and sort models
  const filteredAndSortedModels = useMemo(() => {
    let filtered = models.filter(model => {
      // Search filter
      if (searchTerm && !model.name.toLowerCase().includes(searchTerm.toLowerCase())) {
        return false;
      }
      
      // Type filter
      if (filters.modelType && model.type !== filters.modelType) {
        return false;
      }
      
      // Status filter
      if (filters.status && model.status !== filters.status) {
        return false;
      }
      
      // Dataset filter
      if (filters.datasetId && model.datasetId !== filters.datasetId) {
        return false;
      }
      
      // Accuracy filter
      if (filters.minAccuracy && (model.metrics.accuracy || 0) < filters.minAccuracy) {
        return false;
      }
      
      // Training time filter
      if (filters.maxTrainingTime && (model.trainingTime || 0) > filters.maxTrainingTime) {
        return false;
      }
      
      return true;
    });

    // Sort
    filtered.sort((a, b) => {
      let aValue: any;
      let bValue: any;
      
      if (sort.field.includes('.')) {
        // Handle nested fields like 'metrics.accuracy'
        const [parent, child] = sort.field.split('.');
        aValue = (a as any)[parent]?.[child];
        bValue = (b as any)[parent]?.[child];
      } else {
        aValue = (a as any)[sort.field];
        bValue = (b as any)[sort.field];
      }
      
      // Handle undefined values
      if (aValue === undefined && bValue === undefined) return 0;
      if (aValue === undefined) return 1;
      if (bValue === undefined) return -1;
      
      // Compare values
      if (typeof aValue === 'string') {
        const comparison = aValue.localeCompare(bValue);
        return sort.direction === 'asc' ? comparison : -comparison;
      } else {
        const comparison = aValue - bValue;
        return sort.direction === 'asc' ? comparison : -comparison;
      }
    });

    return filtered;
  }, [models, searchTerm, filters, sort]);

  const handleModelToggle = (modelId: string) => {
    const isSelected = selectedModels.includes(modelId);
    
    if (isSelected) {
      // Remove from selection
      onSelectionChange(selectedModels.filter(id => id !== modelId));
    } else {
      // Add to selection (if under limit)
      if (selectedModels.length < maxSelection) {
        onSelectionChange([...selectedModels, modelId]);
      }
    }
  };

  const handleSelectAll = () => {
    const visibleModelIds = filteredAndSortedModels.slice(0, maxSelection).map(m => m.id);
    onSelectionChange(visibleModelIds);
  };

  const handleClearSelection = () => {
    onSelectionChange([]);
  };

  const handleFilterChange = (field: keyof ModelComparisonFilter, value: any) => {
    setFilters(prev => ({ ...prev, [field]: value }));
  };

  const handleSortChange = (field: ModelComparisonSort['field']) => {
    setSort(prev => ({
      field,
      direction: prev.field === field && prev.direction === 'asc' ? 'desc' : 'asc',
    }));
  };

  const clearFilters = () => {
    setFilters({
      modelType: '',
      status: '',
      datasetId: '',
    });
    setSearchTerm('');
  };

  const formatMetricValue = (value: number | undefined, isPercentage = false) => {
    if (value === undefined) return 'N/A';
    if (isPercentage) {
      return `${(value * 100).toFixed(1)}%`;
    }
    return value.toFixed(3);
  };

  const getModelTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      'random_forest': 'primary',
      'logistic_regression': 'secondary',
      'svm': 'success',
      'neural_network': 'warning',
      'gradient_boosting': 'info',
      'decision_tree': 'error',
    };
    return colors[type] || 'default';
  };

  return (
    <Card>
      <CardContent>
        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">
            Select Models to Compare ({selectedModels.length}/{maxSelection})
          </Typography>
          <Box display="flex" gap={1}>
            <Button
              size="small"
              onClick={handleSelectAll}
              disabled={filteredAndSortedModels.length === 0}
            >
              Select All
            </Button>
            <Button
              size="small"
              onClick={handleClearSelection}
              disabled={selectedModels.length === 0}
            >
              Clear All
            </Button>
            <IconButton
              size="small"
              onClick={() => setShowFilters(!showFilters)}
              color={showFilters ? 'primary' : 'default'}
            >
              <FilterListIcon />
            </IconButton>
          </Box>
        </Box>

        {/* Search */}
        <TextField
          fullWidth
          size="small"
          placeholder="Search models..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          InputProps={{
            startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
            endAdornment: searchTerm && (
              <IconButton size="small" onClick={() => setSearchTerm('')}>
                <ClearIcon />
              </IconButton>
            ),
          }}
          sx={{ mb: 2 }}
        />

        {/* Filters */}
        <Collapse in={showFilters}>
          <Paper elevation={0} sx={{ p: 2, mb: 2, bgcolor: 'action.hover' }}>
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: {
                  xs: '1fr',
                  sm: 'repeat(2, 1fr)',
                  md: 'repeat(4, 1fr)',
                },
                gap: 2,
              }}
            >
              <FormControl fullWidth size="small">
                <InputLabel>Model Type</InputLabel>
                <Select
                  value={filters.modelType || ''}
                  onChange={(e) => handleFilterChange('modelType', e.target.value)}
                  label="Model Type"
                >
                  <MenuItem value="">All Types</MenuItem>
                  {filterOptions.modelTypes.map(type => (
                    <MenuItem key={type} value={type}>
                      {type.replace('_', ' ').toUpperCase()}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              <FormControl fullWidth size="small">
                <InputLabel>Status</InputLabel>
                <Select
                  value={filters.status || ''}
                  onChange={(e) => handleFilterChange('status', e.target.value)}
                  label="Status"
                >
                  <MenuItem value="">All Statuses</MenuItem>
                  {filterOptions.statuses.map(status => (
                    <MenuItem key={status} value={status}>
                      {status.charAt(0).toUpperCase() + status.slice(1)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              <FormControl fullWidth size="small">
                <InputLabel>Dataset</InputLabel>
                <Select
                  value={filters.datasetId || ''}
                  onChange={(e) => handleFilterChange('datasetId', e.target.value)}
                  label="Dataset"
                >
                  <MenuItem value="">All Datasets</MenuItem>
                  {filterOptions.datasets.map(dataset => (
                    <MenuItem key={dataset.id} value={dataset.id}>
                      {dataset.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              <Box display="flex" gap={1}>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={clearFilters}
                  startIcon={<ClearIcon />}
                >
                  Clear Filters
                </Button>
              </Box>
            </Box>
          </Paper>
        </Collapse>

        {/* Sort Options */}
        <Box display="flex" gap={1} mb={2} flexWrap="wrap">
          <Typography variant="body2" color="text.secondary" sx={{ alignSelf: 'center' }}>
            Sort by:
          </Typography>
          {[
            { field: 'createdAt' as const, label: 'Date' },
            { field: 'name' as const, label: 'Name' },
            { field: 'metrics.accuracy' as const, label: 'Accuracy' },
            { field: 'trainingTime' as const, label: 'Training Time' },
          ].map(({ field, label }) => (
            <Button
              key={field}
              size="small"
              variant={sort.field === field ? 'contained' : 'outlined'}
              onClick={() => handleSortChange(field)}
              endIcon={sort.field === field ? (
                sort.direction === 'asc' ? <ExpandLessIcon /> : <ExpandMoreIcon />
              ) : <SortIcon />}
            >
              {label}
            </Button>
          ))}
        </Box>

        {/* Selection Limit Warning */}
        {selectedModels.length >= maxSelection && (
          <Alert severity="info" sx={{ mb: 2 }}>
            Maximum {maxSelection} models can be selected for comparison.
          </Alert>
        )}

        {/* Model List */}
        {loading ? (
          <Typography>Loading models...</Typography>
        ) : filteredAndSortedModels.length === 0 ? (
          <Alert severity="info">
            No models found matching the current filters.
          </Alert>
        ) : (
          <List>
            {filteredAndSortedModels.map((model, index) => {
              const isSelected = selectedModels.includes(model.id);
              const isDisabled = !isSelected && selectedModels.length >= maxSelection;
              
              return (
                <ListItem
                  key={model.id}
                  divider={index < filteredAndSortedModels.length - 1}
                  sx={{
                    bgcolor: isSelected ? 'action.selected' : 'transparent',
                    opacity: isDisabled ? 0.6 : 1,
                  }}
                >
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={isSelected}
                        onChange={() => handleModelToggle(model.id)}
                        disabled={isDisabled}
                      />
                    }
                    label=""
                    sx={{ mr: 1 }}
                  />
                  
                  <ListItemText
                    primary={
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography variant="subtitle1">{model.name}</Typography>
                        <Chip
                          label={model.type}
                          color={getModelTypeColor(model.type) as any}
                          size="small"
                        />
                        <Chip
                          label={model.status}
                          color={model.status === 'completed' ? 'success' : 'default'}
                          size="small"
                        />
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Dataset: {model.datasetName}
                        </Typography>
                        <Box display="flex" gap={2} mt={0.5}>
                          {model.metrics.accuracy && (
                            <Typography variant="body2">
                              Accuracy: {formatMetricValue(model.metrics.accuracy, true)}
                            </Typography>
                          )}
                          {model.metrics.r2Score && (
                            <Typography variant="body2">
                              R²: {formatMetricValue(model.metrics.r2Score)}
                            </Typography>
                          )}
                          {model.trainingTime && (
                            <Typography variant="body2">
                              Training: {Math.round(model.trainingTime)}s
                            </Typography>
                          )}
                        </Box>
                      </Box>
                    }
                  />
                  
                  <ListItemSecondaryAction>
                    <Typography variant="body2" color="text.secondary">
                      {new Date(model.createdAt).toLocaleDateString()}
                    </Typography>
                  </ListItemSecondaryAction>
                </ListItem>
              );
            })}
          </List>
        )}

        {/* Results Summary */}
        <Box mt={2} display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="body2" color="text.secondary">
            Showing {filteredAndSortedModels.length} of {models.length} models
          </Typography>
          
          {selectedModels.length > 0 && (
            <Box display="flex" alignItems="center" gap={1}>
              <CompareIcon color="primary" />
              <Typography variant="body2" color="primary">
                {selectedModels.length} selected for comparison
              </Typography>
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default ModelListSelector;
