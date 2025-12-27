/**
 * FeatureSelector Component
 *
 * Allows users to select input features for model training with search,
 * filtering, and bulk selection capabilities.
 */

import React, { useState, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Checkbox,
  FormControlLabel,
  TextField,
  InputAdornment,
  Chip,
  Stack,
  Button,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Tooltip,
  IconButton,
  Collapse,
} from '@mui/material';
import {
  Search as SearchIcon,
  Clear as ClearIcon,
  SelectAll as SelectAllIcon,
  Deselect as DeselectIcon,
  FilterList as FilterIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import type {
  FeatureSelectorProps,
} from '../../types/featureSelection';
import {
  ColumnDataType,
  getColumnDataType,
} from '../../types/featureSelection';

const FeatureSelector: React.FC<FeatureSelectorProps> = ({
  datasetId,
  columns,
  selectedFeatures,
  excludedColumns = [],
  onChange,
  maxFeatures,
  disabled = false,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<ColumnDataType | 'all'>('all');
  const [showFilters, setShowFilters] = useState(false);

  // Filter available columns
  const availableColumns = useMemo(() => {
    return columns.filter((col) => !excludedColumns.includes(col.name));
  }, [columns, excludedColumns]);

  // Filtered and searched columns
  const filteredColumns = useMemo(() => {
    let result = availableColumns;

    // Apply type filter
    if (filterType !== 'all') {
      result = result.filter((col) => getColumnDataType(col.dtype) === filterType);
    }

    // Apply search
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        (col) =>
          col.name.toLowerCase().includes(query) ||
          col.dataType?.toLowerCase().includes(query)
      );
    }

    return result;
  }, [availableColumns, filterType, searchQuery]);

  // Handle feature toggle
  const handleFeatureToggle = (featureName: string) => {
    if (disabled) return;

    const isSelected = selectedFeatures.includes(featureName);

    if (isSelected) {
      onChange(selectedFeatures.filter((f) => f !== featureName));
    } else {
      if (maxFeatures && selectedFeatures.length >= maxFeatures) {
        return; // Don't add if max reached
      }
      onChange([...selectedFeatures, featureName]);
    }
  };

  // Select all visible features
  const handleSelectAll = () => {
    if (disabled) return;

    const visibleFeatureNames = filteredColumns.map((col) => col.name);
    const newSelection = [
      ...new Set([...selectedFeatures, ...visibleFeatureNames]),
    ].slice(0, maxFeatures);

    onChange(newSelection);
  };

  // Deselect all
  const handleDeselectAll = () => {
    if (disabled) return;
    onChange([]);
  };

  // Clear search
  const handleClearSearch = () => {
    setSearchQuery('');
  };

  // Get column type color
  const getTypeColor = (dataType: ColumnDataType): string => {
    switch (dataType) {
      case ColumnDataType.NUMERIC:
        return 'primary';
      case ColumnDataType.CATEGORICAL:
        return 'secondary';
      case ColumnDataType.BOOLEAN:
        return 'success';
      case ColumnDataType.DATETIME:
        return 'warning';
      case ColumnDataType.TEXT:
        return 'info';
      default:
        return 'default';
    }
  };

  return (
    <Card elevation={2}>
      <CardContent>
        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" fontWeight="bold">
            Select Input Features
          </Typography>
          <Tooltip title="Features used as inputs for model training">
            <IconButton size="small">
              <InfoIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Selection summary */}
        <Alert severity="info" sx={{ mb: 2 }}>
          <Typography variant="body2">
            <strong>{selectedFeatures.length}</strong> feature{selectedFeatures.length !== 1 && 's'} selected
            {maxFeatures && ` (max: ${maxFeatures})`}
            {filteredColumns.length !== availableColumns.length &&
              ` · Showing ${filteredColumns.length} of ${availableColumns.length}`}
          </Typography>
        </Alert>

        {/* Search and filter controls */}
        <Stack spacing={2} mb={2}>
          {/* Search */}
          <TextField
            fullWidth
            size="small"
            placeholder="Search features..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            disabled={disabled}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon fontSize="small" />
                </InputAdornment>
              ),
              endAdornment: searchQuery && (
                <InputAdornment position="end">
                  <IconButton size="small" onClick={handleClearSearch}>
                    <ClearIcon fontSize="small" />
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />

          {/* Filter toggle and actions */}
          <Box display="flex" gap={1}>
            <Button
              size="small"
              variant={showFilters ? 'contained' : 'outlined'}
              startIcon={<FilterIcon />}
              onClick={() => setShowFilters(!showFilters)}
              disabled={disabled}
            >
              Filters
            </Button>
            <Button
              size="small"
              variant="outlined"
              startIcon={<SelectAllIcon />}
              onClick={handleSelectAll}
              disabled={disabled || filteredColumns.length === 0}
            >
              Select All
            </Button>
            <Button
              size="small"
              variant="outlined"
              startIcon={<DeselectIcon />}
              onClick={handleDeselectAll}
              disabled={disabled || selectedFeatures.length === 0}
            >
              Clear
            </Button>
          </Box>

          {/* Filter controls */}
          <Collapse in={showFilters}>
            <FormControl fullWidth size="small">
              <InputLabel>Filter by Type</InputLabel>
              <Select
                value={filterType}
                label="Filter by Type"
                onChange={(e) => setFilterType(e.target.value as ColumnDataType | 'all')}
                disabled={disabled}
              >
                <MenuItem value="all">All Types</MenuItem>
                <MenuItem value={ColumnDataType.NUMERIC}>Numeric</MenuItem>
                <MenuItem value={ColumnDataType.CATEGORICAL}>Categorical</MenuItem>
                <MenuItem value={ColumnDataType.BOOLEAN}>Boolean</MenuItem>
                <MenuItem value={ColumnDataType.DATETIME}>DateTime</MenuItem>
                <MenuItem value={ColumnDataType.TEXT}>Text</MenuItem>
              </Select>
            </FormControl>
          </Collapse>
        </Stack>

        <Divider sx={{ mb: 2 }} />

        {/* Feature list */}
        <Box
          sx={{
            maxHeight: 400,
            overflowY: 'auto',
            border: '1px solid',
            borderColor: 'divider',
            borderRadius: 1,
            p: 1,
          }}
        >
          {filteredColumns.length === 0 ? (
            <Box textAlign="center" py={4}>
              <Typography variant="body2" color="text.secondary">
                {searchQuery || filterType !== 'all'
                  ? 'No features match your filters'
                  : 'No features available'}
              </Typography>
            </Box>
          ) : (
            <Stack spacing={1}>
              {filteredColumns.map((column) => {
                const isSelected = selectedFeatures.includes(column.name);
                const isDisabled = Boolean(
                  disabled ||
                  (maxFeatures &&
                    selectedFeatures.length >= maxFeatures &&
                    !isSelected)
                );
                const dataType = getColumnDataType(column.dtype);

                return (
                  <Box
                    key={column.name}
                    sx={{
                      p: 1.5,
                      borderRadius: 1,
                      bgcolor: isSelected ? 'action.selected' : 'background.paper',
                      border: '1px solid',
                      borderColor: isSelected ? 'primary.main' : 'divider',
                      cursor: isDisabled ? 'not-allowed' : 'pointer',
                      opacity: isDisabled ? 0.6 : 1,
                      '&:hover': {
                        bgcolor: isDisabled ? undefined : 'action.hover',
                      },
                      transition: 'all 0.2s',
                    }}
                    onClick={() => !isDisabled && handleFeatureToggle(column.name)}
                  >
                    <Box display="flex" alignItems="center" justifyContent="space-between">
                      <FormControlLabel
                        control={
                          <Checkbox
                            checked={isSelected}
                            disabled={isDisabled}
                            onClick={(e) => e.stopPropagation()}
                          />
                        }
                        label={
                          <Box>
                            <Typography variant="body2" fontWeight={isSelected ? 600 : 400}>
                              {column.name}
                            </Typography>
                            {column.missing_count !== undefined && column.missing_count > 0 && (
                              <Typography variant="caption" color="warning.main">
                                {column.missing_percentage?.toFixed(1)}% missing
                              </Typography>
                            )}
                          </Box>
                        }
                      />
                      <Stack direction="row" spacing={0.5}>
                        <Chip
                          label={dataType}
                          size="small"
                          color={getTypeColor(dataType) as any}
                          sx={{ textTransform: 'capitalize' }}
                        />
                        {column.unique_count !== undefined && (
                          <Chip
                            label={`${column.unique_count} unique`}
                            size="small"
                            variant="outlined"
                          />
                        )}
                      </Stack>
                    </Box>
                  </Box>
                );
              })}
            </Stack>
          )}
        </Box>

        {/* Max features warning */}
        {maxFeatures && selectedFeatures.length >= maxFeatures && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            Maximum number of features ({maxFeatures}) reached. Deselect features to add new ones.
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default FeatureSelector;
