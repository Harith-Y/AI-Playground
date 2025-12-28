/**
 * FeatureRankingTable Component
 *
 * Displays feature importance rankings in a sortable table format
 * Provides detailed statistics and comparison across multiple importance metrics
 */

import React, { useState, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  Chip,
  LinearProgress,
  Tooltip,
  Alert,
  TextField,
  InputAdornment,
} from '@mui/material';
import { Search as SearchIcon, TrendingUp, TrendingDown, Remove } from '@mui/icons-material';

/**
 * Feature ranking entry with multiple metrics
 */
export interface FeatureRanking {
  feature: string;
  importance: number;
  permutationImportance?: number;
  rank?: number;
  permutationRank?: number;
  correlation?: number;
  type?: string;
  stdDev?: number;
}

/**
 * Props for FeatureRankingTable component
 */
export interface FeatureRankingTableProps {
  features: FeatureRanking[];
  title?: string;
  className?: string;
  showSearch?: boolean;
  showPermutation?: boolean;
  showCorrelation?: boolean;
}

type SortField = 'feature' | 'importance' | 'permutationImportance' | 'correlation' | 'rank';
type SortOrder = 'asc' | 'desc';

const FeatureRankingTable: React.FC<FeatureRankingTableProps> = ({
  features,
  title = 'Feature Rankings',
  className,
  showSearch = true,
  showPermutation = true,
  showCorrelation = true,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [sortField, setSortField] = useState<SortField>('importance');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');

  // Validate data
  if (!features || features.length === 0) {
    return (
      <Alert severity="warning">
        <Typography variant="body2">
          No feature ranking data available
        </Typography>
      </Alert>
    );
  }

  // Process features
  const processedFeatures = useMemo(() => {
    let processed = features.map((f, idx) => ({
      ...f,
      rank: f.rank ?? idx + 1,
    }));

    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      processed = processed.filter((f) =>
        f.feature.toLowerCase().includes(query) ||
        f.type?.toLowerCase().includes(query)
      );
    }

    // Sort
    processed.sort((a, b) => {
      let aVal: any = a[sortField];
      let bVal: any = b[sortField];

      // Handle string comparison
      if (typeof aVal === 'string') {
        return sortOrder === 'asc'
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      }

      // Handle numeric comparison (with undefined handling)
      aVal = aVal ?? -Infinity;
      bVal = bVal ?? -Infinity;
      return sortOrder === 'asc' ? aVal - bVal : bVal - aVal;
    });

    return processed;
  }, [features, searchQuery, sortField, sortOrder]);

  // Calculate max values for progress bars
  const maxImportance = useMemo(() => {
    return Math.max(...features.map((f) => f.importance));
  }, [features]);

  const maxPermutation = useMemo(() => {
    if (!showPermutation) return 0;
    const vals = features
      .map((f) => f.permutationImportance)
      .filter((v): v is number => v !== undefined);
    return vals.length > 0 ? Math.max(...vals) : 0;
  }, [features, showPermutation]);

  // Handle sort
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('desc');
    }
  };

  // Get rank change icon
  const getRankChangeIcon = (importance: number, permutation?: number) => {
    if (!showPermutation || permutation === undefined) return null;

    const diff = importance - permutation;
    if (Math.abs(diff) < 0.01) {
      return (
        <Tooltip title="Consistent ranking">
          <Remove fontSize="small" color="action" />
        </Tooltip>
      );
    } else if (diff > 0) {
      return (
        <Tooltip title="Higher in feature importance">
          <TrendingUp fontSize="small" color="success" />
        </Tooltip>
      );
    } else {
      return (
        <Tooltip title="Higher in permutation importance">
          <TrendingDown fontSize="small" color="warning" />
        </Tooltip>
      );
    }
  };

  // Get correlation badge
  const getCorrelationBadge = (correlation?: number) => {
    if (correlation === undefined || !showCorrelation) return null;

    const absCorr = Math.abs(correlation);
    let color: 'error' | 'warning' | 'success' = 'success';
    let label = 'Low';

    if (absCorr > 0.7) {
      color = 'error';
      label = 'High';
    } else if (absCorr > 0.4) {
      color = 'warning';
      label = 'Medium';
    }

    return (
      <Tooltip title={`Correlation with target: ${correlation.toFixed(3)}`}>
        <Chip
          label={label}
          color={color}
          size="small"
          sx={{ minWidth: 60 }}
        />
      </Tooltip>
    );
  };

  return (
    <Paper elevation={2} sx={{ p: 3 }} className={className}>
      {/* Header */}
      <Box mb={2} display="flex" justifyContent="space-between" alignItems="center">
        <Typography variant="h6" fontWeight="bold">
          {title}
        </Typography>
        <Chip
          label={`${processedFeatures.length} / ${features.length} Features`}
          variant="outlined"
          size="small"
        />
      </Box>

      {/* Search */}
      {showSearch && (
        <Box mb={2}>
          <TextField
            fullWidth
            size="small"
            placeholder="Search features..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
            }}
          />
        </Box>
      )}

      {/* Table */}
      <TableContainer sx={{ maxHeight: 600 }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              <TableCell>
                <TableSortLabel
                  active={sortField === 'rank'}
                  direction={sortField === 'rank' ? sortOrder : 'asc'}
                  onClick={() => handleSort('rank')}
                >
                  Rank
                </TableSortLabel>
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={sortField === 'feature'}
                  direction={sortField === 'feature' ? sortOrder : 'asc'}
                  onClick={() => handleSort('feature')}
                >
                  Feature
                </TableSortLabel>
              </TableCell>
              <TableCell align="right">
                <TableSortLabel
                  active={sortField === 'importance'}
                  direction={sortField === 'importance' ? sortOrder : 'asc'}
                  onClick={() => handleSort('importance')}
                >
                  Importance
                </TableSortLabel>
              </TableCell>
              <TableCell align="center" sx={{ minWidth: 200 }}>
                Importance
              </TableCell>
              {showPermutation && (
                <>
                  <TableCell align="right">
                    <TableSortLabel
                      active={sortField === 'permutationImportance'}
                      direction={sortField === 'permutationImportance' ? sortOrder : 'asc'}
                      onClick={() => handleSort('permutationImportance')}
                    >
                      Permutation
                    </TableSortLabel>
                  </TableCell>
                  <TableCell align="center">Trend</TableCell>
                </>
              )}
              {showCorrelation && (
                <TableCell align="center">
                  <TableSortLabel
                    active={sortField === 'correlation'}
                    direction={sortField === 'correlation' ? sortOrder : 'asc'}
                    onClick={() => handleSort('correlation')}
                  >
                    Correlation
                  </TableSortLabel>
                </TableCell>
              )}
            </TableRow>
          </TableHead>
          <TableBody>
            {processedFeatures.map((feature, idx) => (
              <TableRow key={feature.feature} hover>
                {/* Rank */}
                <TableCell>
                  <Chip
                    label={feature.rank}
                    size="small"
                    color={feature.rank <= 5 ? 'primary' : 'default'}
                    sx={{ minWidth: 40 }}
                  />
                </TableCell>

                {/* Feature Name */}
                <TableCell>
                  <Box>
                    <Typography variant="body2" fontWeight="500">
                      {feature.feature}
                    </Typography>
                    {feature.type && (
                      <Typography variant="caption" color="text.secondary">
                        {feature.type}
                      </Typography>
                    )}
                  </Box>
                </TableCell>

                {/* Importance Value */}
                <TableCell align="right">
                  <Typography variant="body2" fontWeight="500">
                    {feature.importance.toFixed(4)}
                  </Typography>
                  {feature.stdDev !== undefined && (
                    <Typography variant="caption" color="text.secondary">
                      ±{feature.stdDev.toFixed(4)}
                    </Typography>
                  )}
                </TableCell>

                {/* Importance Progress Bar */}
                <TableCell>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ flexGrow: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={(feature.importance / maxImportance) * 100}
                        sx={{
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: '#e0e0e0',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor:
                              feature.importance / maxImportance > 0.7
                                ? '#4caf50'
                                : feature.importance / maxImportance > 0.4
                                  ? '#ff9800'
                                  : '#1976d2',
                          },
                        }}
                      />
                    </Box>
                    <Typography variant="caption" color="text.secondary" sx={{ minWidth: 40 }}>
                      {((feature.importance / maxImportance) * 100).toFixed(0)}%
                    </Typography>
                  </Box>
                </TableCell>

                {/* Permutation Importance */}
                {showPermutation && (
                  <>
                    <TableCell align="right">
                      {feature.permutationImportance !== undefined ? (
                        <Typography variant="body2">
                          {feature.permutationImportance.toFixed(4)}
                        </Typography>
                      ) : (
                        <Typography variant="caption" color="text.secondary">
                          N/A
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell align="center">
                      {getRankChangeIcon(feature.importance, feature.permutationImportance)}
                    </TableCell>
                  </>
                )}

                {/* Correlation */}
                {showCorrelation && (
                  <TableCell align="center">
                    {getCorrelationBadge(feature.correlation)}
                  </TableCell>
                )}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* No results */}
      {processedFeatures.length === 0 && searchQuery && (
        <Box py={4} textAlign="center">
          <Typography variant="body2" color="text.secondary">
            No features match "{searchQuery}"
          </Typography>
        </Box>
      )}

      {/* Legend */}
      <Box mt={2} p={2} bgcolor="grey.50" borderRadius={1}>
        <Typography variant="caption" fontWeight="bold" display="block" gutterBottom>
          Table Guide:
        </Typography>
        <Box display="flex" flexDirection="column" gap={0.5}>
          <Typography variant="caption" color="text.secondary">
            • <strong>Rank:</strong> Position in importance ordering (1 = most important)
          </Typography>
          <Typography variant="caption" color="text.secondary">
            • <strong>Importance:</strong> Feature importance score with progress bar visualization
          </Typography>
          {showPermutation && (
            <Typography variant="caption" color="text.secondary">
              • <strong>Permutation:</strong> Model-agnostic importance from permutation testing
            </Typography>
          )}
          {showCorrelation && (
            <Typography variant="caption" color="text.secondary">
              • <strong>Correlation:</strong> Correlation with target variable (High may indicate multicollinearity)
            </Typography>
          )}
          <Typography variant="caption" color="text.secondary">
            • Click column headers to sort. Use search to filter features.
          </Typography>
        </Box>
      </Box>
    </Paper>
  );
};

export default FeatureRankingTable;
