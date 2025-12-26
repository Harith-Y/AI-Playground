import React from 'react';
import { Box, Typography, Button } from '@mui/material';
import { ErrorOutline, Refresh } from '@mui/icons-material';
import LoadingState from '../common/LoadingState';
import EmptyState from '../common/EmptyState';
import MissingValuesHeatmap from './MissingValuesHeatmap';
import type { ColumnInfo } from '../../types/dataset';

interface MissingValuesAnalysisProps {
  columns?: ColumnInfo[];
  totalRows?: number;
  isLoading?: boolean;
  onRefresh?: () => void;
}

const MissingValuesAnalysis: React.FC<MissingValuesAnalysisProps> = ({
  columns = [],
  totalRows = 0,
  isLoading = false,
  onRefresh,
}) => {
  if (isLoading) {
    return <LoadingState message="Analyzing missing values..." variant="linear" />;
  }

  if (!columns || columns.length === 0 || totalRows === 0) {
    return (
      <Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2, justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ErrorOutline sx={{ color: 'primary.main' }} />
            <Typography variant="h6" fontWeight={600}>
              Missing Values Analysis
            </Typography>
          </Box>
          {onRefresh && (
            <Button
              size="small"
              startIcon={<Refresh />}
              onClick={onRefresh}
            >
              Analyze
            </Button>
          )}
        </Box>
        <EmptyState
          icon={<ErrorOutline sx={{ fontSize: 64, color: 'text.secondary', opacity: 0.5 }} />}
          title="No Data Available"
          message="Upload a dataset to analyze missing values"
        />
      </Box>
    );
  }

  // Transform column data to missing values format
  const missingValuesData = columns.map(col => ({
    columnName: col.name,
    totalCount: totalRows,
    missingCount: col.nullCount || 0,
    missingPercentage: totalRows > 0 ? ((col.nullCount || 0) / totalRows) * 100 : 0,
  }));

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3, justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ErrorOutline sx={{ color: 'primary.main' }} />
          <Typography variant="h6" fontWeight={600}>
            Missing Values Analysis
          </Typography>
        </Box>
        {onRefresh && (
          <Button
            size="small"
            startIcon={<Refresh />}
            onClick={onRefresh}
            variant="outlined"
          >
            Refresh
          </Button>
        )}
      </Box>

      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Visualize and identify columns with missing data. Hover over bars for detailed statistics.
      </Typography>

      <MissingValuesHeatmap
        data={missingValuesData}
        totalRows={totalRows}
      />
    </Box>
  );
};

export default MissingValuesAnalysis;
