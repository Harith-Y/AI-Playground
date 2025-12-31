import React, { useState } from 'react';
import { Box, Typography, Tooltip, Paper, Chip } from '@mui/material';
import { Warning, CheckCircle, Error as ErrorIcon } from '@mui/icons-material';

interface MissingValuesData {
  columnName: string;
  totalCount: number;
  missingCount: number;
  missingPercentage: number;
}

interface MissingValuesHeatmapProps {
  data: MissingValuesData[];
  totalRows: number;
  width?: number;
  maxHeight?: number;
}

const MissingValuesHeatmap: React.FC<MissingValuesHeatmapProps> = ({
  data,
  totalRows,
  width = 800,
  maxHeight = 600,
}) => {
  const [hoveredColumn, setHoveredColumn] = useState<string | null>(null);

  // Sort by missing percentage (descending)
  const sortedData = [...data].sort((a, b) => b.missingPercentage - a.missingPercentage);

  // Calculate dimensions
  const columnWidth = Math.max(120, Math.min(200, width / Math.max(data.length, 5)));
  const actualWidth = columnWidth * data.length;
  const _barHeight = 40;
  const labelHeight = 100;
  const chartHeight = Math.min(maxHeight - labelHeight - 150, 400);

  // Color scale function based on missing percentage
  const getColor = (percentage: number): string => {
    if (percentage === 0) {
      return '#10B981'; // Green - no missing values
    } else if (percentage < 5) {
      return '#3B82F6'; // Blue - very low
    } else if (percentage < 20) {
      return '#F59E0B'; // Orange - moderate
    } else if (percentage < 50) {
      return '#EF4444'; // Red - high
    } else {
      return '#991B1B'; // Dark red - critical
    }
  };

  // Get severity level
  const getSeverity = (percentage: number): { label: string; icon: React.ReactNode } => {
    if (percentage === 0) {
      return { label: 'Complete', icon: <CheckCircle sx={{ fontSize: 16 }} /> };
    } else if (percentage < 5) {
      return { label: 'Excellent', icon: <CheckCircle sx={{ fontSize: 16 }} /> };
    } else if (percentage < 20) {
      return { label: 'Good', icon: <Warning sx={{ fontSize: 16 }} /> };
    } else if (percentage < 50) {
      return { label: 'Poor', icon: <Warning sx={{ fontSize: 16 }} /> };
    } else {
      return { label: 'Critical', icon: <ErrorIcon sx={{ fontSize: 16 }} /> };
    }
  };

  // Calculate overall statistics
  const totalMissing = data.reduce((sum, col) => sum + col.missingCount, 0);
  const totalCells = totalRows * data.length;
  const overallPercentage = (totalMissing / totalCells) * 100;
  const completeColumns = data.filter(d => d.missingPercentage === 0).length;
  const criticalColumns = data.filter(d => d.missingPercentage > 50).length;

  return (
    <Box sx={{ width: '100%', overflowX: 'auto' }}>
      {/* Summary Statistics */}
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
          gap: 2,
          mb: 3,
        }}
      >
        <Paper sx={{ p: 2, border: '1px solid #e2e8f0', background: '#f8fafc' }}>
          <Typography variant="caption" color="text.secondary" display="block">
            Overall Completeness
          </Typography>
          <Typography variant="h5" fontWeight={600} color={getColor(overallPercentage)}>
            {(100 - overallPercentage).toFixed(1)}%
          </Typography>
        </Paper>

        <Paper sx={{ p: 2, border: '1px solid #e2e8f0', background: '#f8fafc' }}>
          <Typography variant="caption" color="text.secondary" display="block">
            Complete Columns
          </Typography>
          <Typography variant="h5" fontWeight={600} color="success.main">
            {completeColumns}/{data.length}
          </Typography>
        </Paper>

        <Paper sx={{ p: 2, border: '1px solid #e2e8f0', background: '#f8fafc' }}>
          <Typography variant="caption" color="text.secondary" display="block">
            Total Missing Cells
          </Typography>
          <Typography variant="h5" fontWeight={600} color="error.main">
            {totalMissing.toLocaleString()}
          </Typography>
        </Paper>

        <Paper sx={{ p: 2, border: '1px solid #e2e8f0', background: '#f8fafc' }}>
          <Typography variant="caption" color="text.secondary" display="block">
            Critical Columns (&gt;50%)
          </Typography>
          <Typography variant="h5" fontWeight={600} color={criticalColumns > 0 ? 'error.main' : 'success.main'}>
            {criticalColumns}
          </Typography>
        </Paper>
      </Box>

      {/* Bar Chart Visualization */}
      <Box sx={{ position: 'relative', mb: 3 }}>
        <Typography variant="subtitle2" fontWeight={600} gutterBottom>
          Missing Values by Column
        </Typography>

        <Box
          sx={{
            display: 'flex',
            overflowX: 'auto',
            pb: 2,
          }}
        >
          <Box sx={{ position: 'relative', minWidth: actualWidth }}>
            {/* Chart area */}
            <Box
              sx={{
                height: chartHeight,
                position: 'relative',
                borderLeft: '2px solid #e2e8f0',
                borderBottom: '2px solid #e2e8f0',
                mb: 1,
              }}
            >
              {/* Horizontal grid lines */}
              {[0, 25, 50, 75, 100].map((tick) => (
                <Box
                  key={tick}
                  sx={{
                    position: 'absolute',
                    left: 0,
                    right: 0,
                    bottom: `${tick}%`,
                    borderTop: '1px dashed #e2e8f0',
                    display: 'flex',
                    alignItems: 'center',
                  }}
                >
                  <Typography
                    variant="caption"
                    sx={{
                      position: 'absolute',
                      left: -35,
                      fontSize: '0.7rem',
                      color: 'text.secondary',
                    }}
                  >
                    {tick}%
                  </Typography>
                </Box>
              ))}

              {/* Bars */}
              {sortedData.map((column, index) => {
                const isHovered = hoveredColumn === column.columnName;
                const barHeightPx = (column.missingPercentage / 100) * chartHeight;

                return (
                  <Tooltip
                    key={column.columnName}
                    title={
                      <Box>
                        <Typography variant="caption" display="block" fontWeight={600}>
                          {column.columnName}
                        </Typography>
                        <Typography variant="caption" display="block">
                          Missing: {column.missingCount.toLocaleString()} / {column.totalCount.toLocaleString()}
                        </Typography>
                        <Typography variant="caption" display="block">
                          Percentage: {column.missingPercentage.toFixed(2)}%
                        </Typography>
                      </Box>
                    }
                    placement="top"
                    arrow
                  >
                    <Box
                      sx={{
                        position: 'absolute',
                        left: index * columnWidth + 10,
                        bottom: 0,
                        width: columnWidth - 20,
                        height: barHeightPx,
                        backgroundColor: getColor(column.missingPercentage),
                        cursor: 'pointer',
                        transition: 'all 0.3s ease',
                        borderRadius: '4px 4px 0 0',
                        opacity: isHovered ? 1 : 0.85,
                        transform: isHovered ? 'scaleY(1.02)' : 'scaleY(1)',
                        transformOrigin: 'bottom',
                        '&:hover': {
                          opacity: 1,
                          transform: 'scaleY(1.02)',
                          boxShadow: '0 -4px 12px rgba(0,0,0,0.15)',
                        },
                      }}
                      onMouseEnter={() => setHoveredColumn(column.columnName)}
                      onMouseLeave={() => setHoveredColumn(null)}
                    >
                      {/* Percentage label on bar */}
                      {barHeightPx > 30 && (
                        <Typography
                          variant="caption"
                          sx={{
                            position: 'absolute',
                            top: 8,
                            left: '50%',
                            transform: 'translateX(-50%)',
                            color: '#ffffff',
                            fontWeight: 600,
                            fontSize: '0.75rem',
                            textShadow: '0 1px 2px rgba(0,0,0,0.3)',
                          }}
                        >
                          {column.missingPercentage.toFixed(1)}%
                        </Typography>
                      )}
                    </Box>
                  </Tooltip>
                );
              })}
            </Box>

            {/* Column labels */}
            <Box sx={{ position: 'relative', height: labelHeight }}>
              {sortedData.map((column, index) => {
                const severity = getSeverity(column.missingPercentage);

                return (
                  <Box
                    key={column.columnName}
                    sx={{
                      position: 'absolute',
                      left: index * columnWidth,
                      width: columnWidth,
                      top: 0,
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      gap: 0.5,
                    }}
                  >
                    <Typography
                      variant="caption"
                      sx={{
                        fontSize: '0.7rem',
                        fontWeight: 600,
                        color: 'text.primary',
                        textAlign: 'center',
                        maxWidth: columnWidth - 10,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {column.columnName}
                    </Typography>
                    <Chip
                      icon={severity.icon as React.ReactElement}
                      label={severity.label}
                      size="small"
                      sx={{
                        height: 20,
                        fontSize: '0.65rem',
                        backgroundColor: `${getColor(column.missingPercentage)}20`,
                        color: getColor(column.missingPercentage),
                        '& .MuiChip-icon': {
                          fontSize: 14,
                        },
                      }}
                    />
                  </Box>
                );
              })}
            </Box>
          </Box>
        </Box>
      </Box>

      {/* Legend */}
      <Paper sx={{ p: 2, border: '1px solid #e2e8f0', background: '#f8fafc' }}>
        <Typography variant="caption" fontWeight={600} display="block" gutterBottom>
          Severity Levels:
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mt: 1 }}>
          {[
            { color: '#10B981', label: 'Complete (0%)', range: '0%' },
            { color: '#3B82F6', label: 'Excellent', range: '<5%' },
            { color: '#F59E0B', label: 'Good', range: '5-20%' },
            { color: '#EF4444', label: 'Poor', range: '20-50%' },
            { color: '#991B1B', label: 'Critical', range: '>50%' },
          ].map((item) => (
            <Box key={item.label} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box
                sx={{
                  width: 16,
                  height: 16,
                  backgroundColor: item.color,
                  borderRadius: 0.5,
                }}
              />
              <Typography variant="caption" sx={{ fontSize: '0.7rem' }}>
                {item.label} <span style={{ color: '#64748b' }}>({item.range})</span>
              </Typography>
            </Box>
          ))}
        </Box>
      </Paper>

      {/* Recommendations */}
      {criticalColumns > 0 && (
        <Paper
          sx={{
            mt: 2,
            p: 2,
            border: '1px solid #fecaca',
            background: '#fef2f2',
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
            <Warning sx={{ color: 'error.main', fontSize: 20, mt: 0.5 }} />
            <Box>
              <Typography variant="caption" fontWeight={600} color="error.main" display="block">
                Data Quality Warning
              </Typography>
              <Typography variant="caption" color="text.secondary" display="block">
                {criticalColumns} column{criticalColumns > 1 ? 's have' : ' has'} more than 50% missing values.
                Consider dropping these columns or using advanced imputation techniques.
              </Typography>
            </Box>
          </Box>
        </Paper>
      )}
    </Box>
  );
};

export default MissingValuesHeatmap;
