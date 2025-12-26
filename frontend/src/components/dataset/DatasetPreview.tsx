import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Collapse,
  Chip,
  Alert,
} from '@mui/material';
import {
  ExpandMore,
  ExpandLess,
  TableChart,
  Info as InfoIcon,
} from '@mui/icons-material';

interface Column {
  name: string;
  dataType: string;
  nullCount: number;
  uniqueCount: number;
  sampleValues: any[];
}

interface DatasetPreviewProps {
  preview?: any[][];
  columns?: Column[];
  isLoading?: boolean;
  error?: string | null;
  maxRows?: number;
}

const DatasetPreview: React.FC<DatasetPreviewProps> = ({
  preview = [],
  columns = [],
  isLoading = false,
  error = null,
  maxRows = 10,
}) => {
  const [isExpanded, setIsExpanded] = useState(true);

  // Get column names
  const columnNames = columns.map((col) => col.name);

  // Limit preview rows
  const displayRows = preview.slice(0, maxRows);

  // Get data type color
  const getDataTypeColor = (dataType: string): string => {
    const type = dataType.toLowerCase();
    if (type.includes('int') || type.includes('float') || type.includes('number')) {
      return '#3B82F6'; // Blue for numeric
    }
    if (type.includes('object') || type.includes('str') || type.includes('string')) {
      return '#10B981'; // Green for text
    }
    if (type.includes('bool')) {
      return '#F59E0B'; // Orange for boolean
    }
    if (type.includes('date') || type.includes('time')) {
      return '#8B5CF6'; // Purple for datetime
    }
    return '#6B7280'; // Gray for others
  };

  // Format cell value
  const formatCellValue = (value: any): string => {
    if (value === null || value === undefined) {
      return 'null';
    }
    if (typeof value === 'number') {
      return value.toLocaleString();
    }
    if (typeof value === 'boolean') {
      return value.toString();
    }
    if (typeof value === 'object') {
      return JSON.stringify(value);
    }
    return String(value);
  };

  if (isLoading) {
    return (
      <Paper sx={{ p: 4, border: '1px solid #e2e8f0', background: '#FFFFFF' }}>
        <Typography variant="body1" color="text.secondary">
          Loading preview...
        </Typography>
      </Paper>
    );
  }

  if (error) {
    return (
      <Paper sx={{ p: 4, border: '1px solid #e2e8f0', background: '#FFFFFF' }}>
        <Alert severity="error">{error}</Alert>
      </Paper>
    );
  }

  if (preview.length === 0 || columns.length === 0) {
    return (
      <Paper sx={{ p: 4, border: '1px solid #e2e8f0', background: '#FFFFFF' }}>
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <TableChart sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="text.secondary">
            No data to preview
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Upload a dataset to see the preview
          </Typography>
        </Box>
      </Paper>
    );
  }

  return (
    <Paper sx={{ border: '1px solid #e2e8f0', background: '#FFFFFF' }}>
      {/* Header */}
      <Box
        sx={{
          p: 2,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid #e2e8f0',
          cursor: 'pointer',
        }}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <TableChart sx={{ color: 'primary.main' }} />
          <Typography variant="h6" fontWeight={600}>
            Data Preview
          </Typography>
          <Chip
            label={`${displayRows.length} of ${preview.length} rows`}
            size="small"
            sx={{ background: '#EFF6FF', color: '#2563eb', fontWeight: 600 }}
          />
        </Box>
        <IconButton size="small">
          {isExpanded ? <ExpandLess /> : <ExpandMore />}
        </IconButton>
      </Box>

      {/* Content */}
      <Collapse in={isExpanded}>
        <Box sx={{ p: 3 }}>
          {/* Column Info */}
          <Box sx={{ mb: 3, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {columns.map((col, idx) => (
              <Chip
                key={idx}
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Typography variant="caption" fontWeight={600}>
                      {col.name}
                    </Typography>
                    <Typography
                      variant="caption"
                      sx={{
                        color: getDataTypeColor(col.dataType),
                        fontWeight: 500,
                      }}
                    >
                      ({col.dataType})
                    </Typography>
                  </Box>
                }
                size="small"
                variant="outlined"
                sx={{ borderColor: '#e2e8f0' }}
              />
            ))}
          </Box>

          {/* Info Alert */}
          {preview.length > maxRows && (
            <Alert severity="info" icon={<InfoIcon />} sx={{ mb: 2 }}>
              Showing first {maxRows} rows. Total rows: {preview.length}
            </Alert>
          )}

          {/* Table */}
          <TableContainer
            sx={{
              maxHeight: 600,
              border: '1px solid #e2e8f0',
              borderRadius: 2,
              '&::-webkit-scrollbar': {
                width: '8px',
                height: '8px',
              },
              '&::-webkit-scrollbar-track': {
                background: '#f1f1f1',
              },
              '&::-webkit-scrollbar-thumb': {
                background: '#cbd5e1',
                borderRadius: '4px',
              },
              '&::-webkit-scrollbar-thumb:hover': {
                background: '#94a3b8',
              },
            }}
          >
            <Table stickyHeader size="small">
              <TableHead>
                <TableRow>
                  <TableCell
                    sx={{
                      background: '#F8FAFC',
                      fontWeight: 700,
                      color: '#475569',
                      borderBottom: '2px solid #cbd5e1',
                      width: 60,
                    }}
                  >
                    #
                  </TableCell>
                  {columnNames.map((colName, idx) => (
                    <TableCell
                      key={idx}
                      sx={{
                        background: '#F8FAFC',
                        fontWeight: 700,
                        color: '#475569',
                        borderBottom: '2px solid #cbd5e1',
                        minWidth: 150,
                        maxWidth: 300,
                      }}
                    >
                      <Box>
                        <Typography variant="body2" fontWeight={700}>
                          {colName}
                        </Typography>
                        {columns[idx] && (
                          <Typography
                            variant="caption"
                            sx={{
                              color: getDataTypeColor(columns[idx].dataType),
                              fontWeight: 500,
                            }}
                          >
                            {columns[idx].dataType}
                          </Typography>
                        )}
                      </Box>
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {displayRows.map((row, rowIdx) => (
                  <TableRow
                    key={rowIdx}
                    sx={{
                      '&:hover': {
                        background: '#F8FAFC',
                      },
                      '&:nth-of-type(even)': {
                        background: '#FAFBFC',
                      },
                    }}
                  >
                    <TableCell
                      sx={{
                        fontWeight: 600,
                        color: '#64748b',
                        borderRight: '1px solid #e2e8f0',
                      }}
                    >
                      {rowIdx + 1}
                    </TableCell>
                    {row.map((cell, cellIdx) => (
                      <TableCell
                        key={cellIdx}
                        sx={{
                          maxWidth: 300,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          color:
                            cell === null || cell === undefined
                              ? '#94a3b8'
                              : '#1e293b',
                          fontStyle:
                            cell === null || cell === undefined
                              ? 'italic'
                              : 'normal',
                        }}
                      >
                        {formatCellValue(cell)}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          {/* Footer Info */}
          <Box
            sx={{
              mt: 2,
              p: 2,
              background: '#F8FAFC',
              borderRadius: 1,
              display: 'flex',
              gap: 3,
            }}
          >
            <Typography variant="caption" color="text.secondary">
              <strong>Columns:</strong> {columns.length}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              <strong>Rows:</strong> {preview.length}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              <strong>Showing:</strong> {displayRows.length} rows
            </Typography>
          </Box>
        </Box>
      </Collapse>
    </Paper>
  );
};

export default DatasetPreview;
