import React, { useState, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  ToggleButtonGroup,
  ToggleButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Divider,
  Card,
  CardContent,
  Grid,
  Tooltip,
  IconButton,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
} from '@mui/material';
import {
  CompareArrows,
  TableChart,
  ShowChart,
  Info,
  TrendingUp,
  TrendingDown,
  RemoveCircle,
  Error as ErrorIcon,
  WarningAmber,
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  LineChart as _LineChart,
  Line as _Line,
  PieChart,
  Pie,
  Cell,
} from 'recharts';

interface PreviewData {
  before: {
    data: Record<string, any>[];
    shape: [number, number];
    columns: string[];
    dtypes: Record<string, string>;
    nullCounts: Record<string, number>;
    statistics?: Record<string, any>;
  };
  after: {
    data: Record<string, any>[];
    shape: [number, number];
    columns: string[];
    dtypes: Record<string, string>;
    nullCounts: Record<string, number>;
    statistics?: Record<string, any>;
  };
}

interface PreviewPanelProps {
  previewData: PreviewData | null;
  isLoading?: boolean;
  error?: string | null;
}

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1'];

const PreviewPanel: React.FC<PreviewPanelProps> = ({ previewData, isLoading = false, error = null }) => {
  const [viewMode, setViewMode] = useState<'table' | 'chart'>('table');
  const [selectedColumn, setSelectedColumn] = useState<string>('');

  // Get common columns between before and after
  const commonColumns = useMemo(() => {
    if (!previewData) return [];
    return previewData.before.columns.filter((col) =>
      previewData.after.columns.includes(col)
    );
  }, [previewData]);

  // Set default selected column
  React.useEffect(() => {
    if (commonColumns.length > 0 && !selectedColumn) {
      setSelectedColumn(commonColumns[0]);
    }
  }, [commonColumns, selectedColumn]);

  // Determine if column is numeric
  const isNumericColumn = (col: string): boolean => {
    if (!previewData) return false;
    const dtype = previewData.before.dtypes[col] || previewData.after.dtypes[col];
    return dtype?.includes('int') || dtype?.includes('float');
  };

  // Get value distribution for categorical columns
  const getValueDistribution = (data: Record<string, any>[], column: string) => {
    const counts: Record<string, number> = {};
    data.forEach((row) => {
      const value = String(row[column] ?? 'null');
      counts[value] = (counts[value] || 0) + 1;
    });
    return Object.entries(counts)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 10); // Top 10 categories
  };

  // Get statistics summary for numeric columns
  const getNumericStats = (data: Record<string, any>[], column: string) => {
    const values = data
      .map((row) => row[column])
      .filter((val) => val != null && !isNaN(Number(val)))
      .map(Number);

    if (values.length === 0) return null;

    const sorted = [...values].sort((a, b) => a - b);
    const sum = values.reduce((acc, val) => acc + val, 0);
    const mean = sum / values.length;
    const median = sorted[Math.floor(sorted.length / 2)];
    const min = Math.min(...values);
    const max = Math.max(...values);
    const variance = values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / values.length;
    const std = Math.sqrt(variance);

    return { mean, median, min, max, std, count: values.length };
  };

  // Create histogram data for numeric columns
  const getHistogramData = (data: Record<string, any>[], column: string, bins = 10) => {
    const values = data
      .map((row) => row[column])
      .filter((val) => val != null && !isNaN(Number(val)))
      .map(Number);

    if (values.length === 0) return [];

    const min = Math.min(...values);
    const max = Math.max(...values);
    const binWidth = (max - min) / bins;

    const histogram = Array.from({ length: bins }, (_, i) => ({
      range: `${(min + i * binWidth).toFixed(1)}-${(min + (i + 1) * binWidth).toFixed(1)}`,
      count: 0,
    }));

    values.forEach((value) => {
      const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1);
      histogram[binIndex].count++;
    });

    return histogram;
  };

  // Error state
  if (error) {
    return (
      <Paper sx={{ p: 3, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Box sx={{ textAlign: 'center', maxWidth: 500 }}>
          <ErrorIcon sx={{ fontSize: 64, color: 'error.main', mb: 2 }} />
          <Typography variant="h6" color="error" gutterBottom>
            Preview Error
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            {error}
          </Typography>
          <Alert severity="error" sx={{ mt: 2, textAlign: 'left' }}>
            Common causes:
            <ul style={{ marginTop: 8, marginBottom: 0, paddingLeft: 20 }}>
              <li>Invalid preprocessing configuration</li>
              <li>Empty or corrupted dataset</li>
              <li>Incompatible column types for selected operations</li>
              <li>Backend service error</li>
            </ul>
          </Alert>
        </Box>
      </Paper>
    );
  }

  // Empty data state (no rows)
  if (previewData && previewData.before.shape[0] === 0) {
    return (
      <Paper sx={{ p: 3, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Box sx={{ textAlign: 'center' }}>
          <WarningAmber sx={{ fontSize: 64, color: 'warning.main', mb: 2 }} />
          <Typography variant="h6" color="warning.main" gutterBottom>
            Empty Dataset
          </Typography>
          <Typography variant="body2" color="text.secondary">
            The dataset has no rows to preview. Please upload a dataset with data.
          </Typography>
        </Box>
      </Paper>
    );
  }

  // Empty columns state
  if (previewData && previewData.before.shape[1] === 0) {
    return (
      <Paper sx={{ p: 3, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Box sx={{ textAlign: 'center' }}>
          <WarningAmber sx={{ fontSize: 64, color: 'warning.main', mb: 2 }} />
          <Typography variant="h6" color="warning.main" gutterBottom>
            No Columns
          </Typography>
          <Typography variant="body2" color="text.secondary">
            The dataset has no columns. Please upload a valid dataset.
          </Typography>
        </Box>
      </Paper>
    );
  }

  // No preview data yet
  if (!previewData) {
    return (
      <Paper sx={{ p: 3, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Box sx={{ textAlign: 'center' }}>
          <CompareArrows sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="text.secondary" gutterBottom>
            No Preview Available
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Execute the preprocessing pipeline to see before/after comparison
          </Typography>
        </Box>
      </Paper>
    );
  }

  const beforeStats = selectedColumn ? getNumericStats(previewData.before.data, selectedColumn) : null;
  const afterStats = selectedColumn ? getNumericStats(previewData.after.data, selectedColumn) : null;

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <CompareArrows color="primary" />
          <Typography variant="h6" fontWeight={600}>
            Before/After Preview
          </Typography>
          <Tooltip title="Shows a comparison of your data before and after preprocessing">
            <IconButton size="small">
              <Info fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>

        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel>Column</InputLabel>
            <Select
              value={selectedColumn}
              onChange={(e) => setSelectedColumn(e.target.value)}
              label="Column"
            >
              {commonColumns.map((col) => (
                <MenuItem key={col} value={col}>
                  {col}
                  <Chip
                    label={isNumericColumn(col) ? 'numeric' : 'categorical'}
                    size="small"
                    sx={{ ml: 1, height: 20 }}
                  />
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={(_, value) => value && setViewMode(value)}
            size="small"
          >
            <ToggleButton value="table">
              <TableChart fontSize="small" />
            </ToggleButton>
            <ToggleButton value="chart">
              <ShowChart fontSize="small" />
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>
      </Box>

      {/* Shape Comparison */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid size={{ xs: 6 }}>
          <Card variant="outlined">
            <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Typography variant="caption" color="text.secondary" gutterBottom>
                Before
              </Typography>
              <Typography variant="h6" fontWeight={600}>
                {previewData.before.shape[0]} × {previewData.before.shape[1]}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                rows × columns
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 6 }}>
          <Card variant="outlined">
            <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Typography variant="caption" color="text.secondary" gutterBottom>
                After
              </Typography>
              <Typography variant="h6" fontWeight={600}>
                {previewData.after.shape[0]} × {previewData.after.shape[1]}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                rows × columns
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Column Changes Alert */}
      {previewData.before.columns.length !== previewData.after.columns.length && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Columns changed: {previewData.before.columns.length} → {previewData.after.columns.length}
          {previewData.after.columns.length > previewData.before.columns.length && (
            <Chip
              icon={<TrendingUp />}
              label={`+${previewData.after.columns.length - previewData.before.columns.length} added`}
              size="small"
              color="success"
              sx={{ ml: 2 }}
            />
          )}
          {previewData.after.columns.length < previewData.before.columns.length && (
            <Chip
              icon={<TrendingDown />}
              label={`-${previewData.before.columns.length - previewData.after.columns.length} removed`}
              size="small"
              color="warning"
              sx={{ ml: 2 }}
            />
          )}
        </Alert>
      )}

      <Divider sx={{ mb: 2 }} />

      {/* Main Content */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {viewMode === 'table' ? (
          <Box sx={{ display: 'flex', gap: 2, height: '100%' }}>
            {/* Before Table */}
            <Box sx={{ flex: 1 }}>
              <Typography variant="subtitle2" gutterBottom fontWeight={600} color="text.secondary">
                Before Preprocessing
              </Typography>
              <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 400 }}>
                <Table size="small" stickyHeader>
                  <TableHead>
                    <TableRow>
                      {previewData.before.columns.slice(0, 6).map((col) => (
                        <TableCell key={col}>
                          <Box>
                            <Typography variant="caption" fontWeight={600}>
                              {col}
                            </Typography>
                            <br />
                            <Typography variant="caption" color="text.secondary">
                              {previewData.before.dtypes[col]}
                            </Typography>
                            {previewData.before.nullCounts[col] > 0 && (
                              <Chip
                                icon={<RemoveCircle />}
                                label={`${previewData.before.nullCounts[col]} nulls`}
                                size="small"
                                color="warning"
                                sx={{ mt: 0.5, height: 18, fontSize: '0.65rem' }}
                              />
                            )}
                          </Box>
                        </TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {previewData.before.data.slice(0, 10).map((row, idx) => (
                      <TableRow key={idx} hover>
                        {previewData.before.columns.slice(0, 6).map((col) => (
                          <TableCell key={col}>
                            {row[col] == null ? (
                              <Typography variant="caption" color="text.secondary" fontStyle="italic">
                                null
                              </Typography>
                            ) : (
                              <Typography variant="caption">
                                {String(row[col]).length > 20
                                  ? String(row[col]).substring(0, 20) + '...'
                                  : String(row[col])}
                              </Typography>
                            )}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              {previewData.before.columns.length > 6 && (
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                  Showing 6 of {previewData.before.columns.length} columns
                </Typography>
              )}
            </Box>

            {/* After Table */}
            <Box sx={{ flex: 1 }}>
              <Typography variant="subtitle2" gutterBottom fontWeight={600} color="primary.main">
                After Preprocessing
              </Typography>
              <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 400 }}>
                <Table size="small" stickyHeader>
                  <TableHead>
                    <TableRow>
                      {previewData.after.columns.slice(0, 6).map((col) => (
                        <TableCell key={col}>
                          <Box>
                            <Typography variant="caption" fontWeight={600}>
                              {col}
                            </Typography>
                            <br />
                            <Typography variant="caption" color="text.secondary">
                              {previewData.after.dtypes[col]}
                            </Typography>
                            {previewData.after.nullCounts[col] > 0 && (
                              <Chip
                                icon={<RemoveCircle />}
                                label={`${previewData.after.nullCounts[col]} nulls`}
                                size="small"
                                color="warning"
                                sx={{ mt: 0.5, height: 18, fontSize: '0.65rem' }}
                              />
                            )}
                          </Box>
                        </TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {previewData.after.data.slice(0, 10).map((row, idx) => (
                      <TableRow key={idx} hover>
                        {previewData.after.columns.slice(0, 6).map((col) => (
                          <TableCell key={col}>
                            {row[col] == null ? (
                              <Typography variant="caption" color="text.secondary" fontStyle="italic">
                                null
                              </Typography>
                            ) : (
                              <Typography variant="caption">
                                {String(row[col]).length > 20
                                  ? String(row[col]).substring(0, 20) + '...'
                                  : String(row[col])}
                              </Typography>
                            )}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              {previewData.after.columns.length > 6 && (
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                  Showing 6 of {previewData.after.columns.length} columns
                </Typography>
              )}
            </Box>
          </Box>
        ) : (
          <Box>
            {selectedColumn && (
              <Box>
                {/* Statistics Cards for Numeric Columns */}
                {isNumericColumn(selectedColumn) && beforeStats && afterStats && (
                  <Grid container spacing={2} sx={{ mb: 3 }}>
                    <Grid size={{ xs: 6, md: 2 }}>
                      <Card variant="outlined">
                        <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                          <Typography variant="caption" color="text.secondary">
                            Mean
                          </Typography>
                          <Typography variant="body2" fontWeight={600} color="text.secondary">
                            {beforeStats.mean.toFixed(2)}
                          </Typography>
                          <Typography variant="h6" fontWeight={600} color="primary.main">
                            {afterStats.mean.toFixed(2)}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {((afterStats.mean - beforeStats.mean) / beforeStats.mean * 100).toFixed(1)}%
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid size={{ xs: 6, md: 2 }}>
                      <Card variant="outlined">
                        <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                          <Typography variant="caption" color="text.secondary">
                            Std Dev
                          </Typography>
                          <Typography variant="body2" fontWeight={600} color="text.secondary">
                            {beforeStats.std.toFixed(2)}
                          </Typography>
                          <Typography variant="h6" fontWeight={600} color="primary.main">
                            {afterStats.std.toFixed(2)}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {((afterStats.std - beforeStats.std) / beforeStats.std * 100).toFixed(1)}%
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid size={{ xs: 6, md: 2 }}>
                      <Card variant="outlined">
                        <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                          <Typography variant="caption" color="text.secondary">
                            Min
                          </Typography>
                          <Typography variant="body2" fontWeight={600} color="text.secondary">
                            {beforeStats.min.toFixed(2)}
                          </Typography>
                          <Typography variant="h6" fontWeight={600} color="primary.main">
                            {afterStats.min.toFixed(2)}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid size={{ xs: 6, md: 2 }}>
                      <Card variant="outlined">
                        <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                          <Typography variant="caption" color="text.secondary">
                            Max
                          </Typography>
                          <Typography variant="body2" fontWeight={600} color="text.secondary">
                            {beforeStats.max.toFixed(2)}
                          </Typography>
                          <Typography variant="h6" fontWeight={600} color="primary.main">
                            {afterStats.max.toFixed(2)}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid size={{ xs: 6, md: 2 }}>
                      <Card variant="outlined">
                        <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                          <Typography variant="caption" color="text.secondary">
                            Median
                          </Typography>
                          <Typography variant="body2" fontWeight={600} color="text.secondary">
                            {beforeStats.median.toFixed(2)}
                          </Typography>
                          <Typography variant="h6" fontWeight={600} color="primary.main">
                            {afterStats.median.toFixed(2)}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid size={{ xs: 6, md: 2 }}>
                      <Card variant="outlined">
                        <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                          <Typography variant="caption" color="text.secondary">
                            Count
                          </Typography>
                          <Typography variant="body2" fontWeight={600} color="text.secondary">
                            {beforeStats.count}
                          </Typography>
                          <Typography variant="h6" fontWeight={600} color="primary.main">
                            {afterStats.count}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>
                )}

                {/* Charts */}
                <Grid container spacing={2}>
                  {/* Before Chart */}
                  <Grid size={{ xs: 12, md: 6 }}>
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Typography variant="subtitle2" gutterBottom fontWeight={600} color="text.secondary">
                        Before: {selectedColumn}
                      </Typography>
                      <ResponsiveContainer width="100%" height={300}>
                        {isNumericColumn(selectedColumn) ? (
                          <BarChart data={getHistogramData(previewData.before.data, selectedColumn)}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="range" angle={-45} textAnchor="end" height={80} />
                            <YAxis />
                            <RechartsTooltip />
                            <Legend />
                            <Bar dataKey="count" fill="#8884d8" name="Frequency" />
                          </BarChart>
                        ) : (
                          <PieChart>
                            <Pie
                              data={getValueDistribution(previewData.before.data, selectedColumn)}
                              dataKey="value"
                              nameKey="name"
                              cx="50%"
                              cy="50%"
                              outerRadius={80}
                              label
                            >
                              {getValueDistribution(previewData.before.data, selectedColumn).map((_, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                              ))}
                            </Pie>
                            <RechartsTooltip />
                            <Legend />
                          </PieChart>
                        )}
                      </ResponsiveContainer>
                    </Paper>
                  </Grid>

                  {/* After Chart */}
                  <Grid size={{ xs: 12, md: 6 }}>
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Typography variant="subtitle2" gutterBottom fontWeight={600} color="primary.main">
                        After: {selectedColumn}
                      </Typography>
                      <ResponsiveContainer width="100%" height={300}>
                        {isNumericColumn(selectedColumn) ? (
                          <BarChart data={getHistogramData(previewData.after.data, selectedColumn)}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="range" angle={-45} textAnchor="end" height={80} />
                            <YAxis />
                            <RechartsTooltip />
                            <Legend />
                            <Bar dataKey="count" fill="#82ca9d" name="Frequency" />
                          </BarChart>
                        ) : (
                          <PieChart>
                            <Pie
                              data={getValueDistribution(previewData.after.data, selectedColumn)}
                              dataKey="value"
                              nameKey="name"
                              cx="50%"
                              cy="50%"
                              outerRadius={80}
                              label
                            >
                              {getValueDistribution(previewData.after.data, selectedColumn).map((_, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                              ))}
                            </Pie>
                            <RechartsTooltip />
                            <Legend />
                          </PieChart>
                        )}
                      </ResponsiveContainer>
                    </Paper>
                  </Grid>
                </Grid>
              </Box>
            )}
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default PreviewPanel;
