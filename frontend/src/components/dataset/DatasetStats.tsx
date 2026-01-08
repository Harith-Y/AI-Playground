import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Chip,
  LinearProgress,
  Alert,
} from '@mui/material';
import {
  TableChart,
  ViewColumn,
  Numbers,
  Category,
  ErrorOutline,
  ContentCopy,
  Memory,
} from '@mui/icons-material';
import type { DatasetStats as DatasetStatsType, ColumnInfo } from '../../types/dataset';

interface DatasetStatsProps {
  stats?: DatasetStatsType;
  columns?: ColumnInfo[];
  isLoading?: boolean;
  error?: string | null;
}

interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  color: string;
  subtext?: string;
}

const StatCard: React.FC<StatCardProps> = ({ icon, label, value, color, subtext }) => {
  return (
    <Card
      sx={{
        height: '100%',
        border: '1px solid', borderColor: 'divider',
        bgcolor: 'background.paper',
        transition: 'all 0.3s ease',
        '&:hover': {
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
          transform: 'translateY(-2px)',
        },
      }}
    >
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
          <Box
            sx={{
              p: 1.5,
              borderRadius: 2,
              background: `${color}20`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {icon}
          </Box>
          <Box sx={{ flex: 1 }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
              {label}
            </Typography>
            <Typography variant="h4" fontWeight={700} sx={{ color: '#1e293b' }}>
              {value.toLocaleString()}
            </Typography>
            {subtext && (
              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
                {subtext}
              </Typography>
            )}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

const DatasetStats: React.FC<DatasetStatsProps> = ({
  stats,
  columns = [],
  isLoading = false,
  error = null,
}) => {
  if (isLoading) {
    return (
      <Box>
        <Typography variant="h6" fontWeight={600} gutterBottom>
          Dataset Statistics
        </Typography>
        <LinearProgress sx={{ mb: 2 }} />
        <Typography variant="body2" color="text.secondary">
          Loading statistics...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box>
        <Typography variant="h6" fontWeight={600} gutterBottom>
          Dataset Statistics
        </Typography>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  if (!stats) {
    return (
      <Box>
        <Typography variant="h6" fontWeight={600} gutterBottom>
          Dataset Statistics
        </Typography>
        <Alert severity="info">No statistics available. Upload a dataset to see stats.</Alert>
      </Box>
    );
  }

  // Format memory usage
  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
  };

  // Calculate missing percentage
  const totalCells = stats.rowCount * stats.columnCount;
  const missingPercentage = totalCells > 0 ? (stats.missingValues / totalCells) * 100 : 0;

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
        <TableChart sx={{ color: 'primary.main', fontSize: 32 }} />
        <Box>
          <Typography variant="h6" fontWeight={600}>
            Dataset Statistics
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Overview of your dataset's structure and quality
          </Typography>
        </Box>
      </Box>

      {/* Stats Cards Grid */}
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: {
            xs: '1fr',
            sm: 'repeat(2, 1fr)',
            md: 'repeat(3, 1fr)',
            lg: 'repeat(4, 1fr)',
          },
          gap: 3,
          mb: 3,
        }}
      >
        <StatCard
          icon={<TableChart sx={{ color: '#3B82F6', fontSize: 28 }} />}
          label="Total Rows"
          value={stats.rowCount}
          color="#3B82F6"
          subtext="Number of records"
        />
        <StatCard
          icon={<ViewColumn sx={{ color: '#10B981', fontSize: 28 }} />}
          label="Total Columns"
          value={stats.columnCount}
          color="#10B981"
          subtext="Number of features"
        />
        <StatCard
          icon={<Numbers sx={{ color: '#8B5CF6', fontSize: 28 }} />}
          label="Numeric Columns"
          value={stats.numericColumns}
          color="#8B5CF6"
          subtext={`${((stats.numericColumns / stats.columnCount) * 100).toFixed(1)}% of columns`}
        />
        <StatCard
          icon={<Category sx={{ color: '#F59E0B', fontSize: 28 }} />}
          label="Categorical Columns"
          value={stats.categoricalColumns}
          color="#F59E0B"
          subtext={`${((stats.categoricalColumns / stats.columnCount) * 100).toFixed(1)}% of columns`}
        />
        <StatCard
          icon={<ErrorOutline sx={{ color: '#EF4444', fontSize: 28 }} />}
          label="Missing Values"
          value={stats.missingValues}
          color="#EF4444"
          subtext={`${missingPercentage.toFixed(2)}% of total cells`}
        />
        <StatCard
          icon={<ContentCopy sx={{ color: '#EC4899', fontSize: 28 }} />}
          label="Duplicate Rows"
          value={stats.duplicateRows}
          color="#EC4899"
          subtext={`${((stats.duplicateRows / stats.rowCount) * 100).toFixed(2)}% of rows`}
        />
        <StatCard
          icon={<Memory sx={{ color: '#6366F1', fontSize: 28 }} />}
          label="Memory Usage"
          value={formatBytes(stats.memoryUsage)}
          color="#6366F1"
          subtext="Estimated size in memory"
        />
      </Box>

      {/* Column Types Breakdown */}
      {columns.length > 0 && (
        <Card sx={{ border: '1px solid', borderColor: 'divider', bgcolor: 'background.paper' }}>
          <CardContent>
            <Typography variant="h6" fontWeight={600} gutterBottom>
              Column Types
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 2 }}>
              {columns.map((col, idx) => {
                // Determine color based on data type
                let chipColor = '#6B7280';
                const type = col.dataType.toLowerCase();
                if (type.includes('int') || type.includes('float')) {
                  chipColor = '#3B82F6';
                } else if (type.includes('object') || type.includes('str')) {
                  chipColor = '#10B981';
                } else if (type.includes('bool')) {
                  chipColor = '#F59E0B';
                } else if (type.includes('date') || type.includes('time')) {
                  chipColor = '#8B5CF6';
                }

                return (
                  <Chip
                    key={idx}
                    label={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <Typography variant="caption" fontWeight={600}>
                          {col.name}
                        </Typography>
                        <Typography variant="caption" sx={{ color: chipColor }}>
                          ({col.dataType})
                        </Typography>
                      </Box>
                    }
                    size="small"
                    variant="outlined"
                    sx={{
                      borderColor: '#e2e8f0',
                      '&:hover': {
                        borderColor: chipColor,
                        background: `${chipColor}10`,
                      },
                    }}
                  />
                );
              })}
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Data Quality Indicator */}
      <Card sx={{ mt: 3, border: '1px solid', borderColor: 'divider', bgcolor: 'background.paper' }}>
        <CardContent>
          <Typography variant="h6" fontWeight={600} gutterBottom>
            Data Quality Score
          </Typography>
          <Box sx={{ mt: 2 }}>
            {/* Calculate quality score */}
            {(() => {
              const missingScore = Math.max(0, 100 - missingPercentage * 2);
              const duplicateScore = Math.max(
                0,
                100 - (stats.duplicateRows / stats.rowCount) * 100 * 2
              );
              const qualityScore = (missingScore + duplicateScore) / 2;

              let scoreColor = '#10B981';
              let scoreLabel = 'Excellent';
              if (qualityScore < 50) {
                scoreColor = '#EF4444';
                scoreLabel = 'Poor';
              } else if (qualityScore < 75) {
                scoreColor = '#F59E0B';
                scoreLabel = 'Good';
              } else if (qualityScore < 90) {
                scoreColor = '#3B82F6';
                scoreLabel = 'Very Good';
              }

              return (
                <>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
                    <Typography variant="h3" fontWeight={700} sx={{ color: scoreColor }}>
                      {qualityScore.toFixed(0)}%
                    </Typography>
                    <Chip
                      label={scoreLabel}
                      size="small"
                      sx={{
                        background: `${scoreColor}20`,
                        color: scoreColor,
                        fontWeight: 600,
                      }}
                    />
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={qualityScore}
                    sx={{
                      height: 8,
                      borderRadius: 1,
                      bgcolor: 'action.hover',
                      '& .MuiLinearProgress-bar': {
                        background: scoreColor,
                        borderRadius: 1,
                      },
                    }}
                  />
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                    Based on missing values ({missingPercentage.toFixed(2)}%) and duplicate rows (
                    {((stats.duplicateRows / stats.rowCount) * 100).toFixed(2)}%)
                  </Typography>
                </>
              );
            })()}
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default DatasetStats;
