import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Chip,
  Alert,
  LinearProgress,
  TextField,
  InputAdornment,
  ToggleButtonGroup,
  ToggleButton,
} from '@mui/material';
import {
  BarChart,
  ScatterPlot,
  ShowChart,
  GridOn,
  ErrorOutline,
  Search,
  Fullscreen,
  Close,
  Download,
  ViewModule,
  ViewList,
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import type { VisualizationData } from '../../types/dataset';

interface VisualizationGalleryProps {
  visualizations: VisualizationData[];
  isLoading?: boolean;
  error?: string | null;
  onVisualizationSelect?: (viz: VisualizationData) => void;
  maxColumns?: number;
}

interface VizCardProps {
  visualization: VisualizationData;
  onExpand: () => void;
  onDownload: () => void;
}

const VizCard: React.FC<VizCardProps> = ({ visualization, onExpand, onDownload }) => {
  const getVisualizationIcon = (type: string) => {
    switch (type) {
      case 'histogram':
        return <BarChart sx={{ fontSize: 20 }} />;
      case 'scatter':
        return <ScatterPlot sx={{ fontSize: 20 }} />;
      case 'box':
        return <ShowChart sx={{ fontSize: 20 }} />;
      case 'correlation':
        return <GridOn sx={{ fontSize: 20 }} />;
      case 'missing':
        return <ErrorOutline sx={{ fontSize: 20 }} />;
      default:
        return <BarChart sx={{ fontSize: 20 }} />;
    }
  };

  const getVisualizationColor = (type: string) => {
    switch (type) {
      case 'histogram':
        return '#3B82F6';
      case 'scatter':
        return '#10B981';
      case 'box':
        return '#8B5CF6';
      case 'correlation':
        return '#F59E0B';
      case 'missing':
        return '#EF4444';
      default:
        return '#6B7280';
    }
  };

  const color = getVisualizationColor(visualization.type);

  return (
    <Card
      sx={{
        height: '100%',
        border: '1px solid', borderColor: 'divider',
        bgcolor: 'background.paper',
        transition: 'all 0.3s ease',
        display: 'flex',
        flexDirection: 'column',
        '&:hover': {
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
          transform: 'translateY(-2px)',
        },
      }}
    >
      <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', p: 2 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box
              sx={{
                p: 0.75,
                borderRadius: 1.5,
                background: `${color}20`,
                display: 'flex',
                alignItems: 'center',
                color: color,
              }}
            >
              {getVisualizationIcon(visualization.type)}
            </Box>
            <Chip
              label={visualization.type.toUpperCase()}
              size="small"
              sx={{
                background: `${color}20`,
                color: color,
                fontWeight: 600,
                fontSize: '0.7rem',
              }}
            />
          </Box>
          <Box sx={{ display: 'flex', gap: 0.5 }}>
            <IconButton
              size="small"
              onClick={onDownload}
              sx={{
                color: 'text.secondary',
                '&:hover': { color: 'primary.main' },
              }}
            >
              <Download sx={{ fontSize: 18 }} />
            </IconButton>
            <IconButton
              size="small"
              onClick={onExpand}
              sx={{
                color: 'text.secondary',
                '&:hover': { color: 'primary.main' },
              }}
            >
              <Fullscreen sx={{ fontSize: 18 }} />
            </IconButton>
          </Box>
        </Box>

        {/* Title */}
        <Typography
          variant="subtitle1"
          fontWeight={600}
          sx={{
            mb: 2,
            color: '#1e293b',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
          }}
        >
          {visualization.title}
        </Typography>

        {/* Visualization */}
        <Box
          sx={{
            flex: 1,
            minHeight: 250,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderRadius: 1,
            bgcolor: 'background.default',
            overflow: 'hidden',
          }}
        >
          <Plot
            data={visualization.data}
            layout={{
              autosize: true,
              margin: { l: 40, r: 20, t: 20, b: 40 },
              xaxis: { title: visualization.xLabel },
              yaxis: { title: visualization.yLabel },
              showlegend: false,
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
            }}
            config={{
              displayModeBar: false,
              responsive: true,
            }}
            style={{ width: '100%', height: '100%' }}
          />
        </Box>

        {/* Labels */}
        {(visualization.xLabel || visualization.yLabel) && (
          <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {visualization.xLabel && (
              <Chip
                label={`X: ${visualization.xLabel}`}
                size="small"
                variant="outlined"
                sx={{ fontSize: '0.7rem' }}
              />
            )}
            {visualization.yLabel && (
              <Chip
                label={`Y: ${visualization.yLabel}`}
                size="small"
                variant="outlined"
                sx={{ fontSize: '0.7rem' }}
              />
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

const VisualizationGallery: React.FC<VisualizationGalleryProps> = ({
  visualizations = [],
  isLoading = false,
  error = null,
  onVisualizationSelect,
  maxColumns = 3,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [expandedViz, setExpandedViz] = useState<VisualizationData | null>(null);

  // Filter visualizations
  const filteredVisualizations = visualizations.filter((viz) => {
    const matchesSearch =
      searchQuery === '' ||
      viz.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      viz.type.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesType = filterType === null || viz.type === filterType;

    return matchesSearch && matchesType;
  });

  // Get unique visualization types for filter
  const visualizationTypes = Array.from(new Set(visualizations.map((v) => v.type)));

  const handleExpand = (viz: VisualizationData) => {
    setExpandedViz(viz);
    if (onVisualizationSelect) {
      onVisualizationSelect(viz);
    }
  };

  const handleDownload = (viz: VisualizationData) => {
    // Create a download link for the visualization data
    const dataStr = JSON.stringify(viz, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${viz.title.replace(/\s+/g, '_')}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  if (isLoading) {
    return (
      <Box>
        <Typography variant="h6" fontWeight={600} gutterBottom>
          Visualizations
        </Typography>
        <LinearProgress sx={{ mb: 2 }} />
        <Typography variant="body2" color="text.secondary">
          Generating visualizations...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box>
        <Typography variant="h6" fontWeight={600} gutterBottom>
          Visualizations
        </Typography>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  if (visualizations.length === 0) {
    return (
      <Box>
        <Typography variant="h6" fontWeight={600} gutterBottom>
          Visualizations
        </Typography>
        <Alert severity="info">
          No visualizations available. Upload and analyze a dataset to generate visualizations.
        </Alert>
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <BarChart sx={{ color: 'primary.main', fontSize: 32 }} />
          <Box>
            <Typography variant="h6" fontWeight={600}>
              Visualizations
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {filteredVisualizations.length} of {visualizations.length} visualization
              {visualizations.length !== 1 ? 's' : ''}
            </Typography>
          </Box>
        </Box>

        {/* View Mode Toggle */}
        <ToggleButtonGroup
          value={viewMode}
          exclusive
          onChange={(_, newMode) => newMode && setViewMode(newMode)}
          size="small"
        >
          <ToggleButton value="grid">
            <ViewModule sx={{ fontSize: 18 }} />
          </ToggleButton>
          <ToggleButton value="list">
            <ViewList sx={{ fontSize: 18 }} />
          </ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {/* Controls */}
      <Box sx={{ mb: 3, display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'center' }}>
        {/* Search */}
        <TextField
          size="small"
          placeholder="Search visualizations..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          slotProps={{
            input: {
              startAdornment: (
                <InputAdornment position="start">
                  <Search sx={{ fontSize: 20, color: 'text.secondary' }} />
                </InputAdornment>
              ),
            },
          }}
          sx={{ flex: 1, minWidth: 250 }}
        />

        {/* Filter by Type */}
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Chip
            label="All"
            onClick={() => setFilterType(null)}
            color={filterType === null ? 'primary' : 'default'}
            size="small"
            sx={{ fontWeight: filterType === null ? 600 : 400 }}
          />
          {visualizationTypes.map((type) => (
            <Chip
              key={type}
              label={type.toUpperCase()}
              onClick={() => setFilterType(type)}
              color={filterType === type ? 'primary' : 'default'}
              size="small"
              sx={{ fontWeight: filterType === type ? 600 : 400 }}
            />
          ))}
        </Box>
      </Box>

      {/* Empty State After Filtering */}
      {filteredVisualizations.length === 0 && (
        <Alert severity="info" sx={{ mb: 3 }}>
          No visualizations match your search criteria. Try adjusting your filters.
        </Alert>
      )}

      {/* Visualization Grid */}
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: {
            xs: '1fr',
            sm: viewMode === 'grid' ? 'repeat(2, 1fr)' : '1fr',
            md: viewMode === 'grid' ? `repeat(${Math.min(maxColumns, 3)}, 1fr)` : '1fr',
          },
          gap: 3,
        }}
      >
        {filteredVisualizations.map((viz, idx) => (
          <VizCard
            key={idx}
            visualization={viz}
            onExpand={() => handleExpand(viz)}
            onDownload={() => handleDownload(viz)}
          />
        ))}
      </Box>

      {/* Expanded View Dialog */}
      <Dialog
        open={expandedViz !== null}
        onClose={() => setExpandedViz(null)}
        maxWidth="lg"
        fullWidth
      >
        {expandedViz && (
          <>
            <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box sx={{ flex: 1 }}>
                <Typography variant="h6" fontWeight={600}>
                  {expandedViz.title}
                </Typography>
                <Chip
                  label={expandedViz.type.toUpperCase()}
                  size="small"
                  sx={{ mt: 1 }}
                  color="primary"
                />
              </Box>
              <IconButton onClick={() => setExpandedViz(null)}>
                <Close />
              </IconButton>
            </DialogTitle>
            <DialogContent>
              <Box
                sx={{
                  minHeight: 500,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Plot
                  data={expandedViz.data}
                  layout={{
                    autosize: true,
                    margin: { l: 60, r: 40, t: 40, b: 60 },
                    xaxis: { title: expandedViz.xLabel },
                    yaxis: { title: expandedViz.yLabel },
                    showlegend: true,
                  }}
                  config={{
                    displayModeBar: true,
                    responsive: true,
                  }}
                  style={{ width: '100%', height: '500px' }}
                />
              </Box>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => handleDownload(expandedViz)} startIcon={<Download />}>
                Download Data
              </Button>
              <Button onClick={() => setExpandedViz(null)} variant="contained">
                Close
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Box>
  );
};

export default VisualizationGallery;
