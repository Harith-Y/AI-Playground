/**
 * Plot Viewer Component
 * 
 * Dynamically renders different types of evaluation plots (ROC curve, confusion matrix, residual plots, etc.)
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  ToggleButtonGroup,
  ToggleButton,
  IconButton,
  Tooltip,
  Skeleton,
  Button,
} from '@mui/material';
import {
  ShowChart as ShowChartIcon,
  GridOn as GridOnIcon,
  ScatterPlot as ScatterPlotIcon,
  Download as DownloadIcon,
  ErrorOutline as ErrorOutlineIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import { evaluationService } from '../../services/evaluationService';
import type { PlotDataResponse } from '../../services/evaluationService';

// Define plot types locally
export type PlotType = 
  | 'roc_curve' 
  | 'confusion_matrix' 
  | 'precision_recall_curve' 
  | 'residuals' 
  | 'learning_curve' 
  | 'calibration_curve' 
  | 'feature_importance';

interface PlotViewerProps {
  modelRunId: string;
  availablePlots?: PlotType[];
  defaultPlot?: PlotType;
}

const PlotViewer: React.FC<PlotViewerProps> = ({
  modelRunId,
  availablePlots = ['roc_curve', 'confusion_matrix', 'residuals'],
  defaultPlot = 'roc_curve',
}) => {
  // Ensure defaultPlot is valid given availablePlots
  const effectiveDefaultPlot = availablePlots.includes(defaultPlot) 
    ? defaultPlot 
    : availablePlots[0] || 'roc_curve';

  const [selectedPlot, setSelectedPlot] = useState<PlotType>(effectiveDefaultPlot);
  const [plotData, setPlotData] = useState<PlotDataResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Update selected plot if availablePlots changes and current selection is invalid
  useEffect(() => {
    if (!availablePlots.includes(selectedPlot)) {
      setSelectedPlot(availablePlots[0] || 'roc_curve');
    }
  }, [availablePlots, selectedPlot]);

  useEffect(() => {
    fetchPlotData(selectedPlot);
  }, [modelRunId, selectedPlot]);

  const fetchPlotData = async (plotType: PlotType) => {
    setLoading(true);
    setError(null);

    try {
      const data = await evaluationService.getPlotData(modelRunId, plotType);
      
      // Adaptation for simple structure { x: [], y: [] } returned by residuals logic in backend
      // Cast to any to access potentially missing properties from strict interface
      const rawData = data as any;
      if (rawData.plot_type === 'residuals' && rawData.data && Array.isArray(rawData.data.x) && !Array.isArray(rawData.plot_data)) {
        // Construct Plotly trace for residuals manually
        const scatterTrace = {
          x: rawData.data.x,
          y: rawData.data.y,
          mode: 'markers',
          type: 'scatter',
          name: 'Residuals',
          marker: { color: '#ff9800', opacity: 0.7 }
        };
        
        // Calculate axis ranges properly to ensure data is visible
        const xValues = rawData.data.x as number[];
        const yValues = rawData.data.y as number[];
        
        // Handle empty array case defensively
        let xMin = -1, xMax = 1;
        if (xValues.length > 0) {
            xMin = Math.min(...xValues);
            xMax = Math.max(...xValues);
        }
        
        // Add padding to range
        const xRange = xMax - xMin;
        const xBuffer = xRange > 0 ? xRange * 0.1 : 1;
        
        const zeroLine = {
            x: [xMin - xBuffer, xMax + xBuffer],
            y: [0, 0],
            mode: 'lines',
            type: 'scatter',
            name: 'Zero Line',
            line: { color: 'black', dash: 'dash' }
        };
        
        setPlotData({
          plot_type: 'residuals',
          plot_data: [scatterTrace, zeroLine],
          layout: {
            title: 'Residuals vs Predicted',
            xaxis: { 
                title: 'Predicted Values', 
                autorange: true 
            },
            yaxis: { 
                title: 'Residuals', 
                autorange: true 
            },
            hovermode: 'closest',
            showlegend: true
          }
        });
      } else {
        setPlotData(data);
      }

    } catch (err: any) {
      console.error(`Error fetching ${plotType} plot:`, err);
      
      // If backend doesn't have the plot, show not implemented message
      if (err.response?.status === 404 || err.response?.status === 501) {
        // Special case for Learning Curve which might not be implemented yet
        if (plotType === 'learning_curve') {
             setError('Learning curve generation is currently in development.');
        } else if (plotType === 'residuals') {
             setError('Residuals data was not generated during training.');
        } else {
             setError(`Plot generation is not available. This feature is coming soon.`);
        }
      } else {
        setError(err.response?.data?.detail || 'Failed to load plot data');
      }
    } finally {
      setLoading(false);
    }
  };

  const generatePlotLocally = async (plotType: PlotType): Promise<PlotDataResponse> => {
    // For demonstration, generate sample plot data
    // In production, this would use actual model predictions

    switch (plotType) {
      case 'roc_curve':
        const rocPlot = evaluationService.generateROCCurvePlot(
          Array.from({ length: 100 }, (_, i) => i / 100),
          Array.from({ length: 100 }, (_, i) => Math.pow(i / 100, 0.8)),
          0.95
        );
        return {
          plot_type: 'roc_curve',
          plot_data: rocPlot.data,
          layout: rocPlot.layout,
        };

      case 'confusion_matrix':
        const cmPlot = evaluationService.generateConfusionMatrixPlot(
          [[85, 15], [10, 90]],
          ['Class 0', 'Class 1']
        );
        return {
          plot_type: 'confusion_matrix',
          plot_data: cmPlot.data,
          layout: cmPlot.layout,
        };

      case 'residuals':
        const predictions = Array.from({ length: 100 }, () => Math.random() * 100);
        const actuals = predictions.map(p => p + (Math.random() - 0.5) * 20);
        const residualPlot = evaluationService.generateResidualPlot(predictions, actuals);
        return {
          plot_type: 'residuals',
          plot_data: residualPlot.data,
          layout: residualPlot.layout,
        };

      case 'precision_recall_curve':
        return {
          plot_type: 'precision_recall_curve',
          plot_data: [{
            type: 'scatter',
            x: Array.from({ length: 100 }, (_, i) => 1 - i / 100),
            y: Array.from({ length: 100 }, (_, i) => Math.pow(1 - i / 100, 1.2)),
            mode: 'lines',
            name: 'PR Curve',
            line: { color: '#2196f3', width: 2 },
          }],
          layout: {
            title: 'Precision-Recall Curve',
            xaxis: { title: 'Recall', range: [0, 1] },
            yaxis: { title: 'Precision', range: [0, 1] },
          },
        };

      case 'learning_curve':
        return {
          plot_type: 'learning_curve',
          plot_data: [
            {
              type: 'scatter',
              x: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
              y: [0.65, 0.72, 0.76, 0.79, 0.82, 0.84, 0.86, 0.87, 0.88, 0.89],
              mode: 'lines+markers',
              name: 'Training Score',
              line: { color: '#4caf50' },
            },
            {
              type: 'scatter',
              x: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
              y: [0.60, 0.68, 0.73, 0.76, 0.78, 0.80, 0.81, 0.82, 0.83, 0.84],
              mode: 'lines+markers',
              name: 'Validation Score',
              line: { color: '#ff9800' },
            },
          ],
          layout: {
            title: 'Learning Curve',
            xaxis: { title: 'Training Examples (%)' },
            yaxis: { title: 'Score', range: [0, 1] },
          },
        };

      default:
        throw new Error(`Unsupported plot type: ${plotType}`);
    }
  };

  const handlePlotChange = (_event: React.MouseEvent<HTMLElement>, newPlot: PlotType | null) => {
    if (newPlot !== null) {
      setSelectedPlot(newPlot);
    }
  };

  const handleDownload = () => {
    if (!plotData) return;

    // Create downloadable image using Plotly's built-in functionality
    const plotElement = document.querySelector('.js-plotly-plot');
    if (plotElement) {
      // @ts-ignore
      Plotly.downloadImage(plotElement, {
        format: 'png',
        width: 1200,
        height: 800,
        filename: `${selectedPlot}-${modelRunId}`,
      });
    }
  };

  const getPlotIcon = (plotType: PlotType) => {
    switch (plotType) {
      case 'confusion_matrix':
        return <GridOnIcon />;
      case 'residuals':
        return <ScatterPlotIcon />;
      default:
        return <ShowChartIcon />;
    }
  };

  const getPlotLabel = (plotType: PlotType): string => {
    switch (plotType) {
      case 'roc_curve':
        return 'ROC Curve';
      case 'confusion_matrix':
        return 'Confusion Matrix';
      case 'precision_recall_curve':
        return 'PR Curve';
      case 'residuals':
        return 'Residuals';
      case 'learning_curve':
        return 'Learning Curve';
      case 'calibration_curve':
        return 'Calibration';
      case 'feature_importance':
        return 'Feature Importance';
      default:
        return plotType;
    }
  };

  return (
    <Card>
      <CardContent>
        {/* Header with Plot Selector */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">
            Evaluation Plots
          </Typography>
          <Box display="flex" gap={1}>
            <Tooltip title="Download Plot">
              <span>
                <IconButton onClick={handleDownload} size="small" disabled={!plotData || loading}>
                  <DownloadIcon />
                </IconButton>
              </span>
            </Tooltip>
          </Box>
        </Box>

        {/* Plot Type Selector */}
        <Box mb={2}>
          <ToggleButtonGroup
            value={selectedPlot}
            exclusive
            onChange={handlePlotChange}
            size="small"
            fullWidth
          >
            {availablePlots.map((plotType) => (
              <ToggleButton key={plotType} value={plotType}>
                <Box display="flex" alignItems="center" gap={0.5}>
                  {getPlotIcon(plotType)}
                  <Typography variant="caption">
                    {getPlotLabel(plotType)}
                  </Typography>
                </Box>
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        </Box>

        {/* Plot Content */}
        {loading && (
          <Box>
            <Skeleton variant="rectangular" height={500} sx={{ borderRadius: 1 }} />
            <Box display="flex" justifyContent="space-between" mt={2}>
              <Skeleton variant="text" width="30%" />
              <Skeleton variant="text" width="30%" />
            </Box>
          </Box>
        )}

        {error && !loading && (
          <Box display="flex" flexDirection="column" alignItems="center" py={6}>
            <ErrorOutlineIcon color="warning" sx={{ fontSize: 56, mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              Plot Not Available
            </Typography>
            <Typography variant="body2" color="text.secondary" textAlign="center" mb={3}>
              {error}
            </Typography>
            <Button
              variant="outlined"
              size="small"
              startIcon={<RefreshIcon />}
              onClick={() => fetchPlotData(selectedPlot)}
            >
              Retry
            </Button>
          </Box>
        )}

        {plotData && !loading && !error && (
          <Box>
            <Plot
              data={plotData.plot_data}
              layout={{
                ...plotData.layout,
                autosize: true,
                margin: { l: 60, r: 40, t: 60, b: 60 },
              }}
              config={{
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
              }}
              style={{ width: '100%', height: '500px' }}
              useResizeHandler
            />

            {plotData.config && (
              <Box mt={2}>
                <Typography variant="caption" color="text.secondary">
                  {Object.entries(plotData.config).map(([key, value]) => (
                    <span key={key} style={{ marginRight: '16px' }}>
                      <strong>{key}:</strong> {String(value)}
                    </span>
                  ))}
                </Typography>
              </Box>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default PlotViewer;
