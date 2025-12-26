import React, { useState } from 'react';
import { Box, Typography, Tooltip, Paper } from '@mui/material';

interface CorrelationHeatmapProps {
  features: string[];
  matrix: number[][];
  width?: number;
  height?: number;
}

const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({
  features,
  matrix,
  width = 600,
  height = 600,
}) => {
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);

  // Calculate cell dimensions
  const cellSize = Math.min(width / features.length, height / features.length);
  const actualWidth = cellSize * features.length;
  const actualHeight = cellSize * features.length;
  const labelWidth = 120;
  const labelHeight = 120;

  // Color scale function: -1 (red) to 0 (white) to 1 (blue)
  const getColor = (value: number): string => {
    if (value === null || value === undefined || isNaN(value)) {
      return '#f0f0f0';
    }

    const absValue = Math.abs(value);

    if (value > 0) {
      // Positive correlation: white to blue
      const intensity = Math.floor(255 - absValue * 200);
      return `rgb(${intensity}, ${intensity}, 255)`;
    } else if (value < 0) {
      // Negative correlation: white to red
      const intensity = Math.floor(255 - absValue * 200);
      return `rgb(255, ${intensity}, ${intensity})`;
    } else {
      // Zero correlation: white
      return '#ffffff';
    }
  };

  // Format correlation value
  const formatValue = (value: number): string => {
    if (value === null || value === undefined || isNaN(value)) {
      return 'N/A';
    }
    return value.toFixed(3);
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        width: '100%',
        overflowX: 'auto',
        pb: 2,
      }}
    >
      <Box
        sx={{
          display: 'flex',
          position: 'relative',
        }}
      >
        {/* Top labels */}
        <Box sx={{ width: labelWidth, height: labelHeight }} />
        <Box
          sx={{
            width: actualWidth,
            height: labelHeight,
            position: 'relative',
          }}
        >
          {features.map((feature, i) => (
            <Box
              key={`top-${i}`}
              sx={{
                position: 'absolute',
                left: i * cellSize + cellSize / 2,
                bottom: 0,
                transform: 'rotate(-45deg)',
                transformOrigin: 'left bottom',
                whiteSpace: 'nowrap',
                fontSize: '0.75rem',
                fontWeight: 500,
                color: 'text.primary',
              }}
            >
              {feature}
            </Box>
          ))}
        </Box>
      </Box>

      <Box sx={{ display: 'flex' }}>
        {/* Left labels */}
        <Box
          sx={{
            width: labelWidth,
            height: actualHeight,
            position: 'relative',
          }}
        >
          {features.map((feature, i) => (
            <Box
              key={`left-${i}`}
              sx={{
                position: 'absolute',
                right: 8,
                top: i * cellSize + cellSize / 2,
                transform: 'translateY(-50%)',
                textAlign: 'right',
                fontSize: '0.75rem',
                fontWeight: 500,
                color: 'text.primary',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                maxWidth: labelWidth - 16,
              }}
            >
              {feature}
            </Box>
          ))}
        </Box>

        {/* Heatmap grid */}
        <Box
          sx={{
            width: actualWidth,
            height: actualHeight,
            border: '1px solid #e0e0e0',
            position: 'relative',
          }}
        >
          {matrix.map((row, i) =>
            row.map((value, j) => {
              const isHovered =
                hoveredCell?.row === i && hoveredCell?.col === j;

              return (
                <Tooltip
                  key={`${i}-${j}`}
                  title={
                    <Box>
                      <Typography variant="caption" display="block">
                        {features[i]} × {features[j]}
                      </Typography>
                      <Typography variant="caption" display="block" fontWeight={600}>
                        Correlation: {formatValue(value)}
                      </Typography>
                    </Box>
                  }
                  placement="top"
                  arrow
                >
                  <Box
                    sx={{
                      position: 'absolute',
                      left: j * cellSize,
                      top: i * cellSize,
                      width: cellSize,
                      height: cellSize,
                      backgroundColor: getColor(value),
                      border: isHovered
                        ? '2px solid #000'
                        : '1px solid rgba(0, 0, 0, 0.1)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      '&:hover': {
                        transform: 'scale(1.05)',
                        zIndex: 10,
                        boxShadow: '0 4px 8px rgba(0,0,0,0.2)',
                      },
                    }}
                    onMouseEnter={() => setHoveredCell({ row: i, col: j })}
                    onMouseLeave={() => setHoveredCell(null)}
                  >
                    {cellSize > 40 && (
                      <Typography
                        variant="caption"
                        sx={{
                          fontSize: cellSize > 60 ? '0.7rem' : '0.6rem',
                          fontWeight: 600,
                          color:
                            Math.abs(value) > 0.5
                              ? '#ffffff'
                              : 'text.primary',
                          textShadow:
                            Math.abs(value) > 0.5
                              ? '0 1px 2px rgba(0,0,0,0.3)'
                              : 'none',
                        }}
                      >
                        {formatValue(value)}
                      </Typography>
                    )}
                  </Box>
                </Tooltip>
              );
            })
          )}
        </Box>
      </Box>

      {/* Legend */}
      <Box sx={{ mt: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
        <Typography variant="caption" fontWeight={600}>
          Correlation:
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: 200,
              height: 20,
              background:
                'linear-gradient(to right, rgb(255, 55, 55), rgb(255, 155, 155), rgb(255, 255, 255), rgb(155, 155, 255), rgb(55, 55, 255))',
              border: '1px solid #e0e0e0',
              borderRadius: 1,
            }}
          />
          <Box sx={{ display: 'flex', justifyContent: 'space-between', width: 200 }}>
            <Typography variant="caption" sx={{ fontSize: '0.65rem' }}>
              -1.0
            </Typography>
            <Typography variant="caption" sx={{ fontSize: '0.65rem' }}>
              0.0
            </Typography>
            <Typography variant="caption" sx={{ fontSize: '0.65rem' }}>
              +1.0
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* Interpretation guide */}
      <Paper
        sx={{
          mt: 2,
          p: 2,
          backgroundColor: '#f8fafc',
          border: '1px solid #e2e8f0',
          width: '100%',
          maxWidth: actualWidth + labelWidth,
        }}
      >
        <Typography variant="caption" fontWeight={600} display="block" gutterBottom>
          Interpretation Guide:
        </Typography>
        <Box sx={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 1, fontSize: '0.7rem' }}>
          <Typography variant="caption" color="primary">Strong positive (0.7 to 1.0):</Typography>
          <Typography variant="caption" color="text.secondary">
            Features increase together
          </Typography>

          <Typography variant="caption" color="info.main">Moderate positive (0.3 to 0.7):</Typography>
          <Typography variant="caption" color="text.secondary">
            Weak positive relationship
          </Typography>

          <Typography variant="caption">Weak (-0.3 to 0.3):</Typography>
          <Typography variant="caption" color="text.secondary">
            Little to no linear relationship
          </Typography>

          <Typography variant="caption" color="warning.main">Moderate negative (-0.7 to -0.3):</Typography>
          <Typography variant="caption" color="text.secondary">
            Weak inverse relationship
          </Typography>

          <Typography variant="caption" color="error">Strong negative (-1.0 to -0.7):</Typography>
          <Typography variant="caption" color="text.secondary">
            Features move in opposite directions
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
};

export default CorrelationHeatmap;
