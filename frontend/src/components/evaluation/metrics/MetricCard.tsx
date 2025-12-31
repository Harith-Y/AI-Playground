/**
 * MetricCard Component
 *
 * Displays a single metric with value, label, and optional trend/comparison
 */

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip as _Chip,
  Tooltip,
  LinearProgress,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Info as InfoIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';

/**
 * Metric card variant
 */
export const MetricVariant = {
  DEFAULT: 'default',
  SUCCESS: 'success',
  WARNING: 'warning',
  ERROR: 'error',
} as const;

export type MetricVariant = (typeof MetricVariant)[keyof typeof MetricVariant];

/**
 * Props for MetricCard component
 */
export interface MetricCardProps {
  label: string;
  value: number | string;
  description?: string;
  variant?: MetricVariant;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: number;
  format?: 'number' | 'percentage' | 'decimal' | 'integer';
  precision?: number;
  unit?: string;
  min?: number;
  max?: number;
  showProgress?: boolean;
  icon?: React.ReactNode;
  tooltip?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  description,
  variant = MetricVariant.DEFAULT,
  trend,
  trendValue,
  format = 'decimal',
  precision = 2,
  unit,
  min,
  max,
  showProgress = false,
  icon,
  tooltip,
}) => {
  // Format the metric value
  const formatValue = (val: number | string): string => {
    if (typeof val === 'string') return val;

    switch (format) {
      case 'percentage':
        return `${(val * 100).toFixed(precision)}%`;
      case 'decimal':
        return val.toFixed(precision);
      case 'integer':
        return Math.round(val).toString();
      case 'number':
      default:
        return val.toLocaleString(undefined, {
          minimumFractionDigits: precision,
          maximumFractionDigits: precision,
        });
    }
  };

  // Get variant color
  const getVariantColor = () => {
    switch (variant) {
      case MetricVariant.SUCCESS:
        return 'success.main';
      case MetricVariant.WARNING:
        return 'warning.main';
      case MetricVariant.ERROR:
        return 'error.main';
      default:
        return 'primary.main';
    }
  };

  // Get variant icon
  const getVariantIcon = () => {
    switch (variant) {
      case MetricVariant.SUCCESS:
        return <CheckCircleIcon fontSize="small" />;
      case MetricVariant.WARNING:
        return <WarningIcon fontSize="small" />;
      case MetricVariant.ERROR:
        return <ErrorIcon fontSize="small" />;
      default:
        return null;
    }
  };

  // Calculate progress percentage
  const getProgressPercentage = (): number => {
    if (typeof value !== 'number' || min === undefined || max === undefined) {
      return 0;
    }
    return ((value - min) / (max - min)) * 100;
  };

  // Get trend color
  const getTrendColor = () => {
    if (trend === 'up') return 'success.main';
    if (trend === 'down') return 'error.main';
    return 'text.secondary';
  };

  const formattedValue = formatValue(value);
  const displayValue = unit ? `${formattedValue} ${unit}` : formattedValue;

  return (
    <Card
      elevation={2}
      sx={{
        height: '100%',
        transition: 'transform 0.2s, box-shadow 0.2s',
        '&:hover': {
          transform: 'translateY(-2px)',
          boxShadow: 4,
        },
      }}
    >
      <CardContent>
        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={1}>
          <Box display="flex" alignItems="center" gap={0.5}>
            <Typography variant="caption" color="text.secondary" fontWeight="500">
              {label}
            </Typography>
            {tooltip && (
              <Tooltip title={tooltip} arrow>
                <InfoIcon sx={{ fontSize: 14, color: 'text.disabled', cursor: 'help' }} />
              </Tooltip>
            )}
          </Box>
          {(getVariantIcon() || icon) && (
            <Box sx={{ color: getVariantColor() }}>
              {icon || getVariantIcon()}
            </Box>
          )}
        </Box>

        {/* Value */}
        <Box display="flex" alignItems="baseline" gap={1} mb={1}>
          <Typography
            variant="h4"
            fontWeight="bold"
            sx={{ color: getVariantColor() }}
          >
            {displayValue}
          </Typography>

          {/* Trend Indicator */}
          {trend && (
            <Box display="flex" alignItems="center" gap={0.5}>
              {trend === 'up' && <TrendingUpIcon sx={{ color: getTrendColor(), fontSize: 20 }} />}
              {trend === 'down' && <TrendingDownIcon sx={{ color: getTrendColor(), fontSize: 20 }} />}
              {trendValue !== undefined && (
                <Typography variant="caption" sx={{ color: getTrendColor() }}>
                  {trendValue > 0 ? '+' : ''}{trendValue.toFixed(1)}%
                </Typography>
              )}
            </Box>
          )}
        </Box>

        {/* Description */}
        {description && (
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1, fontSize: '0.75rem' }}>
            {description}
          </Typography>
        )}

        {/* Progress Bar */}
        {showProgress && min !== undefined && max !== undefined && typeof value === 'number' && (
          <Box mt={1}>
            <LinearProgress
              variant="determinate"
              value={Math.min(100, Math.max(0, getProgressPercentage()))}
              sx={{
                height: 6,
                borderRadius: 3,
                backgroundColor: 'grey.200',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: getVariantColor(),
                },
              }}
            />
            <Box display="flex" justifyContent="space-between" mt={0.5}>
              <Typography variant="caption" color="text.secondary">
                {formatValue(min)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {formatValue(max)}
              </Typography>
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default MetricCard;
