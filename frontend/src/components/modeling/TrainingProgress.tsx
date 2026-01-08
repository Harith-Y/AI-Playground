import React from 'react';
import {
  Box,
  LinearProgress,
  Typography,
  Card,
  CardContent,
  Chip,
  Paper,
} from '@mui/material';
import { Timeline, CheckCircle, Error as _ErrorIcon } from '@mui/icons-material';
import type { TrainingMetrics } from '../../types/model';

interface TrainingProgressProps {
  progress: number;
  isTraining: boolean;
  metrics?: TrainingMetrics | null;
  logs: string[];
}

const TrainingProgress: React.FC<TrainingProgressProps> = ({
  progress,
  isTraining,
  metrics,
  logs,
}) => {
  return (
    <Box sx={{ width: '100%' }}>
      {/* Progress Header */}
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Timeline sx={{ color: 'primary.main' }} />
            <Typography variant="h6" fontWeight={600}>
              Training Progress
            </Typography>
          </Box>
          <Chip
            icon={isTraining ? <Timeline /> : <CheckCircle />}
            label={isTraining ? 'Training...' : 'Completed'}
            size="small"
            color={isTraining ? 'primary' : 'success'}
            sx={{ fontWeight: 600 }}
          />
        </Box>
        <LinearProgress
          variant="determinate"
          value={progress}
          sx={{
            height: 8,
            borderRadius: 1,
            backgroundColor: '#e2e8f0',
            '& .MuiLinearProgress-bar': {
              borderRadius: 1,
            },
          }}
        />
        <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
          {progress.toFixed(1)}% Complete
        </Typography>
      </Box>

      {/* Training Metrics */}
      {metrics && (
        <Card sx={{ mb: 3, border: '1px solid', borderColor: 'divider' }}>
          <CardContent>
            <Typography variant="subtitle1" fontWeight={600} gutterBottom>
              Current Metrics
            </Typography>
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(3, 1fr)' },
                gap: 2,
                mt: 2,
              }}
            >
              {Object.entries(metrics).map(([key, value]) => (
                <Box key={key}>
                  <Typography variant="caption" color="text.secondary">
                    {key.replace(/_/g, ' ').toUpperCase()}
                  </Typography>
                  <Typography variant="h6" fontWeight={600}>
                    {typeof value === 'number' ? value.toFixed(4) : value}
                  </Typography>
                </Box>
              ))}
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Training Logs */}
      {logs.length > 0 && (
        <Box>
          <Typography variant="subtitle1" fontWeight={600} gutterBottom>
            Training Logs
          </Typography>
          <Paper
            sx={{
              p: 2,
              maxHeight: 300,
              overflowY: 'auto',
              backgroundColor: '#f8fafc',
              border: '1px solid', borderColor: 'divider',
              fontFamily: 'monospace',
            }}
          >
            {logs.map((log, index) => (
              <Typography
                key={index}
                variant="caption"
                component="div"
                sx={{
                  mb: 0.5,
                  color: log.includes('error') || log.includes('Error') ? 'error.main' : 'text.primary',
                }}
              >
                {log}
              </Typography>
            ))}
          </Paper>
        </Box>
      )}
    </Box>
  );
};

export default TrainingProgress;
