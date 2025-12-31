/**
 * Generation History Component
 * 
 * Displays history of code generations with quick access
 */

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemIcon,
  Chip,
  IconButton,
  Tooltip,
  Divider,
  Alert,
} from '@mui/material';
import {
  History as HistoryIcon,
  Code as CodeIcon,
  Description as DescriptionIcon,
  Api as ApiIcon,
  Download as DownloadIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';

interface GenerationHistoryItem {
  id: string;
  experimentName: string;
  datasetName: string;
  format: 'python' | 'notebook' | 'fastapi';
  timestamp: string;
}

interface GenerationHistoryProps {
  history: GenerationHistoryItem[];
  onSelect?: (item: GenerationHistoryItem) => void;
  onDownload?: (item: GenerationHistoryItem) => void;
  onDelete?: (id: string) => void;
  onClear?: () => void;
}

const GenerationHistory: React.FC<GenerationHistoryProps> = ({
  history,
  onSelect,
  onDownload,
  onDelete,
  onClear,
}) => {
  const getFormatIcon = (format: string) => {
    switch (format) {
      case 'python':
        return <CodeIcon fontSize="small" />;
      case 'notebook':
        return <DescriptionIcon fontSize="small" />;
      case 'fastapi':
        return <ApiIcon fontSize="small" />;
      default:
        return <CodeIcon fontSize="small" />;
    }
  };

  const getFormatLabel = (format: string) => {
    switch (format) {
      case 'python':
        return 'Python';
      case 'notebook':
        return 'Notebook';
      case 'fastapi':
        return 'FastAPI';
      default:
        return format;
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    });
  };

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Box display="flex" alignItems="center" gap={1}>
            <HistoryIcon color="primary" />
            <Typography variant="h6">Generation History</Typography>
          </Box>
          {history.length > 0 && onClear && (
            <Tooltip title="Clear history">
              <IconButton size="small" onClick={onClear}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          )}
        </Box>
        <Divider sx={{ mb: 2 }} />

        {history.length === 0 ? (
          <Alert severity="info">
            No generation history yet. Generated code will appear here for quick access.
          </Alert>
        ) : (
          <List disablePadding>
            {history.map((item, index) => (
              <React.Fragment key={item.id}>
                {index > 0 && <Divider />}
                <ListItem
                  disablePadding
                  secondaryAction={
                    <Box display="flex" gap={0.5}>
                      {onDownload && (
                        <Tooltip title="Download">
                          <IconButton
                            edge="end"
                            size="small"
                            onClick={(e) => {
                              e.stopPropagation();
                              onDownload(item);
                            }}
                          >
                            <DownloadIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      )}
                      {onDelete && (
                        <Tooltip title="Delete">
                          <IconButton
                            edge="end"
                            size="small"
                            onClick={(e) => {
                              e.stopPropagation();
                              onDelete(item.id);
                            }}
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      )}
                    </Box>
                  }
                >
                  <ListItemButton
                    onClick={() => onSelect?.(item)}
                    sx={{ pr: 10 }}
                  >
                    <ListItemIcon>
                      {getFormatIcon(item.format)}
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Box display="flex" alignItems="center" gap={1}>
                          <Typography variant="body2" noWrap>
                            {item.experimentName}
                          </Typography>
                          <Chip
                            label={getFormatLabel(item.format)}
                            size="small"
                            sx={{ height: 20 }}
                          />
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="caption" color="text.secondary" noWrap>
                            {item.datasetName}
                          </Typography>
                          <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                            • {formatTimestamp(item.timestamp)}
                          </Typography>
                        </Box>
                      }
                    />
                  </ListItemButton>
                </ListItem>
              </React.Fragment>
            ))}
          </List>
        )}
      </CardContent>
    </Card>
  );
};

export default GenerationHistory;
