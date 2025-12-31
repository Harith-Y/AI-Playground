/**
 * Code Preview Component
 * 
 * Displays generated code with syntax highlighting and actions
 */

import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Tooltip,
  Snackbar,
  Tabs,
  Tab,
  Chip,
} from '@mui/material';
import {
  ContentCopy as ContentCopyIcon,
  Check as CheckIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  WrapText as WrapTextIcon,
} from '@mui/icons-material';

interface CodeFile {
  filename: string;
  content: string;
  language: string;
}

interface CodePreviewProps {
  files: CodeFile[];
  title?: string;
}

const CodePreview: React.FC<CodePreviewProps> = ({
  files,
  title = 'Generated Code',
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [copied, setCopied] = useState(false);
  const [fontSize, setFontSize] = useState(14);
  const [wrapText, setWrapText] = useState(false);

  const handleCopy = async () => {
    const currentFile = files[activeTab];
    if (currentFile) {
      await navigator.clipboard.writeText(currentFile.content);
      setCopied(true);
    }
  };

  const handleZoomIn = () => {
    setFontSize(prev => Math.min(prev + 2, 24));
  };

  const handleZoomOut = () => {
    setFontSize(prev => Math.max(prev - 2, 10));
  };

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">{title}</Typography>
        <Box display="flex" gap={1}>
          <Tooltip title="Zoom out">
            <IconButton size="small" onClick={handleZoomOut}>
              <ZoomOutIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Zoom in">
            <IconButton size="small" onClick={handleZoomIn}>
              <ZoomInIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Toggle text wrap">
            <IconButton size="small" onClick={() => setWrapText(!wrapText)}>
              <WrapTextIcon color={wrapText ? 'primary' : 'inherit'} />
            </IconButton>
          </Tooltip>
          <Tooltip title={copied ? 'Copied!' : 'Copy to clipboard'}>
            <IconButton size="small" onClick={handleCopy} color={copied ? 'success' : 'default'}>
              {copied ? <CheckIcon /> : <ContentCopyIcon />}
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* File Tabs */}
      {files.length > 1 && (
        <Tabs
          value={activeTab}
          onChange={(_e, newValue) => setActiveTab(newValue)}
          sx={{ mb: 2, borderBottom: 1, borderColor: 'divider' }}
        >
          {files.map((file, index) => (
            <Tab
              key={index}
              label={
                <Box display="flex" alignItems="center" gap={1}>
                  {file.filename}
                  <Chip label={file.language} size="small" />
                </Box>
              }
            />
          ))}
        </Tabs>
      )}

      {/* Code Display */}
      <Paper
        sx={{
          bgcolor: 'grey.900',
          color: 'grey.100',
          p: 2,
          minHeight: 400,
          maxHeight: 600,
          overflow: 'auto',
          fontFamily: 'monospace',
          fontSize: `${fontSize}px`,
          lineHeight: 1.5,
        }}
      >
        <pre style={{ margin: 0, whiteSpace: wrapText ? 'pre-wrap' : 'pre' }}>
          {files[activeTab]?.content || '// No code generated yet'}
        </pre>
      </Paper>

      {/* Snackbar */}
      <Snackbar
        open={copied}
        autoHideDuration={2000}
        onClose={() => setCopied(false)}
        message="Code copied to clipboard"
      />
    </Box>
  );
};

export default CodePreview;
