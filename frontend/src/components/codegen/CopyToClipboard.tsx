/**
 * Copy to Clipboard Component
 * 
 * Reusable button for copying text to clipboard
 */

import React, { useState } from 'react';
import {
  IconButton,
  Button,
  Tooltip,
  Snackbar,
} from '@mui/material';
import {
  ContentCopy as ContentCopyIcon,
  Check as CheckIcon,
} from '@mui/icons-material';

interface CopyToClipboardProps {
  text: string;
  variant?: 'icon' | 'button';
  label?: string;
  size?: 'small' | 'medium' | 'large';
  showSnackbar?: boolean;
  onCopy?: () => void;
}

const CopyToClipboard: React.FC<CopyToClipboardProps> = ({
  text,
  variant = 'icon',
  label = 'Copy',
  size = 'medium',
  showSnackbar = true,
  onCopy,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      onCopy?.();
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handleSnackbarClose = () => {
    setCopied(false);
  };

  if (variant === 'button') {
    return (
      <>
        <Button
          startIcon={copied ? <CheckIcon /> : <ContentCopyIcon />}
          onClick={handleCopy}
          size={size}
          color={copied ? 'success' : 'primary'}
        >
          {copied ? 'Copied!' : label}
        </Button>
        {showSnackbar && (
          <Snackbar
            open={copied}
            autoHideDuration={2000}
            onClose={handleSnackbarClose}
            message="Copied to clipboard"
          />
        )}
      </>
    );
  }

  return (
    <>
      <Tooltip title={copied ? 'Copied!' : 'Copy to clipboard'}>
        <IconButton
          onClick={handleCopy}
          size={size}
          color={copied ? 'success' : 'default'}
        >
          {copied ? <CheckIcon /> : <ContentCopyIcon />}
        </IconButton>
      </Tooltip>
      {showSnackbar && (
        <Snackbar
          open={copied}
          autoHideDuration={2000}
          onClose={handleSnackbarClose}
          message="Copied to clipboard"
        />
      )}
    </>
  );
};

export default CopyToClipboard;
