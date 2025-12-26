import React from 'react';
import { Box, Typography, Button } from '@mui/material';
import { Inbox } from '@mui/icons-material';

interface EmptyStateProps {
  title?: string;
  message?: string;
  icon?: React.ReactNode;
  action?: {
    label: string;
    onClick: () => void;
  };
  fullscreen?: boolean;
}

const EmptyState: React.FC<EmptyStateProps> = ({
  title = 'No Data Available',
  message = 'There is no data to display at the moment',
  icon,
  action,
  fullscreen = false,
}) => {
  const content = (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        p: 4,
      }}
    >
      {icon || (
        <Inbox
          sx={{
            fontSize: 64,
            color: 'text.secondary',
            mb: 2,
            opacity: 0.5,
          }}
        />
      )}
      <Typography variant="h6" fontWeight={600} gutterBottom color="text.primary">
        {title}
      </Typography>
      <Typography
        variant="body2"
        color="text.secondary"
        sx={{ mb: 3, maxWidth: 500 }}
      >
        {message}
      </Typography>
      {action && (
        <Button
          variant="contained"
          onClick={action.onClick}
          sx={{ mt: 1 }}
        >
          {action.label}
        </Button>
      )}
    </Box>
  );

  if (fullscreen) {
    return (
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '100vh',
          width: '100%',
          backgroundColor: 'background.default',
        }}
      >
        {content}
      </Box>
    );
  }

  return content;
};

export default EmptyState;
