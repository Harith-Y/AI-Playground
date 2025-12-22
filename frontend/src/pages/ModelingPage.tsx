import React from 'react';
import { Container, Typography, Box, Paper } from '@mui/material';
import { ModelTraining } from '@mui/icons-material';

const ModelingPage: React.FC = () => {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: 'calc(100vh - 64px)',
        width: '100%',
        textAlign: 'center',
      }}
    >
      <ModelTraining sx={{ fontSize: 80, color: 'primary.main', mb: 3 }} />
      <Typography variant="h3" component="h1" fontWeight={600} gutterBottom>
        Model Training
      </Typography>
      <Paper
        sx={{
          p: 6,
          mt: 4,
          width: '100%',
          maxWidth: 600,
          border: '1px solid #334155',
        }}
      >
        <Typography variant="h5" color="text.primary" gutterBottom>
          Train machine learning models
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mt: 2 }}>
          This page will be implemented in the next steps
        </Typography>
      </Paper>
    </Box>
  );
};

export default ModelingPage;
