import React from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  Grid,
  Button,
} from '@mui/material';
import {
  CloudUpload,
  Transform,
  TrendingUp,
  ModelTraining,
  Assessment,
  Tune,
  Code,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const HomePage: React.FC = () => {
  const navigate = useNavigate();

  const steps = [
    {
      title: 'Upload Dataset',
      description: 'Start by uploading your dataset in CSV format',
      icon: <CloudUpload sx={{ fontSize: 40 }} />,
      path: '/dataset-upload',
      color: '#1976d2',
    },
    {
      title: 'Preprocess Data',
      description: 'Clean and transform your data',
      icon: <Transform sx={{ fontSize: 40 }} />,
      path: '/preprocessing',
      color: '#2e7d32',
    },
    {
      title: 'Engineer Features',
      description: 'Select and create meaningful features',
      icon: <TrendingUp sx={{ fontSize: 40 }} />,
      path: '/features',
      color: '#ed6c02',
    },
    {
      title: 'Train Models',
      description: 'Train ML models with your data',
      icon: <ModelTraining sx={{ fontSize: 40 }} />,
      path: '/modeling',
      color: '#9c27b0',
    },
    {
      title: 'Evaluate Results',
      description: 'Analyze model performance metrics',
      icon: <Assessment sx={{ fontSize: 40 }} />,
      path: '/evaluation',
      color: '#d32f2f',
    },
    {
      title: 'Tune Hyperparameters',
      description: 'Optimize model parameters',
      icon: <Tune sx={{ fontSize: 40 }} />,
      path: '/tuning',
      color: '#0288d1',
    },
    {
      title: 'Generate Code',
      description: 'Export your pipeline as code',
      icon: <Code sx={{ fontSize: 40 }} />,
      path: '/code-generation',
      color: '#7b1fa2',
    },
  ];

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom fontWeight={600}>
          Welcome to AI Playground
        </Typography>
        <Typography variant="h6" color="text.secondary" paragraph>
          Build and experiment with machine learning models using your own datasets
        </Typography>

        <Box sx={{ my: 4 }}>
          <Typography variant="h5" gutterBottom fontWeight={500}>
            Get Started with Your ML Journey
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph>
            Follow these steps to build your complete machine learning pipeline:
          </Typography>
        </Box>

        <Grid container spacing={3}>
          {steps.map((step, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: 4,
                  },
                  cursor: 'pointer',
                }}
                onClick={() => navigate(step.path)}
              >
                <CardContent sx={{ flexGrow: 1 }}>
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      mb: 2,
                      color: step.color,
                    }}
                  >
                    {step.icon}
                    <Typography
                      variant="h6"
                      component="div"
                      sx={{ ml: 1, fontWeight: 600 }}
                    >
                      {index + 1}. {step.title}
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {step.description}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>

        <Box sx={{ mt: 6, textAlign: 'center' }}>
          <Button
            variant="contained"
            size="large"
            startIcon={<CloudUpload />}
            onClick={() => navigate('/dataset-upload')}
            sx={{
              px: 4,
              py: 1.5,
              fontSize: '1.1rem',
              background: 'linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%)',
              '&:hover': {
                background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
                transform: 'translateY(-2px)',
                boxShadow: '0 10px 15px -3px rgba(59, 130, 246, 0.4)',
              },
              transition: 'all 0.2s',
            }}
          >
            Start with Dataset Upload
          </Button>
        </Box>
      </Box>
    </Container>
  );
};

export default HomePage;
