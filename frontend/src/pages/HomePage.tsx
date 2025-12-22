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
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        minHeight: 'calc(100vh - 64px)',
        width: '100%',
      }}
    >
      <Container maxWidth="lg">
        <Box sx={{ py: 6, textAlign: 'center' }}>
        <Typography
          variant="h3"
          component="h1"
          gutterBottom
          fontWeight={700}
          sx={{
            background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            mb: 2,
          }}
        >
          Welcome to AI Playground
        </Typography>
        <Typography variant="h6" color="text.secondary" paragraph sx={{ mb: 6 }}>
          Build and experiment with machine learning models using your own datasets
        </Typography>

        <Box sx={{ mb: 5 }}>
          <Typography variant="h5" gutterBottom fontWeight={600} color="text.primary">
            Get Started with Your ML Journey
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph>
            Follow these steps to build your complete machine learning pipeline:
          </Typography>
        </Box>

        <Grid container spacing={3} justifyContent="center">
          {steps.map((step, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  transition: 'all 0.3s ease-in-out',
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 3,
                  position: 'relative',
                  overflow: 'hidden',
                  '&:hover': {
                    transform: 'translateY(-8px)',
                    boxShadow: '0 12px 24px -4px rgba(0, 0, 0, 0.1)',
                    borderColor: step.color,
                    '& .icon-box': {
                      backgroundColor: step.color,
                      color: 'white',
                      transform: 'scale(1.1) rotate(5deg)',
                    }
                  },
                  cursor: 'pointer',
                }}
                onClick={() => navigate(step.path)}
              >
                <CardContent sx={{ flexGrow: 1, p: 3, textAlign: 'left' }}>
                  <Box
                    className="icon-box"
                    sx={{
                      display: 'inline-flex',
                      p: 1.5,
                      borderRadius: 2,
                      backgroundColor: `${step.color}15`,
                      color: step.color,
                      mb: 2,
                      transition: 'all 0.3s ease-in-out',
                    }}
                  >
                    {step.icon}
                  </Box>
                  <Typography
                    variant="h6"
                    component="div"
                    sx={{ mb: 1, fontWeight: 700, color: 'text.primary' }}
                  >
                    {index + 1}. {step.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {step.description}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>

        <Box sx={{ mt: 6 }}>
          <Button
            variant="contained"
            size="large"
            startIcon={<CloudUpload />}
            onClick={() => navigate('/dataset-upload')}
            sx={{
              px: 5,
              py: 1.5,
              fontSize: '1.1rem',
              fontWeight: 600,
              background: 'linear-gradient(135deg, #2563eb 0%, #3b82f6 100%)',
              '&:hover': {
                background: 'linear-gradient(135deg, #1e40af 0%, #2563eb 100%)',
                transform: 'translateY(-2px)',
                boxShadow: '0 10px 20px -5px rgba(37, 99, 235, 0.5)',
              },
              transition: 'all 0.2s',
              textTransform: 'none',
            }}
          >
            Start with Dataset Upload
          </Button>
        </Box>
      </Box>
      </Container>
    </Box>
  );
};

export default HomePage;
