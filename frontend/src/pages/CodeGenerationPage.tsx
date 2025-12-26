import React, { useState } from 'react';
import { Typography, Box, Container, Grid, Card, CardContent, Button } from '@mui/material';
import { Code, Download } from '@mui/icons-material';
import { useAppSelector } from '../hooks/redux';
import LoadingState from '../components/common/LoadingState';
import ErrorState from '../components/common/ErrorState';
import EmptyState from '../components/common/EmptyState';

const CodeGenerationPage: React.FC = () => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { currentModel } = useAppSelector((state) => state.modeling);
  const { steps } = useAppSelector((state) => state.preprocessing);

  const handleGenerateCode = async () => {
    setIsGenerating(true);
    setError(null);
    // TODO: Implement code generation
    setTimeout(() => {
      setIsGenerating(false);
    }, 2000);
  };

  const handleRetry = () => {
    setError(null);
    handleGenerateCode();
  };

  if (isGenerating) {
    return <LoadingState message="Generating code from your ML pipeline..." />;
  }

  if (error) {
    return (
      <ErrorState
        title="Code Generation Failed"
        message={error}
        onRetry={handleRetry}
      />
    );
  }

  if (!currentModel && steps.length === 0) {
    return (
      <EmptyState
        title="No Pipeline to Export"
        message="Build your ML pipeline first by preprocessing data and training a model"
      />
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Page Header */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <Code sx={{ fontSize: 40, color: 'primary.main' }} />
          <Box>
            <Typography variant="h4" component="h1" fontWeight={600}>
              Code Generation
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Export your ML pipeline as production-ready code
            </Typography>
          </Box>
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* Pipeline Summary */}
        <Grid item xs={12} md={4}>
          <Card sx={{ border: '1px solid #e2e8f0' }}>
            <CardContent>
              <Typography variant="h6" fontWeight={600} gutterBottom>
                Pipeline Summary
              </Typography>
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Preprocessing Steps: {steps.length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Model: {currentModel ? 'Trained' : 'Not trained'}
                </Typography>
              </Box>
              <Button
                variant="contained"
                fullWidth
                sx={{ mt: 3 }}
                onClick={handleGenerateCode}
                disabled={!currentModel && steps.length === 0}
              >
                Generate Code
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Code Preview */}
        <Grid item xs={12} md={8}>
          <Card sx={{ border: '1px solid #e2e8f0' }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" fontWeight={600}>
                  Generated Code
                </Typography>
                <Button
                  variant="outlined"
                  startIcon={<Download />}
                  size="small"
                  disabled
                >
                  Download
                </Button>
              </Box>
              <EmptyState
                title="No Code Generated Yet"
                message="Click 'Generate Code' to export your ML pipeline"
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default CodeGenerationPage;
