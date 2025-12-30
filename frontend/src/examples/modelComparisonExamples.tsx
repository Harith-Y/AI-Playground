/**
 * Model Comparison Usage Examples
 * 
 * Practical examples of using the Model Comparison feature
 */

import React from 'react';
import {
  ModelComparisonViewEnhanced,
  ModelListSelector,
  CompareModelsRequest,
} from '../components/model';
import { modelComparisonService } from '../services/modelComparisonService';

// ============================================================================
// Example 1: Basic Comparison
// ============================================================================

export const BasicComparisonExample: React.FC = () => {
  const modelIds = ['model-uuid-1', 'model-uuid-2', 'model-uuid-3'];

  return (
    <ModelComparisonViewEnhanced
      modelRunIds={modelIds}
      onRefresh={() => console.log('Refreshing comparison...')}
    />
  );
};

// ============================================================================
// Example 2: Custom Metrics Comparison
// ============================================================================

export const CustomMetricsExample: React.FC = () => {
  const modelIds = ['model-uuid-1', 'model-uuid-2'];
  const customMetrics = ['accuracy', 'f1_score', 'precision', 'recall'];

  return (
    <ModelComparisonViewEnhanced
      modelRunIds={modelIds}
      comparisonMetrics={customMetrics}
      rankingCriteria="f1_score"
    />
  );
};

// ============================================================================
// Example 3: Programmatic Comparison
// ============================================================================

export const ProgrammaticComparisonExample: React.FC = () => {
  const [comparisonResult, setComparisonResult] = React.useState<any>(null);
  const [loading, setLoading] = React.useState(false);

  const handleCompare = async () => {
    setLoading(true);
    try {
      const request: CompareModelsRequest = {
        model_run_ids: ['uuid-1', 'uuid-2', 'uuid-3'],
        comparison_metrics: ['accuracy', 'f1_score'],
        ranking_criteria: 'accuracy',
        include_statistical_tests: false,
      };

      const result = await modelComparisonService.compareModels(request);
      setComparisonResult(result);
      
      console.log('Best Model:', result.best_model);
      console.log('Recommendations:', result.recommendations);
    } catch (error) {
      console.error('Comparison failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <button onClick={handleCompare} disabled={loading}>
        {loading ? 'Comparing...' : 'Compare Models'}
      </button>
      
      {comparisonResult && (
        <div>
          <h3>Best Model: {comparisonResult.best_model.model_type}</h3>
          <p>Score: {comparisonResult.best_model.ranking_score}</p>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Example 4: With Model Selection
// ============================================================================

export const SelectionComparisonExample: React.FC = () => {
  const [models, setModels] = React.useState<any[]>([]);
  const [selectedModels, setSelectedModels] = React.useState<string[]>([]);

  React.useEffect(() => {
    // Fetch available models
    const fetchModels = async () => {
      const modelRuns = await modelComparisonService.listModelRuns(
        undefined,
        'completed',
        undefined,
        100,
        0
      );
      setModels(modelRuns);
    };
    fetchModels();
  }, []);

  return (
    <div>
      <h2>Select Models to Compare</h2>
      <ModelListSelector
        models={models}
        selectedModels={selectedModels}
        onSelectionChange={setSelectedModels}
        maxSelection={5}
      />

      {selectedModels.length >= 2 && (
        <>
          <hr />
          <h2>Comparison Results</h2>
          <ModelComparisonViewEnhanced
            modelRunIds={selectedModels}
          />
        </>
      )}
    </div>
  );
};

// ============================================================================
// Example 5: Classification Models Comparison
// ============================================================================

export const ClassificationComparisonExample: React.FC = () => {
  return (
    <ModelComparisonViewEnhanced
      modelRunIds={[
        'random-forest-uuid',
        'logistic-regression-uuid',
        'svm-uuid',
        'gradient-boosting-uuid',
      ]}
      comparisonMetrics={['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']}
      rankingCriteria="f1_score"
    />
  );
};

// ============================================================================
// Example 6: Regression Models Comparison
// ============================================================================

export const RegressionComparisonExample: React.FC = () => {
  return (
    <ModelComparisonViewEnhanced
      modelRunIds={[
        'linear-regression-uuid',
        'random-forest-reg-uuid',
        'gradient-boosting-reg-uuid',
      ]}
      comparisonMetrics={['r2_score', 'rmse', 'mae', 'mse']}
      rankingCriteria="r2_score"
    />
  );
};

// ============================================================================
// Example 7: Time-Performance Trade-off Analysis
// ============================================================================

export const TradeoffAnalysisExample: React.FC = () => {
  const [models, setModels] = React.useState<any[]>([]);

  React.useEffect(() => {
    const analyzeTradeoffs = async () => {
      const allModels = await modelComparisonService.listModelRuns(
        undefined,
        'completed',
        undefined,
        100,
        0
      );

      // Filter for models with good accuracy but different training times
      const filteredModels = allModels
        .filter((m: any) => m.metrics?.accuracy > 0.85)
        .sort((a: any, b: any) => (a.training_time || 0) - (b.training_time || 0));

      setModels(filteredModels);
    };

    analyzeTradeoffs();
  }, []);

  return (
    <div>
      <h2>Training Time vs Performance Analysis</h2>
      <p>Comparing models with accuracy &gt; 85%, sorted by training time</p>
      
      {models.length >= 2 && (
        <ModelComparisonViewEnhanced
          modelRunIds={models.slice(0, 5).map(m => m.id)}
          rankingCriteria="accuracy"
        />
      )}
    </div>
  );
};

// ============================================================================
// Example 8: Error Handling
// ============================================================================

export const ErrorHandlingExample: React.FC = () => {
  const [error, setError] = React.useState<string | null>(null);

  const handleCompare = async () => {
    try {
      const result = await modelComparisonService.compareModels({
        model_run_ids: ['invalid-id-1', 'invalid-id-2'],
      });
      console.log(result);
    } catch (err: any) {
      if (err.response?.status === 404) {
        setError('One or more models not found');
      } else if (err.response?.status === 400) {
        setError('Invalid comparison request. Models must be from the same task type.');
      } else {
        setError('Failed to compare models. Please try again.');
      }
    }
  };

  return (
    <div>
      <button onClick={handleCompare}>Compare Models</button>
      {error && <div style={{ color: 'red' }}>{error}</div>}
    </div>
  );
};

// ============================================================================
// Example 9: Custom Ranking Weights
// ============================================================================

export const CustomRankingExample: React.FC = () => {
  const [rankingResult, setRankingResult] = React.useState<any>(null);

  const handleCustomRanking = async () => {
    const result = await modelComparisonService.rankModels({
      model_run_ids: ['model-1', 'model-2', 'model-3'],
      ranking_weights: {
        f1_score: 0.5,
        precision: 0.3,
        recall: 0.2,
      },
      higher_is_better: {
        f1_score: true,
        precision: true,
        recall: true,
      },
    });

    setRankingResult(result);
  };

  return (
    <div>
      <button onClick={handleCustomRanking}>Custom Ranking</button>
      
      {rankingResult && (
        <div>
          <h3>Custom Ranking Results</h3>
          <ul>
            {rankingResult.ranked_models.map((model: any) => (
              <li key={model.model_run_id}>
                #{model.rank}: {model.model_type} (Score: {model.composite_score.toFixed(4)})
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Example 10: Export Comparison Data
// ============================================================================

export const ExportComparisonExample: React.FC = () => {
  const exportToJSON = async (modelIds: string[]) => {
    const result = await modelComparisonService.compareModels({
      model_run_ids: modelIds,
    });

    const dataStr = JSON.stringify(result, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `model-comparison-${Date.now()}.json`;
    link.click();
  };

  const exportToCSV = async (modelIds: string[]) => {
    const result = await modelComparisonService.compareModels({
      model_run_ids: modelIds,
    });

    // Build CSV
    const headers = ['Model', 'Rank', 'Score', ...result.metric_statistics.map(s => s.metric_name)];
    const rows = result.compared_models.map(model => [
      model.model_type,
      model.rank,
      model.ranking_score,
      ...result.metric_statistics.map(s => model.metrics[s.metric_name] || 'N/A'),
    ]);

    const csv = [
      headers.join(','),
      ...rows.map(row => row.join(',')),
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `model-comparison-${Date.now()}.csv`;
    link.click();
  };

  return (
    <div>
      <button onClick={() => exportToJSON(['model-1', 'model-2'])}>
        Export as JSON
      </button>
      <button onClick={() => exportToCSV(['model-1', 'model-2'])}>
        Export as CSV
      </button>
    </div>
  );
};

// ============================================================================
// Usage Tips
// ============================================================================

/**
 * TIPS FOR EFFECTIVE MODEL COMPARISON:
 * 
 * 1. Compare similar models (same task type)
 * 2. Use consistent data splits across all models
 * 3. Include 2-6 models for optimal visualization
 * 4. Choose ranking criteria based on business needs
 * 5. Check statistical significance for small differences
 * 6. Consider training time vs performance trade-offs
 * 7. Review recommendations from the system
 * 8. Document comparison results for team review
 * 9. Test on independent dataset before deployment
 * 10. Monitor model performance in production
 */

export default {
  BasicComparisonExample,
  CustomMetricsExample,
  ProgrammaticComparisonExample,
  SelectionComparisonExample,
  ClassificationComparisonExample,
  RegressionComparisonExample,
  TradeoffAnalysisExample,
  ErrorHandlingExample,
  CustomRankingExample,
  ExportComparisonExample,
};
