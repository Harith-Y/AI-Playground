import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

interface Metrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  mse?: number;
  rmse?: number;
  r2?: number;
  mae?: number;
  [key: string]: number | undefined;
}

interface ConfusionMatrix {
  matrix: number[][];
  labels: string[];
}

interface ResidualPlot {
  predicted: number[];
  residuals: number[];
}

interface EvaluationState {
  metrics: Metrics | null;
  confusionMatrix: ConfusionMatrix | null;
  residualPlot: ResidualPlot | null;
  predictions: any[];
  isEvaluating: boolean;
  error: string | null;
}

const initialState: EvaluationState = {
  metrics: null,
  confusionMatrix: null,
  residualPlot: null,
  predictions: [],
  isEvaluating: false,
  error: null,
};

// Async thunks
export const evaluateModel = createAsyncThunk(
  'evaluation/evaluateModel',
  async (params: { modelId: string; datasetId: string }, { rejectWithValue }) => {
    try {
      // TODO: Implement API call
      return {
        metrics: {},
        confusionMatrix: null,
        residualPlot: null,
      };
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to evaluate model');
    }
  }
);

export const fetchPredictions = createAsyncThunk(
  'evaluation/fetchPredictions',
  async (params: { modelId: string; datasetId: string }, { rejectWithValue }) => {
    try {
      // TODO: Implement API call
      return [];
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to fetch predictions');
    }
  }
);

export const generateConfusionMatrix = createAsyncThunk(
  'evaluation/generateConfusionMatrix',
  async (modelId: string, { rejectWithValue }) => {
    try {
      // TODO: Implement API call
      return { matrix: [], labels: [] };
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to generate confusion matrix');
    }
  }
);

const evaluationSlice = createSlice({
  name: 'evaluation',
  initialState,
  reducers: {
    setMetrics: (state, action: PayloadAction<Metrics>) => {
      state.metrics = action.payload;
    },
    clearEvaluation: (state) => {
      state.metrics = null;
      state.confusionMatrix = null;
      state.residualPlot = null;
      state.predictions = [];
      state.error = null;
    },
    clearEvaluationError: (state) => {
      state.error = null;
    },
    resetEvaluationState: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      .addCase(evaluateModel.pending, (state) => {
        state.isEvaluating = true;
        state.error = null;
      })
      .addCase(evaluateModel.fulfilled, (state, action) => {
        state.isEvaluating = false;
        state.metrics = action.payload.metrics;
        state.confusionMatrix = action.payload.confusionMatrix;
        state.residualPlot = action.payload.residualPlot;
      })
      .addCase(evaluateModel.rejected, (state, action) => {
        state.isEvaluating = false;
        state.error = action.payload as string;
      })
      .addCase(fetchPredictions.pending, (state) => {
        state.isEvaluating = true;
        state.error = null;
      })
      .addCase(fetchPredictions.fulfilled, (state, action) => {
        state.isEvaluating = false;
        state.predictions = action.payload;
      })
      .addCase(fetchPredictions.rejected, (state, action) => {
        state.isEvaluating = false;
        state.error = action.payload as string;
      })
      .addCase(generateConfusionMatrix.pending, (state) => {
        state.isEvaluating = true;
        state.error = null;
      })
      .addCase(generateConfusionMatrix.fulfilled, (state, action) => {
        state.isEvaluating = false;
        state.confusionMatrix = action.payload;
      })
      .addCase(generateConfusionMatrix.rejected, (state, action) => {
        state.isEvaluating = false;
        state.error = action.payload as string;
      });
  },
});

export const {
  setMetrics,
  clearEvaluation,
  clearEvaluationError,
  resetEvaluationState,
} = evaluationSlice.actions;

export default evaluationSlice.reducer;
