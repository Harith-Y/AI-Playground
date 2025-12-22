import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

interface ParameterRange {
  name: string;
  type: 'int' | 'float' | 'categorical';
  min?: number;
  max?: number;
  values?: any[];
  step?: number;
}

interface TuningConfig {
  method: 'grid' | 'random' | 'bayesian';
  nTrials: number;
  parameterRanges: ParameterRange[];
}

interface TuningResult {
  trial: number;
  parameters: Record<string, any>;
  score: number;
  metrics: Record<string, number>;
}

interface TuningState {
  config: TuningConfig | null;
  results: TuningResult[];
  bestResult: TuningResult | null;
  isTuning: boolean;
  progress: number;
  isLoading: boolean;
  error: string | null;
}

const initialState: TuningState = {
  config: null,
  results: [],
  bestResult: null,
  isTuning: false,
  progress: 0,
  isLoading: false,
  error: null,
};

// Async thunks
export const startTuning = createAsyncThunk(
  'tuning/startTuning',
  async (
    params: {
      modelType: string;
      datasetId: string;
      config: TuningConfig;
    },
    { rejectWithValue }
  ) => {
    try {
      // TODO: Implement API call
      return { jobId: 'job-123' };
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to start tuning');
    }
  }
);

export const fetchTuningResults = createAsyncThunk(
  'tuning/fetchResults',
  async (jobId: string, { rejectWithValue }) => {
    try {
      // TODO: Implement API call
      return { results: [], bestResult: null };
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to fetch tuning results');
    }
  }
);

export const stopTuning = createAsyncThunk(
  'tuning/stopTuning',
  async (jobId: string, { rejectWithValue }) => {
    try {
      // TODO: Implement API call
      return { success: true };
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to stop tuning');
    }
  }
);

const tuningSlice = createSlice({
  name: 'tuning',
  initialState,
  reducers: {
    setTuningConfig: (state, action: PayloadAction<TuningConfig>) => {
      state.config = action.payload;
    },
    updateParameterRange: (state, action: PayloadAction<ParameterRange>) => {
      if (state.config) {
        const index = state.config.parameterRanges.findIndex(
          p => p.name === action.payload.name
        );
        if (index !== -1) {
          state.config.parameterRanges[index] = action.payload;
        } else {
          state.config.parameterRanges.push(action.payload);
        }
      }
    },
    removeParameterRange: (state, action: PayloadAction<string>) => {
      if (state.config) {
        state.config.parameterRanges = state.config.parameterRanges.filter(
          p => p.name !== action.payload
        );
      }
    },
    setTuningProgress: (state, action: PayloadAction<number>) => {
      state.progress = action.payload;
    },
    addTuningResult: (state, action: PayloadAction<TuningResult>) => {
      state.results.push(action.payload);
      if (!state.bestResult || action.payload.score > state.bestResult.score) {
        state.bestResult = action.payload;
      }
    },
    clearTuningError: (state) => {
      state.error = null;
    },
    resetTuningState: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      .addCase(startTuning.pending, (state) => {
        state.isTuning = true;
        state.error = null;
        state.progress = 0;
        state.results = [];
        state.bestResult = null;
      })
      .addCase(startTuning.fulfilled, (state) => {
        // Tuning started successfully, keep isTuning true
      })
      .addCase(startTuning.rejected, (state, action) => {
        state.isTuning = false;
        state.error = action.payload as string;
      })
      .addCase(fetchTuningResults.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchTuningResults.fulfilled, (state, action) => {
        state.isLoading = false;
        state.results = action.payload.results;
        state.bestResult = action.payload.bestResult;
      })
      .addCase(fetchTuningResults.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      .addCase(stopTuning.fulfilled, (state) => {
        state.isTuning = false;
      });
  },
});

export const {
  setTuningConfig,
  updateParameterRange,
  removeParameterRange,
  setTuningProgress,
  addTuningResult,
  clearTuningError,
  resetTuningState,
} = tuningSlice.actions;

export default tuningSlice.reducer;
