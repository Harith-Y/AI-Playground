import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface FeatureImportance {
  feature: string;
  importance: number;
}

interface CorrelationMatrix {
  features: string[];
  matrix: number[][];
}

interface FeatureState {
  selectedFeatures: string[];
  availableFeatures: string[];
  featureImportance: FeatureImportance[];
  correlationMatrix: CorrelationMatrix | null;
  isLoading: boolean;
  error: string | null;
}

const initialState: FeatureState = {
  selectedFeatures: [],
  availableFeatures: [],
  featureImportance: [],
  correlationMatrix: null,
  isLoading: false,
  error: null,
};

// Async thunks
export const fetchFeatureImportance = createAsyncThunk(
  'feature/fetchImportance',
  async (_datasetId: string, { rejectWithValue }) => {
    try {
      // Feature importance is only available after model training
      // This is a placeholder that returns empty data
      return [];
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to fetch feature importance');
    }
  }
);

export const fetchCorrelationMatrix = createAsyncThunk(
  'feature/fetchCorrelation',
  async (datasetId: string, { rejectWithValue }) => {
    try {
      const response = await axios.get(
        `${API_URL}/api/v1/visualizations/${datasetId}/correlation`
      );
      const data = response.data.data;
      return {
        features: data.columns || [],
        matrix: data.matrix || []
      };
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.detail || error.message || 'Failed to fetch correlation matrix');
    }
  }
);

export const performFeatureSelection = createAsyncThunk(
  'feature/selectFeatures',
  async (_params: { datasetId: string; method: string; threshold?: number }, { rejectWithValue }) => {
    try {
      // TODO: Implement API call
      return [];
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to perform feature selection');
    }
  }
);

const featureSlice = createSlice({
  name: 'feature',
  initialState,
  reducers: {
    setSelectedFeatures: (state, action: PayloadAction<string[]>) => {
      state.selectedFeatures = action.payload;
    },
    addFeature: (state, action: PayloadAction<string>) => {
      if (!state.selectedFeatures.includes(action.payload)) {
        state.selectedFeatures.push(action.payload);
      }
    },
    removeFeature: (state, action: PayloadAction<string>) => {
      state.selectedFeatures = state.selectedFeatures.filter(f => f !== action.payload);
    },
    setAvailableFeatures: (state, action: PayloadAction<string[]>) => {
      state.availableFeatures = action.payload;
    },
    clearFeatureError: (state) => {
      state.error = null;
    },
    resetFeatureState: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchFeatureImportance.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchFeatureImportance.fulfilled, (state, action) => {
        state.isLoading = false;
        state.featureImportance = action.payload;
      })
      .addCase(fetchFeatureImportance.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      .addCase(fetchCorrelationMatrix.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchCorrelationMatrix.fulfilled, (state, action) => {
        state.isLoading = false;
        state.correlationMatrix = action.payload;
      })
      .addCase(fetchCorrelationMatrix.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      .addCase(performFeatureSelection.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(performFeatureSelection.fulfilled, (state, action) => {
        state.isLoading = false;
        state.selectedFeatures = action.payload;
      })
      .addCase(performFeatureSelection.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });
  },
});

export const {
  setSelectedFeatures,
  addFeature,
  removeFeature,
  setAvailableFeatures,
  clearFeatureError,
  resetFeatureState,
} = featureSlice.actions;

export default featureSlice.reducer;
