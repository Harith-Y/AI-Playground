import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import type { Model, TrainingMetrics, Hyperparameters } from '../../types/model';
import { modelService } from '../../services/modelService';

interface ModelingState {
  selectedModel: string | null;
  availableModels: string[];
  currentModel: Model | null;
  models: Model[];
  hyperparameters: Hyperparameters;
  trainingProgress: number;
  trainingMetrics: TrainingMetrics | null;
  isTraining: boolean;
  isLoading: boolean;
  error: string | null;
  logs: string[];
}

const initialState: ModelingState = {
  selectedModel: null,
  availableModels: [],
  currentModel: null,
  models: [],
  hyperparameters: {},
  trainingProgress: 0,
  trainingMetrics: null,
  isTraining: false,
  isLoading: false,
  error: null,
  logs: [],
};

// Async thunks
export const trainModel = createAsyncThunk(
  'modeling/trainModel',
  async (
    params: {
      datasetId: string;
      modelType: string;
      hyperparameters: Hyperparameters;
      targetColumn?: string;
      selectedFeatures?: string[];
      experimentId?: string;
    },
    { rejectWithValue }
  ) => {
    try {
      const result = await modelService.trainModel(
        params.datasetId,
        params.modelType,
        params.hyperparameters,
        params.targetColumn,
        params.selectedFeatures,
        params.experimentId
      );
      return result;
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to train model');
    }
  }
);

export const fetchModels = createAsyncThunk(
  'modeling/fetchModels',
  async (_, { rejectWithValue }) => {
    try {
      // TODO: Implement API call
      return [];
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to fetch models');
    }
  }
);

export const fetchModelDetails = createAsyncThunk(
  'modeling/fetchModelDetails',
  async (_modelId: string, { rejectWithValue }) => {
    try {
      // TODO: Implement API call
      return {} as Model;
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to fetch model details');
    }
  }
);

export const stopTraining = createAsyncThunk(
  'modeling/stopTraining',
  async (_modelId: string, { rejectWithValue }) => {
    try {
      // TODO: Implement API call
      return { success: true };
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to stop training');
    }
  }
);

const modelingSlice = createSlice({
  name: 'modeling',
  initialState,
  reducers: {
    setSelectedModel: (state, action: PayloadAction<string>) => {
      state.selectedModel = action.payload;
    },
    setHyperparameters: (state, action: PayloadAction<Hyperparameters>) => {
      state.hyperparameters = action.payload;
    },
    updateHyperparameter: (state, action: PayloadAction<{ key: string; value: any }>) => {
      state.hyperparameters[action.payload.key] = action.payload.value;
    },
    setTrainingProgress: (state, action: PayloadAction<number>) => {
      state.trainingProgress = action.payload;
    },
    updateTrainingMetrics: (state, action: PayloadAction<TrainingMetrics>) => {
      state.trainingMetrics = action.payload;
    },
    addLog: (state, action: PayloadAction<string>) => {
      state.logs.push(action.payload);
    },
    clearLogs: (state) => {
      state.logs = [];
    },
    setAvailableModels: (state, action: PayloadAction<string[]>) => {
      state.availableModels = action.payload;
    },
    clearModelError: (state) => {
      state.error = null;
    },
    resetModelingState: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      .addCase(trainModel.pending, (state) => {
        state.isTraining = true;
        state.error = null;
        state.trainingProgress = 0;
        state.logs = [];
      })
      .addCase(trainModel.fulfilled, (state, action) => {
        state.isTraining = false;
        state.currentModel = action.payload;
        state.models.push(action.payload);
        state.trainingProgress = 100;
      })
      .addCase(trainModel.rejected, (state, action) => {
        state.isTraining = false;
        state.error = action.payload as string;
        state.trainingProgress = 0;
      })
      .addCase(fetchModels.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchModels.fulfilled, (state, action) => {
        state.isLoading = false;
        state.models = action.payload;
      })
      .addCase(fetchModels.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      .addCase(fetchModelDetails.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchModelDetails.fulfilled, (state, action) => {
        state.isLoading = false;
        state.currentModel = action.payload;
      })
      .addCase(fetchModelDetails.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      .addCase(stopTraining.fulfilled, (state) => {
        state.isTraining = false;
      });
  },
});

export const {
  setSelectedModel,
  setHyperparameters,
  updateHyperparameter,
  setTrainingProgress,
  updateTrainingMetrics,
  addLog,
  clearLogs,
  setAvailableModels,
  clearModelError,
  resetModelingState,
} = modelingSlice.actions;

export default modelingSlice.reducer;
