import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import type { PreprocessingStep, PreprocessingStepCreate, PreprocessingStepUpdate } from '../../types/preprocessing';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface PreprocessingState {
  steps: PreprocessingStep[];
  currentStep: PreprocessingStep | null;
  isLoading: boolean;
  isProcessing: boolean;
  error: string | null;
}

const initialState: PreprocessingState = {
  steps: [],
  currentStep: null,
  isLoading: false,
  isProcessing: false,
  error: null,
};

// Async thunks
export const fetchPreprocessingSteps = createAsyncThunk(
  'preprocessing/fetchSteps',
  async (datasetId: string, { rejectWithValue }) => {
    try {
      const response = await axios.get<PreprocessingStep[]>(
        `${API_URL}/api/v1/preprocessing/?dataset_id=${datasetId}`
      );
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.detail || 'Failed to fetch preprocessing steps');
    }
  }
);

export const createPreprocessingStep = createAsyncThunk(
  'preprocessing/createStep',
  async (step: PreprocessingStepCreate, { rejectWithValue }) => {
    try {
      const response = await axios.post<PreprocessingStep>(
        `${API_URL}/api/v1/preprocessing/`,
        step
      );
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.detail || 'Failed to create preprocessing step');
    }
  }
);

export const updatePreprocessingStep = createAsyncThunk(
  'preprocessing/updateStep',
  async ({ id, data }: { id: string; data: PreprocessingStepUpdate }, { rejectWithValue }) => {
    try {
      const response = await axios.put<PreprocessingStep>(
        `${API_URL}/api/v1/preprocessing/${id}`,
        data
      );
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.detail || 'Failed to update preprocessing step');
    }
  }
);

export const deletePreprocessingStep = createAsyncThunk(
  'preprocessing/deleteStep',
  async (id: string, { rejectWithValue }) => {
    try {
      await axios.delete(`${API_URL}/api/v1/preprocessing/${id}`);
      return id;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.detail || 'Failed to delete preprocessing step');
    }
  }
);

export const reorderPreprocessingSteps = createAsyncThunk(
  'preprocessing/reorderSteps',
  async (stepIds: string[], { rejectWithValue }) => {
    try {
      const response = await axios.post<PreprocessingStep[]>(
        `${API_URL}/api/v1/preprocessing/reorder`,
        { step_ids: stepIds }
      );
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.detail || 'Failed to reorder steps');
    }
  }
);

const preprocessingSlice = createSlice({
  name: 'preprocessing',
  initialState,
  reducers: {
    setCurrentStep: (state, action: PayloadAction<PreprocessingStep | null>) => {
      state.currentStep = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
    clearPreprocessing: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      // Fetch preprocessing steps
      .addCase(fetchPreprocessingSteps.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchPreprocessingSteps.fulfilled, (state, action) => {
        state.isLoading = false;
        state.steps = action.payload;
      })
      .addCase(fetchPreprocessingSteps.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      // Create preprocessing step
      .addCase(createPreprocessingStep.pending, (state) => {
        state.isProcessing = true;
        state.error = null;
      })
      .addCase(createPreprocessingStep.fulfilled, (state, action) => {
        state.isProcessing = false;
        state.steps.push(action.payload);
      })
      .addCase(createPreprocessingStep.rejected, (state, action) => {
        state.isProcessing = false;
        state.error = action.payload as string;
      })
      // Update preprocessing step
      .addCase(updatePreprocessingStep.pending, (state) => {
        state.isProcessing = true;
        state.error = null;
      })
      .addCase(updatePreprocessingStep.fulfilled, (state, action) => {
        state.isProcessing = false;
        const index = state.steps.findIndex(step => step.id === action.payload.id);
        if (index !== -1) {
          state.steps[index] = action.payload;
        }
      })
      .addCase(updatePreprocessingStep.rejected, (state, action) => {
        state.isProcessing = false;
        state.error = action.payload as string;
      })
      // Delete preprocessing step
      .addCase(deletePreprocessingStep.pending, (state) => {
        state.isProcessing = true;
        state.error = null;
      })
      .addCase(deletePreprocessingStep.fulfilled, (state, action) => {
        state.isProcessing = false;
        state.steps = state.steps.filter(step => step.id !== action.payload);
      })
      .addCase(deletePreprocessingStep.rejected, (state, action) => {
        state.isProcessing = false;
        state.error = action.payload as string;
      })
      // Reorder preprocessing steps
      .addCase(reorderPreprocessingSteps.pending, (state) => {
        state.isProcessing = true;
        state.error = null;
      })
      .addCase(reorderPreprocessingSteps.fulfilled, (state, action) => {
        state.isProcessing = false;
        state.steps = action.payload;
      })
      .addCase(reorderPreprocessingSteps.rejected, (state, action) => {
        state.isProcessing = false;
        state.error = action.payload as string;
      });
  },
});

export const {
  setCurrentStep,
  clearError,
  clearPreprocessing,
} = preprocessingSlice.actions;

export default preprocessingSlice.reducer;
