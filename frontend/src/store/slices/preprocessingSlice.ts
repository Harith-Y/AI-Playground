import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

interface PreprocessingStep {
  id: string;
  type: string;
  parameters: Record<string, any>;
  order: number;
}

interface PreprocessingState {
  steps: PreprocessingStep[];
  currentStep: PreprocessingStep | null;
  history: PreprocessingStep[][];
  preview: any[];
  isProcessing: boolean;
  error: string | null;
}

const initialState: PreprocessingState = {
  steps: [],
  currentStep: null,
  history: [],
  preview: [],
  isProcessing: false,
  error: null,
};

// Async thunks
export const applyPreprocessingStep = createAsyncThunk(
  'preprocessing/applyStep',
  async (step: PreprocessingStep, { rejectWithValue }) => {
    try {
      // TODO: Implement API call
      return { step, preview: [] };
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to apply preprocessing step');
    }
  }
);

export const executePreprocessingPipeline = createAsyncThunk(
  'preprocessing/executePipeline',
  async (datasetId: string, { getState, rejectWithValue }) => {
    try {
      // TODO: Implement API call with all steps
      return { success: true };
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to execute pipeline');
    }
  }
);

const preprocessingSlice = createSlice({
  name: 'preprocessing',
  initialState,
  reducers: {
    addStep: (state, action: PayloadAction<PreprocessingStep>) => {
      state.steps.push(action.payload);
      state.history.push([...state.steps]);
    },
    removeStep: (state, action: PayloadAction<string>) => {
      state.steps = state.steps.filter(step => step.id !== action.payload);
      state.history.push([...state.steps]);
    },
    updateStep: (state, action: PayloadAction<PreprocessingStep>) => {
      const index = state.steps.findIndex(step => step.id === action.payload.id);
      if (index !== -1) {
        state.steps[index] = action.payload;
        state.history.push([...state.steps]);
      }
    },
    reorderSteps: (state, action: PayloadAction<PreprocessingStep[]>) => {
      state.steps = action.payload;
      state.history.push([...state.steps]);
    },
    setCurrentStep: (state, action: PayloadAction<PreprocessingStep | null>) => {
      state.currentStep = action.payload;
    },
    undoLastStep: (state) => {
      if (state.history.length > 1) {
        state.history.pop();
        state.steps = [...state.history[state.history.length - 1]];
      }
    },
    clearPreprocessing: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      .addCase(applyPreprocessingStep.pending, (state) => {
        state.isProcessing = true;
        state.error = null;
      })
      .addCase(applyPreprocessingStep.fulfilled, (state, action) => {
        state.isProcessing = false;
        state.preview = action.payload.preview;
      })
      .addCase(applyPreprocessingStep.rejected, (state, action) => {
        state.isProcessing = false;
        state.error = action.payload as string;
      })
      .addCase(executePreprocessingPipeline.pending, (state) => {
        state.isProcessing = true;
        state.error = null;
      })
      .addCase(executePreprocessingPipeline.fulfilled, (state) => {
        state.isProcessing = false;
      })
      .addCase(executePreprocessingPipeline.rejected, (state, action) => {
        state.isProcessing = false;
        state.error = action.payload as string;
      });
  },
});

export const {
  addStep,
  removeStep,
  updateStep,
  reorderSteps,
  setCurrentStep,
  undoLastStep,
  clearPreprocessing,
} = preprocessingSlice.actions;

export default preprocessingSlice.reducer;
