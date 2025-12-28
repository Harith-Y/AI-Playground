import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import type {
  PreprocessingStep,
  PreprocessingStepCreate,
  PreprocessingStepUpdate,
  PreprocessingApplyRequest,
  PreprocessingApplyResponse,
  PreprocessingAsyncResponse,
  PreprocessingTaskStatus,
} from '../../types/preprocessing';
import axios from 'axios';
import { getErrorMessage } from '../../utils/preprocessingValidation';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface PreprocessingState {
  steps: PreprocessingStep[];
  currentStep: PreprocessingStep | null;
  isLoading: boolean;
  isProcessing: boolean;
  error: string | null;
  // Pipeline execution state
  pipelineResult: PreprocessingApplyResponse | null;
  currentTaskId: string | null;
  taskStatus: PreprocessingTaskStatus | null;
  isExecuting: boolean;
}

const initialState: PreprocessingState = {
  steps: [],
  currentStep: null,
  isLoading: false,
  isProcessing: false,
  error: null,
  pipelineResult: null,
  currentTaskId: null,
  taskStatus: null,
  isExecuting: false,
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
      return rejectWithValue(getErrorMessage(error));
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
      return rejectWithValue(getErrorMessage(error));
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
      return rejectWithValue(getErrorMessage(error));
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
      return rejectWithValue(getErrorMessage(error));
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
      return rejectWithValue(getErrorMessage(error));
    }
  }
);

// Apply preprocessing pipeline (synchronous)
export const applyPreprocessingPipeline = createAsyncThunk(
  'preprocessing/applyPipeline',
  async (request: PreprocessingApplyRequest, { rejectWithValue }) => {
    try {
      const response = await axios.post<PreprocessingApplyResponse>(
        `${API_URL}/api/v1/preprocessing/apply`,
        request
      );
      return response.data;
    } catch (error: any) {
      return rejectWithValue(getErrorMessage(error));
    }
  }
);

// Apply preprocessing pipeline asynchronously
export const applyPreprocessingPipelineAsync = createAsyncThunk(
  'preprocessing/applyPipelineAsync',
  async (request: PreprocessingApplyRequest, { rejectWithValue }) => {
    try {
      const response = await axios.post<PreprocessingAsyncResponse>(
        `${API_URL}/api/v1/preprocessing/apply/async`,
        request
      );
      return response.data;
    } catch (error: any) {
      return rejectWithValue(getErrorMessage(error));
    }
  }
);

// Get preprocessing task status
export const getPreprocessingTaskStatus = createAsyncThunk(
  'preprocessing/getTaskStatus',
  async (taskId: string, { rejectWithValue }) => {
    try {
      const response = await axios.get<PreprocessingTaskStatus>(
        `${API_URL}/api/v1/preprocessing/task/${taskId}`
      );
      return response.data;
    } catch (error: any) {
      return rejectWithValue(getErrorMessage(error));
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
    clearPipelineResult: (state) => {
      state.pipelineResult = null;
      state.currentTaskId = null;
      state.taskStatus = null;
      state.isExecuting = false;
    },
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
      })
      // Apply preprocessing pipeline (sync)
      .addCase(applyPreprocessingPipeline.pending, (state) => {
        state.isExecuting = true;
        state.error = null;
        state.pipelineResult = null;
      })
      .addCase(applyPreprocessingPipeline.fulfilled, (state, action) => {
        state.isExecuting = false;
        state.pipelineResult = action.payload;
      })
      .addCase(applyPreprocessingPipeline.rejected, (state, action) => {
        state.isExecuting = false;
        state.error = action.payload as string;
      })
      // Apply preprocessing pipeline (async)
      .addCase(applyPreprocessingPipelineAsync.pending, (state) => {
        state.isExecuting = true;
        state.error = null;
        state.pipelineResult = null;
        state.taskStatus = null;
      })
      .addCase(applyPreprocessingPipelineAsync.fulfilled, (state, action) => {
        state.currentTaskId = action.payload.task_id;
        state.taskStatus = {
          task_id: action.payload.task_id,
          state: 'PENDING',
          status: action.payload.status,
          progress: 0,
        };
      })
      .addCase(applyPreprocessingPipelineAsync.rejected, (state, action) => {
        state.isExecuting = false;
        state.error = action.payload as string;
      })
      // Get task status
      .addCase(getPreprocessingTaskStatus.fulfilled, (state, action) => {
        state.taskStatus = action.payload;

        // If task completed successfully
        if (action.payload.state === 'SUCCESS' && action.payload.result) {
          state.isExecuting = false;
          state.pipelineResult = action.payload.result;
        }

        // If task failed
        if (action.payload.state === 'FAILURE') {
          state.isExecuting = false;
          state.error = action.payload.error || 'Preprocessing task failed';
        }
      })
      .addCase(getPreprocessingTaskStatus.rejected, (state, action) => {
        state.error = action.payload as string;
      });
  },
});

export const {
  setCurrentStep,
  clearError,
  clearPreprocessing,
  clearPipelineResult,
} = preprocessingSlice.actions;

export default preprocessingSlice.reducer;
