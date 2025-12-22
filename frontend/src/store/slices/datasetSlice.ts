import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import type { Dataset, DatasetStats, ColumnInfo } from '../../types/dataset';
import { datasetService } from '../../services/datasetService';

interface DatasetState {
  currentDataset: Dataset | null;
  datasets: Dataset[];
  stats: DatasetStats | null;
  columns: ColumnInfo[];
  preview: any[];
  isLoading: boolean;
  error: string | null;
  uploadProgress: number;
}

const initialState: DatasetState = {
  currentDataset: null,
  datasets: [],
  stats: null,
  columns: [],
  preview: [],
  isLoading: false,
  error: null,
  uploadProgress: 0,
};

// Async thunks
export const uploadDataset = createAsyncThunk(
  'dataset/uploadDataset',
  async (file: File, { rejectWithValue }) => {
    try {
      const response = await datasetService.uploadDataset(file);
      return response;
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to upload dataset');
    }
  }
);

export const fetchDatasets = createAsyncThunk(
  'dataset/fetchDatasets',
  async (_, { rejectWithValue }) => {
    try {
      const response = await datasetService.getDatasets();
      return response;
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to fetch datasets');
    }
  }
);

export const fetchDatasetStats = createAsyncThunk(
  'dataset/fetchDatasetStats',
  async (datasetId: string, { rejectWithValue }) => {
    try {
      const response = await datasetService.getDatasetStats(datasetId);
      return response;
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to fetch dataset stats');
    }
  }
);

export const fetchDatasetPreview = createAsyncThunk(
  'dataset/fetchDatasetPreview',
  async (datasetId: string, { rejectWithValue }) => {
    try {
      const response = await datasetService.getDatasetPreview(datasetId);
      return response;
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to fetch dataset preview');
    }
  }
);

const datasetSlice = createSlice({
  name: 'dataset',
  initialState,
  reducers: {
    setCurrentDataset: (state, action: PayloadAction<Dataset | null>) => {
      state.currentDataset = action.payload;
    },
    clearDatasetError: (state) => {
      state.error = null;
    },
    setUploadProgress: (state, action: PayloadAction<number>) => {
      state.uploadProgress = action.payload;
    },
    resetDatasetState: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      // Upload dataset
      .addCase(uploadDataset.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(uploadDataset.fulfilled, (state, action) => {
        state.isLoading = false;
        state.currentDataset = action.payload;
        state.datasets.push(action.payload);
        state.uploadProgress = 0;
      })
      .addCase(uploadDataset.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
        state.uploadProgress = 0;
      })
      // Fetch datasets
      .addCase(fetchDatasets.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchDatasets.fulfilled, (state, action) => {
        state.isLoading = false;
        state.datasets = action.payload;
      })
      .addCase(fetchDatasets.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      // Fetch dataset stats
      .addCase(fetchDatasetStats.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchDatasetStats.fulfilled, (state, action) => {
        state.isLoading = false;
        state.stats = action.payload.stats;
        state.columns = action.payload.columns;
      })
      .addCase(fetchDatasetStats.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      // Fetch dataset preview
      .addCase(fetchDatasetPreview.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchDatasetPreview.fulfilled, (state, action) => {
        state.isLoading = false;
        state.preview = action.payload;
      })
      .addCase(fetchDatasetPreview.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });
  },
});

export const {
  setCurrentDataset,
  clearDatasetError,
  setUploadProgress,
  resetDatasetState,
} = datasetSlice.actions;

export default datasetSlice.reducer;
