import { configureStore } from '@reduxjs/toolkit';
import datasetReducer from './slices/datasetSlice';
import preprocessingReducer from './slices/preprocessingSlice';
import featureReducer from './slices/featureSlice';
import modelingReducer from './slices/modelingSlice';
import evaluationReducer from './slices/evaluationSlice';
import tuningReducer from './slices/tuningSlice';

export const store = configureStore({
  reducer: {
    dataset: datasetReducer,
    preprocessing: preprocessingReducer,
    feature: featureReducer,
    modeling: modelingReducer,
    evaluation: evaluationReducer,
    tuning: tuningReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these action types
        ignoredActions: ['dataset/uploadDataset/pending', 'dataset/uploadDataset/fulfilled'],
        // Ignore these field paths in all actions
        ignoredActionPaths: ['payload.file', 'meta.arg.file'],
        // Ignore these paths in the state
        ignoredPaths: ['dataset.file'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
