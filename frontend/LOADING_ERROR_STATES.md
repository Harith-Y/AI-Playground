# Loading and Error States Implementation

This document provides an overview of the loading and error state implementation across the frontend application.

## Reusable Components Created

### 1. LoadingState Component
**Location:** `src/components/common/LoadingState.tsx`

A flexible loading component that supports:
- Circular or linear progress indicators
- Optional loading messages
- Fullscreen or inline display modes
- Customizable size

**Usage:**
```tsx
<LoadingState message="Loading data..." variant="circular" />
<LoadingState message="Processing..." variant="linear" fullscreen />
```

### 2. ErrorState Component
**Location:** `src/components/common/ErrorState.tsx`

A comprehensive error display component with:
- Error title and message
- Optional retry button with callback
- Alert or banner display variants
- Fullscreen or inline modes

**Usage:**
```tsx
<ErrorState
  title="Failed to Load"
  message={error}
  onRetry={handleRetry}
  variant="alert"
/>
```

### 3. EmptyState Component
**Location:** `src/components/common/EmptyState.tsx`

A user-friendly empty state component featuring:
- Customizable icon
- Title and message
- Optional call-to-action button
- Fullscreen or inline display

**Usage:**
```tsx
<EmptyState
  title="No Data Available"
  message="Upload a dataset to get started"
  action={{ label: "Upload", onClick: handleUpload }}
/>
```

### 4. Loading Component (Updated)
**Location:** `src/components/common/Loading.tsx`

Wrapper around LoadingSpinner for backward compatibility.

## Pages Updated with Loading/Error States

### 1. ModelingPage
**Location:** `src/pages/ModelingPage.tsx`

**Features:**
- Loading state while fetching model configurations
- Error state with retry functionality
- Empty state when no models available
- Training progress display
- Real-time training metrics

**State Management:**
- Uses `isLoading`, `isTraining`, `error` from modeling slice
- Integrates TrainingProgress component
- Handles model selection and training controls

### 2. ExplorationPage (Model Evaluation)
**Location:** `src/pages/ExplorationPage.tsx`

**Features:**
- Loading state during model evaluation
- Error state with retry
- Empty state when no model selected
- Evaluation metrics display
- Confusion matrix visualization

**State Management:**
- Uses `isEvaluating`, `error` from evaluation slice
- Checks for `currentModel` from modeling slice

### 3. PreprocessingPage
**Location:** `src/pages/PreprocessingPage.tsx`

**Features:**
- Loading state during data transformations
- Error state with retry
- Empty state when no dataset selected
- Preprocessing steps management
- Add step functionality

**State Management:**
- Uses `isProcessing`, `error`, `steps` from preprocessing slice
- Checks for `currentDataset` from dataset slice

### 4. FeatureEngineeringPage
**Location:** `src/pages/FeatureEngineeringPage.tsx`

**Features:**
- Loading state during feature analysis
- Error state with retry
- Empty state when no dataset available
- Correlation matrix display
- Feature importance visualization

**State Management:**
- Uses `isLoading`, `error` from feature slice
- Displays CorrelationMatrix and FeatureImportance components

### 5. TuningPage
**Location:** `src/pages/TuningPage.tsx`

**Features:**
- Loading state for tuning configurations
- Error state with retry
- Empty state when no model selected
- Tuning progress tracking
- Start/stop controls
- Results display

**State Management:**
- Uses `isTuning`, `progress`, `isLoading`, `error` from tuning slice
- Real-time progress updates

### 6. CodeGenerationPage
**Location:** `src/pages/CodeGenerationPage.tsx`

**Features:**
- Loading state during code generation
- Error state with retry
- Empty state when no pipeline exists
- Pipeline summary display
- Code preview and download

**State Management:**
- Uses local state for generation status
- Checks for `currentModel` and `steps`

## Component-Level Loading/Error States

### 1. TrainingProgress Component (New)
**Location:** `src/components/modeling/TrainingProgress.tsx`

**Features:**
- Progress bar with percentage
- Training status chip
- Real-time metrics display
- Training logs viewer
- Error highlighting in logs

### 2. CorrelationMatrix Component
**Location:** `src/components/features/CorrelationMatrix.tsx`

**Features:**
- Loading state during calculation
- Empty state when no data
- Placeholder for visualization

### 3. FeatureImportance Component
**Location:** `src/components/features/FeatureImportance.tsx`

**Features:**
- Loading state during calculation
- Empty state when no features
- Placeholder for chart

### 4. ConfusionMatrix Component
**Location:** `src/components/evaluation/ConfusionMatrix.tsx`

**Features:**
- Loading state during generation
- Empty state when no matrix
- Placeholder for visualization

### 5. EvaluationMetrics Component
**Location:** `src/components/evaluation/EvaluationMetrics.tsx`

**Features:**
- Loading state during calculation
- Empty state when no metrics
- Grid display for metrics
- Formatted values (4 decimal places)

## Consistent Patterns

### State Management Pattern
All pages follow this pattern:
```tsx
const { data, isLoading, error } = useAppSelector(state => state.slice);

if (isLoading) {
  return <LoadingState message="Loading..." />;
}

if (error) {
  return <ErrorState message={error} onRetry={handleRetry} />;
}

if (!data) {
  return <EmptyState title="No Data" message="..." />;
}

return <MainContent />;
```

### Error Handling
- All pages include retry functionality
- Error clearing on component unmount
- Inline error alerts for non-blocking errors
- Fullscreen error states for critical failures

### Loading Indicators
- **Circular Progress:** For general loading (fetching data)
- **Linear Progress:** For ongoing processes (uploads, calculations)
- **Progress Bars:** For trackable operations (training, tuning)

### Empty States
- Informative messages guiding users on next steps
- Relevant icons matching the content type
- Optional action buttons to resolve empty state

## Integration with Redux

All pages integrate with existing Redux slices:
- `datasetSlice`: Dataset data and operations
- `modelingSlice`: Model training and management
- `preprocessingSlice`: Data preprocessing
- `featureSlice`: Feature analysis
- `evaluationSlice`: Model evaluation
- `tuningSlice`: Hyperparameter tuning

### Available State Flags
- `isLoading`: General loading state
- `isTraining`: Model training in progress
- `isProcessing`: Data processing in progress
- `isEvaluating`: Model evaluation in progress
- `isTuning`: Hyperparameter tuning in progress
- `error`: Error message (string | null)

### Available Actions
- `clearDatasetError()`
- `clearModelError()`
- `clearFeatureError()`
- `clearEvaluationError()`
- `clearTuningError()`

## Material-UI Components Used

- `CircularProgress`: Circular loading spinner
- `LinearProgress`: Linear progress bar
- `Alert`: Error/warning/info messages
- `Snackbar`: Toast notifications
- `Card/CardContent`: Content containers
- `Button`: Action buttons
- `Typography`: Text elements
- `Box/Grid/Container`: Layout components

## Styling Consistency

All components use consistent styling:
- Border: `1px solid #e2e8f0`
- Background: `#FFFFFF` or `#f8fafc`
- Primary color for icons and accents
- Typography hierarchy: h4 for page titles, h6 for sections
- Spacing: 2-4 units for margins/padding

## Best Practices Implemented

1. **User Feedback:** Clear messages about what's happening
2. **Error Recovery:** Retry buttons for failed operations
3. **Progressive Disclosure:** Show only relevant information
4. **Accessibility:** ARIA-friendly components from Material-UI
5. **Consistency:** Uniform patterns across all pages
6. **Performance:** Conditional rendering to avoid unnecessary work
7. **State Management:** Centralized Redux state
8. **Component Reusability:** DRY principle with shared components

## Future Enhancements

1. **Skeleton Screens:** Add placeholder content during loading
2. **Retry Logic:** Implement automatic retry with exponential backoff
3. **Analytics:** Track loading times and error rates
4. **WebSocket Integration:** Real-time updates for long-running operations
5. **Notifications:** Global notification system with Snackbar
6. **Progress Estimation:** Better progress indicators for async operations

## Testing Recommendations

1. Test loading states with delayed API responses
2. Test error states with failed API calls
3. Test empty states with no data
4. Test retry functionality
5. Test state transitions (loading → success/error)
6. Test component unmounting during loading
7. Test concurrent operations

## Summary

The frontend now has comprehensive loading and error states across all pages and components. This provides:
- Better user experience with clear feedback
- Consistent UI/UX patterns
- Robust error handling
- Easy maintenance and extensibility
- Foundation for future enhancements
