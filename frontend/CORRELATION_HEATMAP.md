# Correlation Heatmap Component

A fully-featured interactive correlation matrix heatmap visualization component for analyzing feature relationships in your dataset.

## Overview

The Correlation Heatmap provides a visual representation of Pearson correlation coefficients between numerical features, making it easy to identify:
- Strong positive correlations (features that increase together)
- Strong negative correlations (features that move in opposite directions)
- Weak or no correlations (independent features)

## Components

### 1. CorrelationHeatmap Component
**Location:** `src/components/features/CorrelationHeatmap.tsx`

The core visualization component that renders the interactive heatmap.

#### Features

**Visual Design:**
- Color-coded cells using a red-white-blue gradient
  - Red shades: Negative correlations (-1.0 to 0.0)
  - White: No correlation (0.0)
  - Blue shades: Positive correlations (0.0 to +1.0)
- Responsive cell sizing based on the number of features
- Diagonal cells always show 1.0 (perfect self-correlation)

**Interactive Features:**
- Hover effects with scale transformation
- Tooltips showing:
  - Feature pair names
  - Exact correlation value
- Cell highlighting on hover
- Smooth transitions and animations

**Layout:**
- Rotated top labels (45-degree angle) for readability
- Right-aligned left labels
- Automatic sizing based on feature count
- Scrollable container for large matrices

**Color Scale Legend:**
- Visual gradient bar showing the full color spectrum
- Labeled endpoints (-1.0, 0.0, +1.0)
- Clear indication of correlation strength

**Interpretation Guide:**
- Built-in reference panel explaining correlation values:
  - Strong positive (0.7 to 1.0)
  - Moderate positive (0.3 to 0.7)
  - Weak (-0.3 to 0.3)
  - Moderate negative (-0.7 to -0.3)
  - Strong negative (-1.0 to -0.7)

#### Props

```typescript
interface CorrelationHeatmapProps {
  features: string[];      // Array of feature names
  matrix: number[][];      // 2D array of correlation values
  width?: number;          // Total width (default: 600)
  height?: number;         // Total height (default: 600)
}
```

#### Usage Example

```tsx
import CorrelationHeatmap from './CorrelationHeatmap';

const MyComponent = () => {
  const data = {
    features: ['Age', 'Income', 'Education'],
    matrix: [
      [1.0, 0.45, 0.32],
      [0.45, 1.0, 0.67],
      [0.32, 0.67, 1.0],
    ],
  };

  return (
    <CorrelationHeatmap
      features={data.features}
      matrix={data.matrix}
      width={700}
      height={700}
    />
  );
};
```

### 2. CorrelationMatrix Component (Updated)
**Location:** `src/components/features/CorrelationMatrix.tsx`

Wrapper component that integrates the heatmap with loading, error, and empty states.

#### Features

- Loading state with linear progress indicator
- Empty state when no data is available
- Optional refresh button to recalculate correlations
- Descriptive subtitle explaining the correlation type
- Full integration with the CorrelationHeatmap component

#### Props

```typescript
interface CorrelationMatrixData {
  features: string[];
  matrix: number[][];
}

interface CorrelationMatrixProps {
  matrix?: CorrelationMatrixData | null;
  isLoading?: boolean;
  onRefresh?: () => void;
}
```

#### Usage Example

```tsx
import CorrelationMatrix from './CorrelationMatrix';

const FeaturePage = () => {
  const { correlationMatrix, isLoading } = useAppSelector(
    (state) => state.feature
  );

  const handleRefresh = () => {
    dispatch(fetchCorrelationMatrix(datasetId));
  };

  return (
    <CorrelationMatrix
      matrix={correlationMatrix}
      isLoading={isLoading}
      onRefresh={handleRefresh}
    />
  );
};
```

## Data Structure

### Input Format

The correlation matrix expects data in this format:

```typescript
{
  features: string[];      // Feature names (e.g., ['Age', 'Income', 'Score'])
  matrix: number[][];      // Symmetric matrix of correlations
}
```

### Matrix Properties

- **Symmetric:** `matrix[i][j] === matrix[j][i]`
- **Diagonal:** `matrix[i][i] === 1.0` (perfect self-correlation)
- **Range:** All values must be between -1.0 and +1.0
- **Size:** Matrix dimensions must match `features.length × features.length`

### Example

```javascript
{
  features: ['A', 'B', 'C'],
  matrix: [
    [1.0,  0.8, -0.3],  // A correlates with A(1.0), B(0.8), C(-0.3)
    [0.8,  1.0,  0.1],  // B correlates with A(0.8), B(1.0), C(0.1)
    [-0.3, 0.1,  1.0],  // C correlates with A(-0.3), B(0.1), C(1.0)
  ]
}
```

## Integration with Redux

### Feature Slice

The correlation matrix integrates with the existing feature slice:

**Location:** `src/store/slices/featureSlice.ts`

```typescript
interface CorrelationMatrix {
  features: string[];
  matrix: number[][];
}

interface FeatureState {
  correlationMatrix: CorrelationMatrix | null;
  isLoading: boolean;
  error: string | null;
  // ... other state
}
```

### Async Actions

```typescript
// Fetch correlation matrix from API
dispatch(fetchCorrelationMatrix(datasetId));
```

## Sample Data Utilities

**Location:** `src/utils/sampleData.ts`

Utility functions for generating sample correlation matrices for testing:

### Functions

1. **`generateSampleCorrelationMatrix(featureCount)`**
   - Generates a random symmetric correlation matrix
   - Parameters: `featureCount` (default: 6)
   - Returns: `{ features, matrix }`

2. **`generateRealisticCorrelationMatrix()`**
   - Generates a realistic correlation matrix with meaningful relationships
   - Uses real-world feature names (Age, Income, Education, etc.)
   - Returns: `{ features, matrix }`

3. **`generateSampleFeatureImportance(featureCount)`**
   - Generates sample feature importance scores
   - Parameters: `featureCount` (default: 10)
   - Returns: Array of `{ feature, importance }`

### Usage Example

```typescript
import { generateRealisticCorrelationMatrix } from '../utils/sampleData';

// For testing/demo purposes
const sampleData = generateRealisticCorrelationMatrix();
console.log(sampleData);
// {
//   features: ['Age', 'Income', 'Education_Years', ...],
//   matrix: [[1.0, 0.45, ...], ...]
// }
```

## Styling and Theming

### Color Scheme

The heatmap uses a diverging color scheme:

**Negative Correlations (Red):**
```
-1.0: rgb(55, 55, 55)      // Deep red
-0.5: rgb(155, 155, 155)   // Light red
 0.0: rgb(255, 255, 255)   // White
```

**Positive Correlations (Blue):**
```
 0.0: rgb(255, 255, 255)   // White
+0.5: rgb(155, 155, 255)   // Light blue
+1.0: rgb(55, 55, 255)     // Deep blue
```

### Text Contrast

- Light text (white) on strong correlations (|r| > 0.5)
- Dark text on weak correlations (|r| ≤ 0.5)
- Text shadow for better readability on colored backgrounds

### Responsive Design

- Cell size automatically adjusts based on feature count
- Text display toggles based on cell size:
  - Large cells (> 60px): Larger font
  - Medium cells (> 40px): Smaller font
  - Small cells (≤ 40px): No text (hover for values)

## Performance Considerations

### Optimization Techniques

1. **Conditional Rendering:**
   - Text only rendered when cells are large enough
   - Reduces DOM complexity for large matrices

2. **Efficient Color Calculation:**
   - Direct RGB computation without library dependencies
   - Memoization opportunities for repeated values

3. **Hover State Management:**
   - Single hover state for the entire grid
   - No individual state per cell

### Recommended Limits

- **Optimal:** 5-15 features
- **Good:** 15-25 features
- **Acceptable:** 25-50 features
- **Consider alternatives:** 50+ features (may need scrolling or different visualization)

## Accessibility

### Features

- Tooltip text descriptions for screen readers
- Keyboard navigation support via Material-UI components
- High contrast color scheme
- Clear labeling of all axes

### ARIA Attributes

The component inherits ARIA support from Material-UI components:
- Tooltips have `role="tooltip"`
- Interactive elements are keyboard accessible

## Browser Compatibility

The component uses modern CSS features:
- CSS Grid for layout
- CSS Transforms for rotations
- CSS Transitions for animations
- CSS Gradients for the legend

**Supported Browsers:**
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Future Enhancements

### Planned Features

1. **Clustering:**
   - Automatic reordering of features to group correlated features together
   - Dendrogram visualization

2. **Filtering:**
   - Show only correlations above a threshold
   - Highlight specific feature relationships

3. **Export:**
   - Download as PNG/SVG
   - Export data as CSV

4. **Statistical Significance:**
   - Display p-values
   - Visual indicators for statistically significant correlations

5. **Interactive Selection:**
   - Click cells to filter dataset
   - Multi-select for feature pair analysis

6. **Advanced Color Schemes:**
   - User-selectable color palettes
   - Colorblind-friendly modes

7. **Comparison Mode:**
   - Compare correlation matrices across different subsets
   - Before/after preprocessing comparisons

## Troubleshooting

### Common Issues

**Issue: Matrix not displaying**
- Check that `features.length === matrix.length === matrix[0].length`
- Verify all matrix values are between -1.0 and +1.0
- Ensure matrix is symmetric

**Issue: Text overlapping**
- Reduce the number of features displayed
- Increase the width/height props
- Consider using shorter feature names

**Issue: Slow rendering with many features**
- Limit display to top N most important features
- Consider using a threshold filter
- Implement virtualization for very large matrices

## Testing

### Unit Tests

Test cases to implement:

```typescript
// Color calculation
test('getColor returns correct colors for correlation values', () => {
  expect(getColor(1.0)).toBe('rgb(55, 55, 255)');    // Strong positive
  expect(getColor(0.0)).toBe('#ffffff');              // No correlation
  expect(getColor(-1.0)).toBe('rgb(255, 55, 55)');   // Strong negative
});

// Value formatting
test('formatValue formats numbers correctly', () => {
  expect(formatValue(0.123456)).toBe('0.123');
  expect(formatValue(NaN)).toBe('N/A');
});
```

### Integration Tests

```typescript
test('renders heatmap with sample data', () => {
  const data = generateSampleCorrelationMatrix(5);
  render(<CorrelationHeatmap {...data} />);
  expect(screen.getAllByRole('tooltip')).toHaveLength(25);
});
```

## API Reference

### CorrelationHeatmap

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| features | string[] | required | Array of feature names |
| matrix | number[][] | required | 2D correlation matrix |
| width | number | 600 | Total width in pixels |
| height | number | 600 | Total height in pixels |

### CorrelationMatrix

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| matrix | CorrelationMatrixData \| null | null | Correlation data object |
| isLoading | boolean | false | Show loading state |
| onRefresh | () => void | undefined | Refresh button callback |

## Examples

### Basic Usage

```tsx
<CorrelationMatrix
  matrix={correlationMatrix}
  isLoading={isLoading}
/>
```

### With Refresh Button

```tsx
<CorrelationMatrix
  matrix={correlationMatrix}
  isLoading={isLoading}
  onRefresh={() => dispatch(fetchCorrelationMatrix(datasetId))}
/>
```

### Custom Size

```tsx
<CorrelationHeatmap
  features={data.features}
  matrix={data.matrix}
  width={800}
  height={800}
/>
```

### In a Card

```tsx
<Card>
  <CardContent>
    <CorrelationMatrix
      matrix={correlationMatrix}
      isLoading={isLoading}
      onRefresh={handleRefresh}
    />
  </CardContent>
</Card>
```

## Summary

The Correlation Heatmap component provides a complete solution for visualizing feature correlations with:

✅ Interactive hover effects and tooltips
✅ Color-coded correlation strengths
✅ Built-in interpretation guide
✅ Loading and error states
✅ Responsive design
✅ Clean, professional appearance
✅ Easy integration with Redux
✅ Sample data generators for testing

The component is production-ready and can handle datasets with 5-50 features efficiently. For larger datasets, consider implementing filtering or clustering features.
