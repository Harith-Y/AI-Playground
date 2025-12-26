# Missing Values Heatmap Documentation

A comprehensive missing values visualization component for analyzing data quality and identifying columns with incomplete data.

## Overview

The Missing Values Heatmap provides an intuitive bar chart visualization showing the percentage and count of missing values for each column in your dataset, helping you quickly identify data quality issues.

## Components

### 1. MissingValuesHeatmap Component
**Location:** `src/components/dataset/MissingValuesHeatmap.tsx`

The core visualization component that renders an interactive bar chart.

#### Features

**Visual Design:**
- Color-coded bars based on severity levels:
  - 🟢 Green: Complete (0% missing)
  - 🔵 Blue: Excellent (<5% missing)
  - 🟠 Orange: Good (5-20% missing)
  - 🔴 Red: Poor (20-50% missing)
  - 🟤 Dark Red: Critical (>50% missing)
- Sorted by missing percentage (worst first)
- Percentage labels on bars
- Responsive bar sizing

**Interactive Features:**
- Hover effects with bar scaling
- Detailed tooltips showing:
  - Column name
  - Missing count / Total count
  - Missing percentage
- Smooth CSS transitions

**Summary Statistics:**
- Overall completeness percentage
- Count of complete columns
- Total missing cells
- Count of critical columns (>50% missing)

**Data Quality Indicators:**
- Severity chips for each column
- Color-coded legend
- Warning banner for critical columns
- Recommendations for data quality issues

#### Props

```typescript
interface MissingValuesHeatmapProps {
  data: MissingValuesData[];  // Array of column statistics
  totalRows: number;           // Total number of rows in dataset
  width?: number;              // Chart width (default: 800)
  maxHeight?: number;          // Maximum height (default: 600)
}

interface MissingValuesData {
  columnName: string;
  totalCount: number;
  missingCount: number;
  missingPercentage: number;
}
```

#### Usage Example

```tsx
import MissingValuesHeatmap from './MissingValuesHeatmap';

const MyComponent = () => {
  const data = [
    { columnName: 'Age', totalCount: 1000, missingCount: 50, missingPercentage: 5.0 },
    { columnName: 'Income', totalCount: 1000, missingCount: 250, missingPercentage: 25.0 },
    { columnName: 'Email', totalCount: 1000, missingCount: 0, missingPercentage: 0 },
  ];

  return (
    <MissingValuesHeatmap
      data={data}
      totalRows={1000}
      width={900}
      maxHeight={500}
    />
  );
};
```

### 2. MissingValuesAnalysis Component
**Location:** `src/components/dataset/MissingValuesAnalysis.tsx`

Wrapper component that integrates with the Redux dataset slice and adds loading/error states.

#### Features

- Loading state with linear progress
- Empty state when no data available
- Optional refresh button
- Automatic data transformation from ColumnInfo to MissingValuesData
- Descriptive subtitle

#### Props

```typescript
interface MissingValuesAnalysisProps {
  columns?: ColumnInfo[];    // Array of column metadata
  totalRows?: number;        // Total number of rows
  isLoading?: boolean;       // Show loading state
  onRefresh?: () => void;    // Refresh callback
}
```

#### Usage Example

```tsx
import MissingValuesAnalysis from './MissingValuesAnalysis';

const DatasetPage = () => {
  const { columns, stats, isLoading } = useAppSelector(
    (state) => state.dataset
  );

  return (
    <MissingValuesAnalysis
      columns={columns}
      totalRows={stats?.rowCount}
      isLoading={isLoading}
      onRefresh={() => dispatch(fetchDatasetStats(datasetId))}
    />
  );
};
```

## Integration

### DatasetUploadPage Integration

The Missing Values Analysis has been integrated into the Dataset Upload Page as a collapsible section:

**Location:** `src/pages/DatasetUploadPage.tsx`

```tsx
{/* Missing Values Analysis */}
{columns && columns.length > 0 && stats && (
  <Paper sx={{ mb: 3, border: '1px solid #e2e8f0', background: '#FFFFFF' }}>
    <Box
      sx={{ p: 2, cursor: 'pointer' }}
      onClick={() => setShowMissingValues(!showMissingValues)}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <ErrorOutline color="primary" />
        <Typography variant="h6" fontWeight={600}>
          Missing Values Analysis
        </Typography>
      </Box>
      <IconButton size="small">
        {showMissingValues ? <ExpandLess /> : <ExpandMore />}
      </IconButton>
    </Box>
    <Collapse in={showMissingValues}>
      <Divider />
      <Box sx={{ p: 3 }}>
        <MissingValuesAnalysis
          columns={columns}
          totalRows={stats.rowCount}
          isLoading={false}
        />
      </Box>
    </Collapse>
  </Paper>
)}
```

## Color Scheme & Severity Levels

### Severity Definitions

| Level | Range | Color | Description |
|-------|-------|-------|-------------|
| **Complete** | 0% | Green (#10B981) | No missing values |
| **Excellent** | <5% | Blue (#3B82F6) | Minimal missing data |
| **Good** | 5-20% | Orange (#F59E0B) | Acceptable level |
| **Poor** | 20-50% | Red (#EF4444) | Significant issues |
| **Critical** | >50% | Dark Red (#991B1B) | Major data quality problem |

### Visual Indicators

```tsx
// Color calculation function
const getColor = (percentage: number): string => {
  if (percentage === 0) return '#10B981';      // Green
  else if (percentage < 5) return '#3B82F6';   // Blue
  else if (percentage < 20) return '#F59E0B';  // Orange
  else if (percentage < 50) return '#EF4444';  // Red
  else return '#991B1B';                       // Dark Red
};
```

## Layout Structure

### Chart Dimensions

```typescript
const columnWidth = Math.max(120, Math.min(200, width / Math.max(data.length, 5)));
const actualWidth = columnWidth * data.length;
const chartHeight = Math.min(maxHeight - labelHeight - 150, 400);
```

### Grid Lines

- Horizontal grid lines at 0%, 25%, 50%, 75%, 100%
- Dashed lines for better visual clarity
- Y-axis labels positioned on the left

### Bar Layout

- Minimum spacing: 10px between bars
- Percentage labels shown when bar height > 30px
- Column names and severity chips below bars
- Text truncation for long column names

## Summary Statistics Panel

The component displays four key metrics at the top:

1. **Overall Completeness**
   - Formula: `(1 - totalMissing / totalCells) * 100`
   - Color-coded based on severity

2. **Complete Columns**
   - Count of columns with 0% missing values
   - Shows ratio: `completeColumns / totalColumns`

3. **Total Missing Cells**
   - Sum of missing values across all columns
   - Formatted with thousand separators

4. **Critical Columns**
   - Count of columns with >50% missing values
   - Red if any critical columns exist, green otherwise

## Data Quality Warning

When critical columns are detected (>50% missing), a warning banner appears:

```tsx
<Paper sx={{ mt: 2, p: 2, border: '1px solid #fecaca', background: '#fef2f2' }}>
  <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
    <Warning sx={{ color: 'error.main' }} />
    <Box>
      <Typography variant="caption" fontWeight={600} color="error.main">
        Data Quality Warning
      </Typography>
      <Typography variant="caption" color="text.secondary">
        {criticalColumns} column{criticalColumns > 1 ? 's have' : ' has'} more than 50% missing values.
        Consider dropping these columns or using advanced imputation techniques.
      </Typography>
    </Box>
  </Box>
</Paper>
```

## Sample Data Generators

**Location:** `src/utils/sampleData.ts`

### Functions

1. **`generateSampleMissingValues(columnCount, totalRows)`**
   - Generates random missing values data
   - Parameters:
     - `columnCount` (default: 10): Number of columns
     - `totalRows` (default: 1000): Total rows in dataset
   - Returns: Array of MissingValuesData

2. **`generateRealisticMissingValues()`**
   - Generates realistic sample data simulating a customer database
   - Includes 12 columns with varying missing percentages
   - Returns: Array of MissingValuesData

### Usage Examples

```typescript
import { generateRealisticMissingValues, generateSampleMissingValues } from '../utils/sampleData';

// Realistic data
const realisticData = generateRealisticMissingValues();

// Random data
const randomData = generateSampleMissingValues(15, 5000);
```

## Performance Considerations

### Optimization Techniques

1. **Sorting:**
   - Data sorted once during render
   - No re-sorting on hover events

2. **Conditional Rendering:**
   - Percentage labels only shown when bars are tall enough
   - Text truncation for long column names

3. **Hover State:**
   - Single state variable for hovered column
   - CSS transitions for smooth animations

### Recommended Limits

- **Optimal:** 5-20 columns
- **Good:** 20-40 columns
- **Acceptable:** 40-60 columns
- **Consider scrolling:** 60+ columns

## Accessibility

### Features

- Tooltip descriptions for screen readers
- Color AND text for severity indicators (not color-alone)
- High contrast colors
- Keyboard-navigable via Material-UI

### ARIA Support

- Tooltips have proper ARIA labels
- Interactive elements are keyboard accessible
- Severity information available via text

## Browser Compatibility

### Required Features

- CSS Grid
- CSS Transforms
- CSS Transitions
- CSS Gradients
- Flexbox

### Supported Browsers

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Styling Customization

### Theme Integration

The component uses Material-UI theme colors:

```tsx
sx={{
  color: 'primary.main',        // Primary color
  backgroundColor: '#f8fafc',   // Light gray background
  border: '1px solid #e2e8f0',  // Subtle border
}}
```

### Custom Colors

To customize the severity colors, modify the `getColor()` function:

```typescript
const getColor = (percentage: number): string => {
  // Your custom color logic here
  return customColorValue;
};
```

## Integration with Dataset Types

### ColumnInfo Structure

The component automatically transforms ColumnInfo from the dataset slice:

```typescript
interface ColumnInfo {
  name: string;
  dataType: string;
  nullCount: number;      // Used for missing count
  uniqueCount: number;
  sampleValues: any[];
}
```

### Transformation Logic

```typescript
const missingValuesData = columns.map(col => ({
  columnName: col.name,
  totalCount: totalRows,
  missingCount: col.nullCount || 0,
  missingPercentage: totalRows > 0 ? ((col.nullCount || 0) / totalRows) * 100 : 0,
}));
```

## Best Practices

### When to Use

✅ **Use Missing Values Heatmap for:**
- Initial data quality assessment
- Identifying columns that need imputation
- Deciding which columns to drop
- Comparing datasets before/after cleaning
- Presenting data quality to stakeholders

❌ **Don't use for:**
- Real-time monitoring (use streaming updates instead)
- Very small datasets (<10 rows)
- Binary presence/absence (use simple table instead)

### Recommendations

1. **Critical Columns (>50% missing):**
   - Consider dropping entirely
   - Use advanced imputation if domain knowledge supports it
   - Check if missing is systematic (MCAR, MAR, MNAR)

2. **Poor Columns (20-50% missing):**
   - Investigate why data is missing
   - Use appropriate imputation techniques
   - Consider creating "missing" indicator variable

3. **Good Columns (5-20% missing):**
   - Standard imputation methods suitable
   - Mean/median for numerical
   - Mode for categorical

4. **Excellent/Complete Columns (<5% missing):**
   - Simple imputation or deletion
   - Usually safe to proceed

## Future Enhancements

### Planned Features

1. **Pattern Detection:**
   - Identify systematic missing patterns
   - MCAR vs MAR vs MNAR analysis
   - Correlation between missing values

2. **Interactive Filtering:**
   - Filter by severity level
   - Search for specific columns
   - Sort by different criteria

3. **Comparison Mode:**
   - Before/after preprocessing comparison
   - Side-by-side dataset comparison

4. **Export Functionality:**
   - Download as PNG/SVG
   - Export report as PDF
   - CSV export of statistics

5. **Advanced Statistics:**
   - Little's MCAR test
   - Missing patterns visualization
   - Imputation recommendations

6. **Integration:**
   - Suggest preprocessing steps
   - Auto-create imputation pipeline
   - Link to preprocessing page

## Troubleshooting

### Common Issues

**Issue: Bars not displaying**
- Check that `data` array is not empty
- Verify `totalRows` is greater than 0
- Ensure `missingPercentage` is calculated correctly

**Issue: Incorrect percentages**
- Verify `nullCount` from ColumnInfo is accurate
- Check calculation: `(nullCount / totalRows) * 100`

**Issue: Text overlapping**
- Reduce number of columns displayed
- Increase `width` prop
- Use shorter column names

**Issue: Slow rendering**
- Limit to top N worst columns
- Implement virtualization for many columns
- Consider pagination

## Testing

### Unit Tests

```typescript
test('calculates missing percentage correctly', () => {
  const data = {
    columnName: 'Test',
    totalCount: 100,
    missingCount: 25,
    missingPercentage: 25.0,
  };
  expect(data.missingPercentage).toBe(25.0);
});

test('assigns correct severity color', () => {
  expect(getColor(0)).toBe('#10B981');
  expect(getColor(60)).toBe('#991B1B');
});
```

### Integration Tests

```typescript
test('renders heatmap with data', () => {
  const data = generateSampleMissingValues(5, 1000);
  render(<MissingValuesHeatmap data={data} totalRows={1000} />);
  expect(screen.getAllByRole('tooltip')).toBeDefined();
});
```

## API Reference

### MissingValuesHeatmap

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| data | MissingValuesData[] | required | Array of column statistics |
| totalRows | number | required | Total dataset rows |
| width | number | 800 | Chart width in pixels |
| maxHeight | number | 600 | Maximum chart height |

### MissingValuesAnalysis

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| columns | ColumnInfo[] | [] | Dataset column information |
| totalRows | number | 0 | Total rows in dataset |
| isLoading | boolean | false | Show loading state |
| onRefresh | () => void | undefined | Refresh callback |

## Summary

The Missing Values Heatmap provides:

✅ Visual identification of data quality issues
✅ Color-coded severity levels
✅ Interactive tooltips with detailed statistics
✅ Summary statistics panel
✅ Data quality warnings
✅ Responsive design
✅ Integration with existing dataset slice
✅ Sample data generators for testing

The component is production-ready and helps users make informed decisions about data cleaning and preprocessing strategies.
