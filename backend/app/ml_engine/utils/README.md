# ML Engine Utilities

This module contains utility functions and classes for the ML Engine.

## Column Type Detector

Automatic detection of column types in pandas DataFrames beyond basic pandas dtypes.

### Features

- **16 distinct column types** including:
  - Identifiers (ID)
  - Temporal (datetime, date, time)
  - Numeric (continuous, discrete, binary)
  - Categorical (ordinal, nominal, binary)
  - Text (long, short)
  - Special (boolean, constant, mixed, unknown)

- **Smart heuristics** based on:
  - Unique value ratios
  - Data patterns (regex matching)
  - Statistical properties
  - String length analysis
  - Name pattern matching

- **Configurable thresholds** for categorical, ID, and text detection
- **Efficient sampling** for large datasets
- **Detailed column information** with statistics

### Quick Start

```python
from app.ml_engine.utils import detect_column_types, ColumnType
import pandas as pd

# Create a DataFrame
df = pd.DataFrame({
    'user_id': range(1000),
    'age': [25, 30, 35, ...],
    'country': ['USA', 'UK', 'Canada', ...],
    'description': ['Long text about...', ...],
})

# Detect column types
types = detect_column_types(df)

# Print results
for col, col_type in types.items():
    print(f"{col}: {col_type.value}")

# Output:
# user_id: id
# age: numeric_discrete
# country: categorical_nominal
# description: text_long
```

### Advanced Usage

```python
from app.ml_engine.utils import ColumnTypeDetector

# Create detector with custom thresholds
detector = ColumnTypeDetector(
    categorical_threshold=0.1,  # 10% unique ratio for categorical
    id_threshold=0.90,          # 90% unique ratio for IDs
    text_length_threshold=30,   # Avg length > 30 for long text
    sample_size=5000,           # Sample size for large datasets
)

# Detect types
types = detector.detect(df)

# Get detailed information
column_info = detector.get_column_info(df)
print(column_info)
```

### Column Types Reference

| Type | Description | Example |
|------|-------------|---------|
| `id` | Unique identifiers | user_id, order_id |
| `datetime` | Timestamp columns | created_at, updated_at |
| `date` | Date-only columns | birth_date, order_date |
| `time` | Time-only columns | appointment_time |
| `numeric_continuous` | Continuous numeric | price, temperature |
| `numeric_discrete` | Discrete numeric | age, count |
| `numeric_binary` | Binary 0/1 numeric | is_active (0/1) |
| `categorical_ordinal` | Ordered categories | education level |
| `categorical_nominal` | Unordered categories | country, color |
| `categorical_binary` | Two-category | gender, status |
| `text_long` | Long text | descriptions, reviews |
| `text_short` | Short text | names, titles |
| `boolean` | Boolean values | is_premium, verified |
| `constant` | Single value | platform='Web' |
| `mixed` | Mixed types | problematic data |
| `unknown` | Cannot determine | edge cases |

### Use Cases

1. **Automatic Preprocessing**
   ```python
   types = detect_column_types(df)

   # Apply appropriate preprocessing
   for col, col_type in types.items():
       if col_type == ColumnType.NUMERIC_CONTINUOUS:
           df[col] = StandardScaler().fit_transform(df[[col]])
       elif col_type == ColumnType.CATEGORICAL_NOMINAL:
           df = pd.get_dummies(df, columns=[col])
   ```

2. **Feature Selection**
   ```python
   types = detect_column_types(df)

   # Remove ID and constant columns
   cols_to_drop = [col for col, t in types.items()
                   if t in [ColumnType.ID, ColumnType.CONSTANT]]
   df = df.drop(columns=cols_to_drop)
   ```

3. **Data Quality Analysis**
   ```python
   detector = ColumnTypeDetector()
   info = detector.get_column_info(df)

   # Find problematic columns
   mixed_cols = info[info['detected_type'] == 'mixed']['column'].tolist()
   high_null_cols = info[info['null_percentage'] > 50]['column'].tolist()
   ```

### Configuration

#### Categorical Threshold
- Default: 0.05 (5%)
- Controls when a column is considered categorical vs text
- Lower = more columns classified as categorical

#### ID Threshold
- Default: 0.95 (95%)
- Controls when a column is considered an ID
- Higher = stricter ID detection

#### Text Length Threshold
- Default: 50 characters
- Separates long text from short text
- Higher = more columns classified as short text

### Performance

- For datasets > 10,000 rows, automatic sampling is used
- Detection typically takes < 1 second for 100k rows
- Memory efficient - only samples data for analysis

### Example Output

```
Column                   Type                  Unique%  Null%
user_id                  id                    100.0    0.0
age                      numeric_discrete      62.0     0.0
income                   numeric_continuous    98.5     5.0
country                  categorical_nominal   4.0      0.0
education                categorical_ordinal   4.0      0.0
gender                   categorical_binary    2.0      0.0
is_premium               boolean               2.0      0.0
signup_date              datetime              100.0    0.0
customer_name            text_short            95.0     0.0
product_review           text_long             87.0     3.0
platform                 constant              1.0      0.0
```

See `examples/column_type_detection_example.py` for a complete working example.
