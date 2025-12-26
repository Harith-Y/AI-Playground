# Column Type Detection - Theory

## Overview

Column type detection is the process of automatically identifying the semantic type of data columns beyond basic data types (int, float, string). This helps in automatically selecting appropriate preprocessing and feature engineering techniques.

## Why is it Needed?

Pandas dtypes (int64, float64, object) don't capture the semantic meaning of data:
- An `int64` column could be: an ID, age, count, or categorical code
- An `object` column could be: categorical data, free text, or identifiers
- Knowing the semantic type enables automatic preprocessing decisions

## Column Types Detected

### 1. **Identifier (ID)**
Columns that uniquely identify records.

**Detection heuristics:**
- High uniqueness ratio (>95% unique values by default)
- Column name patterns: `*_id`, `id_*`, `*key`, `uuid`, `guid`
- Alphanumeric patterns with dashes/underscores

**Example:** `user_id`, `order_id`, `customer_key`

**Preprocessing recommendation:** Drop (not useful for modeling)

---

### 2. **Temporal Types**

#### 2a. Datetime
Date and time information.

**Detection heuristics:**
- Already `datetime64` dtype in pandas
- String columns that can be parsed as dates (>80% valid)
- Has varying time components

**Example:** `2024-01-15 14:30:00`, `signup_timestamp`

**Preprocessing recommendation:** Extract features (year, month, day, hour, day_of_week)

#### 2b. Date
Date-only information without time.

**Detection heuristics:**
- Datetime dtype with constant time component
- Date-like strings without time

**Example:** `2024-01-15`, `birth_date`

**Preprocessing recommendation:** Extract temporal features, calculate intervals

---

### 3. **Numeric Types**

#### 3a. Continuous
Real-valued measurements.

**Detection heuristics:**
- Numeric dtype with floating-point values
- Integer values with high cardinality (>50 unique values)
- Not all integers

**Example:** `price`, `temperature`, `weight_kg`

**Preprocessing recommendation:** StandardScaler, MinMaxScaler, handle outliers

#### 3b. Discrete
Count-based or integer values.

**Detection heuristics:**
- All values are integers
- Low-to-medium cardinality (<100 unique values)
- Non-negative values (often counts)

**Example:** `age`, `num_children`, `rating`

**Preprocessing recommendation:** May treat as categorical if very low cardinality, or scale

#### 3c. Binary
Numeric columns with only two values.

**Detection heuristics:**
- Exactly 2 unique values: {0, 1} or {0.0, 1.0}

**Example:** `is_active`, `has_premium`

**Preprocessing recommendation:** Keep as-is or use for stratification

---

### 4. **Categorical Types**

#### 4a. Nominal
Unordered categories.

**Detection heuristics:**
- Low uniqueness ratio (<5% by default)
- Not in ordinal patterns
- More than 2 unique values

**Example:** `color`, `city`, `product_category`

**Preprocessing recommendation:** OneHotEncoder, TargetEncoder

#### 4b. Ordinal
Ordered categories with inherent ranking.

**Detection heuristics:**
- Low uniqueness ratio
- Matches known ordinal patterns:
  - Education: elementary, high, bachelor, master, phd
  - Size: xs, s, m, l, xl, xxl
  - Rating: poor, fair, good, excellent
  - Agreement scales
- Numeric ordinal patterns (1st, 2nd, 3rd)

**Example:** `education_level`, `t_shirt_size`, `satisfaction_rating`

**Preprocessing recommendation:** OrdinalEncoder with proper ordering

#### 4c. Binary Categorical
Categorical columns with exactly two categories.

**Detection heuristics:**
- Exactly 2 unique string values
- Not numeric binary

**Example:** `gender` (Male/Female), `status` (Active/Inactive)

**Preprocessing recommendation:** LabelEncoder or binary encoding

---

### 5. **Text Types**

#### 5a. Long Text
Free-form text content.

**Detection heuristics:**
- Average string length ≥50 characters (configurable)
- High uniqueness ratio

**Example:** `product_description`, `customer_review`, `comment`

**Preprocessing recommendation:** TF-IDF, text embeddings, sentiment analysis

#### 5b. Short Text
Short text identifiers or labels.

**Detection heuristics:**
- Average string length <50 characters
- High uniqueness ratio (>50%)
- Not matching ID patterns

**Example:** `customer_name`, `title`, `short_description`

**Preprocessing recommendation:** May drop or use for lookup, rarely useful for ML

---

### 6. **Boolean**
True/false values.

**Detection heuristics:**
- Native boolean dtype
- Two values matching boolean patterns:
  - {true, false}
  - {yes, no}
  - {y, n}
  - {t, f}
  - {1, 0} as strings

**Example:** `is_verified`, `email_opt_in`

**Preprocessing recommendation:** Convert to 0/1 for modeling

---

### 7. **Special Types**

#### 7a. Constant
Columns with only one unique value.

**Detection heuristics:**
- Only 1 unique value (excluding nulls)
- All null values

**Example:** `platform: 'Web'` (all rows same)

**Preprocessing recommendation:** Drop (no information gain)

#### 7b. Mixed
Columns with mixed data types.

**Detection heuristics:**
- Column contains multiple incompatible types (strings and numbers)

**Example:** Column with values: `['text', 123, 'more text', 456]`

**Preprocessing recommendation:** Clean data or split into multiple columns

#### 7c. Unknown
Unable to determine type.

**Detection heuristics:**
- Doesn't match any other pattern
- Edge cases

**Preprocessing recommendation:** Manual inspection required

---

## Algorithm Flow

```
For each column:
1. Check for all nulls → CONSTANT
2. Check for single unique value → CONSTANT
3. Calculate uniqueness ratio = unique_values / total_values

4. Check ID patterns (name + high uniqueness) → ID

5. Check datetime:
   - Already datetime dtype or parseable → DATETIME/DATE

6. Check boolean patterns → BOOLEAN

7. If numeric dtype:
   - 2 values {0,1} → NUMERIC_BINARY
   - All integers + low cardinality → NUMERIC_DISCRETE
   - Otherwise → NUMERIC_CONTINUOUS

8. Otherwise (string/object):
   - 2 values → CATEGORICAL_BINARY
   - Low uniqueness ratio:
     - Matches ordinal patterns → CATEGORICAL_ORDINAL
     - Otherwise → CATEGORICAL_NOMINAL
   - High uniqueness ratio:
     - Long average length → TEXT_LONG
     - Short average length → TEXT_SHORT
```

## Key Parameters

- **categorical_threshold** (default: 0.05)
  - Maximum uniqueness ratio to consider categorical
  - Lower = more restrictive (fewer categoricals)
  - Higher = more permissive (more categoricals)

- **id_threshold** (default: 0.95)
  - Minimum uniqueness ratio for ID columns
  - Higher = more restrictive (only very unique columns)
  - Lower = more permissive (more IDs detected)

- **text_length_threshold** (default: 50)
  - Minimum average character length for long text
  - Higher = fewer long text detections
  - Lower = more text classified as long

- **sample_size** (default: 10000)
  - Number of rows to sample for large datasets
  - Improves performance on big data
  - Set to None to use entire dataset

## Benefits

1. **Automation:** Reduces manual inspection and configuration
2. **Accuracy:** More sophisticated than dtype checking
3. **Efficiency:** Sampling enables fast analysis of large datasets
4. **Customization:** Configurable thresholds for different domains
5. **Integration:** Enables automatic preprocessing pipeline construction

## Limitations

1. **Heuristic-based:** May misclassify edge cases
2. **Domain-specific:** Some patterns are domain-dependent
3. **Sampling trade-off:** Large dataset sampling may miss rare patterns
4. **Ordinal detection:** Limited to known patterns, may miss custom ordinals

## Usage Example

```python
from app.ml_engine.utils import detect_column_types, ColumnType

# Detect types
types = detect_column_types(df)

# Use for preprocessing decisions
for col, col_type in types.items():
    if col_type == ColumnType.NUMERIC_CONTINUOUS:
        # Apply StandardScaler
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])
    elif col_type == ColumnType.CATEGORICAL_NOMINAL:
        # Apply OneHotEncoder
        encoder = OneHotEncoder()
        encoded = encoder.fit_transform(df[[col]])
    elif col_type == ColumnType.ID:
        # Drop ID columns
        df = df.drop(columns=[col])
```

## References

- Pandas dtype system
- Feature engineering best practices
- Statistical data type inference
- AutoML systems (auto-sklearn, TPOT, H2O)
