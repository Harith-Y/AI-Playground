# Edge Case Quick Reference Guide

## Quick Checklist

Before training a model, check for these common edge cases:

### ✅ Dataset Size

- [ ] At least 20 samples total
- [ ] At least 10 samples per class (classification)
- [ ] At least 5 samples in test set

### ✅ Categorical Features

- [ ] No features with > 100 unique values
- [ ] High-cardinality features grouped or encoded appropriately

### ✅ Class Balance (Classification)

- [ ] No classes with only 1 sample
- [ ] Imbalance ratio < 100:1
- [ ] Each class has at least 2 samples for stratification

### ✅ Missing Data

- [ ] No columns with > 90% null values
- [ ] Missing value strategy defined

### ✅ Feature Quality

- [ ] No constant features (all same value)
- [ ] No features with only 1 unique value

## Common Edge Cases & Solutions

### 1. "Dataset too small" (< 10 samples)

**Error**: `InsufficientDataError: Dataset has only 8 samples (minimum: 10)`

**Solutions:**

- Collect more data (recommended)
- Use cross-validation instead of train/test split
- Consider simpler models
- Use domain knowledge instead of ML

---

### 2. "High cardinality feature" (> 100 unique values)

**Warning**: `Column 'city' has 125 unique values - will create 124 features`

**Auto-fix**: ✅ Automatically caps to top 50 categories

**Manual solutions:**

- Use target encoding instead of one-hot
- Group rare categories manually
- Use embeddings (neural networks)
- Consider if feature should be used

---

### 3. "Extreme class imbalance" (> 100:1)

**Warning**: `Extreme imbalance: 120:1 ratio detected`

**Auto-fix**: ⚠️ Uses stratification automatically

**Manual solutions:**

- Use class weights in model
- Apply SMOTE (if minority class has > 5 samples)
- Undersample majority class
- Use appropriate metrics (F1, ROC-AUC, not accuracy)
- Collect more minority class samples

---

### 4. "Single-sample class"

**Critical**: `Classes with only 1 sample: ['rare_class']`

**Auto-fix**: ✅ Automatically removes single-sample classes

**Manual solutions:**

- Collect more samples for that class
- Merge with similar class
- Remove the class
- Disable stratification (not recommended)

---

### 5. "Mostly null column" (> 90% null)

**Critical**: `Column 'optional_field' is 95% null`

**Auto-fix**: ✅ Automatically removes mostly-null columns

**Manual solutions:**

- Remove the column
- Investigate why data is missing (MCAR, MAR, MNAR)
- Impute carefully if data is informative
- Create 'is_missing' indicator feature

---

### 6. "Constant feature"

**Error**: `Column 'status' has only 1 unique value (constant feature)`

**Auto-fix**: ✅ Automatically removes constant features

**Manual solutions:**

- Remove the feature
- Check data filtering issues
- Verify this isn't a data quality problem

---

### 7. "Stratification failed"

**Error**: `Cannot stratify with classes having < 2 samples`

**Auto-fix**: ✅ Automatically falls back to non-stratified split

**Manual solutions:**

- Disable stratification
- Remove or merge rare classes
- Collect more data

---

## When Auto-Fixes Are Applied

The system automatically applies fixes in this order:

1. **Remove constant features** (no predictive value)
2. **Remove mostly-null columns** (> 90% null)
3. **Cap high-cardinality features** (keep top 50 + 'Other')
4. **Remove single-sample classes** (cannot split)
5. **Disable stratification** (if unsafe)

## Validation Process

```
Load Dataset
    ↓
Edge Case Validation ← Detects issues
    ↓
Auto-Fix (if possible) ← Applies fixes
    ↓
Re-validate ← Ensures issues resolved
    ↓
Training (or Error if still invalid)
```

## Severity Guide

| Severity    | Meaning                 | Action                   |
| ----------- | ----------------------- | ------------------------ |
| 🔵 INFO     | Informational           | Review, no action needed |
| 🟡 WARNING  | May affect performance  | Review, consider fixing  |
| 🟠 ERROR    | Will cause poor results | Should fix               |
| 🔴 CRITICAL | Will cause crashes      | Must fix                 |

## Log Examples

### ✅ Good Dataset

```
[INFO] Validating dataset: 500 rows, 10 columns
[INFO] Edge case validation passed
```

### ⚠️ Dataset with Warnings

```
[WARNING] Small dataset with 18 samples (recommended: 20+)
[WARNING] Class imbalance detected: 12:1 ratio
[INFO] Using stratified split to preserve class distribution
```

### 🔴 Critical Issues (Auto-Fixed)

```
[CRITICAL] Column 'product_id' has 150 unique values (high cardinality)
[INFO] Auto-fix applied: Capped 'product_id' from 150 to 50 categories
[CRITICAL] Classes with only 1 sample: ['rare_class']
[INFO] Auto-fix applied: Removed 1 rows with single-sample classes
[INFO] Edge case validation passed after auto-fixes
```

### ❌ Critical Issues (Cannot Fix)

```
[CRITICAL] Dataset has only 5 samples (minimum: 10)
ERROR: Dataset has critical issues that could not be auto-fixed
  - [CRITICAL] Dataset has only 5 samples
    Recommendation: Collect more data (target: 100+ samples)
```

## Best Practices

### 1. Data Collection

- Aim for 100+ samples minimum
- Balanced classes when possible
- Avoid columns with > 50 categories
- Minimize missing data

### 2. During Development

- Check validation logs before each training
- Review auto-fixes to understand changes
- Test with edge cases during development

### 3. Production

- Monitor edge case occurrences
- Alert on critical issues
- Document auto-fix decisions
- Regular data quality audits

## Testing Edge Cases

Use the test suite to verify handling:

```bash
pytest tests/ml_engine/test_edge_case_integration.py -v
```

Test cases included:

- Tiny datasets (8, 12, 15 samples)
- High cardinality (75, 120, 150 unique values)
- Extreme imbalance (100:1, 10:1 ratios)
- Single-sample classes
- Mostly-null columns
- Constant features
- Multiple edge cases simultaneously

## API Response with Edge Cases

```json
{
  "status": "completed",
  "edge_cases_detected": [
    {
      "severity": "warning",
      "category": "high_cardinality",
      "message": "Column 'city' has 125 unique values",
      "auto_fixed": true,
      "fix_applied": "Capped to top 50 categories"
    },
    {
      "severity": "warning",
      "category": "class_imbalance",
      "message": "Class imbalance: 15:1 ratio",
      "auto_fixed": false,
      "recommendation": "Using stratified split"
    }
  ]
}
```

## Need Help?

- Check logs for detailed recommendations
- See [EDGE_CASE_HANDLING.md](EDGE_CASE_HANDLING.md) for full documentation
- Review test cases for examples
- Consult domain experts for class merging decisions

---

**Remember**: Edge case handling is automatic but understanding the issues helps make better decisions about your data!
