# RobustScaler Theory

## Overview

RobustScaler is a feature scaling technique that uses statistics robust to outliers: **median** and **Interquartile Range (IQR)**.

---

## Mathematical Foundation

### Formula

```
X_scaled = (X - median(X)) / IQR(X)
```

Where:
- **median(X)** = 50th percentile (Q2)
- **IQR(X)** = Q3 - Q1 (75th percentile - 25th percentile)

### Components

1. **Centering**: Subtract the median
   - Median is the middle value when data is sorted
   - Not affected by extreme values
   - More robust than mean

2. **Scaling**: Divide by IQR
   - IQR measures the spread of the middle 50% of data
   - Ignores the extreme 25% on each end
   - More robust than standard deviation

---

## Why Robust?

### Problem with Standard Scaling

**StandardScaler** uses mean and standard deviation:
```
X_scaled = (X - mean) / std
```

**Issue**: Both mean and std are heavily influenced by outliers.

**Example**:
```
Data: [1, 2, 3, 4, 5, 100]  # 100 is an outlier

StandardScaler:
  mean = 19.17
  std = 39.67
  Result: All normal values become negative!

RobustScaler:
  median = 3.5
  IQR = 3 (Q3=4.5, Q1=1.5)
  Result: Normal values scaled reasonably, outlier remains outlier
```

---

## Quantiles Explained

### Quartiles

Data is divided into 4 equal parts:

```
|----Q1----|----Q2----|----Q3----|
   25%       50%       75%      100%
  (25th)   (median)  (75th)
```

- **Q1 (25th percentile)**: 25% of data is below this value
- **Q2 (50th percentile)**: Median, 50% below
- **Q3 (75th percentile)**: 75% of data is below this value

### IQR Calculation

```
IQR = Q3 - Q1
```

This captures the spread of the **middle 50%** of data, ignoring extreme values.

---

## Comparison with Other Scalers

| Scaler | Centering | Scaling | Outlier Sensitive |
|--------|-----------|---------|-------------------|
| **StandardScaler** | Mean | Std Dev | ✅ Yes (both) |
| **MinMaxScaler** | Min | Max - Min | ✅ Yes (both) |
| **RobustScaler** | Median | IQR | ❌ No (neither) |

---

## Visual Example

```
Original Data: [1, 2, 3, 4, 5, 100]

StandardScaler:
  [-0.46, -0.43, -0.41, -0.38, -0.36, 2.04]
  ↑ Normal values compressed, outlier dominates

RobustScaler:
  [-0.83, -0.5, -0.17, 0.17, 0.5, 32.17]
  ↑ Normal values well-scaled, outlier clearly identified
```

---

## When to Use RobustScaler

### ✅ Use When:

1. **Data has outliers** that are valid (not errors)
2. **Skewed distributions** (income, prices, etc.)
3. **Outliers are meaningful** and should be preserved
4. **Robust preprocessing** is needed before modeling

### ❌ Don't Use When:

1. **No outliers** in data (StandardScaler is fine)
2. **Outliers are errors** (remove them first)
3. **Need specific range** like [0,1] (use MinMaxScaler)
4. **Gaussian assumption** is important (use StandardScaler)

---

## Real-World Applications

### 1. Income Data
```
Incomes: [$30k, $35k, $40k, $45k, $50k, $500k]
                                          ↑ CEO salary

RobustScaler: Scales based on typical salaries ($30k-$50k)
StandardScaler: Heavily influenced by CEO salary
```

### 2. House Prices
```
Prices: [$200k, $250k, $300k, $350k, $5M]
                                      ↑ Mansion

RobustScaler: Scales based on typical homes
StandardScaler: Distorted by mansion price
```

### 3. Sensor Data
```
Readings: [20°C, 21°C, 22°C, 23°C, 150°C]
                                    ↑ Spike/anomaly

RobustScaler: Scales based on normal readings
StandardScaler: Affected by spike
```

---

## Properties

### Advantages

1. **Outlier Resistant**: Not affected by extreme values
2. **Preserves Outliers**: Doesn't remove or clip them
3. **Interpretable**: Median and IQR are easy to understand
4. **Stable**: Less variance in scaling parameters

### Limitations

1. **Not Bounded**: Output range is not fixed (unlike MinMaxScaler)
2. **Assumes Symmetry**: Works best with roughly symmetric distributions
3. **Requires Sufficient Data**: Needs enough data for reliable quartiles
4. **Not Gaussian**: Doesn't produce normally distributed output

---

## Customization

### Quantile Range

Default: (25, 75) for standard IQR

Can be customized:
```python
# Wider range (more data included, less robust)
RobustScaler(quantile_range=(10, 90))  # 80% of data

# Narrower range (less data, more robust)
RobustScaler(quantile_range=(30, 70))  # 40% of data
```

**Trade-off**: 
- Wider range → Less robust but more data
- Narrower range → More robust but less data

---

## Mathematical Properties

### Median Properties

1. **Breakdown Point**: 50% (can handle up to 50% outliers)
2. **Efficiency**: ~64% as efficient as mean for normal data
3. **Robustness**: Not affected by extreme values

### IQR Properties

1. **Breakdown Point**: 25% (can handle up to 25% outliers)
2. **Scale Invariant**: IQR(aX) = a × IQR(X)
3. **Translation Invariant**: IQR(X + b) = IQR(X)

---

## Implementation Details

### Handling Edge Cases

1. **Constant Columns** (IQR = 0):
   ```
   Set IQR = 1 to avoid division by zero
   ```

2. **Missing Values**:
   ```
   Compute median and IQR ignoring NaN values
   ```

3. **Small Datasets**:
   ```
   Ensure at least 4 unique values for reliable quartiles
   ```

---

## Summary

**RobustScaler** is the go-to choice when:
- Your data has outliers
- Outliers are valid and meaningful
- You want scaling that's not distorted by extremes

**Key Insight**: By using median and IQR, RobustScaler focuses on the "typical" data range, making it ideal for real-world datasets with natural outliers.

---

## References

- Robust Statistics: Huber, P. J. (1981). "Robust Statistics"
- Quartiles: Tukey, J. W. (1977). "Exploratory Data Analysis"
- Scikit-learn: `sklearn.preprocessing.RobustScaler`
