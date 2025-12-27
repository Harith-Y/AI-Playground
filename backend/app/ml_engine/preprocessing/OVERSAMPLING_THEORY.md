

# Oversampling Methods - Theory

## Overview

Oversampling is a technique to address **class imbalance** in classification datasets by increasing the number of samples in the minority class(es) through the generation of **synthetic samples**.

Unlike undersampling which discards majority class data, oversampling creates new minority class instances, preserving all original information while balancing the dataset.

---

## Why Oversampling?

### The Duplication Problem

Simple duplication of minority samples causes **overfitting**:
```
Original minority samples: [A, B, C]
After duplication: [A, A, A, A, B, B, B, B, C, C, C, C]
Problem: Model memorizes exact duplicates, doesn't generalize
```

### The SMOTE Solution

Generate **synthetic** (artificial) samples instead:
```
Original minority samples: [A, B, C]
After SMOTE: [A, B, C, A', B', C', AB, BC, AC]
           where A' ≈ A, AB is between A and B
Benefit: Model learns general patterns, better generalization
```

---

## Method 1: SMOTE (Synthetic Minority Over-sampling TEchnique)

### Theory

SMOTE generates synthetic samples by **interpolating** between existing minority class samples and their nearest neighbors.

**Key Insight:** New samples created along the line connecting a sample to its neighbors are likely to also belong to the same class.

### Algorithm

```
For i = 1 to N_synthetic:
    1. Randomly select a minority sample x
    2. Find k nearest neighbors of x (all from minority class)
    3. Randomly select one neighbor x_nn from the k neighbors
    4. Generate synthetic sample:
       x_new = x + λ × (x_nn - x)
       where λ ~ Uniform(0, 1)
    5. Add x_new to dataset
```

### Mathematical Formulation

**Interpolation Formula:**
```
x_synthetic = x_i + λ × (x_nn - x_i)

Where:
- x_i: Selected minority sample
- x_nn: Randomly chosen nearest neighbor
- λ ∈ [0, 1]: Random interpolation factor
- x_synthetic: New synthetic sample
```

**Why it works:**
- λ = 0: x_synthetic = x_i (original sample)
- λ = 1: x_synthetic = x_nn (neighbor sample)
- λ ∈ (0,1): x_synthetic is between x_i and x_nn

### Visual Explanation

```
Feature Space (2D example):

Minority class samples:
  A ●────────────● B
    │            │
    │    ●       │    ● = Original minority sample
    │   C│       │    ○ = Synthetic sample
    │    │       │
    ○ ○  ○  ○ ○  ○    Generated along lines connecting samples
   A' A''C' C''B' B''

SMOTE creates samples in the "convex hull" of minority samples
```

### K-Neighbors Parameter

**Effect of k:**
- **Small k (e.g., 3):**
  - More local interpolation
  - Synthetic samples closer to originals
  - Less diversity
  - Good for small minority class

- **Large k (e.g., 10):**
  - More global interpolation
  - Greater diversity in synthetic samples
  - Risk of crossing into majority region
  - Good for large minority class

### Sampling Strategies

#### Auto / Minority
Balance all minority classes to majority class count.
```
Before: {0: 50, 1: 200}
After:  {0: 200, 1: 200}
Synthetic: 150 samples of class 0
```

#### Ratio
Specify desired minority/majority ratio.
```
Ratio = 0.5 (minority:majority = 1:2)
Before: {0: 50, 1: 200}
After:  {0: 100, 1: 200}
Synthetic: 50 samples of class 0
```

#### Dictionary
Specify exact target count for each class.
```
Strategy: {0: 150, 1: 200}
Before: {0: 50, 1: 200}
After:  {0: 150, 1: 200}
Synthetic: 100 samples of class 0
```

### Advantages

✅ **No information loss:** Keeps all original data
✅ **Generalization:** Creates diverse new samples
✅ **Simple and effective:** Widely used baseline
✅ **Multi-class support:** Works with any number of classes
✅ **Configurable:** Adjustable sampling strategy and k
✅ **Well-studied:** Extensive research and validation

### Disadvantages

❌ **Noise generation:** Can create samples in majority region
❌ **Overlapping classes:** May worsen class overlap
❌ **Uniform treatment:** Treats all minority samples equally
❌ **Computational cost:** KNN search can be slow
❌ **Curse of dimensionality:** Struggles with high dimensions
❌ **Feature scaling required:** Distance-based, needs normalization

### When to Use SMOTE

✅ **Use SMOTE when:**
- Small minority class (few samples)
- Classes are somewhat separated
- Need quick, effective baseline
- Have numeric features
- Working with medium-sized datasets

❌ **Avoid SMOTE when:**
- Classes heavily overlap
- High-dimensional data (>50 features)
- Categorical features (need encoding first)
- Real-time constraints (too slow)

---

## Method 2: Borderline-SMOTE

### Theory

**Motivation:** Not all minority samples are equally important. Samples far from the decision boundary (safe samples) don't need oversampling. Only **borderline samples** (near the boundary) are critical for classification.

**Borderline Sample:** A minority sample where approximately half of its m nearest neighbors belong to the majority class.

### Classification of Minority Samples

```
For each minority sample:
  Find m nearest neighbors (from all classes)
  Count majority neighbors: n_maj

  If n_maj == 0:
      → SAFE (surrounded by minority)
  If 0 < n_maj < m:
      → BORDERLINE (mixed neighborhood)
  If n_maj == m:
      → NOISE (surrounded by majority)
```

### Borderline Detection

```
Decision regions:

Safe region (don't oversample):
  ● ● ●        All neighbors are minority
  ● ● ●        Low misclassification risk
  ● ● ●

Borderline region (DO oversample):
  ● ○ ●        Mixed neighbors
  ○ ● ○        High misclassification risk
  ● ○ ●        FOCUS HERE!

Noise region (optionally ignore):
  ○ ○ ○        All neighbors are majority
  ○ ● ○        Likely outlier/noise
  ○ ○ ○
```

### Two Variants

#### Borderline-1 (Conservative)

**Generate synthetic samples between borderline minority samples only.**

```
Algorithm:
1. Identify borderline samples B ⊂ minority_samples
2. For each b in B:
   - Find k nearest neighbors from B (minority only)
   - Generate synthetic samples:
     s_new = b + λ × (b_nn - b)
```

**Effect:** Strengthens the decision boundary without crossing into majority region.

#### Borderline-2 (Aggressive)

**Generate synthetic samples between borderline minority and ANY neighbors (including majority).**

```
Algorithm:
1. Identify borderline samples B ⊂ minority_samples
2. For each b in B:
   - Find k nearest neighbors from ALL samples
   - Can include majority neighbors!
   - Generate synthetic samples:
     s_new = b + λ × (neighbor - b)
```

**Effect:** Can create samples slightly into majority region, making a smoother boundary.

### Visual Comparison

```
Original data:
  ● ● ●            ○ ○ ○ ○ ○
  ● B ●          ○ ○ ○ ○ ○ ○
  ● ● ●        ○ ○ ○ ○ ○ ○ ○
  (minority)    (majority)

Standard SMOTE (generates everywhere):
  ● ● ●          ○ ○ ○ ○ ○
  ●●●●●        ○ ○ ○ ○ ○ ○
  ●●●●●      ○ ○ ○ ○ ○ ○ ○
  (many samples, including safe region)

Borderline-SMOTE-1 (only at boundary):
  ● ● ●            ○ ○ ○ ○ ○
  ● ●●●          ○ ○ ○ ○ ○ ○
  ● ● ●        ○ ○ ○ ○ ○ ○ ○
  (focused on boundary)

Borderline-SMOTE-2 (crosses boundary):
  ● ● ●            ○ ○ ○ ○ ○
  ● ●●●○        ○ ○ ○ ○ ○ ○
  ● ● ●        ○ ○ ○ ○ ○ ○ ○
  (some samples toward majority)
```

### Parameters

- **k_neighbors:** Number of neighbors for synthetic generation (default: 5)
- **m_neighbors:** Number of neighbors for borderline detection (default: 10)
- **kind:** 'borderline-1' or 'borderline-2'

### Advantages

✅ **Targeted:** Only oversamples important regions
✅ **Reduces noise:** Ignores safe samples
✅ **Better boundaries:** Focuses on decision boundary
✅ **Often better than SMOTE:** Improved classifier performance
✅ **Two variants:** Choose conservative or aggressive

### Disadvantages

❌ **More complex:** Requires borderline detection step
❌ **Parameter sensitive:** Need to tune m_neighbors
❌ **Binary classification:** Primarily designed for 2 classes
❌ **Computation:** Extra KNN search for borderline detection
❌ **May miss samples:** If no borderline samples found

### When to Use Borderline-SMOTE

✅ **Use Borderline-SMOTE when:**
- Have clear class separation
- Want to focus computational resources
- Concerned about noise from standard SMOTE
- Binary classification problem
- Willing to tune parameters

**Borderline-1 vs Borderline-2:**
- Use **Borderline-1** for conservative approach (stay in minority region)
- Use **Borderline-2** for aggressive approach (smooth boundary)

---

## Method 3: ADASYN (Adaptive Synthetic Sampling)

### Theory

**Core Idea:** Different minority samples have different **learning difficulties**. Samples surrounded by majority neighbors are harder to classify and should receive more synthetic samples.

**Adaptive:** The number of synthetic samples generated for each minority instance is proportional to its difficulty.

### Difficulty Measurement

**Density Distribution:**
```
For each minority sample x_i:
  1. Find k nearest neighbors (from all classes)
  2. Count ratio of majority neighbors:

     difficulty(x_i) = n_majority_neighbors / k

  3. Normalize across all minority samples:

     weight(x_i) = difficulty(x_i) / Σ difficulty(x_j)
```

**Intuition:**
- **Low difficulty (0.0):** All neighbors are minority → Easy to classify → Few synthetic samples
- **High difficulty (1.0):** All neighbors are majority → Hard to classify → Many synthetic samples

### Algorithm

```
Given: N_synthetic total samples to generate

1. Calculate difficulty for each minority sample x_i
2. Normalize to get weights w_i (sum = 1)
3. Allocate samples: n_i = w_i × N_synthetic
4. For each minority sample x_i:
   - Generate n_i synthetic samples
   - Use standard SMOTE interpolation with neighbors
```

### Visual Example

```
Scenario: Need 100 synthetic samples

Minority samples with difficulties:
  A: difficulty = 0.1 (1/10 neighbors are majority)  → 10 synthetic samples
  B: difficulty = 0.3 (3/10 neighbors are majority)  → 30 synthetic samples
  C: difficulty = 0.6 (6/10 neighbors are majority)  → 60 synthetic samples

Feature space:
  ○ ○ ○ ○ ○
  ○ C ○ ○ ○      ← C surrounded by majority (hard)
  ○○○○○         → Generate MANY samples from C

    ○ B ○        ← B has some majority neighbors
    ○ ● ○        → Generate MODERATE samples from B

  ● A ●          ← A surrounded by minority (easy)
  ● ● ●          → Generate FEW samples from A
```

### Mathematical Formulation

**Difficulty (Density):**
```
Γ_i = Δ_i / k

Where:
- Δ_i: Number of majority class samples in k-NN of x_i
- k: Number of neighbors to consider
- Γ_i ∈ [0, 1]: Difficulty score
```

**Normalized Weight:**
```
r_i = Γ_i / Σⱼ Γ_j

Where:
- Σⱼ Γ_j: Sum of difficulties across all minority samples
- r_i: Proportion of synthetic samples for x_i
```

**Number of Synthetic Samples:**
```
g_i = r_i × G

Where:
- G: Total synthetic samples needed
- g_i: Samples to generate for x_i
```

### Comparison with SMOTE

| Aspect | SMOTE | ADASYN |
|--------|-------|--------|
| **Sample allocation** | Uniform (equal probability) | Adaptive (proportional to difficulty) |
| **Easy samples** | Oversampled equally | Less oversampling |
| **Hard samples** | Oversampled equally | More oversampling |
| **Focus** | Overall balance | Difficult regions |
| **Complexity** | Simple | More complex |

### Advantages

✅ **Adaptive:** Focuses on hard-to-learn samples
✅ **Efficient:** Generates samples where needed most
✅ **Better performance:** Often outperforms SMOTE
✅ **Automatic:** No need to identify borderline manually
✅ **Density-aware:** Considers local class distribution

### Disadvantages

❌ **Computational cost:** Must calculate densities for all samples
❌ **Noise amplification:** Can generate many samples in noisy regions
❌ **Complexity:** More parameters and steps than SMOTE
❌ **Sensitivity:** Results depend heavily on k_neighbors
❌ **Imbalanced generation:** May over-generate in some regions

### When to Use ADASYN

✅ **Use ADASYN when:**
- Have varying difficulty across minority samples
- Want best possible classifier performance
- Can afford extra computation
- Have clear hard/easy regions in data
- Optimizing for F1-score or AUC

❌ **Avoid ADASYN when:**
- Data is very noisy
- Need fast training
- All minority samples equally important
- Simple baseline sufficient

---

## Comparison Table

| Method | Intelligence | Computation | Noise | Focus | Best For |
|--------|-------------|-------------|-------|-------|----------|
| **SMOTE** | Low | ⚡ Fast | Medium | Uniform | General use, baseline |
| **Borderline-1** | Medium | 🐌 Slow | Low | Boundary | Clear separation |
| **Borderline-2** | Medium | 🐌 Slow | Medium | Boundary | Smooth boundary |
| **ADASYN** | High | 🐌🐌 Very Slow | High | Difficult | Max performance |

---

## Oversampling vs Undersampling

### When to Oversample vs Undersample?

| Scenario | Recommendation |
|----------|----------------|
| **Small dataset (<1000 samples)** | Oversample (SMOTE) |
| **Large dataset (>100K samples)** | Undersample (Random) |
| **Moderate dataset** | Combined approach |
| **Minority class <20 samples** | Oversample (SMOTE/ADASYN) |
| **Redundant majority** | Undersample |
| **Diverse majority** | Oversample |
| **Training time critical** | Undersample |
| **Performance critical** | Try both |

### Combined Approach (Best Practice)

**Hybrid: Undersampling + Oversampling**

```
Step 1: Moderate Undersampling
Before: {fraud: 50, normal: 10,000}
After:  {fraud: 50, normal: 500}
Effect: Reduce dataset size, keep diversity

Step 2: Oversampling with SMOTE/ADASYN
Before: {fraud: 50, normal: 500}
After:  {fraud: 500, normal: 500}
Effect: Generate synthetic minority, balance classes

Result: Balanced, diverse, manageable size
```

**Advantages of Combined:**
✅ Balanced dataset
✅ Manageable size
✅ Reduced redundancy (from undersampling)
✅ Sufficient minority samples (from oversampling)
✅ Often best performance

---

## Best Practices

### 1. Feature Scaling (CRITICAL)

**Always scale features before SMOTE:**

```python
from sklearn.preprocessing import StandardScaler

# ✅ Correct
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
smote = SMOTE()
X_res, y_res = smote.fit_resample(X_scaled, y_train)

# ❌ Wrong
smote = SMOTE()
X_res, y_res = smote.fit_resample(X_train, y_train)
# Features not scaled → biased distance calculations!
```

**Why:** SMOTE uses Euclidean distance. Unscaled features dominate distance calculations.

### 2. Cross-Validation (CRITICAL)

**Apply oversampling INSIDE CV folds:**

```python
# ✅ Correct
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import Pipeline

pipeline = Pipeline([
    ('smote', SMOTE()),
    ('classifier', RandomForestClassifier())
])

scores = cross_val_score(pipeline, X, y, cv=5)

# ❌ Wrong (Data Leakage!)
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)
scores = cross_val_score(classifier, X_res, y_res, cv=5)
# Synthetic samples leak across folds!
```

### 3. Test Set (CRITICAL)

**Never oversample test data:**

```python
# ✅ Correct
X_train, X_test, y_train, y_test = train_test_split(X, y)
smote = SMOTE()
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
model.fit(X_train_res, y_train_res)
score = model.score(X_test, y_test)  # Original imbalanced test set

# ❌ Wrong
X_res, y_res = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)
# Test set contains synthetic samples → inflated scores!
```

### 4. Evaluation Metrics

**Don't use accuracy for imbalanced data:**

```python
# ❌ Misleading
accuracy = accuracy_score(y_test, y_pred)  # Can be high but useless

# ✅ Use these instead
from sklearn.metrics import (
    f1_score,           # Harmonic mean of precision/recall
    roc_auc_score,      # Area under ROC curve
    average_precision_score,  # Area under PR curve
    classification_report,    # Comprehensive view
)

f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
```

### 5. Hyperparameter Tuning

**k_neighbors guidelines:**
- Small minority (<50 samples): k=3
- Medium minority (50-200): k=5
- Large minority (>200): k=10
- Always: k < minority_count

**sampling_strategy:**
- Start with 'auto' (full balance)
- If overfitting: Use ratio (e.g., 0.5)
- If underfitting: Use 'auto'

### 6. Method Selection

**Decision tree:**
```
Is your dataset small (<1000)?
├─ Yes → Use SMOTE (preserve all data)
└─ No → Is training time critical?
    ├─ Yes → Use Undersampling
    └─ No → Do you have clear class boundaries?
        ├─ Yes → Use Borderline-SMOTE
        └─ No → Try ADASYN for best performance
```

### 7. Categorical Features

**Handle categoricals before SMOTE:**

```python
# SMOTE requires numeric features
# Option 1: Encode first
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_categorical)
smote = SMOTE()
X_res, y_res = smote.fit_resample(X_encoded, y)

# Option 2: Use SMOTE-NC (SMOTE for mixed data)
# Not implemented in this module yet
```

---

## Common Pitfalls

### Pitfall 1: Applying SMOTE to Test Data
```python
# ❌ WRONG
smote = SMOTE()
X_res, y_res = smote.fit_resample(X_all, y_all)
X_train, X_test = train_test_split(X_res, y_res)
# Test set has synthetic samples → overly optimistic results
```

### Pitfall 2: Data Leakage in CV
```python
# ❌ WRONG
X_res, y_res = smote.fit_resample(X, y)
cross_val_score(model, X_res, y_res, cv=5)
# Synthetic samples from test folds leak into training
```

### Pitfall 3: Ignoring Feature Scaling
```python
# ❌ WRONG
# Features: [age: 0-100, salary: 0-100000]
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)
# Salary dominates distance → poor synthetic samples
```

### Pitfall 4: Using Only Accuracy
```python
# ❌ WRONG
# Dataset: 99% class 0, 1% class 1
accuracy = 0.99  # Looks great!
# But model predicts all class 0 → useless for class 1
```

### Pitfall 5: Over-balancing
```python
# ❌ Questionable
# Original: {minority: 10, majority: 10000}
# After SMOTE: {minority: 10000, majority: 10000}
# Created 9990 synthetic samples from just 10 originals!
# Too much extrapolation → likely overfitting
```

---

## Advanced Topics

### SMOTE for Multi-Class

SMOTE naturally extends to multi-class:
```
Original: {class_0: 50, class_1: 100, class_2: 200}
Strategy: 'auto'
After: {class_0: 200, class_1: 200, class_2: 200}

Process:
- Oversample class_0: 50 → 200 (150 synthetic)
- Oversample class_1: 100 → 200 (100 synthetic)
- Keep class_2: 200 (no oversampling)
```

### Clustering-Based SMOTE

**Idea:** First cluster minority class, then apply SMOTE within clusters.

**Benefits:**
- Respects local structure
- Avoids cross-cluster synthetic samples
- Better for multi-modal minority distributions

### Safe-Level SMOTE

**Idea:** Adjust synthetic sample position based on "safety level."

```
If minority sample is very close to majority region:
  → Generate synthetic sample closer to minority centroid (safer)
If minority sample is far from majority:
  → Generate synthetic sample anywhere in neighborhood
```

### Poly-Kernel SMOTE

**Idea:** Use different distance metrics or kernels for different regions.

**Application:** When Euclidean distance doesn't capture feature relationships well.

---

## References

1. **SMOTE:**
   - Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." Journal of Artificial Intelligence Research, 16, 321-357.

2. **Borderline-SMOTE:**
   - Han, H., Wang, W. Y., & Mao, B. H. (2005). "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning." International Conference on Intelligent Computing.

3. **ADASYN:**
   - He, H., Bai, Y., Garcia, E. A., & Li, S. (2008). "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning." IEEE International Joint Conference on Neural Networks.

4. **imbalanced-learn Library:**
   - Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). "Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning." Journal of Machine Learning Research, 18(17), 1-5.
