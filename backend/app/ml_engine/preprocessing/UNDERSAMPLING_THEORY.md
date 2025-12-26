# Undersampling Methods - Theory

## Overview

Undersampling is a technique to address **class imbalance** in classification datasets by reducing the number of samples in the majority class(es) to match or approach the minority class count.

## The Class Imbalance Problem

### What is Class Imbalance?

A dataset is imbalanced when classes are not represented equally:
- **Example:** Fraud detection (99% normal, 1% fraud)
- **Example:** Disease diagnosis (95% healthy, 5% diseased)
- **Example:** Customer churn (80% retained, 20% churned)

### Why is it a Problem?

1. **Biased Models:** Classifiers tend to favor the majority class
2. **Poor Metrics:** High accuracy can be misleading (99% accuracy by always predicting majority)
3. **Missed Minority Cases:** Important rare events get ignored
4. **Threshold Issues:** Decision boundaries shift toward majority class

### Imbalance Ratio

```
Imbalance Ratio = Count(Majority Class) / Count(Minority Class)

Examples:
- 1:1 = Balanced
- 2:1 = Slightly imbalanced
- 10:1 = Moderately imbalanced
- 100:1 = Severely imbalanced
```

---

## Undersampling vs Oversampling

### Undersampling (This Module)
- **Reduce** majority class samples
- **Pros:** Faster training, reduced memory, removes redundancy
- **Cons:** May lose information, risk of underfitting

### Oversampling (SMOTE - ML-25)
- **Increase** minority class samples
- **Pros:** No information loss, better for small datasets
- **Cons:** Slower training, risk of overfitting, increased memory

### When to Use Undersampling?

✅ **Use undersampling when:**
- Large dataset with redundant majority samples
- Training time/memory is a concern
- Majority class has noisy or overlapping samples
- Quick baseline needed

❌ **Avoid undersampling when:**
- Dataset is already small
- Majority samples contain diverse information
- Cannot afford to lose any data

---

## Method 1: Random Undersampling

### Theory

The simplest approach: randomly select a subset of majority class samples without considering their characteristics or relationships.

### How it Works

```
Given:
- Minority class (0): 100 samples
- Majority class (1): 400 samples

Process:
1. Keep all minority samples (100)
2. Randomly select 100 from majority (400)
3. Result: 100 + 100 = 200 balanced samples
```

### Strategies

#### Auto (Default)
Balance all classes to minority class count.
```
Before: {0: 100, 1: 400}
After:  {0: 100, 1: 100}
```

#### Majority
Undersample only the majority class to minority count.
```
Before: {0: 100, 1: 400}
After:  {0: 100, 1: 100}
```

#### Ratio
Specify desired minority-to-majority ratio.
```
Ratio = 0.5 (minority:majority = 1:2)
Before: {0: 100, 1: 400}
After:  {0: 100, 1: 200}
```

#### Custom Dictionary
Specify exact count for each class.
```
Strategy: {0: 80, 1: 150}
Before: {0: 100, 1: 400}
After:  {0: 80, 1: 150}
```

### Parameters

- **sampling_strategy:** 'auto', 'majority', float ratio, or dict
- **random_state:** Seed for reproducibility
- **replacement:** Whether to sample with replacement

### Advantages

✅ Very fast and simple
✅ Works with any classifier
✅ Easy to understand and explain
✅ Reduces training time significantly
✅ Good baseline approach

### Disadvantages

❌ May discard useful information
❌ No control over which samples are removed
❌ Can underfit if majority class is diverse
❌ Purely random (no intelligence)

### Use Cases

- Quick prototyping and baseline
- Large datasets with redundant samples
- When speed is critical
- As part of ensemble methods (different random samples for each model)

---

## Method 2: NearMiss Undersampling

### Theory

**Intelligent** undersampling that selects majority class samples based on their **distance to minority class samples**. Unlike random selection, NearMiss focuses on the decision boundary where classification happens.

### Why Distance Matters?

The decision boundary between classes is where the classifier makes decisions. Samples near this boundary are more informative than samples far from it.

```
Majority samples far from minority:
  ● ● ● ● ● ● ●        (Less informative - clearly majority)

Decision boundary:
  ● ○ ● ○ ● ○ ●        (Most informative - hard to classify)

Minority samples:
        ○ ○ ○ ○ ○      (All important - keep all)
```

### Three Versions

#### NearMiss-1: Closest to Nearest Minority

**Selection criterion:** Select majority samples with the **smallest average distance to k nearest minority samples**.

**Intuition:** Keep majority samples **closest to the minority class** (near the boundary).

```
Process:
1. For each majority sample M:
   - Find k nearest minority samples
   - Calculate average distance to these k samples
2. Select N majority samples with smallest average distances
```

**Effect:** Focuses on the decision boundary from the majority side.

**Best for:** Improving boundary discrimination, reducing majority class overlap.

#### NearMiss-2: Closest to Farthest Minority

**Selection criterion:** Select majority samples with the **smallest average distance to k farthest minority samples**.

**Intuition:** Keep majority samples closest to the **far edge** of the minority class distribution.

```
Process:
1. For each majority sample M:
   - Find k farthest minority samples
   - Calculate average distance to these k samples
2. Select N majority samples with smallest average distances
```

**Effect:** Creates more balanced representation across minority class range.

**Best for:** Alternative boundary approach, experimental comparison with v1.

#### NearMiss-3: Per-Minority Coverage

**Selection criterion:** For **each minority sample**, select its k **nearest majority samples**.

**Intuition:** Ensure even **coverage around each minority sample**.

```
Process:
1. For each minority sample m:
   - Find k nearest majority samples
   - Add these to selected set
2. Result: Up to k × N_minority samples (may overlap)
```

**Effect:** Maintains local structure around minority samples.

**Best for:** Ensuring even representation, preserving local neighborhoods.

### Parameters

- **version:** 1, 2, or 3 (which NearMiss variant)
- **n_neighbors:** Number of neighbors to consider (k)
- **sampling_strategy:** How to balance (same as random)
- **n_jobs:** Parallel processing (for speed)

### Distance Metric

Uses **Euclidean distance** by default (via scikit-learn's NearestNeighbors):

```
distance(x, y) = √(Σ(xi - yi)²)
```

### Advantages

✅ Intelligent sample selection
✅ Focuses on decision boundary
✅ Can improve classifier performance
✅ Reduces noise near boundaries
✅ Three variants for different scenarios

### Disadvantages

❌ Computationally expensive (distance calculations)
❌ Sensitive to feature scaling (must standardize)
❌ May not work well with high-dimensional data
❌ Only works for binary classification
❌ Requires numeric features

### Computational Complexity

- **Time:** O(n × m × d) where n=majority, m=minority, d=dimensions
- **Space:** O(n × m) for distance matrix
- **Bottleneck:** KNN search (can use KD-trees for speedup)

### Use Cases

- Clean decision boundaries
- Remove noisy majority samples
- When training performance matters more than speed
- Medium-sized datasets (not too large)
- After feature scaling/normalization

---

## Method 3: Tomek Links Removal

### Theory

A **data cleaning** technique that removes noisy samples at the class boundary by identifying and removing **Tomek links** - pairs of samples from different classes that are each other's nearest neighbors.

### What is a Tomek Link?

**Definition:** Two samples (x, y) form a Tomek link if:
1. They belong to **different classes**
2. x is the **nearest neighbor** of y
3. y is the **nearest neighbor** of x
4. They are **mutual nearest neighbors**

```
Visualization:

Good separation (No Tomek links):
  ● ● ● ●              ○ ○ ○ ○
    (Class 1)            (Class 0)
    ↑                    ↑
    Clear boundary

Poor separation (Tomek links exist):
  ● ● ● ○ ●            ○ ● ○ ○
         ↑↑
    Tomek link (mutual nearest neighbors from different classes)
```

### How it Works

```
Process:
1. For each sample x:
   - Find its nearest neighbor y

2. Check if mutual:
   - If nearest neighbor of y is also x
   - AND x and y have different classes
   - Then (x, y) is a Tomek link

3. Remove samples according to strategy:
   - 'auto': Remove only majority class sample
   - 'majority': Remove only majority class sample
   - 'all': Remove both samples
```

### Example

```
Sample A (class 1): nearest neighbor is B
Sample B (class 0): nearest neighbor is A
Classes differ? Yes
Mutual nearest neighbors? Yes
→ Tomek link detected!

Strategy 'auto': Remove B (majority)
Strategy 'all': Remove both A and B
```

### Strategies

#### Auto (Default)
Remove only **majority class** samples from Tomek links.

```
Effect: Cleans boundary without affecting minority class
Use when: Want to preserve all minority samples
```

#### All
Remove **both samples** from each Tomek link.

```
Effect: Aggressive boundary cleaning
Use when: Both classes have noise at boundary
```

#### Majority
Remove only **majority class** samples (same as auto).

```
Effect: Conservative cleaning, preserve minority
Use when: Minority class is very small
```

### Parameters

- **sampling_strategy:** 'auto', 'all', or 'majority'
- **n_jobs:** Number of parallel jobs for distance computation

### Key Properties

1. **Not a balancing technique:** May remove very few samples
2. **Boundary cleaning:** Focuses on data quality, not quantity
3. **Works with multi-class:** Can handle more than 2 classes
4. **Symmetric detection:** Finds mutual relationships

### Advantages

✅ Removes noisy boundary samples
✅ Improves class separation
✅ Can improve any classifier
✅ Works for multi-class problems
✅ Minimal data loss (only boundary samples)
✅ Combines well with other methods

### Disadvantages

❌ May not significantly reduce imbalance
❌ Computationally expensive (O(n²) in worst case)
❌ Sensitive to feature scaling
❌ May find no links in well-separated data
❌ Requires numeric features

### Computational Complexity

- **Time:** O(n²) in worst case, O(n log n) with KD-trees
- **Space:** O(n) for neighbor storage
- **Bottleneck:** Nearest neighbor search for each sample

### Use Cases

- Data cleaning before training
- Improving class separation
- Combining with other undersampling (Tomek + Random)
- When you can't afford to lose minority samples
- Datasets with noisy boundaries

---

## Combined Approach: Tomek + Random

### Why Combine?

Leverage **both** data quality (Tomek) and class balance (Random):

```
Step 1: Tomek Links (Clean)
Before:  {0: 100, 1: 400} with noisy boundaries
After:   {0: 100, 1: 385} cleaner boundaries
Effect:  Remove noisy samples

Step 2: Random Undersampling (Balance)
Before:  {0: 100, 1: 385}
After:   {0: 100, 1: 100} balanced
Effect:  Achieve desired balance
```

### Pipeline

```python
# Step 1: Clean boundaries
tomek = TomekLinksRemover(sampling_strategy='auto')
X_clean, y_clean = tomek.fit_resample(X, y)

# Step 2: Balance classes
rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_final, y_final = rus.fit_resample(X_clean, y_clean)
```

### Benefits

✅ Best of both worlds
✅ Clean + balanced data
✅ Often superior to single method
✅ Reduces both noise and imbalance

---

## Comparison Table

| Method | Speed | Intelligence | Data Loss | Boundary Focus | Multi-class |
|--------|-------|--------------|-----------|----------------|-------------|
| **Random** | ⚡⚡⚡ Very Fast | ❌ None | ⚠️ High | ❌ No | ✅ Yes |
| **NearMiss-1** | 🐌 Slow | ✅ High | ⚠️ High | ✅ Yes | ❌ Binary only |
| **NearMiss-2** | 🐌 Slow | ✅ High | ⚠️ High | ✅ Yes | ❌ Binary only |
| **NearMiss-3** | 🐌 Slow | ✅ High | ⚠️ Medium | ✅ Yes | ❌ Binary only |
| **Tomek Links** | 🐌 Slow | ✅ Medium | ✅ Low | ✅ Yes | ✅ Yes |
| **Combined** | 🐌 Slow | ✅ High | ⚠️ High | ✅ Yes | ✅ Yes |

---

## Decision Guide

### Choose Random Undersampling if:
- Need a quick baseline
- Dataset is very large
- Speed is critical
- Majority class is homogeneous

### Choose NearMiss if:
- Classifier performance is critical
- Have time for computation
- Data is numeric and scaled
- Binary classification problem
- Want to focus on decision boundary

### Choose Tomek Links if:
- Data quality matters
- Can't lose minority samples
- Want minimal data loss
- Noisy boundaries suspected
- Multi-class problem

### Choose Combined (Tomek + Random) if:
- Want best results
- Have computation budget
- Both noise and imbalance exist
- Production model (worth the effort)

---

## Best Practices

### 1. Feature Scaling
Always scale features before distance-based methods:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Then apply NearMiss or Tomek
```

### 2. Evaluation
Use appropriate metrics for imbalanced data:
- ❌ **Don't use:** Accuracy
- ✅ **Use:** Precision, Recall, F1-score, ROC-AUC, PR-AUC

### 3. Cross-Validation
Apply undersampling **inside** CV folds, not before:
```python
# ✅ Correct
for train_idx, val_idx in kfold.split(X):
    X_train, y_train = X[train_idx], y[train_idx]
    sampler = RandomUnderSampler()
    X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
    model.fit(X_train_res, y_train_res)

# ❌ Wrong
X_res, y_res = sampler.fit_resample(X, y)  # Data leakage!
for train_idx, val_idx in kfold.split(X_res):
    ...
```

### 4. Preserve Test Set
Never undersample test data:
```python
# Split first
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Undersample only training
sampler = RandomUnderSampler()
X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)

# Test on original imbalanced test set
model.fit(X_train_res, y_train_res)
predictions = model.predict(X_test)  # Original imbalanced test
```

### 5. Try Multiple Methods
Different methods work better for different datasets:
```python
methods = [
    RandomUnderSampler(),
    NearMissUnderSampler(version=1),
    NearMissUnderSampler(version=3),
    TomekLinksRemover(),
]

for method in methods:
    X_res, y_res = method.fit_resample(X_train, y_train)
    score = evaluate(X_res, y_res, X_val, y_val)
    print(f"{method.__class__.__name__}: {score}")
```

---

## Mathematical Foundations

### Distance Metrics

**Euclidean Distance (L2):**
```
d(x, y) = √(Σᵢ(xᵢ - yᵢ)²)
```

**Manhattan Distance (L1):**
```
d(x, y) = Σᵢ|xᵢ - yᵢ|
```

**Why distance matters:**
- Samples close in feature space are similar
- Decision boundaries separate classes in feature space
- Distance-based selection focuses on hard-to-classify regions

### Sampling Probability

**Random Undersampling:**
```
P(sample selected) = n_target / n_original

Example: Select 100 from 400
P(sample selected) = 100/400 = 0.25 = 25%
```

**NearMiss Selection:**
```
P(sample selected) = rank(distance) / n_candidates

Closer samples have higher probability
```

---

## Practical Example

### Scenario: Credit Card Fraud Detection

```
Dataset:
- Total transactions: 100,000
- Fraudulent (minority): 500 (0.5%)
- Normal (majority): 99,500 (99.5%)
- Imbalance ratio: 199:1

Problem:
- Model predicts all as "normal" → 99.5% accuracy (useless!)
- Need to detect rare frauds

Solution 1: Random Undersampling
Before: {fraud: 500, normal: 99,500}
After:  {fraud: 500, normal: 500}
Result: Balanced, but discarded 99,000 normal transactions
        (May lose information about normal patterns)

Solution 2: Random with Ratio
Strategy: 0.1 (fraud:normal = 1:10)
After:  {fraud: 500, normal: 5,000}
Result: Still imbalanced but manageable
        (Keeps more information)

Solution 3: Tomek + NearMiss
Step 1 - Tomek: Remove noisy boundaries
After:  {fraud: 500, normal: 99,200} (removed 300 noisy)

Step 2 - NearMiss-1: Focus on boundary
After:  {fraud: 500, normal: 500}
Result: Clean, balanced, boundary-focused
        (Best for classifier performance)

Recommendation: Solution 3 for production
- Clean data (Tomek)
- Intelligent selection (NearMiss)
- Best fraud detection performance
```

---

## References

- Batista, G. E., Prati, R. C., & Monard, M. C. (2004). "A study of the behavior of several methods for balancing machine learning training data"
- Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
- Tomek, I. (1976). "Two modifications of CNN"
- imbalanced-learn library documentation
- scikit-learn neighbors module
