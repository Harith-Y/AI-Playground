# ML Engine - Theory Documentation Index

This directory contains theoretical documentation for various ML engine components. These documents explain the "why" and "how" behind the implementations.

## Available Theory Documents

### 1. [Column Type Detection Theory](utils/THEORY.md)
**Location:** `app/ml_engine/utils/THEORY.md`

**Topics covered:**
- What is column type detection and why is it needed
- 16 different column types (ID, datetime, numeric, categorical, text, boolean, special)
- Detection heuristics and algorithms
- Configuration parameters and their effects
- Benefits, limitations, and use cases
- Practical examples

**Read this if you want to understand:**
- How automatic type inference works
- Why pandas dtypes aren't enough
- When to use different detection thresholds
- How to build automatic preprocessing pipelines

---

### 2. [Undersampling Methods Theory](preprocessing/UNDERSAMPLING_THEORY.md)
**Location:** `app/ml_engine/preprocessing/UNDERSAMPLING_THEORY.md`

**Topics covered:**
- The class imbalance problem and its impact
- Undersampling vs oversampling strategies
- Random Undersampling (simple baseline)
- NearMiss Undersampling (3 intelligent variants)
- Tomek Links Removal (boundary cleaning)
- Combined approaches
- Distance metrics and mathematical foundations
- Best practices and evaluation strategies

**Read this if you want to understand:**
- When and why to use undersampling
- How each method works differently
- Which method to choose for your problem
- How to evaluate imbalanced classification
- Common pitfalls and how to avoid them

---

## Quick Reference

### Column Type Detection
```python
from app.ml_engine.utils import detect_column_types

# Basic usage
types = detect_column_types(df)

# Custom configuration
types = detect_column_types(
    df,
    categorical_threshold=0.05,  # Max uniqueness for categorical
    id_threshold=0.95,           # Min uniqueness for ID
    text_length_threshold=50,    # Min length for long text
    sample_size=10000            # Sampling for large datasets
)
```

**Key concept:** Goes beyond basic dtypes to understand semantic meaning of data.

---

### Undersampling Methods

```python
from app.ml_engine.preprocessing.undersampling import (
    RandomUnderSampler,
    NearMissUnderSampler,
    TomekLinksRemover
)

# Random (fast baseline)
rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_res, y_res = rus.fit_resample(X, y)

# NearMiss (intelligent boundary focus)
nm = NearMissUnderSampler(version=1, n_neighbors=3, random_state=42)
X_res, y_res = nm.fit_resample(X, y)

# Tomek Links (boundary cleaning)
tl = TomekLinksRemover(sampling_strategy='auto')
X_res, y_res = tl.fit_resample(X, y)

# Combined (clean then balance)
X_clean, y_clean = tl.fit_resample(X, y)
X_final, y_final = rus.fit_resample(X_clean, y_clean)
```

**Key concept:** Address class imbalance by intelligently reducing majority class samples.

---

## Learning Path

### For Beginners
1. Start with the **Column Type Detection** theory
   - Easier concept to grasp
   - Directly useful for data exploration
   - Foundation for preprocessing

2. Then read **Undersampling Methods**
   - Focuses on specific ML problem (imbalance)
   - Multiple algorithms to compare
   - Practical decision-making guidance

### For Advanced Users
- Read both documents for comprehensive understanding
- Focus on "Algorithm Flow" and "Mathematical Foundations" sections
- Study "Best Practices" for production deployment
- Review "Limitations" to understand when NOT to use these methods

---

## Related Documentation

### Code Examples
- `backend/examples/column_type_detection_example.py` - Practical usage of type detection
- `backend/examples/undersampling_example.py` - Comparing all undersampling methods

### API Documentation
- `backend/app/ml_engine/utils/README.md` - Column type detector API reference
- `backend/app/ml_engine/preprocessing/undersampling.py` - Inline docstrings

### Tests
- `backend/tests/ml_engine/utils/test_column_type_detector.py` - Type detection tests
- `backend/tests/ml_engine/preprocessing/test_undersampling.py` - Undersampling tests

---

## Contributing to Theory Docs

When adding new ML components, please:

1. **Create a theory document** explaining:
   - What problem it solves
   - How it works (algorithm/heuristics)
   - When to use it (use cases)
   - Advantages and limitations
   - Mathematical foundations (if applicable)
   - Best practices

2. **Update this index** with:
   - Link to your theory document
   - Brief description
   - Quick reference code snippet

3. **Include practical examples**:
   - Real-world scenarios
   - Code snippets
   - Visual explanations (if possible)

---

## Future Theory Documents

Planned theory documentation for upcoming features:

- **SMOTE (Oversampling)** - ML-25
  - Synthetic minority oversampling technique
  - When to oversample vs undersample
  - Variants (Borderline-SMOTE, ADASYN)

- **Feature Selection Methods** - ML-15, ML-16
  - Variance threshold
  - Correlation-based selection
  - Mutual information
  - RFE (Recursive Feature Elimination)
  - Statistical tests (F-test, chi-square)

- **Preprocessing Pipelines** - ML-21, ML-22
  - Pipeline architecture
  - Step composition
  - Configuration management
  - Serialization/deserialization

- **Model Training & Evaluation** - ML-31 to ML-61
  - Model registry concepts
  - Training workflows
  - Hyperparameter tuning strategies
  - Evaluation metrics for different tasks

---

## Feedback

If you find errors, have suggestions, or want clarification on any theory documentation:
- Open an issue in the repository
- Add inline comments in code reviews
- Discuss in team meetings

Good theory documentation helps everyone understand not just "what" the code does, but "why" and "when" to use it.
