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

### 3. [Oversampling Methods Theory](preprocessing/OVERSAMPLING_THEORY.md)
**Location:** `app/ml_engine/preprocessing/OVERSAMPLING_THEORY.md`

**Topics covered:**
- SMOTE (Synthetic Minority Over-sampling TEchnique)
- Borderline-SMOTE (boundary-focused variant)
- ADASYN (Adaptive Synthetic Sampling)
- How synthetic sample generation works
- Interpolation vs duplication
- Mathematical foundations and algorithms
- Comparison of all three methods
- Oversampling vs undersampling decision guide
- Combined hybrid approaches
- Best practices and common pitfalls

**Read this if you want to understand:**
- How SMOTE creates synthetic samples
- Why interpolation prevents overfitting
- When to use each oversampling variant
- How to combine over- and undersampling
- Cross-validation with oversampling
- Feature scaling requirements

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

### Oversampling Methods

```python
from app.ml_engine.preprocessing.oversampling import (
    SMOTE,
    BorderlineSMOTE,
    ADASYN
)

# SMOTE (general-purpose)
smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Borderline-SMOTE (boundary-focused)
bsmote = BorderlineSMOTE(kind='borderline-1', k_neighbors=5, m_neighbors=10, random_state=42)
X_res, y_res = bsmote.fit_resample(X, y)

# ADASYN (adaptive density-based)
adasyn = ADASYN(k_neighbors=5, random_state=42)
X_res, y_res = adasyn.fit_resample(X, y)

# Combined (undersample + oversample)
from app.ml_engine.preprocessing.undersampling import RandomUnderSampler

# Step 1: Undersample majority
rus = RandomUnderSampler(sampling_strategy={0: 50, 1: 500}, random_state=42)
X_under, y_under = rus.fit_resample(X, y)

# Step 2: Oversample minority
smote = SMOTE(random_state=42)
X_final, y_final = smote.fit_resample(X_under, y_under)
```

**Key concept:** Generate synthetic minority samples via interpolation to balance classes without losing data.

---

### Model Training & Selection

```python
from app.ml_engine.models import create_model, ModelFactory

# Regression example
model = create_model('random_forest_regressor', n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2_score = model.score(X_test, y_test)

# Classification example
model = create_model('gradient_boosting_classifier', n_estimators=200, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)

# Clustering example
model = create_model('kmeans', n_clusters=5, random_state=42)
model.fit(X)
labels = model.predict(X)
inertia = model.get_inertia()

# List all available models
available_models = ModelFactory.get_available_models()
print(list(available_models.keys()))

# Save and load models
model.save('models/my_model.joblib')
loaded_model = type(model).load('models/my_model.joblib')
```

**Key concept:** Unified wrapper interface for 25+ ML algorithms with consistent API for training, prediction, and evaluation.

---

## Learning Path

### For Beginners

**Recommended order:**

1. **Column Type Detection Theory**
   - Easier concept to grasp
   - Directly useful for data exploration
   - Foundation for preprocessing

2. **Regression Models Theory** OR **Classification Models Theory** (choose based on your task)
   - Learn about supervised learning
   - Understand model selection
   - Study hyperparameters

3. **Undersampling/Oversampling Methods Theory**
   - Address imbalanced datasets
   - Learn resampling techniques
   - Practical decision-making guidance

4. **Clustering Models Theory**
   - Explore unsupervised learning
   - Learn pattern discovery
   - Understand evaluation without labels

### For Intermediate Users

**Focus areas:**

1. **Preprocessing pipeline:**
   - Column Type Detection → Undersampling/Oversampling → Model Training

2. **Model selection:**
   - Study "Model Selection Guide" sections in Regression, Classification, and Clustering theory docs
   - Compare performance tiers
   - Learn when to use each algorithm

3. **Hyperparameter tuning:**
   - Review "Key Hyperparameters" for each model
   - Understand defaults and when to change them

### For Advanced Users

**Deep dive topics:**

- Read all six documents for comprehensive understanding
- Focus on "Algorithm Flow" and "Mathematical Foundations" sections
- Study "Best Practices" for production deployment
- Review "Limitations" and "Common Pitfalls" sections
- Compare trade-offs between algorithms
- Experiment with combining techniques (e.g., SMOTE + Random Forest with class_weight)
- Understand when NOT to use certain methods

---

## Related Documentation

### Code Examples
- `backend/examples/column_type_detection_example.py` - Practical usage of type detection
- `backend/examples/undersampling_example.py` - Comparing all undersampling methods
- `backend/examples/oversampling_example.py` - Comparing SMOTE, Borderline-SMOTE, and ADASYN

### API Documentation
- `backend/app/ml_engine/utils/README.md` - Column type detector API reference
- `backend/app/ml_engine/preprocessing/undersampling.py` - Undersampling inline docstrings
- `backend/app/ml_engine/preprocessing/oversampling.py` - Oversampling inline docstrings
- `backend/app/ml_engine/models/base.py` - Model wrapper base classes
- `backend/app/ml_engine/models/registry.py` - Model factory and registry
- `backend/app/ml_engine/models/regression.py` - Regression model wrappers
- `backend/app/ml_engine/models/classification.py` - Classification model wrappers
- `backend/app/ml_engine/models/clustering.py` - Clustering model wrappers

### Tests
- `backend/tests/ml_engine/utils/test_column_type_detector.py` - Type detection tests
- `backend/tests/ml_engine/preprocessing/test_undersampling.py` - Undersampling tests
- `backend/tests/ml_engine/preprocessing/test_oversampling.py` - Oversampling tests

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

### 4. [Regression Models Theory](models/REGRESSION_THEORY.md)
**Location:** `app/ml_engine/models/REGRESSION_THEORY.md`

**Topics covered:**
- 11 regression algorithms (Linear, Ridge, Lasso, Elastic Net, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, SVR, KNN)
- What each model does and how it works
- When to use each algorithm
- Strengths, weaknesses, and key hyperparameters
- Model selection decision trees
- Performance tiers and common pitfalls

**Read this if you want to understand:**
- How to choose the right regression algorithm
- Regularization techniques (L1, L2, Elastic Net)
- Ensemble methods vs single models
- When to use tree-based vs linear models
- Hyperparameter tuning strategies

---

### 5. [Classification Models Theory](models/CLASSIFICATION_THEORY.md)
**Location:** `app/ml_engine/models/CLASSIFICATION_THEORY.md`

**Topics covered:**
- 9 classification algorithms (Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, SVM, KNN, Gaussian Naive Bayes)
- Algorithm explanations and use cases
- Strengths, weaknesses, and hyperparameters
- Model selection guide
- Handling class imbalance
- Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)

**Read this if you want to understand:**
- Choosing classification algorithms for your problem
- Handling imbalanced datasets
- Linear vs non-linear decision boundaries
- Ensemble methods for classification
- Evaluation metrics selection

---

### 6. [Clustering Models Theory](models/CLUSTERING_THEORY.md)
**Location:** `app/ml_engine/models/CLUSTERING_THEORY.md`

**Topics covered:**
- 4 clustering algorithms (K-Means, DBSCAN, Agglomerative Clustering, Gaussian Mixture Model)
- How each clustering method works
- When to use each algorithm
- Strengths, weaknesses, and hyperparameters
- Determining optimal number of clusters
- Internal and external evaluation metrics
- Practical tips and comparison table

**Read this if you want to understand:**
- Unsupervised learning approaches
- Choosing clustering algorithms
- Elbow method, silhouette score, AIC/BIC
- Density-based vs centroid-based clustering
- Soft vs hard clustering assignments

---

## Future Theory Documents

Planned theory documentation for upcoming features:

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

- **Model Evaluation & Metrics** - ML-51 to ML-61
  - Cross-validation strategies
  - Metrics for regression, classification, clustering
  - Model comparison techniques
  - Statistical significance testing

- **Hyperparameter Tuning** - ML-71 to ML-80
  - Grid search vs random search
  - Bayesian optimization
  - Automated tuning strategies
  - Early stopping techniques

---

## Feedback

If you find errors, have suggestions, or want clarification on any theory documentation:
- Open an issue in the repository
- Add inline comments in code reviews
- Discuss in team meetings

Good theory documentation helps everyone understand not just "what" the code does, but "why" and "when" to use it.
