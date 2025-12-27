# Regression Models Theory

## Overview

Regression models predict continuous numerical values. This document covers the regression algorithms implemented in our ML engine.

---

## 1. Linear Regression

### What It Does
Fits a straight line through data points to predict continuous outcomes. The relationship is modeled as:
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

### When to Use
- Linear relationship between features and target
- Interpretability is important
- Quick baseline model
- Small to medium datasets

### Strengths
- Fast to train
- Highly interpretable
- No hyperparameters to tune
- Works well with linearly separable data

### Weaknesses
- Assumes linear relationships
- Sensitive to outliers
- Can't capture complex patterns
- Prone to overfitting with many features

### Key Hyperparameters
- `fit_intercept`: Whether to calculate the intercept (default: True)

---

## 2. Ridge Regression (L2 Regularization)

### What It Does
Linear regression with L2 penalty that shrinks coefficients toward zero:
```
Loss = MSE + α × Σ(βᵢ²)
```

### When to Use
- Multicollinearity in features
- More features than samples
- Prevent overfitting
- All features are potentially relevant

### Strengths
- Handles multicollinearity well
- Reduces overfitting
- Keeps all features (doesn't zero out coefficients)
- Stable solutions

### Weaknesses
- Doesn't perform feature selection
- Less interpretable than plain linear regression
- Requires tuning alpha

### Key Hyperparameters
- `alpha`: Regularization strength (higher = more regularization)
- `fit_intercept`: Calculate intercept (default: True)

---

## 3. Lasso Regression (L1 Regularization)

### What It Does
Linear regression with L1 penalty that can zero out coefficients:
```
Loss = MSE + α × Σ|βᵢ|
```

### When to Use
- Feature selection needed
- Sparse models preferred
- Many irrelevant features
- Interpretability is important

### Strengths
- Automatic feature selection
- Creates sparse models
- Works well with high-dimensional data
- Interpretable results

### Weaknesses
- Can struggle with correlated features
- May arbitrarily select one feature from correlated group
- Requires tuning alpha

### Key Hyperparameters
- `alpha`: Regularization strength
- `max_iter`: Maximum iterations (default: 1000)

---

## 4. Elastic Net

### What It Does
Combines L1 and L2 regularization:
```
Loss = MSE + α × (l1_ratio × Σ|βᵢ| + (1 - l1_ratio) × Σ(βᵢ²))
```

### When to Use
- Both feature selection and handling multicollinearity
- Correlated features exist
- Balance between Ridge and Lasso needed

### Strengths
- Best of both Ridge and Lasso
- Handles correlated features better than Lasso
- Performs feature selection
- More stable than Lasso

### Weaknesses
- Two hyperparameters to tune
- More computationally expensive
- Less interpretable than simple linear models

### Key Hyperparameters
- `alpha`: Overall regularization strength
- `l1_ratio`: Mix of L1/L2 (0=Ridge, 1=Lasso)

---

## 5. Decision Tree Regressor

### What It Does
Creates a tree structure that splits data based on feature values to minimize variance in leaf nodes.

### When to Use
- Non-linear relationships
- Feature interactions important
- Minimal preprocessing needed
- Interpretability desired

### Strengths
- Handles non-linear relationships
- No feature scaling required
- Captures feature interactions
- Handles mixed data types

### Weaknesses
- Prone to overfitting
- Unstable (small data changes affect tree)
- Not great for extrapolation
- Can create biased trees with imbalanced data

### Key Hyperparameters
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples to split a node (default: 2)
- `min_samples_leaf`: Minimum samples in leaf (default: 1)

---

## 6. Random Forest Regressor

### What It Does
Ensemble of decision trees trained on random subsets of data and features. Predictions are averaged.

### When to Use
- Non-linear relationships
- High-dimensional data
- Feature importance needed
- Robust model required

### Strengths
- Handles non-linear patterns well
- Resistant to overfitting
- Provides feature importance
- Handles missing values
- Minimal hyperparameter tuning needed

### Weaknesses
- Less interpretable than single trees
- Slower to train and predict
- Large memory footprint
- Can overfit on noisy data

### Key Hyperparameters
- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples to split
- `max_features`: Features to consider for splits

---

## 7. Extra Trees Regressor

### What It Does
Similar to Random Forest but uses random splits instead of optimal splits. "Extremely Randomized Trees."

### When to Use
- Similar to Random Forest
- When training speed is important
- Reduce overfitting further

### Strengths
- Faster training than Random Forest
- Often better generalization
- More randomization reduces variance
- Similar performance to Random Forest

### Weaknesses
- Same as Random Forest
- May need more trees for same performance
- Less interpretable

### Key Hyperparameters
- Same as Random Forest

---

## 8. Gradient Boosting Regressor

### What It Does
Builds trees sequentially, each correcting errors of previous trees. Uses gradient descent in function space.

### When to Use
- High predictive accuracy needed
- Willing to tune hyperparameters
- Structured/tabular data
- Competition/production models

### Strengths
- Excellent predictive performance
- Handles mixed data types
- Captures complex patterns
- Feature importance available

### Weaknesses
- Prone to overfitting
- Requires careful tuning
- Slower to train
- Sequential (can't parallelize easily)
- Sensitive to outliers

### Key Hyperparameters
- `n_estimators`: Number of boosting stages (default: 100)
- `learning_rate`: Shrinks contribution of each tree (default: 0.1)
- `max_depth`: Maximum tree depth (default: 3)
- `subsample`: Fraction of samples for training each tree

---

## 9. AdaBoost Regressor

### What It Does
Adaptive Boosting that focuses on difficult samples by adjusting weights based on previous errors.

### When to Use
- Simpler alternative to Gradient Boosting
- When you have weak learners
- Less prone to overfitting than other boosting

### Strengths
- Less prone to overfitting than Gradient Boosting
- Few hyperparameters
- Can use any base estimator
- Good for reducing bias

### Weaknesses
- Sensitive to outliers
- Can perform worse than Gradient Boosting
- Sequential training
- Slower than simple models

### Key Hyperparameters
- `n_estimators`: Number of estimators (default: 50)
- `learning_rate`: Weight applied to each estimator (default: 1.0)
- `loss`: Loss function ('linear', 'square', 'exponential')

---

## 10. Support Vector Regression (SVR)

### What It Does
Finds a hyperplane that fits data within a margin of tolerance (epsilon), using kernel trick for non-linear relationships.

### When to Use
- Small to medium datasets
- Non-linear relationships
- High-dimensional spaces
- Robust predictions needed

### Strengths
- Effective in high dimensions
- Memory efficient (uses support vectors)
- Robust to outliers
- Versatile (different kernels)

### Weaknesses
- Doesn't scale well to large datasets
- Requires feature scaling
- Hyperparameter tuning crucial
- Difficult to interpret
- No probability estimates

### Key Hyperparameters
- `kernel`: 'linear', 'poly', 'rbf', 'sigmoid' (default: 'rbf')
- `C`: Regularization parameter (default: 1.0)
- `epsilon`: Epsilon in the epsilon-SVR model (default: 0.1)
- `gamma`: Kernel coefficient (default: 'scale')

---

## 11. K-Nearest Neighbors Regressor

### What It Does
Predicts by averaging the target values of K nearest neighbors in feature space.

### When to Use
- Simple baseline
- Small datasets
- Non-linear relationships
- No training time required

### Strengths
- No training phase
- Simple and intuitive
- No assumptions about data distribution
- Handles multi-output regression

### Weaknesses
- Slow predictions on large datasets
- Sensitive to feature scaling
- Curse of dimensionality
- Memory intensive
- No feature importance
- Sensitive to irrelevant features

### Key Hyperparameters
- `n_neighbors`: Number of neighbors (default: 5)
- `weights`: 'uniform' or 'distance' (default: 'uniform')
- `p`: Power parameter for Minkowski metric (1=Manhattan, 2=Euclidean)

---

## Model Selection Guide

### Quick Decision Tree

```
Need interpretability + linear relationship?
├─ Yes → Linear Regression / Ridge / Lasso
└─ No → Continue

Large dataset (>10K samples)?
├─ Yes → Random Forest / Gradient Boosting
└─ No → Continue

Need feature selection?
├─ Yes → Lasso / Elastic Net
└─ No → Continue

Complex non-linear patterns?
├─ Yes → Gradient Boosting / Random Forest
└─ No → Ridge / Linear Regression

High-dimensional data?
├─ Yes → Ridge / Lasso / Random Forest
└─ No → Any model suitable
```

### Performance Tiers (General)

**Tier 1 (Best Performance):**
- Gradient Boosting
- Random Forest / Extra Trees

**Tier 2 (Good Performance):**
- SVR (small datasets)
- AdaBoost
- Decision Tree (with tuning)

**Tier 3 (Baseline):**
- Ridge / Lasso / Elastic Net
- Linear Regression
- KNN

---

## Common Pitfalls

1. **Using linear models on non-linear data** - Try tree-based or SVR
2. **Not scaling features for SVR/KNN** - Always scale!
3. **Overfitting with deep trees** - Limit max_depth or use ensemble
4. **Gradient Boosting without tuning** - Requires careful hyperparameter selection
5. **KNN on high-dimensional data** - Curse of dimensionality affects performance

---

## References

- Scikit-learn Documentation: https://scikit-learn.org/stable/supervised_learning.html
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
- "Hands-On Machine Learning" by Aurélien Géron
