# Classification Models Theory

## Overview

Classification models predict categorical outcomes (classes). This document covers the classification algorithms implemented in our ML engine.

---

## 1. Logistic Regression

### What It Does
Despite its name, it's a classification algorithm that uses the logistic function to model probability of class membership:
```
P(y=1|x) = 1 / (1 + e^-(β₀ + β₁x₁ + ... + βₙxₙ))
```

### When to Use
- Binary or multiclass classification
- Linear decision boundaries
- Probabilistic predictions needed
- Interpretability required

### Strengths
- Fast to train
- Probabilistic outputs
- Works well with linearly separable data
- Low risk of overfitting
- Interpretable coefficients

### Weaknesses
- Assumes linear decision boundary
- Struggles with complex patterns
- Sensitive to outliers
- Requires feature scaling

### Key Hyperparameters
- `C`: Inverse regularization strength (default: 1.0)
- `penalty`: 'l1', 'l2', 'elasticnet', or 'none' (default: 'l2')
- `solver`: Optimization algorithm (default: 'lbfgs')
- `max_iter`: Maximum iterations (default: 100)

---

## 2. Decision Tree Classifier

### What It Does
Creates tree structure with if-then-else rules by splitting data to maximize information gain or minimize Gini impurity.

### When to Use
- Non-linear decision boundaries
- Feature interactions important
- Interpretability needed
- Mixed feature types

### Strengths
- Highly interpretable
- No feature scaling needed
- Handles non-linear patterns
- Automatic feature selection
- Fast predictions

### Weaknesses
- Prone to overfitting
- Unstable (high variance)
- Biased with imbalanced classes
- Not optimal for complex patterns

### Key Hyperparameters
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples to split (default: 2)
- `min_samples_leaf`: Minimum samples in leaf (default: 1)
- `criterion`: 'gini' or 'entropy' (default: 'gini')

---

## 3. Random Forest Classifier

### What It Does
Ensemble of decision trees where each tree votes for a class. Final prediction is majority vote.

### When to Use
- Default choice for classification
- High-dimensional data
- Feature importance needed
- Robust predictions required

### Strengths
- Excellent performance out-of-the-box
- Handles non-linear patterns
- Robust to overfitting
- Feature importance available
- Handles missing values
- Minimal preprocessing needed

### Weaknesses
- Less interpretable than single tree
- Slower predictions
- Large memory footprint
- Can struggle with very imbalanced data

### Key Hyperparameters
- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples to split
- `max_features`: Features per split (default: 'sqrt')
- `class_weight`: Handle imbalanced data

---

## 4. Extra Trees Classifier

### What It Does
Like Random Forest but uses random thresholds for splits instead of optimal ones.

### When to Use
- Similar to Random Forest
- Faster training needed
- More variance reduction desired

### Strengths
- Faster than Random Forest
- Often better generalization
- Same advantages as Random Forest
- More randomization reduces overfitting

### Weaknesses
- Same as Random Forest
- May need more trees

### Key Hyperparameters
- Same as Random Forest

---

## 5. Gradient Boosting Classifier

### What It Does
Builds trees sequentially where each tree corrects errors of previous trees using gradient descent.

### When to Use
- Maximum accuracy needed
- Kaggle competitions
- Production models
- Structured/tabular data

### Strengths
- State-of-the-art performance
- Handles mixed data types
- Feature importance available
- Captures complex patterns

### Weaknesses
- Prone to overfitting
- Requires extensive tuning
- Slower training
- Can't parallelize easily
- Sensitive to outliers

### Key Hyperparameters
- `n_estimators`: Number of boosting stages (default: 100)
- `learning_rate`: Shrinkage parameter (default: 0.1)
- `max_depth`: Tree depth (default: 3)
- `subsample`: Fraction of samples per tree

---

## 6. AdaBoost Classifier

### What It Does
Adaptive Boosting that adjusts weights of misclassified samples, focusing on hard examples.

### When to Use
- Simpler boosting alternative
- Binary classification
- Less tuning desired

### Strengths
- Simple boosting algorithm
- Fewer hyperparameters than Gradient Boosting
- Less prone to overfitting
- Works with any base estimator

### Weaknesses
- Sensitive to outliers and noise
- Sequential (slow)
- Can perform worse than Gradient Boosting
- Struggles with very complex patterns

### Key Hyperparameters
- `n_estimators`: Number of estimators (default: 50)
- `learning_rate`: Contribution of each estimator (default: 1.0)
- `algorithm`: 'SAMME' or 'SAMME.R' (default: 'SAMME.R')

---

## 7. Support Vector Machine (SVM)

### What It Does
Finds optimal hyperplane that maximizes margin between classes. Uses kernel trick for non-linear boundaries.

### When to Use
- Small to medium datasets
- High-dimensional data
- Clear margin of separation
- Non-linear decision boundaries

### Strengths
- Effective in high dimensions
- Memory efficient (uses support vectors)
- Versatile (different kernels)
- Works well with clear margins

### Weaknesses
- Doesn't scale to large datasets
- Requires feature scaling
- No probability estimates by default
- Difficult to interpret
- Long training time

### Key Hyperparameters
- `C`: Regularization parameter (default: 1.0)
- `kernel`: 'linear', 'poly', 'rbf', 'sigmoid' (default: 'rbf')
- `gamma`: Kernel coefficient (default: 'scale')
- `probability`: Enable probability estimates (default: False)

---

## 8. K-Nearest Neighbors (KNN)

### What It Does
Classifies based on majority vote of K nearest neighbors in feature space.

### When to Use
- Simple baseline
- Small datasets
- Non-linear decision boundaries
- No training time acceptable

### Strengths
- No training required
- Simple and intuitive
- No assumptions about data
- Naturally handles multiclass

### Weaknesses
- Slow predictions on large data
- Sensitive to feature scaling
- Curse of dimensionality
- Memory intensive
- Sensitive to irrelevant features
- No feature importance

### Key Hyperparameters
- `n_neighbors`: Number of neighbors (default: 5)
- `weights`: 'uniform' or 'distance' (default: 'uniform')
- `metric`: Distance metric (default: 'minkowski')
- `p`: Power for Minkowski metric (default: 2)

---

## 9. Gaussian Naive Bayes

### What It Does
Applies Bayes' theorem assuming features are independent and follow Gaussian distribution.

### When to Use
- Text classification
- Quick baseline
- Small datasets
- Features are reasonably independent

### Strengths
- Very fast training and prediction
- Works well with small data
- Handles multiclass naturally
- Probabilistic predictions
- Few hyperparameters

### Weaknesses
- Strong independence assumption
- Assumes Gaussian distribution
- Can be outperformed by other methods
- Sensitive to irrelevant features

### Key Hyperparameters
- `var_smoothing`: Portion of largest variance added to variances (default: 1e-9)

---

## Model Selection Guide

### Quick Decision Tree

```
Small dataset (<1000 samples)?
├─ Yes → Logistic Regression / Naive Bayes / KNN
└─ No → Continue

Need interpretability?
├─ Yes → Logistic Regression / Decision Tree
└─ No → Continue

Maximum accuracy goal?
├─ Yes → Gradient Boosting / Random Forest
└─ No → Continue

Linear decision boundary?
├─ Yes → Logistic Regression / SVM (linear kernel)
└─ No → Random Forest / Gradient Boosting / SVM (RBF)

High-dimensional data (>100 features)?
├─ Yes → Logistic Regression / SVM / Random Forest
└─ No → Any model suitable
```

### Performance Tiers (General)

**Tier 1 (Best Performance):**
- Gradient Boosting
- Random Forest / Extra Trees

**Tier 2 (Good Performance):**
- SVM (tuned)
- AdaBoost
- Logistic Regression (linear problems)

**Tier 3 (Baseline):**
- Decision Tree
- KNN
- Naive Bayes

---

## Handling Class Imbalance

### Techniques by Model:

**Random Forest / Tree-based:**
- Use `class_weight='balanced'`
- Adjust `class_weight` manually

**Logistic Regression:**
- Use `class_weight='balanced'`

**SVM:**
- Use `class_weight='balanced'`

**General Approaches:**
- Oversample minority class (SMOTE)
- Undersample majority class
- Use ensemble methods
- Adjust decision threshold

---

## Common Pitfalls

1. **Not scaling features for SVM/KNN** - Always standardize!
2. **Using Naive Bayes with correlated features** - Violates independence assumption
3. **Deep decision trees without pruning** - Leads to overfitting
4. **Gradient Boosting default parameters** - Almost always needs tuning
5. **Ignoring class imbalance** - Use class_weight or resampling
6. **KNN with many features** - Curse of dimensionality

---

## Evaluation Metrics Guide

### Binary Classification:
- **Accuracy**: Overall correctness (use when balanced)
- **Precision**: Of predicted positives, how many correct?
- **Recall**: Of actual positives, how many found?
- **F1-Score**: Harmonic mean of precision/recall
- **ROC-AUC**: Overall discrimination ability
- **PR-AUC**: Better for imbalanced data

### Multiclass:
- **Accuracy**: Overall correctness
- **Macro F1**: Unweighted mean F1 across classes
- **Weighted F1**: Weighted by class frequency
- **Confusion Matrix**: Detailed class-wise errors

---

## References

- Scikit-learn Documentation: https://scikit-learn.org/stable/supervised_learning.html
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "Hands-On Machine Learning" by Aurélien Géron
