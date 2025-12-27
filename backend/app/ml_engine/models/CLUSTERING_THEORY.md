# Clustering Models Theory

## Overview

Clustering models group similar data points together without labeled outcomes. This document covers the unsupervised clustering algorithms implemented in our ML engine.

---

## 1. K-Means

### What It Does
Partitions data into K clusters by iteratively assigning points to nearest centroids and updating centroids:
```
1. Initialize K centroids randomly
2. Assign each point to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat until convergence
```

Minimizes within-cluster sum of squares (inertia):
```
inertia = Σ min(||x - μᵢ||²)
```

### When to Use
- Know the number of clusters K in advance
- Spherical/globular cluster shapes
- Similar-sized clusters expected
- Fast clustering needed
- Large datasets

### Strengths
- Fast and efficient (O(n·k·t) where t=iterations)
- Scalable to large datasets
- Simple to understand and implement
- Works well for spherical clusters
- Guaranteed to converge

### Weaknesses
- Must specify K beforehand
- Sensitive to initial centroid placement
- Assumes spherical clusters
- Struggles with varying cluster sizes
- Affected by outliers
- Only finds linear cluster boundaries

### Key Hyperparameters
- `n_clusters`: Number of clusters (default: 8)
- `init`: Initialization method - 'k-means++', 'random' (default: 'k-means++')
- `n_init`: Number of initializations (default: 10)
- `max_iter`: Maximum iterations (default: 300)
- `random_state`: Seed for reproducibility

### Special Methods
- `get_inertia()`: Within-cluster sum of squares (lower is better)
- `cluster_centers_`: Coordinates of cluster centers

---

## 2. DBSCAN

### What It Does
Density-Based Spatial Clustering of Applications with Noise. Groups points that are closely packed together, marking outliers as noise.

**Core concepts:**
- **Core point**: Has at least `min_samples` points within `eps` distance
- **Border point**: Within `eps` of core point but has fewer than `min_samples` neighbors
- **Noise point**: Neither core nor border

### When to Use
- Unknown number of clusters
- Arbitrary cluster shapes (non-spherical)
- Noise/outliers expected in data
- Varying cluster densities acceptable
- Geographic/spatial data

### Strengths
- No need to specify number of clusters
- Finds arbitrarily shaped clusters
- Robust to outliers (identifies them as noise)
- Only 2 hyperparameters
- Good for spatial data

### Weaknesses
- Struggles with varying densities
- Sensitive to eps and min_samples
- High-dimensional data issues
- Not deterministic on border points
- Difficult to tune on high-dimensional data
- Doesn't work well with large differences in density

### Key Hyperparameters
- `eps`: Maximum distance between two samples (default: 0.5)
- `min_samples`: Minimum samples in neighborhood for core point (default: 5)
- `metric`: Distance metric (default: 'euclidean')
- `algorithm`: 'auto', 'ball_tree', 'kd_tree', 'brute' (default: 'auto')

### Special Methods
- `get_core_sample_indices()`: Indices of core samples
- `labels_`: Cluster labels (-1 for noise)

---

## 3. Agglomerative Clustering

### What It Does
Hierarchical clustering using bottom-up approach:
```
1. Start with each point as its own cluster
2. Repeatedly merge the two closest clusters
3. Continue until reaching desired number of clusters
```

Creates a dendrogram showing hierarchical relationships.

### When to Use
- Hierarchical structure matters
- Need dendrogram visualization
- Small to medium datasets
- Want to explore different cluster numbers
- Connectivity constraints needed

### Strengths
- No need to specify clusters upfront (can cut dendrogram at any level)
- Deterministic results
- Works with any distance metric
- Can incorporate connectivity constraints
- Creates interpretable dendrogram
- Handles non-spherical clusters

### Weaknesses
- Computationally expensive O(n²) or O(n³)
- Doesn't scale to large datasets
- Cannot undo merges (greedy)
- Sensitive to noise and outliers
- Memory intensive
- Final clusters depend on linkage method

### Key Hyperparameters
- `n_clusters`: Number of clusters to find (default: 2)
- `linkage`: 'ward', 'complete', 'average', 'single' (default: 'ward')
- `metric`: Distance metric when linkage is not 'ward' (default: 'euclidean')
- `connectivity`: Connectivity matrix for constrained clustering

### Linkage Methods
- **Ward**: Minimizes variance of merged clusters (only works with Euclidean)
- **Complete**: Maximum distance between cluster pairs
- **Average**: Average distance between all pairs
- **Single**: Minimum distance between cluster pairs

### Special Methods
- `get_n_leaves()`: Number of leaves in hierarchical tree
- `get_n_connected_components()`: Number of connected components

---

## 4. Gaussian Mixture Model (GMM)

### What It Does
Probabilistic model assuming data comes from a mixture of Gaussian distributions. Uses Expectation-Maximization (EM) algorithm:
```
1. E-step: Compute probability each point belongs to each Gaussian
2. M-step: Update Gaussian parameters based on probabilities
3. Repeat until convergence
```

Provides soft clustering (probabilities) rather than hard assignments.

### When to Use
- Need probability estimates
- Ellipsoidal cluster shapes
- Overlapping clusters
- Model-based clustering desired
- Cluster uncertainty matters

### Strengths
- Provides probability distributions
- Flexible cluster shapes (ellipsoids)
- Theoretically grounded
- Can use AIC/BIC for model selection
- Handles overlapping clusters
- Soft clustering (membership probabilities)

### Weaknesses
- Computationally expensive
- Sensitive to initialization
- May converge to local optima
- Requires specifying number of components
- Can overfit with too many components
- Assumes Gaussian distributions

### Key Hyperparameters
- `n_components`: Number of mixture components (default: 1)
- `covariance_type`: 'full', 'tied', 'diag', 'spherical' (default: 'full')
- `max_iter`: Maximum EM iterations (default: 100)
- `n_init`: Number of initializations (default: 1)
- `init_params`: 'kmeans', 'random' (default: 'kmeans')

### Covariance Types
- **Full**: Each component has its own general covariance matrix
- **Tied**: All components share same covariance matrix
- **Diag**: Diagonal covariance matrices (axis-aligned ellipsoids)
- **Spherical**: Circular covariance (single variance per component)

### Special Methods
- `predict_proba()`: Probability of each cluster for each sample
- `get_aic()`: Akaike Information Criterion (lower is better)
- `get_bic()`: Bayesian Information Criterion (lower is better)

---

## Model Selection Guide

### Quick Decision Tree

```
Do you know the number of clusters?
├─ No
│  ├─ Have outliers/noise? → DBSCAN
│  └─ No outliers → Agglomerative (use dendrogram)
└─ Yes
   ├─ Need probability estimates? → Gaussian Mixture
   ├─ Arbitrary shapes needed? → DBSCAN or Agglomerative
   ├─ Large dataset (>10K samples)? → K-Means
   └─ Hierarchical structure important? → Agglomerative

Need soft clustering (probabilities)?
├─ Yes → Gaussian Mixture
└─ No → K-Means or DBSCAN or Agglomerative

Dataset size?
├─ Large (>10K) → K-Means
├─ Medium (1K-10K) → Any method
└─ Small (<1K) → Agglomerative or Gaussian Mixture
```

### By Use Case

**Customer Segmentation:**
- K-Means (fast, interpretable)
- Gaussian Mixture (soft assignments)

**Anomaly Detection:**
- DBSCAN (identifies noise points)

**Image Segmentation:**
- K-Means (fast)
- Gaussian Mixture (better boundaries)

**Geographic Clustering:**
- DBSCAN (arbitrary shapes, handles noise)
- Agglomerative with connectivity

**Document Clustering:**
- K-Means (high dimensions)
- Agglomerative (hierarchical topics)

---

## Determining Number of Clusters

### Elbow Method (K-Means)
Plot inertia vs. number of clusters, look for "elbow" where improvement diminishes.

### Silhouette Score (All Methods)
Measures how similar points are to their own cluster vs. other clusters. Range: [-1, 1], higher is better.
```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)
```

### AIC/BIC (Gaussian Mixture)
Information criteria that balance fit and complexity. Lower is better.

### Dendrogram (Agglomerative)
Visual inspection of hierarchical tree to choose cut height.

### Gap Statistic
Compares within-cluster dispersion to reference distribution.

---

## Evaluation Metrics

### Internal Metrics (No ground truth needed)

**Silhouette Score:**
- Measures cluster cohesion and separation
- Range: [-1, 1], higher is better
- -1: Wrong cluster, 0: Overlapping, +1: Well-separated

**Calinski-Harabasz Index:**
- Ratio of between-cluster to within-cluster dispersion
- Higher is better
- Fast to compute

**Davies-Bouldin Index:**
- Average similarity between each cluster and its most similar cluster
- Lower is better
- Range: [0, ∞)

**Inertia (K-Means only):**
- Within-cluster sum of squares
- Lower is better
- Use for elbow method

### External Metrics (Ground truth available)

**Adjusted Rand Index (ARI):**
- Similarity between true and predicted labels
- Range: [-1, 1], 1 is perfect
- Accounts for chance

**Normalized Mutual Information (NMI):**
- Information shared between labels
- Range: [0, 1], 1 is perfect
- Less sensitive to cluster sizes

**Fowlkes-Mallows Score:**
- Geometric mean of precision and recall
- Range: [0, 1], 1 is perfect

---

## Common Pitfalls

1. **Not scaling features** - Different scales can dominate distance calculations. Always standardize!
2. **Using K-Means for non-spherical clusters** - K-Means assumes globular shapes
3. **Wrong DBSCAN parameters** - Use k-distance plot to choose eps
4. **Too many GMM components** - Leads to overfitting, use AIC/BIC
5. **Ignoring cluster validation** - Always validate with silhouette or other metrics
6. **Treating DBSCAN noise as clusters** - Noise points (label -1) should be handled separately
7. **High-dimensional data without reduction** - Use PCA or other reduction first
8. **Not checking for convergence** - Especially important for K-Means and GMM

---

## Practical Tips

### Preprocessing
```python
from sklearn.preprocessing import StandardScaler

# Always scale your features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### K-Means Optimization
```python
# Use k-means++ initialization (default)
# Run multiple times with different seeds
kmeans = KMeans(n_clusters=3, n_init=10, init='k-means++')
```

### DBSCAN Parameter Selection
```python
from sklearn.neighbors import NearestNeighbors

# Plot k-distance graph to find eps
neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(X)
distances, indices = neighbors.kneighbors(X)
# Plot sorted distances, look for "elbow"
```

### GMM Model Selection
```python
# Compare different numbers of components
bic_scores = []
for n in range(1, 11):
    gmm = GaussianMixture(n_components=n)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
# Choose n with lowest BIC
```

---

## Comparison Table

| Algorithm | Cluster Shape | Scalability | Needs K? | Handles Noise | Deterministic |
|-----------|---------------|-------------|----------|---------------|---------------|
| K-Means | Spherical | Excellent | Yes | No | No* |
| DBSCAN | Arbitrary | Good | No | Yes | Mostly** |
| Agglomerative | Any | Poor | Yes*** | No | Yes |
| GMM | Ellipsoidal | Fair | Yes | Partial | No |

\* Different initializations may yield different results (use `random_state`)
\*\* Border points may vary
\*\*\* Can cut dendrogram at any level after fitting

---

## References

- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/clustering.html
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "Introduction to Data Mining" by Tan, Steinbach, Kumar
- Original DBSCAN paper: Ester et al. (1996)
- K-Means++: Arthur & Vassilvitskii (2007)
