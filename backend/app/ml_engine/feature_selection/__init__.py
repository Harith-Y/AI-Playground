"""
Feature selection module for ML engine.

Provides methods to select the most relevant features for model training:
- Variance Threshold: Remove low-variance features
- Correlation Selector: Select/remove features based on correlation analysis
- Univariate Selection: Select features based on statistical tests
- Recursive Feature Elimination: Iteratively remove features
- Feature Importance: Select based on tree-based model importance
"""

from .variance_threshold import VarianceThreshold
from .correlation_selector import CorrelationSelector

__all__ = [
    "VarianceThreshold",
    "CorrelationSelector",
]
