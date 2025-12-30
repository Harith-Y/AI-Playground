import numpy as np
import pandas as pd
from sklearn.metrics import *

# ============================================================================
# CLUSTERING EVALUATION
# ============================================================================

def evaluate_model(
    model,
    X_test: np.ndarray,
    labels: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluate clustering model performance.
    
    Args:
        model: Trained clustering model
        X_test: Test features
        labels: Cluster labels (optional, will be computed if not provided)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    print("Evaluating clustering model...")
    print(f"Test samples: {len(X_test)}")
    
    # Get cluster labels if not provided
    if labels is None:
        if hasattr(model, 'labels_'):
            labels = model.labels_
        elif hasattr(model, 'predict'):
            labels = model.predict(X_test)
        else:
            raise ValueError("Could not get cluster labels from model")
    
    results = {
        'task_type': 'clustering',
        'n_samples': len(X_test),
        'n_clusters': len(np.unique(labels[labels != -1])),  # Exclude noise points
        'metrics': {}
    }
    
    # Silhouette Score
    try:
        # Filter out noise points (label -1) for DBSCAN
        mask = labels != -1
        if np.sum(mask) > 1 and len(np.unique(labels[mask])) > 1:
            silhouette = silhouette_score(X_test[mask], labels[mask])
            results['metrics']['silhouette_score'] = float(silhouette)
            print(f"Silhouette Score: {silhouette:.4f}")
        else:
            print("Not enough valid clusters for silhouette score")
    except Exception as e:
        print(f"Could not compute silhouette score: {e}")
    
    # Cluster Sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))
    results['cluster_sizes'] = cluster_sizes
    print(f"\nCluster Sizes: {cluster_sizes}")
    
    print(f"\nEvaluation complete!")
    return results
