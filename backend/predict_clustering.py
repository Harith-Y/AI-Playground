import numpy as np
import pandas as pd
import pickle

# ============================================================================
# CLUSTERING PREDICTION
# ============================================================================

def predict(
    model,
    X: Union[np.ndarray, pd.DataFrame]
) -> Dict[str, Any]:
    """
    Assign cluster labels using the trained clustering model.
    
    Args:
        model: Trained clustering model
        X: Input features (numpy array or pandas DataFrame)
    
    Returns:
        Dictionary containing cluster assignments
    """
    print(f"Assigning clusters for {len(X)} samples...")
    
    # Get cluster assignments
    if hasattr(model, 'predict'):
        labels = model.predict(X)
    elif hasattr(model, 'labels_'):
        # For models that don't have predict (like some clustering algorithms)
        labels = model.labels_
    else:
        raise ValueError("Model does not support prediction")
    
    # Get cluster statistics
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_distribution = dict(zip(unique_labels.tolist(), counts.tolist()))
    
    results = {
        'cluster_labels': labels.tolist() if isinstance(labels, np.ndarray) else labels,
        'n_samples': len(X),
        'n_clusters': len(unique_labels),
        'cluster_distribution': cluster_distribution
    }
    
    print(f"Cluster assignment complete!")
    return results
