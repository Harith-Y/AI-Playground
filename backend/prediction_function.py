import numpy as np
import pandas as pd
import pickle

# ============================================================================
# REGRESSION PREDICTION
# ============================================================================

def predict(
    model,
    X: Union[np.ndarray, pd.DataFrame]
) -> Dict[str, Any]:
    """
    Make predictions using the trained regression model.
    
    Args:
        model: Trained model
        X: Input features (numpy array or pandas DataFrame)
    
    Returns:
        Dictionary containing predictions
    """
    print(f"Making predictions on {len(X)} samples...")
    
    # Get predictions
    predictions = model.predict(X)
    
    results = {
        'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
        'n_samples': len(X),
        'mean_prediction': float(np.mean(predictions)),
        'std_prediction': float(np.std(predictions)),
        'min_prediction': float(np.min(predictions)),
        'max_prediction': float(np.max(predictions))
    }
    
    print(f"Predictions complete!")
    return results
