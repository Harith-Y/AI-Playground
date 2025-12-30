import numpy as np
import pandas as pd
import pickle

# ============================================================================
# CLASSIFICATION PREDICTION
# ============================================================================

def predict(
    model,
    X: Union[np.ndarray, pd.DataFrame],
    return_probabilities: bool = True
) -> Dict[str, Any]:
    """
    Make predictions using the trained classification model.
    
    Args:
        model: Trained model
        X: Input features (numpy array or pandas DataFrame)
        return_probabilities: Whether to return class probabilities
    
    Returns:
        Dictionary containing predictions and optionally probabilities
    """
    print(f"Making predictions on {len(X)} samples...")
    
    # Get predictions
    predictions = model.predict(X)
    
    results = {
        'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
        'n_samples': len(X)
    }
    
    # Get probabilities if requested and available
    if return_probabilities and hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        results['probabilities'] = probabilities.tolist() if isinstance(probabilities, np.ndarray) else probabilities
        
        # Get predicted class names if available
        if hasattr(model, 'classes_'):
            results['classes'] = model.classes_.tolist()
    
    print(f"Predictions complete!")
    return results
