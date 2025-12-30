import numpy as np
import pandas as pd
from sklearn.metrics import *

# ============================================================================
# REGRESSION EVALUATION
# ============================================================================

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluate regression model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True values
        y_pred: Predicted values (optional, will be computed if not provided)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    print("Evaluating regression model...")
    print(f"Test samples: {len(y_test)}")
    
    # Get predictions if not provided
    if y_pred is None:
        y_pred = model.predict(X_test)
    
    results = {
        'task_type': 'regression',
        'n_samples': len(y_test),
        'metrics': {}
    }
    
    # Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)
    results['metrics']['mae'] = float(mae)
    print(f"MAE: {mae:.4f}")
    
    # R² Score
    r2 = r2_score(y_test, y_pred)
    results['metrics']['r2'] = float(r2)
    print(f"R² Score: {r2:.4f}")
    
    # Residual Analysis
    residuals = y_test - y_pred
    results['residuals'] = {
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'min': float(np.min(residuals)),
        'max': float(np.max(residuals))
    }
    print(f"\nResidual Statistics:")
    print(f"  Mean: {results['residuals']['mean']:.4f}")
    print(f"  Std: {results['residuals']['std']:.4f}")
    
    print(f"\nEvaluation complete!")
    return results
