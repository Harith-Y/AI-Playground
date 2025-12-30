import numpy as np
import pandas as pd
from sklearn.metrics import *

# ============================================================================
# CLASSIFICATION EVALUATION
# ============================================================================

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluate classification model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        y_pred: Predicted labels (optional, will be computed if not provided)
        y_proba: Predicted probabilities (optional, for AUC/ROC)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    print("Evaluating classification model...")
    print(f"Test samples: {len(y_test)}")
    
    # Get predictions if not provided
    if y_pred is None:
        y_pred = model.predict(X_test)
    
    # Get probabilities if available and not provided
    if y_proba is None and hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    
    results = {
        'task_type': 'classification',
        'n_samples': len(y_test),
        'n_classes': len(np.unique(y_test)),
        'metrics': {}
    }
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results['metrics']['accuracy'] = float(accuracy)
    print(f"Accuracy: {accuracy:.4f}")
    
    # F1 Score
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    results['metrics']['f1_score'] = float(f1)
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm.tolist()
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    results['classification_report'] = report
    
    print(f"\nEvaluation complete!")
    return results
