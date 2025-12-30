import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(X_train, y_train, X_val=None, y_val=None):
    """
    Train RandomForestClassifier model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
    
    Returns:
        Trained model
    """
    print("Training RandomForestClassifier...")
    
    # Initialize model
    RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Training metrics
    train_score = model.score(X_train, y_train)
    print(f"Training score: {train_score:.4f}")
    
    # Validation metrics
    if X_val is not None and y_val is not None:
        val_score = model.score(X_val, y_val)
        print(f"Validation score: {val_score:.4f}")
    
    return model
