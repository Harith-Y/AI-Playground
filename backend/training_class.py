import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle

class ModelTrainer:
    """
    Model training pipeline.
    
    Auto-generated from AI-Playground experiment.
    Generated: 2025-12-30T20:34:38.782829
    Model: GradientBoostingClassifier
    """
    
    def __init__(self, random_state=123):
        """Initialize model trainer."""
        self.random_state = random_state
        self.model = None
        self.is_trained = False
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        
        Returns:
            Self (for method chaining)
        """
        print("Training GradientBoostingClassifier...")
        
        # Initialize model
        GradientBoostingClassifier
        self.model = GradientBoostingClassifier(
            n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
            random_state=self.random_state
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Training metrics
        train_score = self.model.score(X_train, y_train)
        print(f"Training score: {train_score:.4f}")
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            print(f"Validation score: {val_score:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features to predict on
        
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def save(self, filepath: str):
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load trained model from file.
        
        Args:
            filepath: Path to load model from
        
        Returns:
            Self (for method chaining)
        """
        import pickle
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {filepath}")
        return self
