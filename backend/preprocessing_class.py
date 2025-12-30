import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """
    Data preprocessing pipeline.
    
    Auto-generated from AI-Playground experiment.
    Generated: 2025-12-30T21:33:31.957221
    """
    
    def __init__(self):
        """Initialize preprocessing pipeline."""
        self.fitted = False
        self.transformers = {}
    
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit preprocessing transformers on data.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Self (for method chaining)
        """
        df = df.copy()
        
        # Step 1: Label Encoding
        from sklearn.preprocessing import LabelEncoder
        self.transformers['encoder_1'] = {}
        for col in ['category', 'subcategory']:
            encoder = LabelEncoder()
            encoder.fit(df[col].astype(str))
            self.transformers['encoder_1'][col] = encoder
        
        # Step 2: MinMax Scaling
        from sklearn.preprocessing import MinMaxScaler
        self.transformers['scaler_2'] = MinMaxScaler()
        self.transformers['scaler_2'].fit(df[['length', 'word_count', 'char_count']])
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted transformers.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        df = df.copy()
        
        # Step 1: Label Encoding
        for col in ['category', 'subcategory']:
            df[col] = self.transformers['encoder_1'][col].transform(df[col].astype(str))
        
        # Step 2: MinMax Scaling
        df[['length', 'word_count', 'char_count']] = self.transformers['scaler_2'].transform(df[['length', 'word_count', 'char_count']])
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Transformed DataFrame
        """
        return self.fit(df).transform(df)
