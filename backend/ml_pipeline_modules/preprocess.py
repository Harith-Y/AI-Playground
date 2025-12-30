"""Preprocessing Module - Auto-generated"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing steps to the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()
    print("\nApplying preprocessing steps...")
    
    
    # Step 1: Impute Missing Values
    print(f"  - Impute missing values using mean strategy")
    
    # Handle missing values in age, income, tenure
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    df[['age', 'income', 'tenure']] = imputer.fit_transform(df[['age', 'income', 'tenure']])
    
    
    
    
    # Step 2: Standard Scaling
    print(f"  - Scale features using standard scaler")
    
    # Scale features: age, income, tenure
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    df[['age', 'income', 'tenure']] = scaler.fit_transform(df[['age', 'income', 'tenure']])
    
    
    
    
    # Step 3: OneHot Encoding
    print(f"  - Encode categorical features using onehot encoding")
    
    # Encode categorical features: gender, contract_type
    
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[['gender', 'contract_type']])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(['gender', 'contract_type']),
        index=df.index
    )
    df = df.drop(columns=['gender', 'contract_type'])
    df = pd.concat([df, encoded_df], axis=1)
    
    
    
    
    
    
    print(f"Preprocessing complete. Shape: {df.shape}")
    return df


# Apply preprocessing
df = preprocess_data(df)