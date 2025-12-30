import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
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
    
    
    # Step 1: Remove Price Outliers
    print(f"  - Detect outliers using iqr method")
    
    # Detect and handle outliers using iqr
    
    for col in ['price', 'sqft', 'lot_size']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
    
    
    
    
    
    # Step 2: Impute Missing Values
    print(f"  - Impute missing values using median strategy")
    
    # Handle missing values in bedrooms, bathrooms, garage_size
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    df[['bedrooms', 'bathrooms', 'garage_size']] = imputer.fit_transform(df[['bedrooms', 'bathrooms', 'garage_size']])
    
    
    
    
    # Step 3: Robust Scaling
    print(f"  - Scale features using robust scaler")
    
    # Scale features: price, sqft, lot_size, bedrooms, bathrooms
    
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    
    df[['price', 'sqft', 'lot_size', 'bedrooms', 'bathrooms']] = scaler.fit_transform(df[['price', 'sqft', 'lot_size', 'bedrooms', 'bathrooms']])
    
    
    
    
    # Step 4: Variance Threshold
    print(f"  - Select features using variance_threshold method")
    
    # Feature selection: variance_threshold
    
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    selected_features = df[['sqft', 'lot_size', 'bedrooms', 'bathrooms']].columns[selector.fit(df[['sqft', 'lot_size', 'bedrooms', 'bathrooms']]).get_support()]
    df = df[list(selected_features) + [col for col in df.columns if col not in ['sqft', 'lot_size', 'bedrooms', 'bathrooms']]]
    
    
    
    
    
    print(f"Preprocessing complete. Shape: {df.shape}")
    return df


# Apply preprocessing
df = preprocess_data(df)