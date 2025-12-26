"""
Example usage of the ColumnTypeDetector.

This script demonstrates how to use the automatic column type detection
feature to analyze a pandas DataFrame.
"""

import pandas as pd
import numpy as np
from app.ml_engine.utils import ColumnTypeDetector, detect_column_types, ColumnType


def create_sample_dataframe():
    """Create a sample DataFrame with various column types."""
    np.random.seed(42)

    df = pd.DataFrame({
        # ID columns
        'user_id': range(1000),
        'order_id': [f'ORD-{i:05d}' for i in range(1000)],

        # Numeric columns
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.uniform(30000, 150000, 1000),
        'num_purchases': np.random.randint(0, 50, 1000),

        # Boolean columns
        'is_premium': np.random.choice([True, False], 1000),
        'email_verified': np.random.choice(['yes', 'no'], 1000),

        # Categorical columns
        'country': np.random.choice(['USA', 'UK', 'Canada', 'Australia'], 1000),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 1000),
        'education': np.random.choice(['high', 'bachelor', 'master', 'phd'], 1000),

        # Binary columns
        'gender': np.random.choice(['Male', 'Female'], 1000),
        'subscription_status': np.random.choice(['Active', 'Inactive'], 1000),

        # Text columns
        'customer_name': [f'Customer {i}' for i in range(1000)],
        'review': [f'This is a detailed review about product quality and service. ' * 3 for i in range(1000)],

        # Datetime column
        'signup_date': pd.date_range('2020-01-01', periods=1000, freq='D'),

        # Constant column
        'platform': ['Web'] * 1000,
    })

    # Add some null values
    df.loc[::10, 'income'] = None
    df.loc[::15, 'review'] = None

    return df


def main():
    """Run the example."""
    print("=" * 80)
    print("Column Type Detection Example")
    print("=" * 80)
    print()

    # Create sample data
    df = create_sample_dataframe()
    print(f"Created sample DataFrame with {len(df)} rows and {len(df.columns)} columns")
    print()

    # Example 1: Using the convenience function
    print("Example 1: Using detect_column_types() convenience function")
    print("-" * 80)
    types = detect_column_types(df)

    print("\nDetected Column Types:")
    for col, col_type in types.items():
        print(f"  {col:25} -> {col_type.value}")

    print()

    # Example 2: Using the ColumnTypeDetector class
    print("\nExample 2: Using ColumnTypeDetector class with custom thresholds")
    print("-" * 80)
    detector = ColumnTypeDetector(
        categorical_threshold=0.1,  # Higher threshold for categorical
        id_threshold=0.90,          # Lower threshold for IDs
        text_length_threshold=30,   # Lower threshold for long text
    )

    types = detector.detect(df)
    print("\nDetected types with custom thresholds:")
    for col, col_type in types.items():
        print(f"  {col:25} -> {col_type.value}")

    print()

    # Example 3: Get detailed column information
    print("\nExample 3: Getting detailed column information")
    print("-" * 80)
    column_info = detector.get_column_info(df)

    print("\nDetailed Column Information:")
    print(column_info.to_string())

    print()

    # Example 4: Filtering columns by type
    print("\nExample 4: Filtering columns by detected type")
    print("-" * 80)

    numeric_cols = [col for col, t in types.items()
                   if 'numeric' in t.value]
    categorical_cols = [col for col, t in types.items()
                       if 'categorical' in t.value]
    text_cols = [col for col, t in types.items()
                if 'text' in t.value]

    print(f"\nNumeric columns ({len(numeric_cols)}):")
    for col in numeric_cols:
        print(f"  - {col}")

    print(f"\nCategorical columns ({len(categorical_cols)}):")
    for col in categorical_cols:
        print(f"  - {col}")

    print(f"\nText columns ({len(text_cols)}):")
    for col in text_cols:
        print(f"  - {col}")

    print()

    # Example 5: Type-specific processing
    print("\nExample 5: Type-specific processing recommendations")
    print("-" * 80)
    print("\nRecommended preprocessing steps by type:")

    for col, col_type in types.items():
        if col_type == ColumnType.NUMERIC_CONTINUOUS:
            print(f"  {col}: StandardScaler or MinMaxScaler")
        elif col_type == ColumnType.CATEGORICAL_NOMINAL:
            print(f"  {col}: OneHotEncoder")
        elif col_type == ColumnType.CATEGORICAL_ORDINAL:
            print(f"  {col}: OrdinalEncoder (with ordering)")
        elif col_type == ColumnType.TEXT_LONG:
            print(f"  {col}: TF-IDF or Text Embeddings")
        elif col_type == ColumnType.ID:
            print(f"  {col}: Drop (not useful for modeling)")
        elif col_type == ColumnType.CONSTANT:
            print(f"  {col}: Drop (constant value)")

    print()
    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
