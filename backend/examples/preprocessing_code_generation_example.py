"""
Example: Preprocessing Code Generation

This script demonstrates how to generate Python code for data preprocessing
from experiment configurations.
"""

from app.ml_engine.code_generation import generate_preprocessing_code

print("=" * 80)
print("PREPROCESSING CODE GENERATION EXAMPLES")
print("=" * 80)

# ============================================================================
# Example 1: Basic Preprocessing Pipeline
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: Basic Preprocessing Pipeline")
print("=" * 80)

basic_config = {
    'experiment_name': 'Customer Churn Prediction',
    'task_type': 'classification',
    'random_state': 42,
    'dataset_info': {
        'file_path': 'customer_data.csv',
        'file_format': 'csv'
    },
    'preprocessing_steps': [
        {
            'type': 'missing_value_imputation',
            'name': 'Impute Missing Values',
            'parameters': {
                'strategy': 'mean',
                'columns': ['age', 'income', 'tenure']
            }
        },
        {
            'type': 'scaling',
            'name': 'Standard Scaling',
            'parameters': {
                'scaler': 'standard',
                'columns': ['age', 'income', 'tenure']
            }
        },
        {
            'type': 'encoding',
            'name': 'OneHot Encoding',
            'parameters': {
                'encoder': 'onehot',
                'columns': ['gender', 'contract_type', 'payment_method']
            }
        }
    ]
}

# Generate complete script
script_code = generate_preprocessing_code(basic_config, output_format='script')
print("Generated preprocessing script (first 500 chars):")
print(script_code[:500] + "...")

# ============================================================================
# Example 2: Advanced Preprocessing with Outlier Detection
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Advanced Preprocessing with Outlier Detection")
print("=" * 80)

advanced_config = {
    'experiment_name': 'House Price Prediction',
    'task_type': 'regression',
    'dataset_info': {
        'file_path': 'house_prices.csv',
        'file_format': 'csv'
    },
    'preprocessing_steps': [
        {
            'type': 'outlier_detection',
            'name': 'Remove Price Outliers',
            'parameters': {
                'method': 'iqr',
                'threshold': 1.5,
                'action': 'clip',
                'columns': ['price', 'sqft', 'lot_size']
            }
        },
        {
            'type': 'missing_value_imputation',
            'name': 'Impute Missing Values',
            'parameters': {
                'strategy': 'median',
                'columns': ['bedrooms', 'bathrooms', 'garage_size']
            }
        },
        {
            'type': 'scaling',
            'name': 'Robust Scaling',
            'parameters': {
                'scaler': 'robust',
                'columns': ['price', 'sqft', 'lot_size', 'bedrooms', 'bathrooms']
            }
        },
        {
            'type': 'feature_selection',
            'name': 'Variance Threshold',
            'parameters': {
                'method': 'variance_threshold',
                'threshold': 0.01,
                'columns': ['sqft', 'lot_size', 'bedrooms', 'bathrooms']
            }
        }
    ]
}

# Generate preprocessing function
function_code = generate_preprocessing_code(advanced_config, output_format='function')
print("Generated preprocessing function (first 300 chars):")
print(function_code[:300] + "...")

# ============================================================================
# Example 3: Preprocessing Class (sklearn-style)
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Preprocessing Class (sklearn-style)")
print("=" * 80)

class_config = {
    'experiment_name': 'Text Classification',
    'preprocessing_steps': [
        {
            'type': 'encoding',
            'name': 'Label Encoding',
            'parameters': {
                'encoder': 'label',
                'columns': ['category', 'subcategory']
            }
        },
        {
            'type': 'scaling',
            'name': 'MinMax Scaling',
            'parameters': {
                'scaler': 'minmax',
                'columns': ['length', 'word_count', 'char_count']
            }
        }
    ]
}

# Generate preprocessing class
class_code = generate_preprocessing_code(class_config, output_format='class')
print("Generated preprocessing class (first 400 chars):")
print(class_code[:400] + "...")

# ============================================================================
# Example 4: Multiple Imputation Strategies
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Multiple Imputation Strategies")
print("=" * 80)

imputation_config = {
    'experiment_name': 'Healthcare Analytics',
    'preprocessing_steps': [
        {
            'type': 'missing_value_imputation',
            'name': 'Mean Imputation for Continuous',
            'parameters': {
                'strategy': 'mean',
                'columns': ['blood_pressure', 'heart_rate', 'temperature']
            }
        },
        {
            'type': 'missing_value_imputation',
            'name': 'Mode Imputation for Categorical',
            'parameters': {
                'strategy': 'mode',
                'columns': ['blood_type', 'diagnosis']
            }
        },
        {
            'type': 'missing_value_imputation',
            'name': 'Median Imputation for Skewed',
            'parameters': {
                'strategy': 'median',
                'columns': ['cholesterol', 'glucose']
            }
        }
    ]
}

imputation_code = generate_preprocessing_code(imputation_config, output_format='function')
print("Generated imputation code (first 300 chars):")
print(imputation_code[:300] + "...")

# ============================================================================
# Example 5: Different Scaling Methods
# ============================================================================
print("\n" + "=" * 80)
print("Example 5: Different Scaling Methods")
print("=" * 80)

scaling_config = {
    'experiment_name': 'Financial Analysis',
    'preprocessing_steps': [
        {
            'type': 'scaling',
            'name': 'Standard Scaling',
            'parameters': {
                'scaler': 'standard',
                'columns': ['revenue', 'profit', 'expenses']
            }
        },
        {
            'type': 'scaling',
            'name': 'MinMax Scaling',
            'parameters': {
                'scaler': 'minmax',
                'columns': ['growth_rate', 'market_share']
            }
        },
        {
            'type': 'scaling',
            'name': 'Robust Scaling',
            'parameters': {
                'scaler': 'robust',
                'columns': ['stock_price', 'volume']
            }
        }
    ]
}

scaling_code = generate_preprocessing_code(scaling_config, output_format='function')
print("Generated scaling code (first 300 chars):")
print(scaling_code[:300] + "...")

# ============================================================================
# Example 6: Categorical Encoding Methods
# ============================================================================
print("\n" + "=" * 80)
print("Example 6: Categorical Encoding Methods")
print("=" * 80)

encoding_config = {
    'experiment_name': 'E-commerce Analysis',
    'preprocessing_steps': [
        {
            'type': 'encoding',
            'name': 'OneHot Encoding',
            'parameters': {
                'encoder': 'onehot',
                'columns': ['product_category', 'payment_method']
            }
        },
        {
            'type': 'encoding',
            'name': 'Label Encoding',
            'parameters': {
                'encoder': 'label',
                'columns': ['customer_segment', 'priority']
            }
        }
    ]
}

encoding_code = generate_preprocessing_code(encoding_config, output_format='function')
print("Generated encoding code (first 300 chars):")
print(encoding_code[:300] + "...")

# ============================================================================
# Example 7: Complete Data Science Pipeline
# ============================================================================
print("\n" + "=" * 80)
print("Example 7: Complete Data Science Pipeline")
print("=" * 80)

complete_config = {
    'experiment_name': 'Credit Risk Assessment',
    'task_type': 'classification',
    'random_state': 42,
    'dataset_info': {
        'file_path': 'credit_data.csv',
        'file_format': 'csv'
    },
    'preprocessing_steps': [
        # Step 1: Handle outliers
        {
            'type': 'outlier_detection',
            'name': 'Remove Income Outliers',
            'parameters': {
                'method': 'iqr',
                'threshold': 1.5,
                'action': 'clip',
                'columns': ['annual_income', 'debt_amount']
            }
        },
        # Step 2: Impute missing values
        {
            'type': 'missing_value_imputation',
            'name': 'Impute Numerical',
            'parameters': {
                'strategy': 'median',
                'columns': ['annual_income', 'credit_score', 'debt_amount']
            }
        },
        {
            'type': 'missing_value_imputation',
            'name': 'Impute Categorical',
            'parameters': {
                'strategy': 'mode',
                'columns': ['employment_status', 'education_level']
            }
        },
        # Step 3: Encode categorical variables
        {
            'type': 'encoding',
            'name': 'OneHot Encode',
            'parameters': {
                'encoder': 'onehot',
                'columns': ['employment_status', 'education_level', 'marital_status']
            }
        },
        # Step 4: Scale numerical features
        {
            'type': 'scaling',
            'name': 'Standard Scale',
            'parameters': {
                'scaler': 'standard',
                'columns': ['annual_income', 'credit_score', 'debt_amount', 'age']
            }
        },
        # Step 5: Feature selection
        {
            'type': 'feature_selection',
            'name': 'Remove Low Variance',
            'parameters': {
                'method': 'variance_threshold',
                'threshold': 0.01
            }
        }
    ]
}

complete_code = generate_preprocessing_code(complete_config, output_format='script')
print("Generated complete pipeline (first 500 chars):")
print(complete_code[:500] + "...")

# ============================================================================
# Example 8: Save Generated Code to Files
# ============================================================================
print("\n" + "=" * 80)
print("Example 8: Save Generated Code to Files")
print("=" * 80)

# Save complete script
with open('generated_preprocessing_script.py', 'w', encoding='utf-8') as f:
    f.write(script_code)
print("[OK] Complete preprocessing script saved to 'generated_preprocessing_script.py'")

# Save preprocessing function
with open('preprocessing_function.py', 'w', encoding='utf-8') as f:
    f.write("import pandas as pd\nimport numpy as np\n")
    f.write("from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler\n")
    f.write("from sklearn.preprocessing import OneHotEncoder, LabelEncoder\n")
    f.write("from sklearn.impute import SimpleImputer\n\n")
    f.write(function_code)
print("[OK] Preprocessing function saved to 'preprocessing_function.py'")

# Save preprocessing class
with open('preprocessing_class.py', 'w', encoding='utf-8') as f:
    f.write("import pandas as pd\nimport numpy as np\n")
    f.write("from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler\n")
    f.write("from sklearn.preprocessing import OneHotEncoder, LabelEncoder\n")
    f.write("from sklearn.impute import SimpleImputer\n\n")
    f.write(class_code)
print("[OK] Preprocessing class saved to 'preprocessing_class.py'")

# ============================================================================
# Example 9: Generate for Different Data Formats
# ============================================================================
print("\n" + "=" * 80)
print("Example 9: Generate for Different Data Formats")
print("=" * 80)

formats = [
    ('CSV', 'csv', 'data.csv'),
    ('Excel', 'excel', 'data.xlsx'),
    ('JSON', 'json', 'data.json'),
    ('Parquet', 'parquet', 'data.parquet')
]

for format_name, format_type, file_path in formats:
    config = {
        'experiment_name': f'{format_name} Data Processing',
        'dataset_info': {
            'file_path': file_path,
            'file_format': format_type
        },
        'preprocessing_steps': [
            {
                'type': 'missing_value_imputation',
                'parameters': {'strategy': 'mean', 'columns': ['value']}
            }
        ]
    }
    
    code = generate_preprocessing_code(config, output_format='script')
    filename = f'preprocess_{format_type}.py'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(code)
    
    print(f"[OK] {format_name} preprocessing saved to '{filename}'")

# ============================================================================
# Example 10: Comparison of Output Formats
# ============================================================================
print("\n" + "=" * 80)
print("Example 10: Comparison of Output Formats")
print("=" * 80)

comparison_config = {
    'experiment_name': 'Format Comparison',
    'preprocessing_steps': [
        {
            'type': 'scaling',
            'parameters': {'scaler': 'standard', 'columns': ['feature1', 'feature2']}
        }
    ]
}

print("\n--- Script Format ---")
script = generate_preprocessing_code(comparison_config, output_format='script')
print(f"Length: {len(script)} characters")
print("Includes: imports, data loading, preprocessing, main execution")

print("\n--- Function Format ---")
function = generate_preprocessing_code(comparison_config, output_format='function')
print(f"Length: {len(function)} characters")
print("Includes: preprocessing function only (reusable)")

print("\n--- Class Format ---")
class_format = generate_preprocessing_code(comparison_config, output_format='class')
print(f"Length: {len(class_format)} characters")
print("Includes: sklearn-style class with fit/transform methods")

# ============================================================================
# Example 11: Real-World Use Case - Titanic Dataset
# ============================================================================
print("\n" + "=" * 80)
print("Example 11: Real-World Use Case - Titanic Dataset")
print("=" * 80)

titanic_config = {
    'experiment_name': 'Titanic Survival Prediction',
    'task_type': 'classification',
    'random_state': 42,
    'dataset_info': {
        'file_path': 'titanic.csv',
        'file_format': 'csv'
    },
    'preprocessing_steps': [
        {
            'type': 'missing_value_imputation',
            'name': 'Impute Age',
            'parameters': {
                'strategy': 'median',
                'columns': ['Age']
            }
        },
        {
            'type': 'missing_value_imputation',
            'name': 'Impute Embarked',
            'parameters': {
                'strategy': 'mode',
                'columns': ['Embarked']
            }
        },
        {
            'type': 'encoding',
            'name': 'Encode Categorical',
            'parameters': {
                'encoder': 'onehot',
                'columns': ['Sex', 'Embarked', 'Pclass']
            }
        },
        {
            'type': 'scaling',
            'name': 'Scale Numerical',
            'parameters': {
                'scaler': 'standard',
                'columns': ['Age', 'Fare', 'SibSp', 'Parch']
            }
        }
    ]
}

titanic_code = generate_preprocessing_code(titanic_config, output_format='script')
with open('titanic_preprocessing.py', 'w', encoding='utf-8') as f:
    f.write(titanic_code)
print("[OK] Titanic preprocessing saved to 'titanic_preprocessing.py'")
print(f"Generated {len(titanic_code)} characters of production-ready code")

print("\n" + "=" * 80)
print("Examples completed successfully!")
print("Generated files:")
print("  - generated_preprocessing_script.py: Complete preprocessing pipeline")
print("  - preprocessing_function.py: Standalone preprocessing function")
print("  - preprocessing_class.py: sklearn-style preprocessing class")
print("  - preprocess_*.py: Format-specific preprocessing scripts")
print("  - titanic_preprocessing.py: Real-world example")
print("=" * 80)
