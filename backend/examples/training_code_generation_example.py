"""
Example: Model Training Code Generation

This script demonstrates how to generate Python code for model training
from experiment configurations.
"""

from app.ml_engine.code_generation import generate_training_code

print("=" * 80)
print("MODEL TRAINING CODE GENERATION EXAMPLES")
print("=" * 80)

# ============================================================================
# Example 1: Basic Regression Model
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: Basic Linear Regression")
print("=" * 80)

regression_config = {
    'experiment_name': 'House Price Prediction',
    'model_type': 'linear_regression',
    'task_type': 'regression',
    'hyperparameters': {},
    'target_column': 'price',
    'random_state': 42,
    'test_size': 0.2
}

# Generate complete script
script_code = generate_training_code(regression_config, output_format='script')
print("Generated training script (first 500 chars):")
print(script_code[:500] + "...")

# ============================================================================
# Example 2: Classification with Hyperparameters
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Random Forest Classifier with Hyperparameters")
print("=" * 80)

classification_config = {
    'experiment_name': 'Customer Churn Prediction',
    'model_type': 'random_forest_classifier',
    'task_type': 'classification',
    'hyperparameters': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt'
    },
    'target_column': 'churn',
    'random_state': 42,
    'test_size': 0.25
}

# Generate training function
function_code = generate_training_code(classification_config, output_format='function')
print("Generated training function (first 400 chars):")
print(function_code[:400] + "...")

# ============================================================================
# Example 3: Gradient Boosting with Training Class
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Gradient Boosting Classifier (Class Format)")
print("=" * 80)

boosting_config = {
    'experiment_name': 'Fraud Detection',
    'model_type': 'gradient_boosting_classifier',
    'task_type': 'classification',
    'hyperparameters': {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.8
    },
    'target_column': 'is_fraud',
    'random_state': 123
}

# Generate training class
class_code = generate_training_code(boosting_config, output_format='class')
print("Generated training class (first 500 chars):")
print(class_code[:500] + "...")

# ============================================================================
# Example 4: Regularized Regression
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Ridge Regression with Regularization")
print("=" * 80)

ridge_config = {
    'experiment_name': 'Sales Forecasting',
    'model_type': 'ridge',
    'task_type': 'regression',
    'hyperparameters': {
        'alpha': 1.0,
        'fit_intercept': True,
        'solver': 'auto'
    },
    'target_column': 'sales',
    'test_size': 0.2
}

ridge_code = generate_training_code(ridge_config, output_format='script')
print("Generated Ridge regression script (first 400 chars):")
print(ridge_code[:400] + "...")

# ============================================================================
# Example 5: Support Vector Machine
# ============================================================================
print("\n" + "=" * 80)
print("Example 5: Support Vector Classifier")
print("=" * 80)

svc_config = {
    'experiment_name': 'Image Classification',
    'model_type': 'svc',
    'task_type': 'classification',
    'hyperparameters': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'probability': True
    },
    'target_column': 'category'
}

svc_code = generate_training_code(svc_config, output_format='function')
print("Generated SVC training function (first 400 chars):")
print(svc_code[:400] + "...")

# ============================================================================
# Example 6: Clustering Model
# ============================================================================
print("\n" + "=" * 80)
print("Example 6: K-Means Clustering")
print("=" * 80)

kmeans_config = {
    'experiment_name': 'Customer Segmentation',
    'model_type': 'kmeans',
    'task_type': 'clustering',
    'hyperparameters': {
        'n_clusters': 5,
        'max_iter': 300,
        'n_init': 10
    },
    'random_state': 42
}

kmeans_code = generate_training_code(kmeans_config, output_format='class')
print("Generated K-Means clustering class (first 400 chars):")
print(kmeans_code[:400] + "...")

# ============================================================================
# Example 7: Save Generated Code to Files
# ============================================================================
print("\n" + "=" * 80)
print("Example 7: Save Generated Code to Files")
print("=" * 80)

# Save complete training script
with open('generated_training_script.py', 'w') as f:
    f.write(script_code)
print("[OK] Complete training script saved to 'generated_training_script.py'")

# Save training function
with open('training_function.py', 'w') as f:
    f.write("import pandas as pd\nimport numpy as np\n")
    f.write("from sklearn.model_selection import train_test_split\n")
    f.write("from sklearn.ensemble import RandomForestClassifier\n\n")
    f.write(function_code)
print("[OK] Training function saved to 'training_function.py'")

# Save training class
with open('training_class.py', 'w') as f:
    f.write("import pandas as pd\nimport numpy as np\n")
    f.write("from sklearn.model_selection import train_test_split\n")
    f.write("from sklearn.ensemble import GradientBoostingClassifier\n")
    f.write("import pickle\n\n")
    f.write(class_code)
print("[OK] Training class saved to 'training_class.py'")

# ============================================================================
# Example 8: Multiple Models Comparison
# ============================================================================
print("\n" + "=" * 80)
print("Example 8: Generate Code for Multiple Models")
print("=" * 80)

models_to_compare = [
    {'model_type': 'logistic_regression', 'name': 'Logistic Regression'},
    {'model_type': 'random_forest_classifier', 'name': 'Random Forest'},
    {'model_type': 'gradient_boosting_classifier', 'name': 'Gradient Boosting'},
]

for model_info in models_to_compare:
    config = {
        'model_type': model_info['model_type'],
        'task_type': 'classification',
        'hyperparameters': {},
        'target_column': 'target'
    }
    
    code = generate_training_code(config, output_format='function')
    filename = f"train_{model_info['model_type']}.py"
    
    with open(filename, 'w') as f:
        f.write("import pandas as pd\nimport numpy as np\n")
        f.write("from sklearn.model_selection import train_test_split\n\n")
        f.write(code)
    
    print(f"[OK] {model_info['name']} training code saved to '{filename}'")

print("\n" + "=" * 80)
print("Examples completed successfully!")
print("Generated files:")
print("  - generated_training_script.py: Complete training pipeline")
print("  - training_function.py: Standalone training function")
print("  - training_class.py: sklearn-style training class")
print("  - train_*.py: Individual model training functions")
print("=" * 80)
