"""
Example: Evaluation Code Generation

This script demonstrates how to generate Python code for model evaluation
from experiment configurations.
"""

from app.ml_engine.code_generation import generate_evaluation_code

print("=" * 80)
print("EVALUATION CODE GENERATION EXAMPLES")
print("=" * 80)

# ============================================================================
# Example 1: Basic Classification Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: Basic Classification Evaluation")
print("=" * 80)

classification_config = {
    'experiment_name': 'Customer Churn Prediction',
    'task_type': 'classification',
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    'include_confusion_matrix': True,
    'include_roc_curve': True,
    'random_state': 42
}

# Generate complete script
script_code = generate_evaluation_code(classification_config, output_format='script')
print("Generated evaluation script (first 500 chars):")
print(script_code[:500] + "...")

# ============================================================================
# Example 2: Regression Evaluation with Residual Analysis
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Regression Evaluation with Residual Analysis")
print("=" * 80)

regression_config = {
    'experiment_name': 'House Price Prediction',
    'task_type': 'regression',
    'metrics': ['mae', 'mse', 'rmse', 'r2', 'mape'],
    'include_residual_analysis': True,
    'save_results': True,
    'results_path': 'regression_results.json'
}

# Generate evaluation function
function_code = generate_evaluation_code(regression_config, output_format='function')
print("Generated evaluation function (first 400 chars):")
print(function_code[:400] + "...")

# ============================================================================
# Example 3: Clustering Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Clustering Evaluation")
print("=" * 80)

clustering_config = {
    'experiment_name': 'Customer Segmentation',
    'task_type': 'clustering',
    'metrics': ['silhouette', 'calinski_harabasz', 'davies_bouldin'],
    'random_state': 42
}

# Generate evaluation module
module_code = generate_evaluation_code(clustering_config, output_format='module')
print("Generated evaluation module (first 400 chars):")
print(module_code[:400] + "...")

# ============================================================================
# Example 4: Minimal Classification Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Minimal Classification Evaluation (Default Metrics)")
print("=" * 80)

minimal_config = {
    'task_type': 'classification',
    'metrics': []  # Will use default metrics
}

minimal_code = generate_evaluation_code(minimal_config, output_format='function')
print("Generated minimal evaluation (first 300 chars):")
print(minimal_code[:300] + "...")

# ============================================================================
# Example 5: Binary Classification with ROC/PR Curves
# ============================================================================
print("\n" + "=" * 80)
print("Example 5: Binary Classification with ROC/PR Curves")
print("=" * 80)

binary_config = {
    'experiment_name': 'Fraud Detection',
    'task_type': 'classification',
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    'include_confusion_matrix': True,
    'include_roc_curve': True,
    'include_pr_curve': True,
    'save_results': True
}

binary_code = generate_evaluation_code(binary_config, output_format='script')
print("Generated binary classification evaluation (first 400 chars):")
print(binary_code[:400] + "...")

# ============================================================================
# Example 6: Multi-class Classification
# ============================================================================
print("\n" + "=" * 80)
print("Example 6: Multi-class Classification")
print("=" * 80)

multiclass_config = {
    'experiment_name': 'Image Classification',
    'task_type': 'classification',
    'metrics': ['accuracy', 'precision', 'recall', 'f1'],
    'include_confusion_matrix': True,
    'save_results': True,
    'results_path': 'multiclass_results.json'
}

multiclass_code = generate_evaluation_code(multiclass_config, output_format='script')
print("Generated multi-class evaluation (first 400 chars):")
print(multiclass_code[:400] + "...")

# ============================================================================
# Example 7: Regression with Custom Metrics
# ============================================================================
print("\n" + "=" * 80)
print("Example 7: Regression with Custom Metrics")
print("=" * 80)

custom_regression_config = {
    'experiment_name': 'Sales Forecasting',
    'task_type': 'regression',
    'metrics': ['mae', 'rmse', 'r2'],  # Subset of metrics
    'include_residual_analysis': True,
    'save_results': True
}

custom_regression_code = generate_evaluation_code(custom_regression_config, output_format='function')
print("Generated custom regression evaluation (first 300 chars):")
print(custom_regression_code[:300] + "...")

# ============================================================================
# Example 8: Save Generated Code to Files
# ============================================================================
print("\n" + "=" * 80)
print("Example 8: Save Generated Code to Files")
print("=" * 80)

# Save complete classification script
with open('generated_classification_evaluation.py', 'w', encoding='utf-8') as f:
    f.write(script_code)
print("[OK] Classification evaluation script saved to 'generated_classification_evaluation.py'")

# Save regression function
with open('regression_evaluation_function.py', 'w', encoding='utf-8') as f:
    f.write("import numpy as np\nimport pandas as pd\n")
    f.write("from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n\n")
    f.write(function_code)
print("[OK] Regression evaluation function saved to 'regression_evaluation_function.py'")

# Save clustering module
with open('clustering_evaluation_module.py', 'w', encoding='utf-8') as f:
    f.write(module_code)
print("[OK] Clustering evaluation module saved to 'clustering_evaluation_module.py'")

# ============================================================================
# Example 9: Generate for Different Task Types
# ============================================================================
print("\n" + "=" * 80)
print("Example 9: Generate for Different Task Types")
print("=" * 80)

task_types = [
    ('classification', ['accuracy', 'f1']),
    ('regression', ['mae', 'r2']),
    ('clustering', ['silhouette'])
]

for task_type, metrics in task_types:
    config = {
        'experiment_name': f'{task_type.title()} Evaluation',
        'task_type': task_type,
        'metrics': metrics
    }
    
    code = generate_evaluation_code(config, output_format='function')
    filename = f'evaluate_{task_type}.py'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("import numpy as np\nimport pandas as pd\n")
        f.write("from sklearn.metrics import *\n\n")
        f.write(code)
    
    print(f"[OK] {task_type.title()} evaluation saved to '{filename}'")

# ============================================================================
# Example 10: Comparison of Output Formats
# ============================================================================
print("\n" + "=" * 80)
print("Example 10: Comparison of Output Formats")
print("=" * 80)

comparison_config = {
    'experiment_name': 'Format Comparison',
    'task_type': 'classification',
    'metrics': ['accuracy', 'f1']
}

print("\n--- Script Format ---")
script = generate_evaluation_code(comparison_config, output_format='script')
print(f"Length: {len(script)} characters")
print("Includes: imports, evaluation function, visualization, save results, main execution")

print("\n--- Function Format ---")
function = generate_evaluation_code(comparison_config, output_format='function')
print(f"Length: {len(function)} characters")
print("Includes: evaluation function only (reusable)")

print("\n--- Module Format ---")
module = generate_evaluation_code(comparison_config, output_format='module')
print(f"Length: {len(module)} characters")
print("Includes: module docstring, imports, config, functions, main block")

# ============================================================================
# Example 11: Real-World Use Case - Titanic Survival Prediction
# ============================================================================
print("\n" + "=" * 80)
print("Example 11: Real-World Use Case - Titanic Survival Prediction")
print("=" * 80)

titanic_config = {
    'experiment_name': 'Titanic Survival Prediction',
    'task_type': 'classification',
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    'include_confusion_matrix': True,
    'include_roc_curve': True,
    'save_results': True,
    'results_path': 'titanic_evaluation_results.json',
    'random_state': 42
}

titanic_code = generate_evaluation_code(titanic_config, output_format='script')
with open('titanic_evaluation.py', 'w', encoding='utf-8') as f:
    f.write(titanic_code)
print("[OK] Titanic evaluation saved to 'titanic_evaluation.py'")
print(f"Generated {len(titanic_code)} characters of production-ready code")

# ============================================================================
# Example 12: Feature Importance Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("Example 12: Evaluation with Feature Importance")
print("=" * 80)

feature_importance_config = {
    'experiment_name': 'Random Forest Analysis',
    'task_type': 'classification',
    'metrics': ['accuracy', 'f1'],
    'include_feature_importance': True,
    'save_results': True
}

fi_code = generate_evaluation_code(feature_importance_config, output_format='script')
print("Generated evaluation with feature importance (first 400 chars):")
print(fi_code[:400] + "...")

print("\n" + "=" * 80)
print("Examples completed successfully!")
print("Generated files:")
print("  - generated_classification_evaluation.py: Complete classification evaluation")
print("  - regression_evaluation_function.py: Regression evaluation function")
print("  - clustering_evaluation_module.py: Clustering evaluation module")
print("  - evaluate_*.py: Task-specific evaluation functions")
print("  - titanic_evaluation.py: Real-world example")
print("=" * 80)
