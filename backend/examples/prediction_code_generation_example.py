"""
Example: Prediction Code Generation

This script demonstrates how to generate Python code for model predictions
from experiment configurations.
"""

from app.ml_engine.code_generation import generate_prediction_code
import os

# Create output directory for generated code
OUTPUT_DIR = 'generated_code/prediction'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("PREDICTION CODE GENERATION EXAMPLES")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 80)

# ============================================================================
# Example 1: Basic Classification Prediction
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: Basic Classification Prediction")
print("=" * 80)

classification_config = {
    'experiment_name': 'Customer Churn Prediction',
    'task_type': 'classification',
    'model_path': 'models/churn_model.pkl',
    'include_probabilities': True,
    'save_predictions': True,
    'output_path': 'predictions/churn_predictions.csv'
}

# Generate complete script
script_code = generate_prediction_code(classification_config, output_format='script')
print("Generated prediction script (first 500 chars):")
print(script_code[:500] + "...")

# ============================================================================
# Example 2: Regression Prediction
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Regression Prediction")
print("=" * 80)

regression_config = {
    'experiment_name': 'House Price Prediction',
    'task_type': 'regression',
    'model_path': 'models/price_model.pkl',
    'save_predictions': True,
    'output_path': 'predictions/price_predictions.csv'
}

# Generate prediction function
function_code = generate_prediction_code(regression_config, output_format='function')
print("Generated prediction function (first 400 chars):")
print(function_code[:400] + "...")

# ============================================================================
# Example 3: Clustering Prediction
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Clustering Prediction")
print("=" * 80)

clustering_config = {
    'experiment_name': 'Customer Segmentation',
    'task_type': 'clustering',
    'model_path': 'models/clustering_model.pkl',
    'save_predictions': True
}

# Generate prediction module
module_code = generate_prediction_code(clustering_config, output_format='module')
print("Generated prediction module (first 400 chars):")
print(module_code[:400] + "...")

# ============================================================================
# Example 4: Prediction with Preprocessing
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Prediction with Preprocessing")
print("=" * 80)

preprocessing_config = {
    'experiment_name': 'Fraud Detection',
    'task_type': 'classification',
    'model_path': 'models/fraud_model.pkl',
    'include_preprocessing': True,
    'preprocessing_path': 'models/preprocessor.pkl',
    'include_probabilities': True
}

preprocessing_code = generate_prediction_code(preprocessing_config, output_format='script')
print("Generated prediction with preprocessing (first 400 chars):")
print(preprocessing_code[:400] + "...")

# ============================================================================
# Example 5: Batch Prediction
# ============================================================================
print("\n" + "=" * 80)
print("Example 5: Batch Prediction for Large Datasets")
print("=" * 80)

batch_config = {
    'experiment_name': 'Large Scale Prediction',
    'task_type': 'classification',
    'model_path': 'models/model.pkl',
    'batch_prediction': True,
    'save_predictions': True,
    'output_path': 'predictions/batch_predictions.csv'
}

batch_code = generate_prediction_code(batch_config, output_format='script')
print("Generated batch prediction script (first 400 chars):")
print(batch_code[:400] + "...")

# ============================================================================
# Example 6: FastAPI Prediction Service
# ============================================================================
print("\n" + "=" * 80)
print("Example 6: FastAPI Prediction Service")
print("=" * 80)

api_config = {
    'experiment_name': 'ML Prediction API',
    'task_type': 'classification',
    'model_path': 'models/api_model.pkl'
}

api_code = generate_prediction_code(api_config, output_format='api')
print("Generated FastAPI service (first 500 chars):")
print(api_code[:500] + "...")

# ============================================================================
# Example 7: Save Generated Code to Files
# ============================================================================
print("\n" + "=" * 80)
print("Example 7: Save Generated Code to Files")
print("=" * 80)

# Save complete prediction script
with open(f'{OUTPUT_DIR}/prediction_script.py', 'w', encoding='utf-8') as f:
    f.write(script_code)
print(f"[OK] Prediction script saved to '{OUTPUT_DIR}/prediction_script.py'")

# Save prediction function
with open(f'{OUTPUT_DIR}/prediction_function.py', 'w', encoding='utf-8') as f:
    f.write("import numpy as np\nimport pandas as pd\nimport pickle\n\n")
    f.write(function_code)
print(f"[OK] Prediction function saved to '{OUTPUT_DIR}/prediction_function.py'")

# Save prediction module
with open(f'{OUTPUT_DIR}/prediction_module.py', 'w', encoding='utf-8') as f:
    f.write(module_code)
print(f"[OK] Prediction module saved to '{OUTPUT_DIR}/prediction_module.py'")

# Save FastAPI service
with open(f'{OUTPUT_DIR}/prediction_api.py', 'w', encoding='utf-8') as f:
    f.write(api_code)
print(f"[OK] FastAPI service saved to '{OUTPUT_DIR}/prediction_api.py'")

# ============================================================================
# Example 8: Generate for Different Task Types
# ============================================================================
print("\n" + "=" * 80)
print("Example 8: Generate for Different Task Types")
print("=" * 80)

task_types = ['classification', 'regression', 'clustering']

for task_type in task_types:
    config = {
        'experiment_name': f'{task_type.title()} Prediction',
        'task_type': task_type,
        'model_path': f'models/{task_type}_model.pkl',
        'save_predictions': True
    }
    
    code = generate_prediction_code(config, output_format='function')
    filename = f'{OUTPUT_DIR}/predict_{task_type}.py'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("import numpy as np\nimport pandas as pd\nimport pickle\n\n")
        f.write(code)
    
    print(f"[OK] {task_type.title()} prediction saved to '{filename}'")

# ============================================================================
# Example 9: Comparison of Output Formats
# ============================================================================
print("\n" + "=" * 80)
print("Example 9: Comparison of Output Formats")
print("=" * 80)

comparison_config = {
    'experiment_name': 'Format Comparison',
    'task_type': 'classification',
    'model_path': 'model.pkl'
}

print("\n--- Script Format ---")
script = generate_prediction_code(comparison_config, output_format='script')
print(f"Length: {len(script)} characters")
print("Includes: imports, load model, predict, save, main execution")

print("\n--- Function Format ---")
function = generate_prediction_code(comparison_config, output_format='function')
print(f"Length: {len(function)} characters")
print("Includes: prediction function only (reusable)")

print("\n--- API Format ---")
api = generate_prediction_code(comparison_config, output_format='api')
print(f"Length: {len(api)} characters")
print("Includes: FastAPI app, endpoints, request/response models")

print("\n--- Module Format ---")
module = generate_prediction_code(comparison_config, output_format='module')
print(f"Length: {len(module)} characters")
print("Includes: module docstring, imports, config, functions, main block")

# ============================================================================
# Example 10: Real-World Use Case - Titanic Survival Prediction
# ============================================================================
print("\n" + "=" * 80)
print("Example 10: Real-World Use Case - Titanic Survival Prediction")
print("=" * 80)

titanic_config = {
    'experiment_name': 'Titanic Survival Prediction',
    'task_type': 'classification',
    'model_path': 'models/titanic_model.pkl',
    'include_preprocessing': True,
    'preprocessing_path': 'models/titanic_preprocessor.pkl',
    'include_probabilities': True,
    'batch_prediction': True,
    'save_predictions': True,
    'output_path': 'predictions/titanic_predictions.csv'
}

titanic_code = generate_prediction_code(titanic_config, output_format='script')
with open(f'{OUTPUT_DIR}/titanic_prediction.py', 'w', encoding='utf-8') as f:
    f.write(titanic_code)
print(f"[OK] Titanic prediction saved to '{OUTPUT_DIR}/titanic_prediction.py'")
print(f"Generated {len(titanic_code)} characters of production-ready code")

# ============================================================================
# Example 11: Microservice Deployment
# ============================================================================
print("\n" + "=" * 80)
print("Example 11: Microservice Deployment (FastAPI)")
print("=" * 80)

microservice_config = {
    'experiment_name': 'Churn Prediction Microservice',
    'task_type': 'classification',
    'model_path': 'models/churn_model.pkl'
}

microservice_code = generate_prediction_code(microservice_config, output_format='api')
with open(f'{OUTPUT_DIR}/churn_microservice.py', 'w', encoding='utf-8') as f:
    f.write(microservice_code)
print(f"[OK] Microservice saved to '{OUTPUT_DIR}/churn_microservice.py'")
print("\nTo run the microservice:")
print(f"  cd {OUTPUT_DIR}")
print("  uvicorn churn_microservice:app --reload")
print("  Then visit: http://localhost:8000/docs")

# ============================================================================
# Example 12: Complete Inference Pipeline
# ============================================================================
print("\n" + "=" * 80)
print("Example 12: Complete Inference Pipeline")
print("=" * 80)

inference_config = {
    'experiment_name': 'Complete Inference Pipeline',
    'task_type': 'classification',
    'model_path': 'models/final_model.pkl',
    'include_preprocessing': True,
    'preprocessing_path': 'models/final_preprocessor.pkl',
    'batch_prediction': True,
    'save_predictions': True,
    'output_path': 'predictions/final_predictions.csv',
    'input_format': 'csv'
}

inference_code = generate_prediction_code(inference_config, output_format='module')
with open(f'{OUTPUT_DIR}/inference_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(inference_code)
print(f"[OK] Complete inference pipeline saved to '{OUTPUT_DIR}/inference_pipeline.py'")

print("\n" + "=" * 80)
print("Examples completed successfully!")
print(f"All files saved to: {OUTPUT_DIR}/")
print("Generated files:")
print("  - prediction_script.py: Complete prediction script")
print("  - prediction_function.py: Reusable prediction function")
print("  - prediction_module.py: Importable prediction module")
print("  - prediction_api.py: FastAPI prediction service")
print("  - predict_*.py: Task-specific prediction functions")
print("  - titanic_prediction.py: Real-world example")
print("  - churn_microservice.py: Deployable microservice")
print("  - inference_pipeline.py: Complete inference pipeline")
print("=" * 80)
