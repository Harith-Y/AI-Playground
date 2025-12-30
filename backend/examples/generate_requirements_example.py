"""
Requirements Generator Example

Demonstrates how to use the requirements generator to create
minimal, modular requirements.txt files.
"""

from app.ml_engine.code_generation import generate_requirements
from pathlib import Path


def example_1_basic_requirements():
    """Example 1: Generate basic requirements.txt"""
    print("=" * 80)
    print("Example 1: Basic Requirements")
    print("=" * 80)
    
    config = {
        'model_type': 'random_forest_classifier',
        'task_type': 'classification',
        'preprocessing_steps': [
            {'type': 'missing_value_imputation'},
            {'type': 'scaling'}
        ]
    }
    
    requirements = generate_requirements(config)
    
    print("\nGenerated requirements.txt:")
    print("-" * 80)
    print(requirements['requirements.txt'])
    
    # Save to file
    output_dir = Path('generated_requirements')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'requirements.txt', 'w') as f:
        f.write(requirements['requirements.txt'])
    
    print(f"\n✓ Saved to {output_dir / 'requirements.txt'}")


def example_2_modular_requirements():
    """Example 2: Generate modular requirements files"""
    print("\n" + "=" * 80)
    print("Example 2: Modular Requirements")
    print("=" * 80)
    
    config = {
        'model_type': 'xgboost_classifier',
        'task_type': 'classification',
        'preprocessing_steps': [
            {'type': 'missing_value_imputation', 'strategy': 'mean'},
            {'type': 'outlier_detection', 'method': 'iqr'},
            {'type': 'scaling', 'scaler': 'standard'},
            {'type': 'encoding', 'encoder': 'onehot'}
        ],
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 10,
            'learning_rate': 0.1
        },
        'include_evaluation': True
    }
    
    requirements_files = generate_requirements(config, modular=True)
    
    print(f"\nGenerated {len(requirements_files)} modular files:")
    for filename in requirements_files.keys():
        print(f"  - {filename}")
    
    # Save all files
    output_dir = Path('generated_requirements/modular')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, content in requirements_files.items():
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ Saved {filepath}")
    
    # Show core requirements
    print("\nCore requirements.txt:")
    print("-" * 80)
    print(requirements_files['requirements.txt'][:500] + "...")


def example_3_docker_requirements():
    """Example 3: Generate Docker-optimized requirements"""
    print("\n" + "=" * 80)
    print("Example 3: Docker Requirements")
    print("=" * 80)
    
    config = {
        'model_type': 'lightgbm_regressor',
        'task_type': 'regression',
        'preprocessing_steps': [{'type': 'scaling'}],
        'include_evaluation': True
    }
    
    requirements = generate_requirements(config, output_format='docker')
    
    print("\nGenerated Docker requirements.txt:")
    print("-" * 80)
    print(requirements['requirements.txt'])
    
    # Save to file
    output_dir = Path('generated_requirements/docker')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'requirements.txt', 'w') as f:
        f.write(requirements['requirements.txt'])
    
    print(f"\n✓ Saved to {output_dir / 'requirements.txt'}")


def example_4_conda_environment():
    """Example 4: Generate conda environment.yml"""
    print("\n" + "=" * 80)
    print("Example 4: Conda Environment")
    print("=" * 80)
    
    config = {
        'model_type': 'catboost_classifier',
        'task_type': 'classification',
        'preprocessing_steps': [
            {'type': 'missing_value_imputation'},
            {'type': 'encoding'}
        ],
        'include_evaluation': True
    }
    
    env = generate_requirements(config, output_format='conda')
    
    print("\nGenerated environment.yml:")
    print("-" * 80)
    print(env['environment.yml'])
    
    # Save to file
    output_dir = Path('generated_requirements/conda')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'environment.yml', 'w') as f:
        f.write(env['environment.yml'])
    
    print(f"\n✓ Saved to {output_dir / 'environment.yml'}")


def example_5_code_analysis():
    """Example 5: Generate requirements from code analysis"""
    print("\n" + "=" * 80)
    print("Example 5: Code Analysis")
    print("=" * 80)
    
    # Simulate generated code sections
    preprocessing_code = """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
"""
    
    training_code = """
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
"""
    
    evaluation_code = """
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
"""
    
    config = {
        'model_type': 'random_forest_classifier',
        'task_type': 'classification'
    }
    
    code_sections = {
        'preprocessing': preprocessing_code,
        'training': training_code,
        'evaluation': evaluation_code
    }
    
    requirements_files = generate_requirements(
        config,
        modular=True,
        code_sections=code_sections
    )
    
    print(f"\nAnalyzed code and generated {len(requirements_files)} files")
    
    # Show preprocessing requirements (analyzed from code)
    print("\nPreprocessing requirements (from code analysis):")
    print("-" * 80)
    print(requirements_files['requirements-preprocessing.txt'][:400] + "...")
    
    # Save all files
    output_dir = Path('generated_requirements/code_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, content in requirements_files.items():
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
    
    print(f"\n✓ Saved all files to {output_dir}/")


def example_6_minimal_prediction():
    """Example 6: Minimal requirements for prediction/inference"""
    print("\n" + "=" * 80)
    print("Example 6: Minimal Prediction Requirements")
    print("=" * 80)
    
    config = {
        'model_type': 'xgboost_classifier',
        'task_type': 'classification',
        'include_evaluation': False  # No evaluation needed for inference
    }
    
    requirements_files = generate_requirements(config, modular=True)
    
    # Show only prediction requirements
    print("\nMinimal requirements for prediction/inference:")
    print("-" * 80)
    print(requirements_files['requirements-prediction.txt'])
    
    # Save prediction requirements
    output_dir = Path('generated_requirements/prediction')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'requirements.txt', 'w') as f:
        f.write(requirements_files['requirements-prediction.txt'])
    
    print(f"\n✓ Saved to {output_dir / 'requirements.txt'}")
    print("\nThis minimal file is perfect for production inference containers!")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("REQUIREMENTS GENERATOR EXAMPLES")
    print("=" * 80)
    
    try:
        example_1_basic_requirements()
        example_2_modular_requirements()
        example_3_docker_requirements()
        example_4_conda_environment()
        example_5_code_analysis()
        example_6_minimal_prediction()
        
        print("\n" + "=" * 80)
        print("✓ All examples completed successfully!")
        print("=" * 80)
        print("\nGenerated files are in: generated_requirements/")
        print("\nNext steps:")
        print("  1. Review the generated requirements files")
        print("  2. Install dependencies: pip install -r requirements.txt")
        print("  3. Use modular files for specific use cases")
        print("  4. Customize as needed for your project")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
