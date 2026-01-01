"""
Example: Model and Pipeline Serialization

This example demonstrates how to save and load ML models, preprocessing pipelines,
and complete workflows using the serialization module.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import serialization utilities
from app.ml_engine.utils.serialization import (
    save_model, load_model, get_model_info,
    save_pipeline, load_pipeline, get_pipeline_info,
    save_workflow, load_workflow, get_workflow_info
)

# Import ML components
from app.ml_engine.models.classification import RandomForestClassifierWrapper
from app.ml_engine.models.regression import LinearRegressionWrapper
from app.ml_engine.models.base import ModelConfig
from app.ml_engine.preprocessing.pipeline import Pipeline
from app.ml_engine.preprocessing.scaler import StandardScaler
from app.ml_engine.preprocessing.imputer import MeanImputer
from app.ml_engine.preprocessing.encoder import OneHotEncoder


def example_1_save_load_model():
    """Example 1: Save and load a classification model."""
    print("=" * 80)
    print("Example 1: Save and Load a Classification Model")
    print("=" * 80)
    
    # Create sample data
    np.random.seed(42)
    X_train = pd.DataFrame({
        'age': np.random.randint(20, 60, 100),
        'income': np.random.randint(30000, 100000, 100),
        'experience': np.random.randint(0, 20, 100)
    })
    y_train = pd.Series(np.random.randint(0, 2, 100), name='churn')
    
    # Train a model
    print("\n1. Training a Random Forest classifier...")
    config = ModelConfig(
        model_type='random_forest_classifier',
        hyperparameters={'n_estimators': 50, 'max_depth': 5, 'random_state': 42}
    )
    model = RandomForestClassifierWrapper(config)
    model.fit(X_train, y_train)
    print(f"   Model trained on {len(X_train)} samples")
    
    # Save the model
    print("\n2. Saving the model...")
    save_path = Path('models/example_rf_model.pkl')
    save_model(model, save_path, compression=True, metadata={
        'experiment_id': 'exp_001',
        'dataset': 'customer_churn',
        'notes': 'Example model for serialization demo'
    })
    print(f"   Model saved to: {save_path}.gz")
    
    # Get model info without loading
    print("\n3. Getting model info without loading...")
    info = get_model_info(save_path)
    print(f"   Model type: {info['model_type']}")
    print(f"   Features: {info['n_features']}")
    print(f"   Samples: {info['n_train_samples']}")
    print(f"   File size: {info['file_size_kb']:.2f} KB")
    print(f"   Compressed: {info['compressed']}")
    
    # Load the model
    print("\n4. Loading the model...")
    loaded_model = load_model(save_path)
    print(f"   Model loaded successfully")
    print(f"   Is fitted: {loaded_model.is_fitted}")
    
    # Test predictions
    print("\n5. Testing predictions...")
    X_test = X_train.iloc[:5]
    predictions = loaded_model.predict(X_test)
    print(f"   Predictions: {predictions}")
    
    print("\n✅ Example 1 completed successfully!\n")


def example_2_save_load_pipeline():
    """Example 2: Save and load a preprocessing pipeline."""
    print("=" * 80)
    print("Example 2: Save and Load a Preprocessing Pipeline")
    print("=" * 80)
    
    # Create sample data with missing values
    np.random.seed(42)
    X_train = pd.DataFrame({
        'age': [25, np.nan, 35, 40, np.nan, 50],
        'income': [50000, 60000, np.nan, 80000, 90000, 100000],
        'category': ['A', 'B', 'A', 'C', 'B', 'A']
    })
    
    # Create preprocessing pipeline
    print("\n1. Creating preprocessing pipeline...")
    pipeline = Pipeline(steps=[
        MeanImputer(columns=['age', 'income']),
        StandardScaler(columns=['age', 'income']),
        OneHotEncoder(columns=['category'])
    ], name='CustomerPreprocessing')
    
    # Fit the pipeline
    print("\n2. Fitting the pipeline...")
    pipeline.fit(X_train)
    print(f"   Pipeline fitted with {len(pipeline.steps)} steps")
    
    # Save the pipeline
    print("\n3. Saving the pipeline...")
    save_path = Path('pipelines/example_preprocessing.pkl')
    save_pipeline(pipeline, save_path, metadata={
        'version': '1.0.0',
        'description': 'Example preprocessing pipeline'
    })
    print(f"   Pipeline saved to: {save_path}")
    
    # Get pipeline info
    print("\n4. Getting pipeline info...")
    info = get_pipeline_info(save_path)
    print(f"   Pipeline name: {info['name']}")
    print(f"   Number of steps: {info['num_steps']}")
    print(f"   Steps: {', '.join(info['step_names'])}")
    
    # Load the pipeline
    print("\n5. Loading the pipeline...")
    loaded_pipeline = load_pipeline(save_path)
    print(f"   Pipeline loaded successfully")
    
    # Test transformation
    print("\n6. Testing transformation...")
    X_test = X_train.iloc[:2]
    X_transformed = loaded_pipeline.transform(X_test)
    print(f"   Original shape: {X_test.shape}")
    print(f"   Transformed shape: {X_transformed.shape}")
    
    print("\n✅ Example 2 completed successfully!\n")


def example_3_save_load_workflow():
    """Example 3: Save and load a complete ML workflow."""
    print("=" * 80)
    print("Example 3: Save and Load a Complete ML Workflow")
    print("=" * 80)
    
    # Create sample data
    np.random.seed(42)
    X_train = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })
    y_train = pd.Series(2 * X_train['feature1'] + X_train['feature2'] + np.random.randn(100) * 0.1)
    
    # Create preprocessing pipeline
    print("\n1. Creating preprocessing pipeline...")
    pipeline = Pipeline(steps=[
        StandardScaler()
    ], name='RegressionPreprocessing')
    pipeline.fit(X_train)
    
    # Train model
    print("\n2. Training regression model...")
    config = ModelConfig(
        model_type='linear_regression',
        hyperparameters={}
    )
    model = LinearRegressionWrapper(config)
    X_train_scaled = pipeline.transform(X_train)
    model.fit(X_train_scaled, y_train)
    
    # Save complete workflow
    print("\n3. Saving complete workflow...")
    save_path = Path('workflows/example_regression_workflow.pkl')
    save_workflow(
        pipeline=pipeline,
        model=model,
        path=save_path,
        workflow_name='RegressionWorkflow_v1',
        metadata={
            'model_type': 'linear_regression',
            'preprocessing': 'standard_scaling',
            'r2_score': model.score(X_train_scaled, y_train)
        }
    )
    print(f"   Workflow saved to: {save_path}.gz")
    
    # Get workflow info
    print("\n4. Getting workflow info...")
    info = get_workflow_info(save_path)
    print(f"   Workflow name: {info['workflow_name']}")
    print(f"   Pipeline: {info['pipeline_name']}")
    print(f"   Model: {info['model_class']}")
    print(f"   Features: {info['n_features']}")
    
    # Load workflow
    print("\n5. Loading workflow...")
    loaded_pipeline, loaded_model = load_workflow(save_path)
    print(f"   Workflow loaded successfully")
    
    # Test end-to-end prediction
    print("\n6. Testing end-to-end prediction...")
    X_test = X_train.iloc[:5]
    X_test_scaled = loaded_pipeline.transform(X_test)
    predictions = loaded_model.predict(X_test_scaled)
    print(f"   Predictions: {predictions[:3]}")
    
    print("\n✅ Example 3 completed successfully!\n")


def example_4_model_registry():
    """Example 4: Simple model registry using serialization."""
    print("=" * 80)
    print("Example 4: Simple Model Registry")
    print("=" * 80)
    
    class SimpleModelRegistry:
        """Simple model registry for managing multiple model versions."""
        
        def __init__(self, base_path='models/registry'):
            self.base_path = Path(base_path)
            self.base_path.mkdir(parents=True, exist_ok=True)
        
        def register_model(self, model, name, version, metadata=None):
            """Register a model with version."""
            path = self.base_path / f"{name}_v{version}.pkl"
            save_model(model, path, compression=True, metadata=metadata, overwrite=True)
            print(f"   Registered: {name} v{version}")
            return path
        
        def get_model(self, name, version):
            """Retrieve a specific model version."""
            path = self.base_path / f"{name}_v{version}.pkl"
            return load_model(path)
        
        def list_models(self):
            """List all registered models."""
            models = []
            for path in self.base_path.glob('*.pkl*'):
                try:
                    info = get_model_info(path)
                    models.append({
                        'name': path.stem,
                        'path': path,
                        **info
                    })
                except:
                    pass
            return models
    
    # Create registry
    print("\n1. Creating model registry...")
    registry = SimpleModelRegistry()
    
    # Train and register multiple models
    print("\n2. Training and registering models...")
    X = pd.DataFrame(np.random.randn(50, 3), columns=['f1', 'f2', 'f3'])
    y = pd.Series(np.random.randint(0, 2, 50))
    
    for version in ['1.0.0', '1.1.0', '2.0.0']:
        config = ModelConfig(
            model_type='random_forest_classifier',
            hyperparameters={'n_estimators': int(version.split('.')[0]) * 50, 'random_state': 42}
        )
        model = RandomForestClassifierWrapper(config)
        model.fit(X, y)
        
        registry.register_model(
            model, 'customer_churn', version,
            metadata={'accuracy': model.score(X, y)}
        )
    
    # List all models
    print("\n3. Listing all registered models...")
    models = registry.list_models()
    for model_info in models:
        print(f"   - {model_info['name']}: {model_info['n_features']} features, "
              f"{model_info['file_size_kb']:.2f} KB")
    
    # Load specific version
    print("\n4. Loading specific version...")
    model_v2 = registry.get_model('customer_churn', '2.0.0')
    print(f"   Loaded: customer_churn v2.0.0")
    print(f"   Is fitted: {model_v2.is_fitted}")
    
    print("\n✅ Example 4 completed successfully!\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("MODEL AND PIPELINE SERIALIZATION EXAMPLES")
    print("=" * 80 + "\n")
    
    # Create directories
    Path('models').mkdir(exist_ok=True)
    Path('pipelines').mkdir(exist_ok=True)
    Path('workflows').mkdir(exist_ok=True)
    
    # Run examples
    example_1_save_load_model()
    example_2_save_load_pipeline()
    example_3_save_load_workflow()
    example_4_model_registry()
    
    print("=" * 80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✅")
    print("=" * 80)
    print("\nSaved files:")
    print("  - models/example_rf_model.pkl.gz")
    print("  - pipelines/example_preprocessing.pkl")
    print("  - workflows/example_regression_workflow.pkl.gz")
    print("  - models/registry/customer_churn_v*.pkl.gz")
    print("\nFor more information, see:")
    print("  - app/ml_engine/utils/SERIALIZATION_GUIDE.md")
    print("  - app/ml_engine/utils/SERIALIZATION_README.md")
    print()


if __name__ == '__main__':
    main()
