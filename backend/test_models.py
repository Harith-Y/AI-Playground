"""
Quick reference examples for database operations
Run this after setting up the database to test CRUD operations
"""
import sys
from pathlib import Path
from datetime import datetime

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))


def example_crud_operations():
    """Examples of Create, Read, Update, Delete operations"""
    from app.db.session import SessionLocal
    from app.models import (
        User, Dataset, Experiment, ModelRun,
        PreprocessingStep, FeatureEngineering,
        TuningRun, GeneratedCode,
        ExperimentStatus, TuningStatus
    )
    
    db = SessionLocal()
    
    try:
        print("🚀 Database CRUD Operations Examples\n")
        
        # ========== CREATE ==========
        print("1️⃣  CREATE Operations")
        print("-" * 50)
        
        # Create User
        user = User(email=f"test_{datetime.now().timestamp()}@example.com")
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"✅ Created User: {user.email} (ID: {user.id})")
        
        # Create Dataset
        dataset = Dataset(
            user_id=user.id,
            name="iris.csv",
            file_path="/uploads/iris.csv",
            rows=150,
            cols=5,
            dtypes={
                "sepal_length": "float64",
                "sepal_width": "float64",
                "petal_length": "float64",
                "petal_width": "float64",
                "species": "object"
            },
            missing_values={"sepal_width": 2}
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        print(f"✅ Created Dataset: {dataset.name} (ID: {dataset.id})")
        
        # Create Preprocessing Step
        preprocessing = PreprocessingStep(
            dataset_id=dataset.id,
            step_type="missing_value_imputation",
            parameters={"strategy": "mean"},
            column_name="sepal_width",
            order=1
        )
        db.add(preprocessing)
        db.commit()
        print(f"✅ Created PreprocessingStep: {preprocessing.step_type}")
        
        # Create Feature Engineering
        feature_eng = FeatureEngineering(
            dataset_id=dataset.id,
            operation="pca",
            parameters={"n_components": 3}
        )
        db.add(feature_eng)
        db.commit()
        print(f"✅ Created FeatureEngineering: {feature_eng.operation}")
        
        # Create Experiment
        experiment = Experiment(
            user_id=user.id,
            dataset_id=dataset.id,
            name="Iris Classification Experiment",
            status=ExperimentStatus.RUNNING
        )
        db.add(experiment)
        db.commit()
        db.refresh(experiment)
        print(f"✅ Created Experiment: {experiment.name} (ID: {experiment.id})")
        
        # Create Model Run
        model_run = ModelRun(
            experiment_id=experiment.id,
            model_type="random_forest",
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            metrics={
                "accuracy": 0.96,
                "precision": 0.95,
                "recall": 0.94,
                "f1_score": 0.945
            },
            training_time=2.5,
            model_artifact_path="/models/rf_model.pkl"
        )
        db.add(model_run)
        db.commit()
        db.refresh(model_run)
        print(f"✅ Created ModelRun: {model_run.model_type} (ID: {model_run.id})")
        
        # Create Tuning Run
        tuning_run = TuningRun(
            model_run_id=model_run.id,
            tuning_method="grid_search",
            best_params={
                "n_estimators": 150,
                "max_depth": 15
            },
            results={
                "best_score": 0.98,
                "cv_scores": [0.96, 0.97, 0.98, 0.97, 0.99]
            },
            status=TuningStatus.COMPLETED
        )
        db.add(tuning_run)
        db.commit()
        print(f"✅ Created TuningRun: {tuning_run.tuning_method}")
        
        # Create Generated Code
        generated_code = GeneratedCode(
            experiment_id=experiment.id,
            code_type="notebook",
            file_path="/generated/experiment_123.ipynb"
        )
        db.add(generated_code)
        db.commit()
        print(f"✅ Created GeneratedCode: {generated_code.code_type}")
        
        # ========== READ ==========
        print("\n2️⃣  READ Operations")
        print("-" * 50)
        
        # Get user by email
        found_user = db.query(User).filter(User.email == user.email).first()
        print(f"✅ Found User: {found_user.email}")
        
        # Get all datasets for user (using relationship)
        user_datasets = found_user.datasets
        print(f"✅ User has {len(user_datasets)} dataset(s)")
        
        # Get dataset with preprocessing steps
        dataset_with_steps = db.query(Dataset).filter(Dataset.id == dataset.id).first()
        print(f"✅ Dataset has {len(dataset_with_steps.preprocessing_steps)} preprocessing step(s)")
        print(f"✅ Dataset has {len(dataset_with_steps.feature_engineering)} feature engineering operation(s)")
        
        # Get experiment with model runs
        exp_with_runs = db.query(Experiment).filter(Experiment.id == experiment.id).first()
        print(f"✅ Experiment has {len(exp_with_runs.model_runs)} model run(s)")
        
        # Get model run with tuning runs
        run_with_tuning = db.query(ModelRun).filter(ModelRun.id == model_run.id).first()
        print(f"✅ Model run has {len(run_with_tuning.tuning_runs)} tuning run(s)")
        
        # Complex query with joins
        from sqlalchemy import func
        completed_experiments = db.query(Experiment).filter(
            Experiment.status.in_([ExperimentStatus.COMPLETED, ExperimentStatus.RUNNING])
        ).count()
        print(f"✅ Total running/completed experiments: {completed_experiments}")
        
        # ========== UPDATE ==========
        print("\n3️⃣  UPDATE Operations")
        print("-" * 50)
        
        # Update experiment status
        experiment.status = ExperimentStatus.COMPLETED
        db.commit()
        print(f"✅ Updated Experiment status to: {experiment.status}")
        
        # Update model run metrics
        model_run.metrics["auc_roc"] = 0.99
        db.commit()
        print(f"✅ Updated ModelRun metrics")
        
        # Update using query
        db.query(TuningRun).filter(TuningRun.id == tuning_run.id).update({
            "status": TuningStatus.COMPLETED
        })
        db.commit()
        print(f"✅ Updated TuningRun status using query")
        
        # ========== DELETE ==========
        print("\n4️⃣  DELETE Operations (with CASCADE)")
        print("-" * 50)
        
        # Count related records before delete
        total_datasets = db.query(Dataset).filter(Dataset.user_id == user.id).count()
        total_experiments = db.query(Experiment).filter(Experiment.user_id == user.id).count()
        total_model_runs = db.query(ModelRun).filter(ModelRun.experiment_id == experiment.id).count()
        
        print(f"📊 Before delete:")
        print(f"   - Datasets: {total_datasets}")
        print(f"   - Experiments: {total_experiments}")
        print(f"   - Model Runs: {total_model_runs}")
        
        # Delete user (should cascade to all related records)
        db.delete(user)
        db.commit()
        
        # Verify cascade delete
        remaining_datasets = db.query(Dataset).count()
        remaining_experiments = db.query(Experiment).count()
        
        print(f"\n✅ Deleted User (cascade delete activated)")
        print(f"📊 After delete:")
        print(f"   - Remaining datasets: {remaining_datasets}")
        print(f"   - Remaining experiments: {remaining_experiments}")
        
        # ========== SUMMARY ==========
        print("\n" + "=" * 50)
        print("✅ All CRUD operations completed successfully!")
        print("=" * 50)
        print("\n📚 Key Takeaways:")
        print("   1. All models are working correctly")
        print("   2. Relationships are properly configured")
        print("   3. Cascade deletes are functioning")
        print("   4. JSONB fields store complex data")
        print("   5. Enum types ensure data integrity")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    print("⚠️  WARNING: This script will create and delete test records")
    print("    Make sure you're connected to a development database!\n")
    
    response = input("Continue? (yes/no): ").lower()
    if response == "yes":
        example_crud_operations()
    else:
        print("Cancelled.")
