"""
Verify database models are correctly set up
"""
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

def verify_models():
    """Verify all models are importable and have correct attributes"""
    print("🔍 Verifying database models...\n")
    
    try:
        # Import all models
        from app.models import (
            User, Dataset, Experiment, ModelRun,
            PreprocessingStep, FeatureEngineering,
            TuningRun, GeneratedCode,
            ExperimentStatus, TuningStatus
        )
        from app.db.base import Base
        
        print("✅ All models imported successfully!")
        
        # Verify model attributes
        models_to_check = {
            "User": User,
            "Dataset": Dataset,
            "Experiment": Experiment,
            "ModelRun": ModelRun,
            "PreprocessingStep": PreprocessingStep,
            "FeatureEngineering": FeatureEngineering,
            "TuningRun": TuningRun,
            "GeneratedCode": GeneratedCode,
        }
        
        print("\n📋 Model Details:")
        print("-" * 80)
        
        for name, model in models_to_check.items():
            table_name = model.__tablename__
            columns = [col.name for col in model.__table__.columns]
            print(f"\n{name} (table: {table_name})")
            print(f"  Columns: {', '.join(columns)}")
            
            # Check relationships
            if hasattr(model, '__mapper__'):
                relationships = [rel.key for rel in model.__mapper__.relationships]
                if relationships:
                    print(f"  Relationships: {', '.join(relationships)}")
        
        print("\n" + "-" * 80)
        print("\n✅ All models verified successfully!")
        print("\n📚 Next Steps:")
        print("  1. Copy .env.example to .env and configure your NeonDB connection")
        print("  2. Run: alembic revision --autogenerate -m 'Create initial tables'")
        print("  3. Run: alembic upgrade head")
        print("  4. Or simply run: python init_db.py")
        print("\n📖 See DATABASE_SETUP.md for detailed instructions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying models: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_models()
    sys.exit(0 if success else 1)
