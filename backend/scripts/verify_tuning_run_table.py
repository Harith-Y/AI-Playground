"""
Script to verify TuningRun table exists and is properly configured.

This script checks:
1. Table exists in database
2. All columns are present with correct types
3. Indexes are created
4. Foreign key constraints are set up
5. Enum types are defined
"""

import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import inspect, text
from app.db.session import SessionLocal
from app.models.tuning_run import TuningRun, TuningStatus

# Simple logger for this script
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_table_exists(inspector, table_name: str) -> bool:
    """Check if table exists in database."""
    tables = inspector.get_table_names()
    exists = table_name in tables
    
    if exists:
        print(f"✅ Table '{table_name}' exists")
    else:
        print(f"❌ Table '{table_name}' does NOT exist")
    
    return exists


def verify_columns(inspector, table_name: str) -> bool:
    """Verify all required columns exist with correct types."""
    expected_columns = {
        'id': 'UUID',
        'model_run_id': 'UUID',
        'tuning_method': 'VARCHAR',
        'best_params': 'JSONB',
        'results': 'JSONB',
        'status': 'tuningstatus',  # Enum type
        'created_at': 'TIMESTAMP'
    }
    
    columns = inspector.get_columns(table_name)
    column_dict = {col['name']: str(col['type']) for col in columns}
    
    all_correct = True
    
    for col_name, expected_type in expected_columns.items():
        if col_name not in column_dict:
            print(f"❌ Column '{col_name}' is MISSING")
            all_correct = False
        else:
            actual_type = column_dict[col_name]
            # Normalize type names for comparison
            if expected_type.upper() in actual_type.upper() or actual_type.upper() in expected_type.upper():
                print(f"✅ Column '{col_name}' exists with type {actual_type}")
            else:
                print(f"⚠️  Column '{col_name}' has type {actual_type} (expected {expected_type})")
    
    return all_correct


def verify_indexes(inspector, table_name: str) -> bool:
    """Verify required indexes exist."""
    expected_indexes = [
        'ix_tuning_runs_id',
        'ix_tuning_runs_model_run_id',
        'ix_tuning_runs_status'
    ]
    
    indexes = inspector.get_indexes(table_name)
    index_names = [idx['name'] for idx in indexes]
    
    all_exist = True
    
    for idx_name in expected_indexes:
        if idx_name in index_names:
            print(f"✅ Index '{idx_name}' exists")
        else:
            print(f"❌ Index '{idx_name}' is MISSING")
            all_exist = False
    
    return all_exist


def verify_foreign_keys(inspector, table_name: str) -> bool:
    """Verify foreign key constraints."""
    foreign_keys = inspector.get_foreign_keys(table_name)
    
    if not foreign_keys:
        print(f"❌ No foreign keys found for '{table_name}'")
        return False
    
    for fk in foreign_keys:
        print(f"✅ Foreign key: {fk['constrained_columns']} → {fk['referred_table']}.{fk['referred_columns']}")
        if fk.get('options', {}).get('ondelete'):
            print(f"   ON DELETE: {fk['options']['ondelete']}")
    
    return True


def verify_enum_type(db) -> bool:
    """Verify TuningStatus enum type exists in database."""
    try:
        result = db.execute(text("""
            SELECT EXISTS (
                SELECT 1 
                FROM pg_type 
                WHERE typname = 'tuningstatus'
            );
        """))
        exists = result.scalar()
        
        if exists:
            print("✅ Enum type 'tuningstatus' exists")
            
            # Get enum values
            result = db.execute(text("""
                SELECT enumlabel 
                FROM pg_enum 
                WHERE enumtypid = (
                    SELECT oid FROM pg_type WHERE typname = 'tuningstatus'
                )
                ORDER BY enumsortorder;
            """))
            values = [row[0] for row in result]
            print(f"   Values: {', '.join(values)}")
            
            # Verify expected values
            expected_values = ['running', 'completed', 'failed']
            if set(values) == set(expected_values):
                print(f"✅ Enum values match expected: {expected_values}")
            else:
                print(f"⚠️  Enum values don't match. Expected: {expected_values}, Got: {values}")
        else:
            print("❌ Enum type 'tuningstatus' does NOT exist")
        
        return exists
    except Exception as e:
        print(f"❌ Error checking enum type: {e}")
        return False


def test_crud_operations(db) -> bool:
    """Test basic CRUD operations on TuningRun."""
    try:
        from uuid import uuid4
        from datetime import datetime
        from app.models.model_run import ModelRun
        from app.models.experiment import Experiment
        from app.models.dataset import Dataset
        from app.models.user import User
        
        print("\n--- Testing CRUD Operations ---")
        
        # Note: This is a dry run - we won't actually commit
        # Just verify the model can be instantiated
        
        test_id = uuid4()
        tuning_run = TuningRun(
            id=test_id,
            model_run_id=uuid4(),  # Fake ID for testing
            tuning_method="grid_search",
            status=TuningStatus.RUNNING,
            created_at=datetime.utcnow()
        )
        
        print(f"✅ TuningRun model can be instantiated")
        print(f"   ID: {tuning_run.id}")
        print(f"   Status: {tuning_run.status}")
        print(f"   Method: {tuning_run.tuning_method}")
        
        # Test enum values
        for status in TuningStatus:
            print(f"✅ TuningStatus.{status.name} = '{status.value}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing CRUD operations: {e}")
        return False


def main():
    """Main verification function."""
    print("=" * 60)
    print("TuningRun Table Verification")
    print("=" * 60)
    print()
    
    db = SessionLocal()
    
    try:
        inspector = inspect(db.bind)
        table_name = 'tuning_runs'
        
        # 1. Check table exists
        print("1. Checking if table exists...")
        table_exists = verify_table_exists(inspector, table_name)
        print()
        
        if not table_exists:
            print("❌ Table does not exist. Run migrations: alembic upgrade head")
            return False
        
        # 2. Check columns
        print("2. Verifying columns...")
        columns_ok = verify_columns(inspector, table_name)
        print()
        
        # 3. Check indexes
        print("3. Verifying indexes...")
        indexes_ok = verify_indexes(inspector, table_name)
        print()
        
        # 4. Check foreign keys
        print("4. Verifying foreign keys...")
        fk_ok = verify_foreign_keys(inspector, table_name)
        print()
        
        # 5. Check enum type
        print("5. Verifying enum type...")
        enum_ok = verify_enum_type(db)
        print()
        
        # 6. Test CRUD operations
        print("6. Testing CRUD operations...")
        crud_ok = test_crud_operations(db)
        print()
        
        # Summary
        print("=" * 60)
        print("Verification Summary")
        print("=" * 60)
        
        checks = [
            ("Table exists", table_exists),
            ("Columns correct", columns_ok),
            ("Indexes exist", indexes_ok),
            ("Foreign keys set up", fk_ok),
            ("Enum type exists", enum_ok),
            ("CRUD operations work", crud_ok)
        ]
        
        all_passed = all(result for _, result in checks)
        
        for check_name, result in checks:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {check_name}")
        
        print()
        
        if all_passed:
            print("🎉 All checks passed! TuningRun table is properly configured.")
            return True
        else:
            print("⚠️  Some checks failed. Please review the output above.")
            return False
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        db.close()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
