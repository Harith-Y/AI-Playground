"""
Database initialization script
"""
import subprocess
import sys
from pathlib import Path
from app.core.config import Settings
from sqlalchemy import create_engine, inspect

def check_tables_exist():
    """Check if database tables already exist"""
    settings = Settings()
    engine = create_engine(settings.DATABASE_URL)
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    return len(tables) > 0

def check_migrations_exist():
    """Check if migration files exist"""
    versions_dir = Path(__file__).parent / "alembic" / "versions"
    migration_files = list(versions_dir.glob("*.py"))
    # Exclude __init__.py
    migration_files = [f for f in migration_files if f.name != "__init__.py"]
    return len(migration_files) > 0

def main():
    print("🚀 Initializing database...")
    
    # Check if tables already exist
    if check_tables_exist():
        print("⚠️  Tables already exist in the database!")
        response = input("Do you want to continue? This might cause errors. (y/N): ")
        if response.lower() != 'y':
            print("❌ Aborted.")
            return
    
    # Check if migrations already exist
    if check_migrations_exist():
        print("⚠️  Migration files already exist!")
        response = input("Apply existing migrations instead of creating new ones? (Y/n): ")
        if response.lower() != 'n':
            print("\n📦 Applying existing migrations...")
            result = subprocess.run(
                ["alembic", "upgrade", "head"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ Migrations applied successfully!")
                print(result.stdout)
            else:
                print("❌ Error applying migrations:")
                print(result.stderr)
                sys.exit(1)
            return
    
    # Create new migration
    print("\n📝 Generating migration...")
    result = subprocess.run(
        ["alembic", "revision", "--autogenerate", "-m", "Create initial tables"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("❌ Error generating migration:")
        print(result.stderr)
        sys.exit(1)
    
    print("✅ Migration generated!")
    print(result.stdout)
    
    # Apply migration
    print("\n📦 Applying migration...")
    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Database initialized successfully!")
        print(result.stdout)
        
        # Show created tables
        from sqlalchemy import create_engine, inspect
        settings = Settings()
        engine = create_engine(settings.DATABASE_URL)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print("\n📊 Created tables:")
        for table in tables:
            print(f"   • {table}")
    else:
        print("❌ Error applying migration:")
        print(result.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()