import sys
import os
from sqlalchemy import create_engine, text, inspect
from app.core.config import settings

def check_db_state():
    print(f"Connecting to {settings.DATABASE_URL}")
    engine = create_engine(settings.DATABASE_URL)
    
    try:
        with engine.connect() as conn:
            # 1. List all tables
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            print("\nExisting Tables:")
            for table in tables:
                print(f" - {table}")
            
            if not tables:
                print(" (No tables found)")

            # 2. Check alembic_version
            if 'alembic_version' in tables:
                result = conn.execute(text("SELECT version_num FROM alembic_version"))
                version = result.scalar()
                print(f"\nAlembic Version: {version}")
            else:
                print("\nAlembic Version table not found.")

    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    check_db_state()
