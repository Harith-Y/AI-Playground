"""
Database session management
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import Settings

settings = Settings()

# Detect database type and set appropriate connection args
is_sqlite = settings.DATABASE_URL.startswith("sqlite")
is_postgres = settings.DATABASE_URL.startswith("postgresql")

# Build engine configuration based on database type
engine_config = {
    "pool_pre_ping": True,
}

if is_postgres:
    # PostgreSQL-specific configuration (optimized for NeonDB serverless)
    engine_config.update({
        "pool_size": 5,  # Lower pool size for serverless
        "max_overflow": 10,  # Reduced overflow for NeonDB
        "pool_recycle": 300,  # Recycle connections after 5 minutes (NeonDB idle timeout)
        "connect_args": {"connect_timeout": 10}  # 10 second connection timeout
    })
elif is_sqlite:
    # SQLite-specific configuration
    engine_config.update({
        "connect_args": {"check_same_thread": False}  # Allow multi-threading
    })

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    **engine_config
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """
    Dependency for getting database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
