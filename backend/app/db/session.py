"""
Database session management
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import Settings

settings = Settings()

# Create database engine
# Optimized for NeonDB (serverless Postgres)
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,  # Lower pool size for serverless
    max_overflow=10,  # Reduced overflow for NeonDB
    pool_recycle=300,  # Recycle connections after 5 minutes (NeonDB idle timeout)
    connect_args={"connect_timeout": 10}  # 10 second connection timeout
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
