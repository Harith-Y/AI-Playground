"""
Database session management
"""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import Pool
from app.core.config import Settings
import logging

logger = logging.getLogger(__name__)
settings = Settings()

# Detect database type and set appropriate connection args
is_sqlite = settings.DATABASE_URL.startswith("sqlite")
is_postgres = settings.DATABASE_URL.startswith("postgresql")

# Build engine configuration based on database type
engine_config = {}

if is_postgres:
    # PostgreSQL-specific configuration (optimized for NeonDB serverless)
    engine_config.update({
        "pool_size": 3,  # Small pool for serverless (Neon autoscales)
        "max_overflow": 7,  # Allow burst traffic (total 10 connections max)
        "pool_recycle": 60,  # Recycle connections after 1 minute (Neon hibernates quickly)
        "pool_timeout": 15,  # Wait up to 15 seconds for a connection
        "pool_pre_ping": True,  # Always verify connections before using
        "connect_args": {
            "connect_timeout": 10  # Connection timeout
        }
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

# Add connection pool listener to log pool statistics for debugging
@event.listens_for(Pool, "connect")
def receive_connect(dbapi_conn, connection_record):
    logger.debug(f"New DB connection established")

@event.listens_for(Pool, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    logger.debug(f"Connection checked out from pool. Pool size: {engine.pool.size()}, Checked out: {engine.pool.checkedout()}")

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """
    Dependency for getting database session with proper cleanup
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        logger.debug("Database session closed")
