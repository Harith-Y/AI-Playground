"""
FastAPI server startup script with automatic database migrations
"""
import sys
import uvicorn
from app.db.migration_manager import run_migrations_on_startup
from app.utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    # Run database migrations before starting server
    logger.info("Checking database migrations...")
    
    try:
        migration_success = run_migrations_on_startup(
            auto_upgrade=True,      # Automatically upgrade to latest
            wait_for_db=True,       # Wait for database to be available
            fail_on_error=False     # Don't fail startup on migration error
        )
        
        if not migration_success:
            logger.warning("Migration failed or skipped, starting server anyway")
    
    except Exception as e:
        logger.error(f"Migration error: {e}")
        logger.warning("Starting server despite migration error")
    
    # Start FastAPI server
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
