#!/usr/bin/env python3
"""
CI/CD Database Migration Script

Use this script in CI/CD pipelines to ensure database migrations
are applied before deploying the application.
"""

import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.db.migration_manager import MigrationManager, MigrationError
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main CI/CD migration function."""
    logger.info("=" * 70)
    logger.info("CI/CD Database Migration Script")
    logger.info("=" * 70)
    
    # Check for environment variable flags
    dry_run = os.getenv("MIGRATION_DRY_RUN", "false").lower() == "true"
    wait_timeout = int(os.getenv("MIGRATION_WAIT_TIMEOUT", "60"))
    fail_on_error = os.getenv("MIGRATION_FAIL_ON_ERROR", "true").lower() == "true"
    create_backup = os.getenv("MIGRATION_CREATE_BACKUP", "false").lower() == "true"
    
    logger.info(f"Configuration:")
    logger.info(f"  Dry Run: {dry_run}")
    logger.info(f"  Wait Timeout: {wait_timeout}s")
    logger.info(f"  Fail on Error: {fail_on_error}")
    logger.info(f"  Create Backup: {create_backup}")
    logger.info("")
    
    try:
        # Create migration manager
        manager = MigrationManager(
            auto_upgrade=True,
            backup_before_upgrade=create_backup
        )
        
        # Wait for database
        logger.info(f"Waiting for database (timeout: {wait_timeout}s)...")
        if not manager.wait_for_database(timeout=wait_timeout):
            logger.error("Database not available within timeout")
            if fail_on_error:
                sys.exit(1)
            else:
                logger.warning("Continuing despite database unavailability")
                sys.exit(0)
        
        # Check migration status
        logger.info("\nChecking migration status...")
        status = manager.get_migration_status()
        
        logger.info(f"Current Revision: {status['current_revision']}")
        logger.info(f"Head Revision: {status['head_revision']}")
        logger.info(f"Up to Date: {status['is_up_to_date']}")
        logger.info(f"Pending Migrations: {status['pending_count']}")
        
        if status['pending_migrations']:
            logger.info("\nPending migrations:")
            for rev in status['pending_migrations']:
                logger.info(f"  - {rev}")
        
        # Run migration
        if not status['is_up_to_date']:
            logger.info("\n" + "=" * 70)
            logger.info("Applying Migrations")
            logger.info("=" * 70)
            
            if dry_run:
                logger.info("DRY RUN: Would apply migrations")
                success = True
            else:
                success = manager.upgrade_with_retry()
            
            if success:
                logger.info("\n" + "=" * 70)
                logger.info("✓ Migration completed successfully")
                logger.info("=" * 70)
                
                # Verify final state
                final_status = manager.get_migration_status()
                logger.info(f"\nFinal Revision: {final_status['current_revision']}")
                logger.info(f"Up to Date: {final_status['is_up_to_date']}")
                
                sys.exit(0)
            else:
                logger.error("\n" + "=" * 70)
                logger.error("✗ Migration failed")
                logger.error("=" * 70)
                
                if fail_on_error:
                    sys.exit(1)
                else:
                    logger.warning("Continuing despite migration failure")
                    sys.exit(0)
        else:
            logger.info("\n" + "=" * 70)
            logger.info("✓ Database already up to date")
            logger.info("=" * 70)
            sys.exit(0)
    
    except Exception as e:
        logger.error(f"\nMigration error: {e}", exc_info=True)
        
        if fail_on_error:
            logger.error("Exiting with error status")
            sys.exit(1)
        else:
            logger.warning("Exiting with success status despite error")
            sys.exit(0)


if __name__ == "__main__":
    main()
