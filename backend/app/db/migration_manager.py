"""
Automated Database Migration Manager

Handles automatic database migrations on startup, with safety checks,
rollback capabilities, and CI/CD integration.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import time

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import OperationalError, ProgrammingError

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MigrationError(Exception):
    """Base exception for migration errors."""
    pass


class MigrationManager:
    """
    Automated database migration manager.
    
    Features:
    - Automatic migration on startup
    - Safety checks before migration
    - Rollback capabilities
    - Migration status tracking
    - Backup creation (optional)
    - CI/CD integration
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        alembic_ini_path: Optional[str] = None,
        auto_upgrade: bool = True,
        backup_before_upgrade: bool = False,
        max_retry_attempts: int = 3,
        retry_delay: int = 5
    ):
        """
        Initialize migration manager.
        
        Args:
            database_url: Database connection URL (uses settings if None)
            alembic_ini_path: Path to alembic.ini (auto-detected if None)
            auto_upgrade: Automatically upgrade to head on startup
            backup_before_upgrade: Create backup before upgrading
            max_retry_attempts: Maximum retry attempts for migration
            retry_delay: Delay between retry attempts (seconds)
        """
        self.database_url = database_url or settings.DATABASE_URL
        self.auto_upgrade = auto_upgrade
        self.backup_before_upgrade = backup_before_upgrade
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay = retry_delay
        
        # Find alembic.ini
        if alembic_ini_path:
            self.alembic_ini_path = Path(alembic_ini_path)
        else:
            # Auto-detect alembic.ini - check Render's /app structure first
            render_path = Path("/app/alembic.ini")
            if render_path.exists():
                self.alembic_ini_path = render_path
            else:
                # Fall back to local development structure
                backend_dir = Path(__file__).resolve().parent.parent
                self.alembic_ini_path = backend_dir / "alembic.ini"
        
        if not self.alembic_ini_path.exists():
            raise MigrationError(f"Alembic config not found: {self.alembic_ini_path}")
        
        # Create Alembic config
        self.alembic_cfg = Config(str(self.alembic_ini_path))
        self.alembic_cfg.set_main_option("sqlalchemy.url", self.database_url)
        
        # Create engine
        self.engine = None
        
        logger.info(f"Migration manager initialized with config: {self.alembic_ini_path}")
    
    def _create_engine(self):
        """Create database engine."""
        if not self.engine:
            self.engine = create_engine(
                self.database_url,
                pool_pre_ping=True,  # Verify connections before using
                echo=False
            )
        return self.engine
    
    def check_database_connection(self) -> bool:
        """
        Check if database is accessible.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            engine = self._create_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def wait_for_database(self, timeout: int = 30) -> bool:
        """
        Wait for database to be available.
        
        Args:
            timeout: Maximum time to wait (seconds)
        
        Returns:
            True if database becomes available, False if timeout
        """
        start_time = time.time()
        attempt = 0
        
        while time.time() - start_time < timeout:
            attempt += 1
            logger.info(f"Waiting for database... (attempt {attempt})")
            
            if self.check_database_connection():
                return True
            
            time.sleep(2)
        
        logger.error(f"Database not available after {timeout}s")
        return False
    
    def get_current_revision(self) -> Optional[str]:
        """
        Get current database revision.
        
        Returns:
            Current revision ID or None if no migrations applied
        """
        try:
            engine = self._create_engine()
            with engine.connect() as conn:
                migration_context = MigrationContext.configure(conn)
                current_rev = migration_context.get_current_revision()
                return current_rev
        except Exception as e:
            logger.error(f"Failed to get current revision: {e}")
            return None
    
    def get_head_revision(self) -> str:
        """
        Get head (latest) revision from migration scripts.
        
        Returns:
            Head revision ID
        """
        script = ScriptDirectory.from_config(self.alembic_cfg)
        head = script.get_current_head()
        return head
    
    def get_pending_migrations(self) -> List[str]:
        """
        Get list of pending migrations.
        
        Returns:
            List of pending revision IDs
        """
        try:
            current = self.get_current_revision()
            head = self.get_head_revision()
            
            if current == head:
                return []
            
            script = ScriptDirectory.from_config(self.alembic_cfg)
            
            # Get all revisions between current and head
            if current is None:
                # No migrations applied yet
                revisions = []
                for rev in script.iterate_revisions(head, current):
                    revisions.append(rev.revision)
                return list(reversed(revisions))
            else:
                revisions = []
                for rev in script.iterate_revisions(head, current):
                    if rev.revision != current:
                        revisions.append(rev.revision)
                return list(reversed(revisions))
        
        except Exception as e:
            logger.error(f"Failed to get pending migrations: {e}")
            return []
    
    def is_migration_needed(self) -> bool:
        """
        Check if migration is needed.
        
        Returns:
            True if migrations pending, False otherwise
        """
        current = self.get_current_revision()
        head = self.get_head_revision()
        
        if current is None:
            logger.info("No migrations applied yet")
            return True
        
        if current != head:
            logger.info(f"Migration needed: current={current}, head={head}")
            return True
        
        logger.info("Database is up to date")
        return False
    
    def get_migration_status(self) -> Dict[str, Any]:
        """
        Get detailed migration status.
        
        Returns:
            Dictionary with migration status information
        """
        current = self.get_current_revision()
        head = self.get_head_revision()
        pending = self.get_pending_migrations()
        
        status = {
            "current_revision": current,
            "head_revision": head,
            "is_up_to_date": current == head,
            "pending_migrations": pending,
            "pending_count": len(pending),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return status
    
    def create_backup(self) -> Optional[str]:
        """
        Create database backup before migration.
        
        Returns:
            Backup file path or None if backup not supported/failed
        """
        # Note: This is a placeholder. Actual backup implementation
        # depends on the database type (PostgreSQL, MySQL, SQLite, etc.)
        logger.warning("Database backup not implemented. Proceeding without backup.")
        return None
    
    def upgrade_to_head(self, dry_run: bool = False) -> bool:
        """
        Upgrade database to latest revision.
        
        Args:
            dry_run: If True, only show what would be done
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if dry_run:
                logger.info("DRY RUN: Would upgrade database to head")
                pending = self.get_pending_migrations()
                if pending:
                    logger.info(f"Would apply {len(pending)} migrations: {pending}")
                return True
            
            # Check if migration needed
            if not self.is_migration_needed():
                logger.info("Database already at head revision")
                return True
            
            # Create backup if requested
            if self.backup_before_upgrade:
                backup_path = self.create_backup()
                if backup_path:
                    logger.info(f"Backup created: {backup_path}")
            
            # Run migration
            logger.info("Starting database migration to head...")
            command.upgrade(self.alembic_cfg, "head")
            logger.info("Database migration completed successfully")
            
            # Verify migration
            new_revision = self.get_current_revision()
            head_revision = self.get_head_revision()
            
            if new_revision == head_revision:
                logger.info(f"Migration verified: database at revision {new_revision}")
                return True
            else:
                logger.error(f"Migration verification failed: expected {head_revision}, got {new_revision}")
                return False
        
        except Exception as e:
            import traceback
            error_msg = f"Migration failed: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            # Re-raise the exception so the caller knows it failed
            raise MigrationError(error_msg) from e
    
    def downgrade(self, revision: str = "-1") -> bool:
        """
        Downgrade database to specified revision.
        
        Args:
            revision: Target revision ("-1" for previous, "-2" for two back, etc.)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downgrading database to revision: {revision}")
            command.downgrade(self.alembic_cfg, revision)
            logger.info("Database downgrade completed successfully")
            return True
        except Exception as e:
            logger.error(f"Downgrade failed: {e}", exc_info=True)
            return False
    
    def upgrade_with_retry(self) -> bool:
        """
        Upgrade database with retry logic.
        
        Returns:
            True if successful, False otherwise
        """
        for attempt in range(1, self.max_retry_attempts + 1):
            logger.info(f"Migration attempt {attempt}/{self.max_retry_attempts}")
            
            if self.upgrade_to_head():
                return True
            
            if attempt < self.max_retry_attempts:
                logger.warning(f"Migration failed, retrying in {self.retry_delay}s...")
                time.sleep(self.retry_delay)
        
        logger.error("Migration failed after all retry attempts")
        return False
    
    def auto_migrate(self, wait_for_db: bool = True, raise_exceptions: bool = False) -> bool:
        """
        Automatically run migrations on startup.
        
        Args:
            wait_for_db: Wait for database to be available
            raise_exceptions: Whether to raise exceptions instead of returning False
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("=== Starting automatic database migration ===")
            
            # Wait for database
            if wait_for_db:
                if not self.wait_for_database(timeout=30):
                    logger.error("Database not available, skipping migration")
                    return False
            
            # Check migration status
            status = self.get_migration_status()
            logger.info(f"Migration status: {status}")
            
            # Run migration if needed
            if not status["is_up_to_date"]:
                logger.info(f"Applying {status['pending_count']} pending migration(s)...")
                
                if self.auto_upgrade:
                    # We need to handle the retry logic here to respect raise_exceptions
                    success = False
                    last_error = None
                    
                    for attempt in range(1, self.max_retry_attempts + 1):
                        try:
                            if self.upgrade_to_head():
                                success = True
                                break
                        except Exception as e:
                            last_error = e
                            if attempt < self.max_retry_attempts:
                                logger.warning(f"Migration attempt {attempt} failed, retrying...")
                                time.sleep(self.retry_delay)
                    
                    if not success and last_error and raise_exceptions:
                        raise last_error
                    
                    if success:
                        logger.info("=== Automatic migration completed successfully ===")
                    else:
                        logger.error("=== Automatic migration failed ===")
                    
                    return success
                else:
                    logger.warning("Auto-upgrade disabled, migration skipped")
                    return True
            else:
                logger.info("=== Database already up to date ===")
                return True
        
        except Exception as e:
            logger.error(f"Auto-migration error: {e}", exc_info=True)
            if raise_exceptions:
                raise
            return False
    
    def create_revision(
        self,
        message: str,
        autogenerate: bool = True
    ) -> Optional[str]:
        """
        Create new migration revision.
        
        Args:
            message: Migration message/description
            autogenerate: Auto-detect model changes
        
        Returns:
            Revision ID or None if failed
        """
        try:
            logger.info(f"Creating new migration: {message}")
            
            if autogenerate:
                command.revision(
                    self.alembic_cfg,
                    message=message,
                    autogenerate=True
                )
            else:
                command.revision(
                    self.alembic_cfg,
                    message=message
                )
            
            logger.info("Migration revision created successfully")
            return self.get_head_revision()
        
        except Exception as e:
            logger.error(f"Failed to create revision: {e}", exc_info=True)
            return None
    
    def show_current_revision(self):
        """Display current revision information."""
        current = self.get_current_revision()
        head = self.get_head_revision()
        
        print("\n" + "=" * 60)
        print("Database Migration Status")
        print("=" * 60)
        print(f"Current Revision: {current or 'None (no migrations applied)'}")
        print(f"Head Revision:    {head}")
        print(f"Status:           {'UP TO DATE' if current == head else 'MIGRATION NEEDED'}")
        
        pending = self.get_pending_migrations()
        if pending:
            print(f"\nPending Migrations ({len(pending)}):")
            for rev in pending:
                print(f"  - {rev}")
        
        print("=" * 60 + "\n")
    
    def show_history(self, limit: int = 10):
        """
        Display migration history.
        
        Args:
            limit: Maximum number of revisions to show
        """
        try:
            command.history(self.alembic_cfg, rev_range=f"-{limit}:")
        except Exception as e:
            logger.error(f"Failed to show history: {e}")


def run_migrations_on_startup(
    auto_upgrade: bool = True,
    wait_for_db: bool = True,
    fail_on_error: bool = False
) -> bool:
    """
    Run migrations automatically on application startup.
    
    Args:
        auto_upgrade: Automatically upgrade to head
        wait_for_db: Wait for database to be available
        fail_on_error: Raise exception on migration failure
    
    Returns:
        True if successful, False otherwise
    
    Raises:
        MigrationError: If fail_on_error=True and migration fails
    """
    try:
        manager = MigrationManager(auto_upgrade=auto_upgrade)
        # Pass fail_on_error as raise_exceptions to get the real error
        success = manager.auto_migrate(wait_for_db=wait_for_db, raise_exceptions=fail_on_error)
        
        if not success and fail_on_error:
            raise MigrationError("Database migration failed (unknown error)")
        
        return success
    
    except Exception as e:
        logger.error(f"Migration startup error: {e}", exc_info=True)
        
        if fail_on_error:
            raise
        
        return False


def check_migration_status() -> Dict[str, Any]:
    """
    Check migration status (for health checks).
    
    Returns:
        Migration status dictionary
    """
    try:
        manager = MigrationManager(auto_upgrade=False)
        return manager.get_migration_status()
    except Exception as e:
        logger.error(f"Failed to check migration status: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


if __name__ == "__main__":
    """CLI interface for migration management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database Migration Manager")
    parser.add_argument(
        "command",
        choices=["status", "upgrade", "downgrade", "history", "auto"],
        help="Command to execute"
    )
    parser.add_argument(
        "--revision",
        default="head",
        help="Target revision (for upgrade/downgrade)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    
    args = parser.parse_args()
    
    manager = MigrationManager()
    
    if args.command == "status":
        manager.show_current_revision()
    
    elif args.command == "upgrade":
        if args.revision == "head":
            manager.upgrade_to_head(dry_run=args.dry_run)
        else:
            command.upgrade(manager.alembic_cfg, args.revision)
    
    elif args.command == "downgrade":
        manager.downgrade(args.revision)
    
    elif args.command == "history":
        manager.show_history()
    
    elif args.command == "auto":
        success = manager.auto_migrate()
        sys.exit(0 if success else 1)
