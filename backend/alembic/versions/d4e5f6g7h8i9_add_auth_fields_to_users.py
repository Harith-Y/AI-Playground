"""add auth fields to users

Revision ID: d4e5f6g7h8i9
Revises: c3d4e5f6g7h8
Create Date: 2026-01-01 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'd4e5f6g7h8i9'
down_revision = 'c3d4e5f6g7h8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Add authentication and authorization fields to users table.

    New fields:
    - password_hash: Hashed password for authentication
    - is_active: Whether the user account is active
    - is_admin: Whether the user has admin privileges
    - api_key: Optional API key for programmatic access
    - updated_at: Timestamp of last update
    - last_login: Timestamp of last login
    """
    # Add password_hash column (not nullable, requires default for existing rows)
    # Note: Existing users will get a placeholder hash - they'll need to reset password
    op.add_column('users', sa.Column(
        'password_hash',
        sa.String(),
        nullable=False,
        server_default='$2b$12$placeholder_hash_requires_password_reset'
    ))

    # Add boolean columns for access control
    op.add_column('users', sa.Column(
        'is_active',
        sa.Boolean(),
        nullable=False,
        server_default=sa.text('true')
    ))

    op.add_column('users', sa.Column(
        'is_admin',
        sa.Boolean(),
        nullable=False,
        server_default=sa.text('false')
    ))

    # Add API key column (nullable, for optional programmatic access)
    op.add_column('users', sa.Column(
        'api_key',
        sa.String(),
        nullable=True
    ))

    # Add timestamp columns
    op.add_column('users', sa.Column(
        'updated_at',
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text('now()')
    ))

    op.add_column('users', sa.Column(
        'last_login',
        sa.DateTime(timezone=True),
        nullable=True
    ))

    # Create indexes for performance
    op.create_index('ix_users_api_key', 'users', ['api_key'], unique=True)
    op.create_index('ix_users_is_active', 'users', ['is_active'])
    op.create_index('ix_users_is_admin', 'users', ['is_admin'])


def downgrade() -> None:
    """Remove authentication fields from users table."""
    # Drop indexes first
    op.drop_index('ix_users_is_admin', table_name='users')
    op.drop_index('ix_users_is_active', table_name='users')
    op.drop_index('ix_users_api_key', table_name='users')

    # Drop columns
    op.drop_column('users', 'last_login')
    op.drop_column('users', 'updated_at')
    op.drop_column('users', 'api_key')
    op.drop_column('users', 'is_admin')
    op.drop_column('users', 'is_active')
    op.drop_column('users', 'password_hash')
