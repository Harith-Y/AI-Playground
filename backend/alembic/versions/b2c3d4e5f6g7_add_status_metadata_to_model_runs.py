"""Add status and run_metadata columns to model_runs

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
Create Date: 2025-12-29 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6g7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Add status and run_metadata columns to model_runs."""
    # Add status column with default value 'pending'
    op.add_column('model_runs', sa.Column('status', sa.String(), nullable=False, server_default='pending'))

    # Add run_metadata column for storing task_id, feature_importance, errors, etc.
    op.add_column('model_runs', sa.Column('run_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True))

    # Create index on status for faster filtering
    op.create_index(op.f('ix_model_runs_status'), 'model_runs', ['status'], unique=False)


def downgrade() -> None:
    """Downgrade schema - Remove status and run_metadata columns from model_runs."""
    # Drop index first
    op.drop_index(op.f('ix_model_runs_status'), table_name='model_runs')

    # Drop columns
    op.drop_column('model_runs', 'run_metadata')
    op.drop_column('model_runs', 'status')
