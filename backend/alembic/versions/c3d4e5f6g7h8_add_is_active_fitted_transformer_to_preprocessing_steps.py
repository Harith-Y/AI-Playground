"""Add is_active and fitted_transformer to preprocessing_steps

Revision ID: c3d4e5f6g7h8
Revises: b2c3d4e5f6g7
Create Date: 2025-12-29 00:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6g7h8'
down_revision: Union[str, Sequence[str], None] = 'b2c3d4e5f6g7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Add is_active and fitted_transformer columns to preprocessing_steps."""
    # Add is_active column with default value True
    op.add_column('preprocessing_steps', sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'))

    # Add fitted_transformer column for storing serialized transformer objects
    op.add_column('preprocessing_steps', sa.Column('fitted_transformer', sa.LargeBinary(), nullable=True))

    # Create index on is_active for faster filtering
    op.create_index(op.f('ix_preprocessing_steps_is_active'), 'preprocessing_steps', ['is_active'], unique=False)


def downgrade() -> None:
    """Downgrade schema - Remove is_active and fitted_transformer columns from preprocessing_steps."""
    # Drop index first
    op.drop_index(op.f('ix_preprocessing_steps_is_active'), table_name='preprocessing_steps')

    # Drop columns
    op.drop_column('preprocessing_steps', 'fitted_transformer')
    op.drop_column('preprocessing_steps', 'is_active')
