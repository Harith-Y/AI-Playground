"""Add preprocessing history table

Revision ID: a1b2c3d4e5f6
Revises: 5c1b6116de51
Create Date: 2025-12-28 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '5c1b6116de51'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Add preprocessing_history table."""
    op.create_table('preprocessing_history',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('dataset_id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('operation_type', sa.String(), nullable=False),
        sa.Column('steps_applied', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('original_shape', postgresql.ARRAY(sa.Integer()), nullable=False),
        sa.Column('transformed_shape', postgresql.ARRAY(sa.Integer()), nullable=False),
        sa.Column('execution_time', sa.Float(), nullable=True),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('error_message', sa.String(), nullable=True),
        sa.Column('output_dataset_id', sa.UUID(), nullable=True),
        sa.Column('statistics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['output_dataset_id'], ['datasets.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_preprocessing_history_id'), 'preprocessing_history', ['id'], unique=False)
    op.create_index(op.f('ix_preprocessing_history_dataset_id'), 'preprocessing_history', ['dataset_id'], unique=False)
    op.create_index(op.f('ix_preprocessing_history_user_id'), 'preprocessing_history', ['user_id'], unique=False)
    op.create_index(op.f('ix_preprocessing_history_created_at'), 'preprocessing_history', ['created_at'], unique=False)


def downgrade() -> None:
    """Downgrade schema - Remove preprocessing_history table."""
    op.drop_index(op.f('ix_preprocessing_history_created_at'), table_name='preprocessing_history')
    op.drop_index(op.f('ix_preprocessing_history_user_id'), table_name='preprocessing_history')
    op.drop_index(op.f('ix_preprocessing_history_dataset_id'), table_name='preprocessing_history')
    op.drop_index(op.f('ix_preprocessing_history_id'), table_name='preprocessing_history')
    op.drop_table('preprocessing_history')
