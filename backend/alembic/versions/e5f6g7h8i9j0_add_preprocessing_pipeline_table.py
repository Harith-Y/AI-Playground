"""add preprocessing pipeline table

Revision ID: e5f6g7h8i9j0
Revises: d4e5f6g7h8i9
Create Date: 2026-01-05 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'e5f6g7h8i9j0'
down_revision = 'd4e5f6g7h8i9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('preprocessing_pipelines',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('dataset_id', sa.UUID(), nullable=True),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('fitted', sa.Boolean(), nullable=False),
        sa.Column('pickle_path', sa.String(), nullable=True),
        sa.Column('num_steps', sa.Integer(), nullable=False),
        sa.Column('step_summary', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('statistics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('fitted_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_preprocessing_pipelines_dataset_id'), 'preprocessing_pipelines', ['dataset_id'], unique=False)
    op.create_index(op.f('ix_preprocessing_pipelines_id'), 'preprocessing_pipelines', ['id'], unique=False)
    op.create_index(op.f('ix_preprocessing_pipelines_user_id'), 'preprocessing_pipelines', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_preprocessing_pipelines_user_id'), table_name='preprocessing_pipelines')
    op.drop_index(op.f('ix_preprocessing_pipelines_id'), table_name='preprocessing_pipelines')
    op.drop_index(op.f('ix_preprocessing_pipelines_dataset_id'), table_name='preprocessing_pipelines')
    op.drop_table('preprocessing_pipelines')
