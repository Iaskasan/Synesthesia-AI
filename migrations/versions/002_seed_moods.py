"""Seed the documented vocabulary, preserving the inactive inspiring label."""
from alembic import op
import sqlalchemy as sa

revision = "002_seed_moods"
down_revision = "6dd30323d3bb"
branch_labels = None
depends_on = None

LABELS = "happy energetic relaxing emotional dark epic dream inspiring sad meditative uplifting motivational romantic fun calm adventure melancholic dramatic powerful hopeful".split()


def upgrade():
    table = sa.table("moods", sa.column("id", sa.Integer), sa.column("label", sa.String), sa.column("active", sa.Boolean))
    op.bulk_insert(table, [{"id": i, "label": label, "active": label != "inspiring"} for i, label in enumerate(LABELS, 1)])


def downgrade():
    # Foreign keys intentionally prevent removing vocabulary used by saved results.
    table = sa.table("moods", sa.column("label", sa.String))
    op.execute(table.delete().where(table.c.label.in_(LABELS)))
