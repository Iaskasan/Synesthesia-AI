"""Database connection shared by the API, migrations and GPU worker."""
import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[2]
STORAGE = Path(os.getenv("SYNESTHESIA_STORAGE", str(ROOT / "outputs/web"))).resolve()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://synesthesia:synesthesia@localhost:5432/synesthesia")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
if engine.dialect.name == "sqlite":
    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

Session = sessionmaker(engine, expire_on_commit=False)


def get_db():
    with Session() as db:
        yield db


def storage_path(key):
    path = (STORAGE / key).resolve()
    if not path.is_relative_to(STORAGE):
        raise ValueError("Invalid storage key")
    return path
