"""Explicit relational records. JSON is reserved for versioned scientific settings."""
import time
import uuid

from sqlalchemy import Boolean, CheckConstraint, Column, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def identifier():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=identifier)
    email = Column(String(254), nullable=False, unique=True)
    password_hash = Column(Text, nullable=False)
    created_at = Column(Float, nullable=False, default=time.time)


class LoginSession(Base):
    __tablename__ = "sessions"
    token_hash = Column(String(64), primary_key=True)
    user_id = Column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    csrf = Column(String(64), nullable=False)
    expires_at = Column(Float, nullable=False, index=True)


class Project(Base):
    __tablename__ = "projects"
    id = Column(String(36), primary_key=True, default=identifier)
    owner_id = Column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(120), nullable=False)
    created_at = Column(Float, nullable=False, default=time.time)


class AudioAsset(Base):
    __tablename__ = "audio_assets"
    id = Column(String(36), primary_key=True, default=identifier)
    project_id = Column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    storage_key = Column(String(255), unique=True)
    sha256 = Column(String(64), nullable=False)
    size_bytes = Column(Integer, nullable=False)
    retain = Column(Boolean, nullable=False, default=False)
    __table_args__ = (CheckConstraint("size_bytes > 0"),)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(String(36), primary_key=True, default=identifier)
    kind = Column(String(20), nullable=False)
    name = Column(String(255), nullable=False)
    revision = Column(String(128), nullable=False)
    __table_args__ = (UniqueConstraint("kind", "name", "revision"),)


class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(String(36), primary_key=True, default=identifier)
    audio_id = Column(ForeignKey("audio_assets.id", ondelete="CASCADE"), nullable=False, index=True)
    model_id = Column(ForeignKey("model_versions.id", ondelete="RESTRICT"))
    start_seconds = Column(Float, nullable=False, default=0)
    duration_seconds = Column(Float, nullable=False, default=30)
    features = Column(JSON)
    status = Column(String(16), nullable=False, default="queued", index=True)
    error = Column(Text)
    lease_until = Column(Float)
    worker_token = Column(String(36))
    created_at = Column(Float, nullable=False, default=time.time)
    __table_args__ = (CheckConstraint("start_seconds >= 0 AND duration_seconds > 0 AND duration_seconds <= 30"), CheckConstraint("status IN ('queued', 'running', 'succeeded', 'failed')"))


class Mood(Base):
    __tablename__ = "moods"
    id = Column(Integer, primary_key=True)
    label = Column(String(40), nullable=False, unique=True)
    active = Column(Boolean, nullable=False, default=True)


class AnalysisMood(Base):
    __tablename__ = "analysis_moods"
    analysis_id = Column(ForeignKey("analyses.id", ondelete="CASCADE"), primary_key=True)
    mood_id = Column(ForeignKey("moods.id", ondelete="RESTRICT"), primary_key=True)
    confidence = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    detected = Column(Boolean, nullable=False)
    __table_args__ = (CheckConstraint("confidence >= 0 AND confidence <= 1"), CheckConstraint("threshold >= 0 AND threshold <= 1"))


class Generation(Base):
    __tablename__ = "generations"
    id = Column(String(36), primary_key=True, default=identifier)
    analysis_id = Column(ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True)
    model_id = Column(ForeignKey("model_versions.id", ondelete="RESTRICT"))
    prompt = Column(Text, nullable=False)
    settings = Column(JSON, nullable=False)
    seed = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    status = Column(String(16), nullable=False, default="queued", index=True)
    error = Column(Text)
    lease_until = Column(Float)
    worker_token = Column(String(36))
    created_at = Column(Float, nullable=False, default=time.time)
    __table_args__ = (CheckConstraint("seed >= 0 AND seed <= 2147483647"), CheckConstraint("width >= 256 AND width <= 1024 AND width % 8 = 0 AND height >= 256 AND height <= 1024 AND height % 8 = 0"), CheckConstraint("status IN ('queued', 'running', 'succeeded', 'failed')"))


class GenerationMood(Base):
    __tablename__ = "generation_moods"
    generation_id = Column(ForeignKey("generations.id", ondelete="CASCADE"), primary_key=True)
    mood_id = Column(ForeignKey("moods.id", ondelete="RESTRICT"), primary_key=True)
    position = Column(Integer, nullable=False)
    __table_args__ = (UniqueConstraint("generation_id", "position"),)


class GeneratedImage(Base):
    __tablename__ = "generated_images"
    id = Column(String(36), primary_key=True, default=identifier)
    generation_id = Column(ForeignKey("generations.id", ondelete="CASCADE"), nullable=False, unique=True)
    storage_key = Column(String(255), nullable=False, unique=True)
    sha256 = Column(String(64), nullable=False)


class FileDeletion(Base):
    """Transactional outbox: deleting a project schedules retryable file cleanup."""
    __tablename__ = "file_deletions"
    storage_key = Column(String(255), primary_key=True)
