"""Database/API integration tests. AI compute is explicitly stubbed in tests only."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("httpx")
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def system(tmp_path, monkeypatch):
    from src.web import database, worker
    from src.web.app import app
    engine = create_engine(f"sqlite:///{tmp_path / 'test.sqlite'}", connect_args={"check_same_thread": False})
    @event.listens_for(engine, "connect")
    def foreign_keys(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")
    factory = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "Session", factory)
    monkeypatch.setattr(database, "STORAGE", tmp_path / "files")
    monkeypatch.setattr(worker, "Session", factory)
    config = Config(str(database.ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(database.ROOT / "migrations"))
    command.upgrade(config, "head")
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, db=factory, worker=worker, config=config, engine=engine, root=tmp_path)
    engine.dispose()


def register(client, email="owner@example.com"):
    response = client.post("/api/register", json={"email": email, "password": "a strong test password"})
    assert response.status_code == 201, response.text
    client.headers["X-CSRF-Token"] = response.json()["csrf"]
    return response


def project(client):
    response = client.post("/api/projects", json={"title": "Night drive"})
    assert response.status_code == 201
    return response.json()["id"]


def upload(client, project_id, retain=False):
    response = client.post(f"/api/projects/{project_id}/audio", files={"file": ("song.wav", b"test audio bytes", "audio/wav")}, data={"retain": str(retain).lower()})
    assert response.status_code == 202, response.text
    return response.json()["id"]


def fake_analysis(job_id):
    return {"tempo_bpm": 90.0, "rms_energy": 0.1, "spectral_centroid_hz": 1500}, [SimpleNamespace(label="calm", confidence=0.8, threshold=0.5, detected=True)], "a" * 64, 12.5


def test_sessions_csrf_and_project_ownership(system):
    client = system.client
    assert client.get("/api/projects").status_code == 401
    result = register(client)
    assert "httponly" in result.headers["set-cookie"].lower()
    pid = project(client)
    assert client.patch(f"/api/projects/{pid}", json={"title": "Updated"}).status_code == 200
    client.headers.pop("X-CSRF-Token")
    assert client.post("/api/projects", json={"title": "Blocked"}).status_code == 403
    register(client, "second@example.com")
    assert client.get("/api/projects").json() == []
    assert client.get(f"/api/projects/{pid}").status_code == 404
    assert client.delete(f"/api/projects/{pid}").status_code == 404
    assert client.post(f"/api/projects/{pid}/audio", files={"file": ("a.wav", b"a")}).status_code == 404
    assert client.post("/api/logout").status_code == 204
    assert client.get("/api/me").status_code == 401
    assert client.post("/api/login", json={"email": "owner@example.com", "password": "wrong password here"}).status_code == 401
    assert client.post("/api/login", json={"email": "OWNER@example.com", "password": "a strong test password"}).status_code == 200


def test_cross_origin_and_validation(system, monkeypatch):
    client = system.client
    assert client.post("/api/register", headers={"Origin": "https://evil.example"}, json={"email": "a@b", "password": "a strong test password"}).status_code == 403
    register(client)
    pid = project(client)
    assert client.post("/api/projects", json={"title": "  "}).status_code == 422
    assert client.post(f"/api/projects/{pid}/audio", files={"file": ("file.exe", b"x")}).status_code == 422
    assert client.post(f"/api/projects/{pid}/audio", files={"file": ("empty.wav", b"")}).status_code == 422
    assert client.post(f"/api/projects/{pid}/audio", files={"file": ("a.wav", b"abc")}, data={"start_seconds": "nan"}).status_code == 422
    import importlib
    app_module = importlib.import_module("src.web.app")
    monkeypatch.setattr(app_module, "MAX_UPLOAD", 2)
    assert client.post(f"/api/projects/{pid}/audio", files={"file": ("a.wav", b"abc")}).status_code == 413
    assert not list((system.root / "files/audio").glob("*"))


def test_full_persisted_journey_and_cascade_cleanup(system, monkeypatch):
    from src.web.database import storage_path
    from src.web.models import Analysis, AnalysisMood, AudioAsset, Generation, GenerationMood, GeneratedImage, Project
    client = system.client
    register(client)
    pid = project(client)
    aid = upload(client, pid)
    monkeypatch.setattr(system.worker, "analyze", fake_analysis)
    assert system.worker.run_once()
    detail = client.get(f"/api/projects/{pid}").json()
    analysis = detail["analyses"][0]
    assert analysis["status"] == "succeeded"
    assert analysis["duration_seconds"] == 12.5
    assert not analysis["audio_available"]
    assert analysis["predictions"][0]["label"] == "calm"
    assert not list((system.root / "files/audio").glob("*"))
    creative = {"moods": ["calm"], "style": "watercolor"}
    prompt = client.post(f"/api/analyses/{aid}/prompt", json=creative)
    assert prompt.status_code == 200
    assert "watercolor" in prompt.json()["prompt"]
    assert client.post(f"/api/analyses/{aid}/prompt", json={"moods": ["inspiring"]}).status_code == 422
    response = client.post(f"/api/analyses/{aid}/generations", json={**creative, "prompt": prompt.json()["prompt"], "seed": 123})
    assert response.status_code == 202
    gid = response.json()["id"]
    def fake_render(job_id, token):
        key = f"images/{job_id}-{token}.png"
        path = storage_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"explicit test fixture image")
        return key, hashlib.sha256(path.read_bytes()).hexdigest(), "test-model", "test-revision", {"test": True}
    monkeypatch.setattr(system.worker, "render", fake_render)
    assert system.worker.run_once()
    assert client.get(f"/api/generations/{gid}/image").status_code == 200
    metadata = client.get(f"/api/generations/{gid}/metadata").json()
    assert metadata["selected_moods"] == ["calm"]
    assert metadata["generation"]["seed"] == 123
    assert metadata["diffusion"]["revision"] == "test-revision"
    with system.db() as db:
        assert db.scalar(select(AnalysisMood)).confidence == 0.8
        assert db.scalar(select(GenerationMood)).position == 0
    register(client, "stranger@example.com")
    assert client.post(f"/api/analyses/{aid}/prompt", json=creative).status_code == 404
    assert client.get(f"/api/generations/{gid}/metadata").status_code == 404
    assert client.get(f"/api/generations/{gid}/image").status_code == 404
    result = client.post("/api/login", json={"email": "owner@example.com", "password": "a strong test password"})
    client.headers["X-CSRF-Token"] = result.json()["csrf"]
    assert client.delete(f"/api/projects/{pid}").status_code == 204
    system.worker.cleanup_files()
    with system.db() as db:
        for model in (Project, AudioAsset, Analysis, AnalysisMood, Generation, GenerationMood, GeneratedImage):
            assert db.scalar(select(model)) is None
    assert not list((system.root / "files/images").glob("*"))


def test_worker_failure_and_lease_recovery(system, monkeypatch):
    from src.web.models import Analysis
    register(system.client)
    aid = upload(system.client, project(system.client))
    first = system.worker.claim(Analysis)
    assert system.worker.claim(Analysis) is None
    with system.db() as db:
        db.get(Analysis, aid).lease_until = 0
        db.commit()
    second = system.worker.claim(Analysis)
    assert first[1] != second[1]
    monkeypatch.setattr(system.worker, "analyze", fake_analysis)
    system.worker.process(Analysis, first)
    with system.db() as db:
        assert db.get(Analysis, aid).status == "running"
    def failure(_):
        raise FileNotFoundError("secret internal path must not be exposed")
    monkeypatch.setattr(system.worker, "analyze", failure)
    system.worker.process(Analysis, second)
    system.worker.cleanup_files()
    with system.db() as db:
        job = db.get(Analysis, aid)
        assert job.status == "failed"
        assert "secret" not in job.error
    assert not list((system.root / "files/audio").glob("*"))


def test_relational_constraints_and_migration_round_trip(system):
    from src.web.models import AnalysisMood, Mood, Project
    with system.db() as db:
        assert len(db.scalars(select(Mood)).all()) == 20
        assert not db.scalar(select(Mood).where(Mood.label == "inspiring")).active
        db.add(Project(owner_id="nonexistent-user", title="Cannot be orphaned"))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()
        db.add(AnalysisMood(analysis_id="missing", mood_id=1, confidence=1.5, threshold=0.5, detected=True))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()
    command.check(system.config)
    command.downgrade(system.config, "base")
    command.upgrade(system.config, "head")
    with system.db() as db:
        assert len(db.scalars(select(Mood)).all()) == 20


def test_browser_shell_and_assets(system):
    response = system.client.get("/")
    assert response.status_code == 200
    assert "Your music." in response.text
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    for asset in ("app.js", "style.css"):
        assert system.client.get(f"/static/{asset}").status_code == 200
