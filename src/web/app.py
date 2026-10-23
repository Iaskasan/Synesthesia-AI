"""HTTP API and browser shell. GPU dependencies are loaded only by the worker."""
import hashlib
import json
import math
import os
import secrets
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DBSession

from src.generation.prompt_builder import build_prompt
from src.web.database import get_db, storage_path
from src.web.models import Analysis, AnalysisMood, AudioAsset, FileDeletion, GeneratedImage, Generation, GenerationMood, LoginSession, ModelVersion, Mood, Project, User, identifier
from src.web.security import password_hash, password_matches, token_hash

app = FastAPI(title="Synesthesia-AI", version="0.1.0")
STATIC = Path(__file__).parent / "static"
COOKIE = "synesthesia_session"
MAX_UPLOAD = 25 * 1024 * 1024
DUMMY_PASSWORD = password_hash("unused timing equalizer")


@app.middleware("http")
async def browser_security(request, call_next):
    # Reject cross-origin mutations, including login CSRF. No permissive CORS.
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("origin")
        expected = os.getenv("APP_ORIGIN", str(request.base_url).rstrip("/"))
        if origin and origin != expected:
            return Response("Cross-origin request rejected", status_code=403)
        if request.headers.get("sec-fetch-site") == "cross-site":
            return Response("Cross-site request rejected", status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' blob:; media-src 'self' blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    if request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-store"
    return response


def authenticated(request: Request, db: DBSession = Depends(get_db)):
    session = db.get(LoginSession, token_hash(request.cookies.get(COOKIE, "")))
    if session is None or session.expires_at <= time.time():
        raise HTTPException(401, "Please sign in.")
    if request.method not in {"GET", "HEAD"} and not secrets.compare_digest(request.headers.get("x-csrf-token", ""), session.csrf):
        raise HTTPException(403, "Session verification failed. Refresh and try again.")
    return session


def owned_project(db, project_id, user_id):
    project = db.scalar(select(Project).where(Project.id == project_id, Project.owner_id == user_id))
    if project is None:
        raise HTTPException(404, "Project not found.")
    return project


def owned_analysis(db, analysis_id, user_id):
    analysis = db.scalar(select(Analysis).join(AudioAsset).join(Project).where(Analysis.id == analysis_id, Project.owner_id == user_id))
    if analysis is None:
        raise HTTPException(404, "Analysis not found.")
    return analysis


def owned_generation(db, generation_id, user_id):
    generation = db.scalar(select(Generation).join(Analysis).join(AudioAsset).join(Project).where(Generation.id == generation_id, Project.owner_id == user_id))
    if generation is None:
        raise HTTPException(404, "Generation not found.")
    return generation


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=12, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value):
        value = value.strip().lower()
        if value.count("@") != 1 or any(c.isspace() for c in value) or not all(value.split("@")):
            raise ValueError("Enter a valid email address")
        return value


class ProjectInput(BaseModel):
    title: str = Field(min_length=1, max_length=120)

    @field_validator("title")
    @classmethod
    def trim_title(cls, value):
        if not value.strip():
            raise ValueError("Enter a project title")
        return value.strip()


class CreativeInput(BaseModel):
    moods: list[str] = Field(min_length=1, max_length=5)
    style: str = Field(default="digital art", min_length=1, max_length=100)
    subject: str = Field(default="an abstract landscape", min_length=1, max_length=300)
    abstraction: float = Field(default=0.7, ge=0, le=1)
    music_influence: float = Field(default=0.8, ge=0, le=1)


class GenerationInput(CreativeInput):
    prompt: str = Field(min_length=1, max_length=3000)
    seed: int = Field(default=42, ge=0, le=2147483647)
    width: int = Field(default=512, ge=256, le=1024, multiple_of=8)
    height: int = Field(default=512, ge=256, le=1024, multiple_of=8)

    @field_validator("prompt")
    @classmethod
    def nonempty_prompt(cls, value):
        if not value.strip():
            raise ValueError("Prompt cannot be blank")
        return value.strip()


def start_session(db, user, request, response):
    old = request.cookies.get(COOKIE)
    if old:
        db.execute(delete(LoginSession).where(LoginSession.token_hash == token_hash(old)))
    db.execute(delete(LoginSession).where(LoginSession.expires_at < time.time()))
    token, csrf = secrets.token_urlsafe(32), secrets.token_hex(32)
    db.add(LoginSession(token_hash=token_hash(token), user_id=user.id, csrf=csrf, expires_at=time.time() + 86400))
    db.commit()
    response.set_cookie(COOKIE, token, httponly=True, samesite="strict", secure=os.getenv("COOKIE_SECURE", "0") == "1", max_age=86400)
    return {"email": user.email, "csrf": csrf}


@app.post("/api/register", status_code=201)
def register(data: Credentials, request: Request, response: Response, db: DBSession = Depends(get_db)):
    user = User(email=data.email, password_hash=password_hash(data.password))
    db.add(user)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "This email is already registered.")
    return start_session(db, user, request, response)


@app.post("/api/login")
def login(data: Credentials, request: Request, response: Response, db: DBSession = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == data.email))
    valid = password_matches(data.password, user.password_hash if user else DUMMY_PASSWORD)
    if not user or not valid:
        raise HTTPException(401, "Incorrect email or password.")
    return start_session(db, user, request, response)


@app.get("/api/me")
def me(session=Depends(authenticated), db: DBSession = Depends(get_db)):
    return {"email": db.get(User, session.user_id).email, "csrf": session.csrf}


@app.post("/api/logout", status_code=204)
def logout(response: Response, session=Depends(authenticated), db: DBSession = Depends(get_db)):
    db.delete(session)
    db.commit()
    response.delete_cookie(COOKIE)


@app.get("/api/moods")
def moods(session=Depends(authenticated), db: DBSession = Depends(get_db)):
    return list(db.scalars(select(Mood.label).where(Mood.active.is_(True)).order_by(Mood.label)))


def project_json(project):
    return {"id": project.id, "title": project.title, "created_at": project.created_at}


@app.get("/api/projects")
def projects(session=Depends(authenticated), db: DBSession = Depends(get_db)):
    return [project_json(p) for p in db.scalars(select(Project).where(Project.owner_id == session.user_id).order_by(Project.created_at.desc()))]


@app.post("/api/projects", status_code=201)
def create_project(data: ProjectInput, session=Depends(authenticated), db: DBSession = Depends(get_db)):
    project = Project(owner_id=session.user_id, title=data.title)
    db.add(project)
    db.commit()
    return project_json(project)


@app.patch("/api/projects/{project_id}")
def rename_project(project_id: str, data: ProjectInput, session=Depends(authenticated), db: DBSession = Depends(get_db)):
    project = owned_project(db, project_id, session.user_id)
    project.title = data.title
    db.commit()
    return project_json(project)


@app.delete("/api/projects/{project_id}", status_code=204)
def delete_project(project_id: str, session=Depends(authenticated), db: DBSession = Depends(get_db)):
    project = owned_project(db, project_id, session.user_id)
    keys = list(db.scalars(select(AudioAsset.storage_key).where(AudioAsset.project_id == project_id, AudioAsset.storage_key.is_not(None))))
    keys += list(db.scalars(select(GeneratedImage.storage_key).join(Generation).join(Analysis).join(AudioAsset).where(AudioAsset.project_id == project_id)))
    for key in keys:
        db.merge(FileDeletion(storage_key=key))
    db.delete(project)
    db.commit()


def analysis_json(db, item):
    audio = db.get(AudioAsset, item.audio_id)
    predictions = [{"label": mood.label, "confidence": prediction.confidence, "threshold": prediction.threshold, "detected": prediction.detected} for prediction, mood in db.execute(select(AnalysisMood, Mood).join(Mood).where(AnalysisMood.analysis_id == item.id).order_by(AnalysisMood.confidence.desc()))]
    return {"id": item.id, "filename": audio.filename, "status": item.status, "error": item.error, "features": item.features, "predictions": predictions, "start_seconds": item.start_seconds, "duration_seconds": item.duration_seconds, "audio_available": audio.storage_key is not None}


def generation_json(db, item):
    image = db.scalar(select(GeneratedImage).where(GeneratedImage.generation_id == item.id))
    return {"id": item.id, "analysis_id": item.analysis_id, "status": item.status, "error": item.error, "prompt": item.prompt, "seed": item.seed, "width": item.width, "height": item.height, "created_at": item.created_at, "image_url": f"/api/generations/{item.id}/image" if image else None}


@app.get("/api/projects/{project_id}")
def project_detail(project_id: str, session=Depends(authenticated), db: DBSession = Depends(get_db)):
    project = owned_project(db, project_id, session.user_id)
    analyses = db.scalars(select(Analysis).join(AudioAsset).where(AudioAsset.project_id == project_id).order_by(Analysis.created_at.desc())).all()
    generations = db.scalars(select(Generation).join(Analysis).join(AudioAsset).where(AudioAsset.project_id == project_id).order_by(Generation.created_at.desc())).all()
    return {**project_json(project), "analyses": [analysis_json(db, a) for a in analyses], "generations": [generation_json(db, g) for g in generations]}


@app.post("/api/projects/{project_id}/audio", status_code=202)
async def upload(project_id: str, file: UploadFile = File(...), start_seconds: float = Form(0), duration_seconds: float = Form(30), retain: bool = Form(False), session=Depends(authenticated), db: DBSession = Depends(get_db)):
    owned_project(db, project_id, session.user_id)
    if not math.isfinite(start_seconds) or not math.isfinite(duration_seconds) or not 0 <= start_seconds <= 86400 or not 0 < duration_seconds <= 30:
        raise HTTPException(422, "Choose a valid excerpt of up to 30 seconds.")
    filename = Path((file.filename or "audio").replace("\\", "/")).name[:255]
    suffix = Path(filename).suffix.lower()
    if suffix not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
        raise HTTPException(422, "Upload a WAV, MP3, FLAC, OGG or M4A file.")
    key = f"audio/{identifier()}{suffix}"
    path = storage_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest, size = hashlib.sha256(), 0
    try:
        with path.open("xb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "Audio files must be at most 25 MB.")
                digest.update(chunk)
                output.write(chunk)
        if not size:
            raise HTTPException(422, "The uploaded file is empty.")
        audio = AudioAsset(project_id=project_id, filename=filename, storage_key=key, sha256=digest.hexdigest(), size_bytes=size, retain=retain)
        db.add(audio)
        db.flush()
        analysis = Analysis(audio_id=audio.id, start_seconds=start_seconds, duration_seconds=duration_seconds)
        db.add(analysis)
        db.commit()
    except Exception:
        db.rollback()
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return {"id": analysis.id, "status": analysis.status}


def creative_moods(db, data):
    names = list(dict.fromkeys(data.moods))
    moods = db.scalars(select(Mood).where(Mood.label.in_(names), Mood.active.is_(True))).all()
    by_label = {m.label: m for m in moods}
    if len(by_label) != len(names):
        raise HTTPException(422, "Select active mood labels.")
    return [by_label[name] for name in names]


@app.post("/api/analyses/{analysis_id}/prompt")
def compose(analysis_id: str, data: CreativeInput, session=Depends(authenticated), db: DBSession = Depends(get_db)):
    analysis = owned_analysis(db, analysis_id, session.user_id)
    if analysis.status != "succeeded":
        raise HTTPException(409, "Wait for a successful analysis.")
    selected = creative_moods(db, data)
    return {"prompt": build_prompt([m.label for m in selected], tempo_bpm=analysis.features["tempo_bpm"], style=data.style, subject=data.subject, abstraction=data.abstraction, music_influence=data.music_influence)}


@app.post("/api/analyses/{analysis_id}/generations", status_code=202)
def generate(analysis_id: str, data: GenerationInput, session=Depends(authenticated), db: DBSession = Depends(get_db)):
    analysis = owned_analysis(db, analysis_id, session.user_id)
    if analysis.status != "succeeded":
        raise HTTPException(409, "Wait for a successful analysis.")
    selected = creative_moods(db, data)
    item = Generation(analysis_id=analysis_id, prompt=data.prompt, seed=data.seed, width=data.width, height=data.height, settings=data.model_dump(include={"style", "subject", "abstraction", "music_influence"}))
    db.add(item)
    db.flush()
    for position, mood in enumerate(selected):
        db.add(GenerationMood(generation_id=item.id, mood_id=mood.id, position=position))
    db.commit()
    return generation_json(db, item)


@app.get("/api/generations/{generation_id}/image")
def download_image(generation_id: str, session=Depends(authenticated), db: DBSession = Depends(get_db)):
    owned_generation(db, generation_id, session.user_id)
    image = db.scalar(select(GeneratedImage).where(GeneratedImage.generation_id == generation_id))
    if image is None or not storage_path(image.storage_key).is_file():
        raise HTTPException(404, "Image not available.")
    return FileResponse(storage_path(image.storage_key), media_type="image/png", filename="synesthesia.png", content_disposition_type="inline")


@app.get("/api/generations/{generation_id}/metadata")
def metadata(generation_id: str, session=Depends(authenticated), db: DBSession = Depends(get_db)):
    item = owned_generation(db, generation_id, session.user_id)
    analysis = db.get(Analysis, item.analysis_id)
    audio = db.get(AudioAsset, analysis.audio_id)
    def model(model_id):
        value = db.get(ModelVersion, model_id) if model_id else None
        return {"name": value.name, "revision": value.revision} if value else None
    result = {"schema_version": 1, "audio_sha256": audio.sha256, "analysis": analysis_json(db, analysis), "generation": generation_json(db, item), "visual_settings": item.settings, "selected_moods": list(db.scalars(select(Mood.label).join(GenerationMood).where(GenerationMood.generation_id == item.id).order_by(GenerationMood.position))), "classifier": model(analysis.model_id), "diffusion": model(item.model_id), "preprocessing": "librosa-mono-22050-v1", "reproducibility_note": "Identical settings do not guarantee pixel-identical output across GPU and library versions."}
    return Response(json.dumps(result, indent=2), media_type="application/json", headers={"Content-Disposition": 'attachment; filename="synesthesia-metadata.json"'})


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
