"""Run with python -m src.web.worker on the GPU machine."""
import argparse
import hashlib
import logging
import os
import platform
import threading
import time
from functools import lru_cache
from importlib.metadata import version

from sqlalchemy import or_, select, update

from src.web.database import ROOT, Session, storage_path
from src.web.models import Analysis, AnalysisMood, AudioAsset, FileDeletion, GeneratedImage, Generation, ModelVersion, Mood, identifier

log = logging.getLogger(__name__)
LEASE_SECONDS = 180


def cleanup_files():
    with Session() as db:
        for item in db.scalars(select(FileDeletion)).all():
            try:
                storage_path(item.storage_key).unlink(missing_ok=True)
                db.delete(item)
            except OSError:
                log.exception("File cleanup will be retried")
        db.commit()


def claim(model):
    """Compare-and-swap claims also fence off workers whose leases expired."""
    now, token = time.time(), identifier()
    with Session() as db:
        eligible = or_(model.status == "queued", (model.status == "running") & (model.lease_until < now))
        job_id = db.scalar(select(model.id).where(eligible).order_by(model.created_at).limit(1))
        if job_id is None:
            return None
        result = db.execute(update(model).where(model.id == job_id, eligible).values(status="running", worker_token=token, lease_until=now + LEASE_SECONDS, error=None))
        db.commit()
        return (job_id, token) if result.rowcount == 1 else None


def heartbeat(model, job_id, token, stop):
    while not stop.wait(30):
        try:
            with Session() as db:
                db.execute(update(model).where(model.id == job_id, model.worker_token == token, model.status == "running").values(lease_until=time.time() + LEASE_SECONDS))
                db.commit()
        except Exception:
            log.exception("Could not renew job lease")


def model_version(db, kind, name, revision):
    item = db.scalar(select(ModelVersion).where(ModelVersion.kind == kind, ModelVersion.name == name, ModelVersion.revision == revision))
    if item is None:
        item = ModelVersion(kind=kind, name=name, revision=revision)
        db.add(item)
        db.flush()
    return item.id


@lru_cache(maxsize=1)
def classifier(path, checksum):
    from src.ml.inference import ClapMoodClassifier
    return ClapMoodClassifier(path)


def analyze(job_id):
    import librosa
    import numpy as np
    from src.audio.analysis import analyze_audio
    with Session() as db:
        job = db.get(Analysis, job_id)
        audio = db.get(AudioAsset, job.audio_id)
        if not audio.storage_key:
            raise ValueError("Source audio is no longer available")
        path, start, duration = storage_path(audio.storage_key), job.start_seconds, job.duration_seconds
    values, sr = librosa.load(path, sr=22050, mono=True, offset=start, duration=duration)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("The selected excerpt contains no valid audio")
    # Short tracks are supported: record the actual analyzed duration.
    actual_duration = len(values) / sr
    features = analyze_audio(values, sr).to_dict()
    checkpoint = ROOT / os.getenv("CLASSIFIER_PATH", "artifacts/clap_diagnostics/selected_head.joblib")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    predictions = classifier(str(checkpoint), digest).predict(values, sr)
    return features, predictions, digest, actual_duration


def render(job_id, token):
    from huggingface_hub import model_info
    from src.generation.generate_image import generate_image
    name = os.getenv("DIFFUSION_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    revision = model_info(name, revision=os.getenv("DIFFUSION_REVISION", "main")).sha
    if not revision:
        raise RuntimeError("Could not resolve an immutable diffusion revision")
    with Session() as db:
        job = db.get(Generation, job_id)
        prompt, seed, width, height = job.prompt, job.seed, job.width, job.height
    image = generate_image(prompt, name, seed=seed, width=width, height=height, revision=revision)
    key = f"images/{job_id}-{token}.png"
    path = storage_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        image.save(path, format="PNG")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        path.unlink(missing_ok=True)
        raise
    environment = {p: version(p) for p in ("torch", "diffusers", "transformers", "librosa")}
    environment["python"] = platform.python_version()
    return key, digest, name, revision, environment


def release_temporary_audio(db, analysis):
    audio = db.get(AudioAsset, analysis.audio_id)
    if audio and not audio.retain and audio.storage_key:
        db.merge(FileDeletion(storage_key=audio.storage_key))
        audio.storage_key = None


def process(model, claimed):
    job_id, token = claimed
    stop = threading.Event()
    thread = threading.Thread(target=heartbeat, args=(model, job_id, token, stop), daemon=True)
    thread.start()
    output_key = None
    committed = False
    try:
        result = analyze(job_id) if model is Analysis else render(job_id, token)
        if model is Generation:
            output_key = result[0]
        with Session() as db:
            job = db.scalar(select(model).where(model.id == job_id, model.worker_token == token, model.status == "running").with_for_update())
            if job is None:
                return
            if model is Analysis:
                features, predictions, checksum, duration = result
                job.features, job.duration_seconds = features, duration
                job.model_id = model_version(db, "classifier", "CLAP selected head", checksum)
                for prediction in predictions:
                    mood = db.scalar(select(Mood).where(Mood.label == prediction.label, Mood.active.is_(True)))
                    if mood:
                        db.add(AnalysisMood(analysis_id=job.id, mood_id=mood.id, confidence=prediction.confidence, threshold=prediction.threshold, detected=prediction.detected))
                release_temporary_audio(db, job)
            else:
                key, checksum, name, revision, environment = result
                job.model_id = model_version(db, "diffusion", name, revision)
                job.settings = {**job.settings, "environment": environment, "num_inference_steps": 50, "guidance_scale": 7.5}
                db.add(GeneratedImage(generation_id=job.id, storage_key=key, sha256=checksum))
            job.status, job.lease_until = "succeeded", None
            db.commit()
            committed = True
    except Exception:
        log.exception("Job %s failed", job_id)
        with Session() as db:
            job = db.scalar(select(model).where(model.id == job_id, model.worker_token == token, model.status == "running").with_for_update())
            if job:
                job.status = "failed"
                job.error = "Processing failed. Check the worker log, model files and GPU setup, then submit again."
                job.lease_until = None
                if model is Analysis:
                    release_temporary_audio(db, job)
                db.commit()
    finally:
        stop.set()
        thread.join(timeout=5)
        if output_key and not committed:
            storage_path(output_key).unlink(missing_ok=True)


def run_once():
    cleanup_files()
    for model in (Analysis, Generation):
        claimed = claim(model)
        if claimed:
            process(model, claimed)
            cleanup_files()
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Process at most one job, then exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.once:
        run_once()
        return
    log.info("GPU worker ready; polling database for jobs")
    while True:
        if not run_once():
            time.sleep(2)


if __name__ == "__main__":
    main()
