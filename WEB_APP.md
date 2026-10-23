# Synesthesia-AI web application

The CDA application adds a browser interface and relational persistence around
the existing Python AI pipeline. It runs locally with PostgreSQL and a separate
GPU worker. It is not deployed to public hosting.

## Implemented scope

- Registration, sign-in/out, scrypt password hashes, expiring server-side sessions,
  HttpOnly cookies, CSRF checks, and project ownership checks.
- Create, list, rename, and delete private projects.
- Upload audio (25 MB maximum), choose an excerpt of up to 30 seconds, and decide
  explicitly whether to retain source audio after processing.
- Queue real audio-feature extraction and CLAP classification in a Python worker.
- Display mood scores; select moods and visual controls; compose and edit a prompt.
- Queue Stable Diffusion generation with a seed and canvas dimensions.
- Save generated images; download images and JSON metadata containing analysis,
  selected moods, settings, classifier checksum, diffusion revision, and runtime versions.
- Retry interrupted jobs after their worker lease expires. Deleted projects enqueue
  file removal; the worker retries filesystem cleanup independently of database deletion.

The frontend uses HTML, CSS and JavaScript, served by FastAPI. There is no Node
build step. The API does not import PyTorch; only the GPU worker needs AI packages.
The interface polls database-backed job states every four seconds while work is pending.

## Run locally with PostgreSQL

Run commands from the `Synesthesia-AI` directory, not its parent repository.
Use Python 3.10 or later. Docker Compose is used only for PostgreSQL; the API and
worker run in your Python environment so the worker can use your existing CUDA setup.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-web.txt
docker compose up -d db

export DATABASE_URL='postgresql+psycopg://synesthesia:synesthesia@localhost:5432/synesthesia'
alembic upgrade head
uvicorn src.web.app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** and create an account. Interactive API documentation
is at **http://127.0.0.1:8000/docs**. Mutating authenticated API requests also need
the `X-CSRF-Token` returned by login or `/api/me`.

In a second terminal, activate the same environment (or your existing CUDA-enabled
AI environment), install both sets of dependencies, and start the worker:

```bash
source .venv/bin/activate
pip install -r requirements.txt -r requirements-web.txt
export DATABASE_URL='postgresql+psycopg://synesthesia:synesthesia@localhost:5432/synesthesia'
python -m src.web.worker
```

Preserve the CUDA-compatible PyTorch installation already used by the project.
The classifier artifact must exist at
`artifacts/clap_diagnostics/selected_head.joblib`. Set `CLASSIFIER_PATH` to use
another trusted artifact. Joblib artifacts execute Python during loading: use
your own trained artifacts, not arbitrary user uploads.

The first AI run requires the pretrained models to be available or downloaded.
Generation resolves the diffusion model to an immutable Hugging Face revision
before loading it. Network access is currently required for that revision lookup.
The worker reports detailed failures in its terminal and a safe error message in
the interface. A stopped worker leaves requests visibly queued.

For a shorter source track, the worker analyzes the available excerpt and stores
its actual duration. An empty or undecodable excerpt fails. Temporary audio is
removed after success or failure; to retry a failed analysis, upload it again.

## Configuration and storage

See `.env.web.example` for variables. It is a reference file, not automatically
loaded configuration; export the variables in each terminal.

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | Shared database for API, migrations and worker. Defaults to the local Compose database. |
| `SYNESTHESIA_STORAGE` | Shared private storage directory. Defaults to `outputs/web` relative to the project. |
| `CLASSIFIER_PATH` | Classifier artifact, relative to the project or absolute. |
| `DIFFUSION_MODEL` | Hugging Face model ID; defaults to the existing Stable Diffusion 1.5 model. |
| `DIFFUSION_REVISION` | Branch or immutable model revision; defaults to `main`, resolved and recorded per run. |
| `APP_ORIGIN` | Optional explicit browser origin, e.g. `http://127.0.0.1:8000`. Must match the browser URL. |
| `COOKIE_SECURE` | Set to `1` when serving over HTTPS; defaults to `0` for local HTTP. |

Both processes must share storage and database configuration. Keep the worker
running for cleanup even when no AI jobs are queued. Back up the database and
private file directory together. Model files are retained independently.

The Compose credentials and loopback binding are for local development. Public
deployment requires deployment-specific credentials, HTTPS, login throttling,
account recovery, request-size limits at the proxy, and storage/resource quotas.
These are not implemented as a public-service deployment in this version.

## Relational database evidence

The executable schema is versioned in `migrations/versions/`. No tables are created
implicitly at API startup. `alembic upgrade head` applies the schema and seeds the
20-label vocabulary, keeping `inspiring` inactive for new output.

The principal relationships are:

```mermaid
erDiagram
    users ||--o{ sessions : authenticates
    users ||--o{ projects : owns
    projects ||--o{ audio_assets : contains
    audio_assets ||--o{ analyses : analyzed_as
    model_versions o|--o{ analyses : classifies
    analyses ||--o{ analysis_moods : predicts
    moods ||--o{ analysis_moods : labels
    analyses ||--o{ generations : inspires
    generations ||--o{ generation_moods : selects
    moods ||--o{ generation_moods : labels
    model_versions o|--o{ generations : renders
    generations ||--o| generated_images : produces
```

There are 12 application tables, including the independent `file_deletions`
cleanup outbox. The two mood junction tables use composite primary keys.
Foreign keys prevent orphaned records, model references restrict deletion,
and check constraints enforce score ranges, image dimensions, seeds and job states.
Original predictions remain separate from user-selected moods.

An example SQL query for the assessment joins the user's project history to
saved analyses and images:

```sql
SELECT p.title, a.filename, an.status AS analysis_status,
       g.status AS generation_status, g.seed, gi.sha256 AS image_checksum
FROM projects p
JOIN audio_assets a ON a.project_id = p.id
JOIN analyses an ON an.audio_id = a.id
LEFT JOIN generations g ON g.analysis_id = an.id
LEFT JOIN generated_images gi ON gi.generation_id = g.id
WHERE p.owner_id = :user_id
ORDER BY p.created_at DESC, g.created_at DESC;
```

Use bound parameters, as the API does through SQLAlchemy. Scientific vectors and
supplemental visual settings use JSON columns; identity, ownership, moods, analyses,
generations and file relationships are explicit relational records.

## Verification

```bash
pip install -r requirements-web-dev.txt
python -m pytest tests/test_web_app.py tests/test_prompt_builder.py
alembic check
```

Web integration tests migrate isolated SQLite databases with foreign keys enabled.
They exercise sessions, CSRF, isolation between users, upload rejection, persistence,
mood associations, metadata exports, cascade deletion and file cleanup, failed jobs,
lease recovery, and migration upgrade/downgrade. AI compute is stubbed only in tests;
the actual application worker calls the original AI pipeline.

For a database-only smoke test without PostgreSQL installed:

```bash
export DATABASE_URL=sqlite:///./outputs/web-smoke.sqlite
mkdir -p outputs
alembic upgrade head
uvicorn src.web.app:app --host 127.0.0.1 --port 8000
```

Use PostgreSQL for the CDA database demonstration. SQLite validation does not
establish PostgreSQL concurrency behavior. Run a real audio-to-image acceptance
test on the RTX 4090 with the trained artifact before presenting the app as fully
validated. Pixel-identical output across different hardware/library versions is
not guaranteed even with saved seeds and revisions.

## Code map

- `src/web/app.py`: HTTP API, authentication, authorization, uploads and exports.
- `src/web/models.py`: SQLAlchemy entity definitions and relational constraints.
- `src/web/security.py`: password and session-token hashing.
- `src/web/worker.py`: job claiming, lease renewal, AI execution and file cleanup.
- `src/web/static/`: responsive browser interface.
- `migrations/`: versioned schema and seed data.
- `tests/test_web_app.py`: isolated integration tests.
- `docs/full-application-design.md`: design decisions and follow-up roadmap.

The original Streamlit entry point is preserved. The full web application extends
the original portfolio scope; map its demonstrated behavior to your actual CDA
assessment rubric rather than treating this document as certification evidence.

### Validation performed during implementation

- 9 tests passed across the web integration suite and existing prompt-builder tests.
- PostgreSQL upgrade SQL compiled successfully; no live PostgreSQL server was
  available in the implementation environment.
- Chromium verified registration, project creation, upload queuing, renaming,
  deletion, and sign-out without JavaScript errors.
- Desktop screenshots were visually inspected; the studio passed a 390-pixel
  mobile viewport overflow check.
- GPU inference was not run in this environment. The automated workflow tests
  use explicit AI test fixtures, not real model outputs.
