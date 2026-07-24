# Synesthesia AI

Synesthesia AI turns a short audio excerpt into visual artwork:

`audio → audio features → mood/style tags → image prompt → diffusion model`

The first MVP is a web app where a user uploads an excerpt, reviews the
detected musical characteristics, adjusts the visual direction, and generates
one image.

## Current state

Implemented:

- audio upload and normalized 30-second loading;
- tempo, energy, spectral, chroma, and MFCC feature extraction;
- structured prompt construction from musical and user-selected attributes;
- local Stable Diffusion integration;
- a Streamlit interface;
- unit tests for the feature and prompt contracts.

The mood selector is deliberately manual for now. It will be replaced by the
first trained multi-label classifier; the UI does not pretend that a model has
already made that prediction.

## Architecture

```text
src/audio/          loading, feature extraction, visualization
src/ml/             training, evaluation, and saved-model inference
src/generation/     prompt construction and image model adapter
src/app/            Streamlit presentation layer
tests/              fast tests that do not require the downloaded dataset
artifacts/          local trained models (ignored by Git)
outputs/            generated images and reports (ignored by Git)
```

Training belongs outside the web request. A training command will read the
MTG-Jamendo annotations and audio, build feature rows, train and evaluate a
multi-label model, then save a versioned artifact. The app should only load
that artifact for inference.

## Run locally

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest
```

Start the app:

```bash
streamlit run src/app/app.py
```

The first image generation downloads model weights and can be slow. A
CUDA-capable GPU is strongly recommended.

## Dataset

The repository includes the MTG-Jamendo dataset toolkit as a vendored source
tree. Downloaded audio belongs in the ignored `dataset/` directory and must
not be committed. See [HowTo.md](HowTo.md) for the current download command.

## Delivery roadmap

1. **Working audio-to-prompt slice** — complete the upload, analysis, prompt,
   and generation path with clear error handling.
2. **Dataset pipeline** — index MTG-Jamendo tracks and labels, extract cached
   fixed-size features, and create reproducible train/validation/test inputs.
3. **Baseline model** — train a multi-label mood/genre classifier, record
   precision/recall/F1, and save the model with its label and feature schema.
4. **Inference integration** — replace the manual mood field with ranked tags
   and confidence scores while retaining user control.
5. **Generation quality** — add seeds, negative prompts, selectable model
   adapters, and reproducible output metadata.
6. **Product polish** — background jobs, progress reporting, history, and
   deployment.

Longer-term ideas such as live microphone input, multi-frame visuals, and
music recommendation remain useful, but are intentionally outside the first
MVP.
