# SynesthesiaAI

**Transforming music into visual worlds through mood recognition**

SynesthesiaAI is a machine-learning portfolio project that converts a short
music excerpt into an original still image. It analyzes the audio, predicts
musical mood and theme tags, calculates signal characteristics, translates
that structured interpretation into an editable image prompt, and sends the
prompt to a pretrained diffusion model.

There is no single objectively correct image for a piece of music.
SynesthesiaAI is therefore designed as a transparent, user-controlled
interpretation rather than a black-box generator.

For the complete rationale, scope, ethics, and delivery plan, see the
[project pitch](SynesthesiaAI%20Project%20Pitch.md).

## Portfolio MVP

The committed MVP will:

1. accept one common audio file and select a 30-second excerpt;
2. predict a focused set of mood or theme tags with confidence scores;
3. calculate tempo, energy, loudness, and spectral brightness;
4. let the user review, remove, or emphasize detected tags;
5. expose controls for visual style, abstraction, music influence, aspect
   ratio, tag count, seed, and variations;
6. compose a visible and editable image-generation prompt;
7. generate one downloadable still image with Stable Diffusion or SDXL; and
8. export the analysis settings, prompt, model versions, and seed required to
   reproduce the result.

Real-time visuals, video, full-song storyboards, user accounts, and
image-to-music discovery are outside the portfolio MVP.

## Machine-learning objective

The main experiment compares two multilabel audio-classification approaches:

- **Baseline:** log-mel or hand-crafted audio features with a small classifier.
- **Main model:** frozen CLAP audio embeddings with a trainable multilabel
  classification head.

The models will be evaluated on the official held-out test split using:

- macro and micro F1;
- precision-recall AUC;
- per-label precision, recall, and F1;
- inference time; and
- qualitative error analysis.

Per-label thresholds, class weighting, or focal loss may be used when label
imbalance causes rare tags to be ignored. The final model will be selected
from measured results rather than model complexity.

## Data

The primary dataset is the
[MTG-Jamendo mood/theme subset](https://mtg.github.io/mtg-jamendo-dataset/),
which contains 18,486 tracks and 59 mood/theme tags.

For the portfolio project, the label space will be reduced to approximately
12–20 frequent and visually meaningful concepts. Candidate labels include
calm, dark, dreamy, energetic, epic, happy, melancholic, romantic, and
uplifting. The final list will be selected after frequency analysis.

The data pipeline will:

- preserve track IDs, artist splits, labels, source information, and licences;
- use the official train, validation, and test partitions;
- audit artist leakage, corrupted files, and label imbalance;
- produce deterministic 30-second excerpts;
- cache log-mel features or CLAP embeddings; and
- version metadata, split definitions, configurations, and manifests.

### Selected mood/theme labels

The initial classifier will use the following 20 labels:

```text
happy, energetic, relaxing, emotional, dark,
epic, dream, inspiring, sad, meditative,
uplifting, motivational, romantic, fun, calm,
adventure, melancholic, dramatic, powerful, hopeful
```

These labels were selected from the official training split using three
criteria:

- **Visual usefulness:** each label can produce a meaningful change in an
  image's atmosphere, composition, colour, lighting, or subject matter.
- **Dataset support:** the labels occur in every official split and generally
  have enough training examples for an initial multilabel classifier.
- **Coverage:** the set spans positive and negative emotion, energy, ambience,
  narrative scale, and reflective moods instead of concentrating on a single
  emotional family.

Related labels such as `sad` and `melancholic`, or `calm` and `meditative`,
remain separate because they can support visibly different interpretations.
Context and usage tags such as `advertising`, `corporate`, `film`, and
`trailer` were excluded because they describe where music may be used rather
than how it feels. Primarily musical descriptors such as `melodic`, `slow`,
and `fast` were also excluded because tempo, energy, and other audio
characteristics are calculated separately by the application.

`powerful` and `hopeful` are the least represented selected labels in the
training split, with 106 and 137 examples respectively. Training will
therefore evaluate class weighting or focal loss, tune thresholds per label,
and report per-label metrics. The vocabulary will only be revised after the
baseline's validation results and error analysis, rather than from test-set
performance.

Audio files, cached features, generated outputs, and model weights are local
artifacts and are excluded from Git.

See [HowTo.md](HowTo.md) for the current dataset download command.

## Application pipeline

```text
audio excerpt
    ↓
validation and deterministic preprocessing
    ↓
tempo, energy, loudness, and spectral features
    +
multilabel mood/theme classifier
    ↓
editable tags and confidence scores
    ↓
structured prompt schema + user controls
    ↓
constrained language-model composer
    or deterministic template fallback
    ↓
Stable Diffusion / SDXL
    ↓
image + reproducibility metadata
```

The project-trained model analyzes the music. The language model is only
responsible for translating structured tags and settings into concise visual
language. The diffusion model remains pretrained.

## Repository structure

```text
src/audio/          audio loading, features, and visualization
src/ml/             training, evaluation, and saved-model inference
src/generation/     prompt composition and diffusion integration
src/app/            Gradio desktop web application
tests/              dataset-independent automated tests
notebooks/          exploration and analysis
artifacts/          local trained models and feature caches (ignored)
outputs/            generated images and reports (ignored)
dataset/            downloaded audio (ignored)
```

Model training runs separately from the web application. Training produces a
versioned artifact containing the classifier, label mapping, thresholds,
feature schema, and evaluation metadata. The application only loads that
artifact for inference.

## Platform

- Windows 11 host with WSL2 Ubuntu
- Python and PyTorch
- librosa or Essentia for audio processing
- Hugging Face Transformers and Diffusers
- Gradio desktop browser interface
- target GPU: NVIDIA RTX 4090 with 24 GB VRAM

The definition-of-done performance target is a complete result in under
90 seconds on the target system.

## Local development

Create and activate an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the automated tests:

```bash
python -m pytest
```

Train a quick multilabel baseline against an extracted MTG-Jamendo dataset
mounted from Windows (for example, drive `G:` in WSL):

```bash
python -m src.ml.train_baseline \
  --dataset-root /mnt/g/AI/Datasets \
  --max-tracks 100
```

Remove `--max-tracks 100` for a full run. The trainer uses the official
train/validation/test split, chooses the 12 most frequent training labels by
default, and writes the model and evaluation report to `artifacts/baseline/`.
The expected extracted layout is `/mnt/g/AI/Datasets/00/7400.low.mp3`.

The application launch command will be added when the Gradio interface
replaces the current prototype.

## Delivery plan

### Week 1 — Data audit and baseline

- review licences, file integrity, official splits, and label frequencies;
- select and freeze the 12–20-label target space;
- complete deterministic preprocessing and caching;
- train and evaluate the handcrafted-feature baseline; and
- freeze the MVP feature scope.

### Week 2 — Pretrained audio representations

- extract and cache CLAP embeddings;
- train the multilabel classification head;
- handle imbalance and tune per-label thresholds; and
- select a checkpoint using held-out metrics and error analysis.

### Week 3 — Creative pipeline

- integrate calculated audio characteristics and model inference;
- implement the structured prompt schema;
- add the constrained language-model composer and template fallback;
- integrate one tested diffusion checkpoint; and
- build the Gradio controls and export flow.

### Week 4 — Evaluation and delivery

- compare language-model prompts with deterministic templates;
- conduct a small, anonymous human evaluation;
- complete accessibility, reliability, and reproducibility checks; and
- prepare the final report, demo, documentation, and presentation.

## Evaluation beyond classifier accuracy

At least 10 voluntary testers will rate a randomized set of outputs for:

- perceived music–image consistency; and
- perceived user control.

The project will compare language-model prompts against deterministic
templates using the same unseen excerpts and visual settings. Success is
defined by correspondence, robustness, transparency, and reproducibility—not
only by image aesthetics.

## Ethics and privacy

- Only recordings distributed for research under documented licences will be
  used, and licence metadata will be preserved.
- Mood predictions are presented as a detected interpretation, not an
  objective description of a listener's emotions.
- Weak labels and representation limitations will be reported.
- Prompt generation will avoid requests to imitate living artists.
- Generated images will be identified as AI-generated.
- Uploaded audio remains local and is deleted after processing unless the
  user explicitly saves it.
- The project does not collect identity, listening history, or unnecessary
  analytics.
- Large pretrained models remain frozen, and cached embeddings reduce repeated
  computation.

## Current status

The repository currently contains an early audio-to-prompt prototype:

- normalized audio loading;
- tempo, energy, spectral, chroma, and MFCC extraction;
- deterministic prompt construction;
- local diffusion integration; and
- dataset-independent tests.

The current interface and manual mood selector are temporary scaffolding.
The next milestone is the MTG-Jamendo data audit, label shortlist, and
reproducible baseline pipeline, followed by the Gradio application described
above.
