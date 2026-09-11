# Presentation readiness

## Scope to freeze

Present SynesthesiaAI as a user-controlled music-to-image prototype: upload audio, inspect its characteristics and suggested moods, edit the interpretation, generate artwork, and export the image and settings.

Pause space experiments, new annotation queues, encoder changes, and parameter searches. The app currently uses the existing CLAP head with nine active moods; inspiring is suppressed and experimental space heads are not deployed. Keep experimental artifacts as evidence of investigation, not a prerequisite for delivery.

## Priority order

1. Rehearse the complete app on the actual presentation machine. Measure audio analysis and image generation time. Verify playback, prompt edits, image generation, and both downloads. No GPU is visible in the current development session; model cache directories exist, but this does not prove all weights are available or generation is fast enough.
2. Prepare two or three contrasting audio examples with permission to use them. Save the actual generated images and exported JSON settings. Use these as a clearly identified recorded/precomputed fallback if live generation is slow. Show one imperfect mood prediction and how editing helps.
3. Prepare a concise presentation and rehearse to the time limit. Focus on the problem, architecture, demonstration, evaluation, limitations, and practical lessons. Confirm deadline, duration, and required deliverables with the presenter.
4. Fix issues that disrupt the rehearsal before cosmetic changes. Track downloader test failures separately; the downloader is not in the inference demo path.

## Suggested demonstration

- Upload a prepared clip; explain that the first 30 seconds are analyzed.
- Show tempo, energy, brightness, and suggested moods. Model scores are not calibrated certainty.
- Review the mood choices and show the editable prompt. Explain that music can support multiple valid visual interpretations.
- Generate one image, then export the image and settings. Use an already generated example if needed, explicitly identifying it as precomputed.
- Briefly show a contrasting result to demonstrate the scope of the idea.

## Presentation outline

1. Problem and objective: make musical characteristics a starting point for visual creation.
2. Architecture: audio excerpt → signal features and frozen CLAP embeddings → trained mood head → editable prompt → pretrained Stable Diffusion → image and metadata.
3. Engineering contribution: data preparation and split discipline, cached features, classifier training and evaluation, interface, prompt construction, and reproducibility exports. CLAP and Stable Diffusion are pretrained components, not models trained from scratch for this project.
4. Live or recorded demonstration with actual project outputs.
5. Evidence and limitations: distinguish dataset-label metrics, subjective manual judgments, and small biased review samples. Explain why inspiring was disabled and space stayed experimental. Existing final-test results belong to the historical model; do not attribute them to later candidates.
6. Conclusion: a working controllable prototype with imperfect mood detection; future work should improve semantic agreement and evaluate visual usefulness.

## Current checks

- Full automated suite: 44 passed, 2 failed. Failures are in `tests/test_download_cli.py`: missing `build_parser` and missing partial-download resume behavior in the bundled downloader.
- App implements upload, analysis, mood editing, image generation and exports; an actual end-to-end rehearsal is still required.
- Current session reports no CUDA GPU. Cached model directories exist. No demo images were found in `outputs/` during this check.
- No production checkpoint changed for presentation preparation.

## Run

From the repository root:

```bash
.venv/bin/python -m streamlit run src/app/app.py
```

Leave space review queues aside until after the presentation unless they are explicitly needed in the written report.

## Confirmed format: 10-minute presentation and 1-minute live demo

Budget the demo inside the ten minutes for a conservative rehearsal; if it is additional, use the extra minute for questions or transitions.

| Slide | Content | Time |
|---|---|---|
| 1 | Objective and one actual audio/image example | 0:45 |
| 2 | User journey and MVP scope | 0:45 |
| 3 | Architecture and pretrained versus custom components | 1:30 |
| 4 | Dataset, official splits, cached embeddings, training approach | 1:15 |
| 5 | Results and honest limitations; keep space investigation brief | 1:30 |
| 6 | Live demo | 1:00 |
| 7 | Engineering decisions and what the experiments taught us | 1:15 |
| 8 | Conclusion and focused future work | 1:00 |
| — | Transition buffer | 1:00 |

### One-minute demo script

Prepare the app and a real generated result before speaking. Keep the actual prompt, selected tags, and exported settings matched to that result.

- 0–10 seconds: show the prepared clip and describe the goal.
- 10–25 seconds: show measured features and mood suggestions.
- 25–40 seconds: show the editable prompt and explain how the user steers it. If changing controls, use “Recompose prompt from controls” so the prompt reflects them.
- 40–55 seconds: show the previously generated image, explicitly saying it was generated before the talk, and demonstrate the exports.
- 55–60 seconds: explain the result as one visual interpretation, not a unique correct answer.

Only click Generate live if a timed rehearsal establishes that it fits comfortably. The precomputed result remains the fallback; do not imply it reflects prompt edits made afterward. Avoid changing uploads during the short demo without verifying which audio the displayed result belongs to.
