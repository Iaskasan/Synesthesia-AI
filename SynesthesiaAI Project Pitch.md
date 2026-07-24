**MACHINE LEARNING PORTFOLIO PROJECT**

# **SynesthesiaAI**

## Transforming music into visual worlds through mood recognition

| Author | Hadrien Tayac | Duration | Two month |
| :---- | :---- | :---- | :---- |
| Role | Sole developer | Target | Desktop web application |

# **Introduction**

SynesthesiaAI is a creative machine learning application that converts a short music excerpt into an original still image. The system will analyze the audio, identify musical moods and characteristics, translate that structured analysis into an image-generation prompt, and send the prompt to a pretrained diffusion model.

The project does not assume that one objectively correct image exists for a piece of music. Its intent is to produce a transparent and controllable visual interpretation. The user remains part of the creative process by choosing the image style, selecting how many detected tags are used, and adjusting how strongly the music should influence the result.

**Portfolio MVP:** analyze one 30-second excerpt, predict a focused set of mood tags, calculate tempo and energy features, compose a controlled prompt, and generate one downloadable image.

## **Project goal**

The central machine learning goal is to compare a simple audio baseline with a classifier built on pretrained audio representations, then determine whether the predicted attributes lead to images that users perceive as emotionally consistent with the source music.

## **Author and responsibilities**

Hadrien Tayac is the sole author and developer. Responsibilities include dataset selection and licensing review, audio preprocessing, model training and evaluation, prompt-composer design, diffusion integration, user-interface development, testing, documentation, and the final presentation. No additional teammate is planned for the portfolio MVP.

# **Description: user experience**

The application will guide the user through a short, visible pipeline rather than hiding every decision behind a single Generate button.

1. **Upload music.** The user uploads a common audio file and selects, or allows the application to select, a 30-second excerpt.  
2. **Analyze the excerpt.** The trained model predicts mood or theme tags with confidence scores. Signal-processing functions estimate tempo, energy, loudness, and spectral brightness.  
3. **Review the interpretation.** The interface displays the predicted tags and allows the user to remove or emphasize them.  
4. **Choose the visual direction.** The user selects an image style, tag count, abstraction level, music-influence level, aspect ratio, and number of variations.  
5. **Compose the prompt.** A pretrained language model translates the structured attributes into a concise prompt for the selected diffusion model. The prompt remains visible and editable.  
6. **Generate and export.** Stable Diffusion or SDXL creates the image. The user can download it together with the analysis settings, prompt, and seed.

## **System boundaries**

*  The trained project model analyzes music; the language model only converts structured tags and user settings into visual language.  
*  The diffusion model is pretrained and will not be trained from scratch.  
*  Tempo and low-level signal properties will be calculated with established audio-processing methods instead of unnecessarily learned by a neural network.  
*  The first release generates a single still image from one excerpt. Real-time visuals, video, and evolving full-song storyboards are outside the portfolio scope.

## **Planned ML experiment**

*  Baseline: log-mel or hand-crafted audio features with a small classifier.  
*  Main model: frozen CLAP audio embeddings followed by a trainable multilabel classifier.  
*  Metrics: macro and micro F1, precision-recall AUC, per-label performance, inference time, and qualitative error analysis.  
*  Final validation: a small human study comparing whether generated images fit the source music and whether LLM prompts outperform deterministic templates.

## **Definition of done**

*  The application accepts a valid audio file and completes the full pipeline without manual code execution.  
*  The final classifier is evaluated on a held-out test set and compared with at least one baseline.  
*  The user can inspect and edit tags, prompt, style controls, and seed.  
*  One result is generated in under 90 seconds on the target RTX 4090 system.

# **Data**

## **Required data**

The project requires music recordings paired with multilabel annotations describing mood, theme, genre, or instrumentation. The primary candidate is the MTG-Jamendo mood/theme subset, which contains 18,486 tracks and 59 mood/theme tags. The full MTG-Jamendo collection contains more than 55,000 tracks and 195 tags. The lower-bitrate mood/theme audio package is approximately 46 GB, making it practical for local experimentation while preserving the relevant labels.

For the two-month portfolio feature, the label space will be reduced to approximately 12-20 frequent, visually meaningful concepts such as calm, dark, dreamy, energetic, epic, happy, melancholic, romantic, and uplifting. Final labels will be chosen only after frequency analysis and a review of their usefulness for visual generation.

## **Collection and licensing**

*  Download metadata and audio only through the dataset's official repository and download scripts; do not scrape commercial streaming platforms.  
*  Retain the original track identifier, source URL, artist split, label list, and per-track licence information.  
*  Use the official train, validation, and test partitions, then check for artist leakage and label imbalance before training.  
*  Use the Song Describer Dataset only as an optional captioning or evaluation reference; its 706 recordings are too small for training the core classifier from scratch.

## **Preprocessing**

*  Validate file integrity and remove unreadable or missing samples.  
*  Decode and resample audio to the sample rate required by the selected encoder.  
*  Create deterministic 30-second excerpts and avoid using different excerpts from the same track across data splits.  
*  Encode labels as multilabel vectors and record class frequencies.  
*  Cache log-mel features or CLAP embeddings so repeated experiments do not decode every audio file again.  
*  Apply class weighting, threshold tuning, or focal loss if rare tags are systematically ignored.

## **Storage and data management**

Raw audio will be stored on a secondary local SSD rather than the Windows system drive. Metadata and split definitions will be stored as versioned CSV or Parquet files; cached features will use NumPy or PyTorch tensor files; experiment summaries will use JSON or CSV. Dataset files and model weights will be excluded from Git, while scripts, configuration files, checksums, and a reproducible data manifest will be committed.

User-uploaded audio will be processed locally in a temporary directory and deleted after the session unless the user explicitly chooses to save it. Generated images will store only the selected tags, prompt, settings, model versions, and random seed needed to reproduce the result.

## **Data review during the project month**

Although most collection and preprocessing will happen before the project month, Week 1 reserves time to audit corrupted files, label distributions, licences, and split quality. Week 2 includes a second review after the first model exposes weak labels or underrepresented categories.

# **Ethics**

* ** Copyright and attribution.** Only data distributed for research under documented licences will be used. Licence metadata will be preserved, attribution requirements will be documented, and the dataset will not be republished with the source code.  
* ** Subjective labels.** Mood recognition reflects annotators and dataset conventions, not an objective reading of a listener's emotions. The interface will show confidence scores, use language such as detected interpretation, and let users correct the tags.  
* ** Representation bias.** Public music datasets may overrepresent Western genres, common instruments, and English-language metadata. Results will be reported per label, weak categories will be disclosed, and the project will not claim universal music understanding.  
* ** Generative-model bias.** The prompt composer will avoid requests to imitate living artists and will use a constrained output schema. Generated images will be labelled as AI-generated, and unsafe or discriminatory outputs found during testing will be documented.  
* ** Privacy.** Uploaded music will remain local to the application and be removed after processing by default. The application will not collect identity, listening history, or unnecessary analytics.  
* ** Accessibility and user control.** Controls will have clear labels and keyboard-friendly inputs. Users can edit the model's interpretation instead of being forced to accept an incorrect prediction.  
* ** Environmental cost.** The audio encoder and diffusion model will remain frozen. Embeddings will be cached, experiment sizes will be limited, and training runs will be logged to reduce repeated computation.

# **Platform**

SynesthesiaAI will target a local desktop web application. Development will use a Windows 11 host with WSL2 Ubuntu, Python, PyTorch, Hugging Face Transformers/Diffusers, and audio libraries such as librosa or Essentia. A Gradio interface will provide audio upload, tag review, controls, prompt editing, and image display in a browser.

*  Target hardware: NVIDIA RTX 4090 with 24 GB VRAM, standard desktop CPU, and secondary SSD storage.  
*  Supported client: modern desktop browser. A mobile-native application is not part of the MVP.  
*  Fallbacks: cached embeddings for faster experiments, a deterministic prompt template if the LLM is unavailable, and a smaller diffusion checkpoint if SDXL exceeds the deployment memory budget.  
*  Reproducibility: pinned dependencies, configuration files, logged seeds, model versions, and a scripted end-to-end demo.

# **Main risks and mitigations**

* ** Noisy or imbalanced mood tags:** reduce the label set, tune per-class thresholds, and report per-label metrics.  
* ** Prompt drift or hallucinated concepts:** enforce JSON output, low temperature, explicit constraints, and a template fallback.  
* ** GPU or integration delays:** freeze large models, cache features, integrate components separately, and keep one tested checkpoint for the final demo.  
* ** Scope expansion:** freeze the one-image MVP at the end of Week 1; treat video, swap to image-to-music option, real-time generation, accounts, galleries, and full-song evolution as future work.

# **Evaluation plan**

*  Measure multilabel performance on the held-out test set using macro/micro F1, precision-recall AUC, per-label scores, and qualitative error analysis.  
*  Compare the LLM prompt composer with a deterministic template on the same unseen excerpts and visual settings.  
*  Ask at least 10 voluntary testers to rate music-image consistency and perceived user control on a short, randomized set of outputs; collect no identifying information.  
*  Treat correspondence, robustness, transparency, and reproducibility as success criteria rather than judging images only by aesthetic appeal.

## **Scope gates**

*  End of Week 1: label set, dataset split, evaluation metrics, and MVP features are frozen.  
*  End of Week 2: if the main classifier is unstable, keep the best validated baseline and prioritize the complete application.  
*  End of Week 3: no new features; Week 4 is reserved for evaluation, reliability, documentation, and presentation.

# **Schedule**

## **Pre-project preparation**

During the weeks before the official month, the objective is to remove data uncertainty: confirm the dataset and licences, download an initial subset, validate files, inspect tag frequencies, prototype one preprocessing path, create the repository, and open the GitHub Projects board. An alumni or mentor should review this pitch before Month Week 1 begins.

| Period | Focus | Key tasks | Exit deliverable |
| :---- | :---- | :---- | :---- |
| Week 1 | Data audit & baseline | Confirm 12-20 labels; review licences and splits; finish preprocessing; train a simple baseline; freeze MVP scope. | Validated dataset card, working loader, baseline metrics, pitch review. |
| Week 2 | Main audio model | Extract/cache CLAP embeddings; train multilabel head; tune thresholds and imbalance handling; review data errors. | Selected checkpoint, held-out metrics, error analysis, inference function. |
| Week 3 | Creative pipeline | Implement tempo/energy features; controlled LLM prompt composer; template fallback; diffusion wrapper; Gradio controls. | End-to-end local MVP generating one image from one excerpt. |
| Week 4 | Evaluation & delivery | Usability fixes; human evaluation; mentor review; final ethics/limitations audit; README, demo, report, and presentation. | Tested portfolio release and final review package. |

## **Project management**

GitHub Projects will be used as the project management tool. The board will contain Backlog, Ready, In Progress, Review, and Done columns. Each card will include an owner, area, priority, target week, acceptance criterion, and link to the related issue or pull request.

## **Initial GitHub Projects cards**

* ** Data:** Create dataset card; licence audit; download validation; EDA and label shortlist; preprocessing and cache pipeline.  
* ** Model:** Baseline classifier; CLAP embedding extraction; multilabel head; threshold tuning; test metrics and error analysis.  
* ** Generation:** Audio-feature extractor; structured prompt schema; LLM composer; deterministic fallback; diffusion wrapper.  
* ** Product:** Gradio upload flow; tag editor; style controls; prompt editor; generation result and metadata export.  
* ** Quality:** Unit/integration tests; human evaluation; accessibility check; reproducibility test; README and demo script.

# **Portfolio feature versus full vision**

The full SynesthesiaAI idea could later analyze whole tracks in segments, preserve visual continuity, and create storyboards or video. Those features are intentionally excluded from the two-month deliverable.

**Committed portfolio feature:** a transparent audio-to-tags-to-prompt-to-image pipeline for one excerpt and one still image, with user-controlled style and tag selection.

# **Expected outcome**

SynesthesiaAI should demonstrate an end-to-end application of machine learning rather than only a notebook result. The final submission will show data management, multilabel audio classification, transfer learning, reproducible evaluation, human-centered controls, ethical limitations, and integration with pretrained generative models.

# **References and technical resources**

*  [MTG-Jamendo Dataset \- metadata, official splits, download scripts, and baseline information](https://mtg.github.io/mtg-jamendo-dataset/)  
*  [LAION CLAP \- contrastive audio-language representations](https://github.com/LAION-AI/CLAP)  
*  [Song Describer Dataset \- permissively licensed music-caption evaluation data](https://github.com/mulab-mir/song-describer-dataset)  
*  [Hugging Face Diffusers documentation](https://huggingface.co/docs/diffusers/)

