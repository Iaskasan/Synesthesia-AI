# 🎨 Synesthesia AI

**Goal**: Build a system that listens to a song, extracts its mood/tempo/features, and generates an AI artwork that matches it.   
**MVP**: Upload a song → system extracts tempo & mood → system generates a Stable Diffusion image.  

---

## 🚀 Roadmap

### Phase 1: Foundations (1–2 months)
- [ ] Install libraries (librosa, numpy, matplotlib, sklearn)  
- [ ] Load and visualize audio waveform  
- [ ] Extract tempo and MFCCs  
- [ ] Understand features (plot + interpret)  

### Phase 2: Mood Classification (3–4 months)
- [ ] Build a small dataset of songs (with mood/genre labels)  
- [ ] Train ML models (logistic regression, random forest, etc.)  
- [ ] Evaluate models (accuracy, confusion matrix)  
- [ ] Visualize with PCA/t-SNE  

### Phase 3: Image Generation (5–6 months)
- [ ] Connect mood classifier → text prompt builder  
- [ ] Generate images with Stable Diffusion (diffusers library or API)  
- [ ] Test different prompt templates (“dreamy watercolor”, “cyberpunk neon”, etc.)  

### Phase 4: Multi-Frame Generation (7–8 months)
- [ ] Split audio into chunks (10–20s)  
- [ ] Generate one image per chunk  
- [ ] Experiment with smooth transitions between images  

### Phase 5: Final Polish (9 months)
- [ ] Build a simple demo app (Streamlit/Gradio)  
- [ ] Allow user to upload a song + pick style  
- [ ] Generate final visuals + music  
- [ ] Prepare presentation + documentation  

---

## ✅ Task Board

| To Do | In Progress | Done |
|-------|-------------|------|
| Set up environment |  | Installed librosa |
| Load sample song |  |  |
| Extract tempo/MFCCs |  |  |

---

## 🧠 Knowledge Notes
- **Tempo** = beats per minute.  
- **MFCCs** = a way to capture timbre of sound.  
- **PCA** = reduces data into 2D/3D for visualization.  
- **Confusion matrix** = compares predictions vs reality.  
- **Stable Diffusion** = AI that generates images from text prompts.  

---

## 🧪 Experiments

| Date | Dataset | Model | Features | Accuracy | Notes |
|------|---------|-------|----------|----------|-------|
| 2025-01-15 | 10 songs | Logistic Regression | Tempo + MFCCs | 65% | Small dataset, need more samples |

---

## 🎨 Prompt Bank

- Calm: “dreamy watercolor landscape, soft pastel colors”  
- Energetic: “abstract neon cyberpunk city, sharp strokes, vibrant”  
- Dark: “misty gothic forest, monochrome, dramatic shadows”  
- Happy: “cartoonish sunny meadow, bright colors, cheerful mood”  

---
