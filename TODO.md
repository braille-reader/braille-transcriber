# TODO - Braille OCR Transcriber

**Last Updated:** March 2026
**Current Phase:** Grade 2 Data Collection + Model Development

---

## Completed

### Phase 1: Grade 1 CLI (January 2026)
- [x] Research and planning (architecture, tech stack, roadmap)
- [x] YOLOv8 integration (DotNeuralNet pre-trained weights)
- [x] Preprocessing pipeline (CLAHE)
- [x] Grade 1 lookup table (A-Z, 0-9, punctuation, indicators)
- [x] CLI tool (`transcribe.py`)
- [x] Tested on AngelinaDataset (91.7% confidence on real photos)
- [x] Optimal confidence threshold: 0.15-0.25

### Grade 2 Planning (January-March 2026)
- [x] Data format design (dot notation with pipe separators)
- [x] Annotation app design (CLI-based, accessibility-friendly)
- [x] Braille codes reference table
- [x] First manual dataset collected (jellybean_jungle.txt)
- [x] Researched synthetic data generation approaches (Liblouis, BRF files)

---

## Current Focus

### Synthetic Data Pipeline
- [ ] Build Liblouis-based data generator (English text -> contracted braille dot patterns)
- [ ] Source English text corpus (Project Gutenberg)
- [ ] Generate large-scale parallel corpus (dot patterns <-> English)
- [ ] Validate synthetic data against manual data

### Seq2Seq Translation Model
- [ ] Design model architecture (dot pattern sequences -> English text)
- [ ] Training pipeline
- [ ] Evaluate against manually collected test data

---

## Backlog

### Testing & Validation
- [ ] Formal accuracy benchmarks on English braille images
- [ ] Test suite (pytest) for pipeline components
- [ ] Document failure cases

### Grade 2 Model Improvements
- [ ] Hybrid approach: Liblouis rules + ML
- [ ] Context-aware contraction resolution
- [ ] Error analysis (which contractions fail?)

### Nemeth Code (Future)
- [ ] Math braille dataset creation
- [ ] Spatial layout parser
- [ ] LaTeX/MathML generation

### Mobile App (Deferred)
- [ ] Flutter app with camera interface
- [ ] Model export (TFLite/CoreML)
- [ ] On-device inference

---

## Known Issues

- Model struggles with synthetic/graphic images (expected — trained on real photos)
- Double-sided braille interference not handled
- No Grade 2 support yet (90-95% of real braille)

## Key Findings

- Optimal confidence threshold: 0.15-0.25
- Model performs best on real embossed braille photos (91%+)
- CLAHE preprocessing helps with low-contrast images
- Liblouis can generate unlimited synthetic parallel data for Grade 2 training
