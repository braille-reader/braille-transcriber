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

### Grade 2 Data Pipeline (January-March 2026)
- [x] Data format design (dot notation with pipe separators)
- [x] Braille entry tool for manual data collection (tools/braille_entry.py)
- [x] First manual dataset collected (jellybean_jungle.txt)
- [x] Researched synthetic data generation approaches (Liblouis, BRF files)
- [x] Collected 5 BRF files from Bookshare (public domain classics)
- [x] Downloaded matching Gutenberg plain texts
- [x] Cell codec module — BRF, dot patterns, cell codes (0-63) all interconvertible
- [x] Design doc for data pipeline (cell codes as canonical representation)
- [x] Installed liblouis 3.37.0 + Python bindings
- [x] Built data generator: English → liblouis Grade 2 → cell codes
- [x] Gutenberg header/footer stripping, sentence splitting
- [x] Validated: Liblouis output closely matches real Bookshare BRF
- [x] Generated synthetic training data from 5 books (27,930 sentences, 2.16M cells)
- [x] Test suite: 44 tests passing (cell codec + data generator)

---

## Current Focus

### Prepare Training/Validation Sets
- [ ] Convert jellybean_jungle.txt to training format (cell codes + English)
- [ ] Split synthetic data into train/val/test sets
- [ ] Use manual data (jellybean) as held-out validation

### Seq2Seq Translation Model
- [ ] Design model architecture (cell code sequences → English text)
- [ ] Training pipeline
- [ ] Evaluate against manual test data
- [ ] Compare model output vs liblouis back-translation

---

## Backlog

### BRF ↔ Gutenberg Alignment (Real Data)
- [ ] Strip Bookshare boilerplate from BRF files
- [ ] Reconstruct paragraphs from BRF line wrapping
- [ ] Align BRF paragraphs with Gutenberg text
- [ ] Use as fine-tuning data or additional test set

### Grade 2 Model Improvements
- [ ] Hybrid approach: Liblouis rules + ML
- [ ] Context-aware contraction resolution
- [ ] Error analysis (which contractions fail?)

### Testing & Validation
- [ ] Formal accuracy benchmarks on English braille images
- [ ] Document failure cases

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
- src/__init__.py eagerly imports detector (needs model files) — causes import issues for tests
- Liblouis lowercase BRF uses `` ` { | } ~ `` instead of `` @ [ \ ] ^ `` — handled in cell_codec
- Minor contraction differences between Liblouis and human transcriptions (e.g., "grows")

## Key Findings

- Optimal confidence threshold: 0.15-0.25
- Model performs best on real embossed braille photos (91%+)
- CLAHE preprocessing helps with low-contrast images
- Liblouis can generate unlimited synthetic parallel data for Grade 2 training
- Cell codes (0-63) are the best canonical representation — BRF and dot patterns both convert trivially
- BRF from Bookshare represents real professional transcriptions (may differ from Liblouis)
- 5 public domain books yield ~28K training sentences with 2.16M braille cells
