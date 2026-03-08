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

### Training Pipeline (March 2026)
- [x] Data preparation script (tools/prepare_data.py)
- [x] Converted jellybean manual data to training format
- [x] Split synthetic data: train (25,138) / val (1,396) / test (1,396)
- [x] Jellybean held out as real-world test set (42 pairs)
- [x] T5 format with custom tokens: c0-c63 for 64 braille cells
- [x] T5-small fine-tuning pipeline (src/trainer.py)
- [x] MPS (Apple Silicon), CUDA, CPU device support
- [x] Gradient accumulation for small-memory training
- [x] Best model checkpointing by validation loss
- [x] Test suite: 67 tests passing (cell codec + data generator + prepare data + trainer)
- [x] Evaluation script (tools/evaluate.py)

### T5-small v1 Training Run (March 2026)
- [x] Ran T5-small fine-tuning: 5 epochs, lr=3e-4, batch=4, grad_accum=8
- [x] Evaluated on synthetic test set: 2.9% exact match (40/1396)
- [x] Evaluated on jellybean held-out: 0% exact match (0/42)
- [x] Result: model did not learn the task — outputs memorized English phrases
- [x] Root cause analysis: no LR warmup/scheduler, static padding, insufficient epochs
- [x] Evaluation report written (docs/evaluation-report-v1.md)

---

## Current Focus

### Fix Training Pipeline (v2)
- [ ] Add learning rate scheduler (linear warmup + cosine decay)
- [ ] Switch to dynamic padding (pad to longest in batch, not max_length)
- [ ] Lower learning rate to 1e-4
- [ ] Increase epochs to 10
- [ ] Re-run training
- [ ] Evaluate v2 on synthetic test set and jellybean
- [ ] Compare against liblouis back-translation baseline

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
- T5-small v1 failed (2.9% accuracy) — needs LR warmup, dynamic padding, more epochs
