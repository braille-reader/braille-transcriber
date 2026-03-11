# TODO - Braille OCR Transcriber

**Last Updated:** March 11, 2026
**Current Phase:** Grade 2 Model v3 Complete — Evaluation & Improvement

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
- [x] Unicode braille format with task prefix (prepare_data.py)
- [x] ByT5-small fine-tuning pipeline (src/trainer.py)
- [x] Seq2SeqTrainer + DataCollatorForSeq2Seq with dynamic padding
- [x] MPS (Apple Silicon), CUDA, CPU device support
- [x] Gradient accumulation for small-memory training
- [x] Best model checkpointing by validation loss
- [x] Test suite: 74 tests passing (cell codec + data generator + prepare data + trainer)
- [x] Evaluation script with CER, BLEU metrics (tools/evaluate.py)

### v1 Training Run — T5-small + Custom Tokens (March 2026)
- [x] Ran T5-small fine-tuning: 5 epochs, lr=3e-4, batch=4, grad_accum=8
- [x] Evaluated on synthetic test set: 2.9% exact match (40/1396)
- [x] Evaluated on jellybean held-out: 0% exact match (0/42)
- [x] Result: model did not learn — outputs memorized English phrases
- [x] Root cause: random custom token embeddings, no LR warmup/scheduler
- [x] Evaluation report written (docs/evaluation-report-v1.md)

### v2 Training Run — T5-small + Unicode Braille (March 2026)
- [x] Switched to Seq2SeqTrainer with warmup + cosine decay
- [x] Unicode braille input format (U+2800-U+283F)
- [x] Trained on Colab T4 with fp16
- [x] Result: 0% accuracy — T5's SentencePiece vocab maps all braille to `<unk>`
- [x] Root cause: T5 tokenizer doesn't include Unicode braille characters
- [x] Evaluation report written (docs/evaluation-report-v2.md)

### v3 Training Run — ByT5-small on A100 (March 2026)
- [x] Switched to ByT5-small (300M params) — byte-level, no tokenizer issues
- [x] Researched BrailleLLM (EMNLP 2025) and related work (docs/v3-proposal.md)
- [x] Debugged T4 infrastructure issues: OOM, gradient checkpointing, fp16 overflow
- [x] Upgraded to Colab Pro with A100 80GB + bf16
- [x] Trained 10 epochs: train loss 10.6 → 0.049, val loss 2.25 → 0.008
- [x] **Jellybean evaluation: 76.2% exact match, 0.01 CER** (first working Grade 2 model)
- [x] 7/10 misses are smart quote differences only — content accuracy ~93%
- [x] Zero hallucinations — all predictions are input-dependent and correct
- [x] Training run documented (docs/v3-training-run.md)

---

## Current Focus

### Evaluate & Improve v3
- [ ] Run full evaluation on synthetic test set (1,396 samples) on Colab
- [ ] Normalize smart quotes in evaluation to get true content accuracy
- [ ] Write formal evaluation-report-v3.md
- [ ] Compare against liblouis back-translation baseline
- [ ] Error analysis — which contractions or patterns cause failures?

---

## Backlog

### More Training Data
- [ ] Add more Gutenberg books (currently 5, target 20+)
- [ ] BRF ↔ Gutenberg alignment for real transcription data
- [ ] Retrain with larger dataset

### Model Improvements
- [ ] Continue training beyond 10 epochs (loss still decreasing)
- [ ] Try ByT5-base (580M params) for higher capacity
- [ ] Hybrid approach: Liblouis rules + ML for edge cases
- [ ] Context-aware contraction resolution for ambiguous cells

### End-to-End Pipeline
- [ ] Connect Stage 1 (YOLOv8 detection) → Stage 2 (ByT5 interpretation)
- [ ] Test on real braille images through full pipeline
- [ ] Formal accuracy benchmarks on real braille photos

### Nemeth Code (Future)
- [ ] Math braille dataset creation
- [ ] Spatial layout parser
- [ ] LaTeX/MathML generation

### Mobile App (Deferred)
- [ ] Flutter app with camera interface
- [ ] Model export (TFLite/CoreML/ONNX)
- [ ] On-device inference

---

## Known Issues

- Stage 1 detector struggles with synthetic/graphic images (expected — trained on real photos)
- Double-sided braille interference not handled
- Smart quote vs straight quote differences cause false misses in evaluation
- src/__init__.py eagerly imports detector (needs model files) — causes import issues for tests
- Liblouis lowercase BRF uses `` ` { | } ~ `` instead of `` @ [ \ ] ^ `` — handled in cell_codec
- Minor contraction differences between Liblouis and human transcriptions (e.g., "grows")
- ByT5 inference is slow on CPU/MPS (~1-2 sec/sample) — needs GPU for batch evaluation

## Key Findings

- **ByT5-small is the right model for braille** — byte-level processing handles Unicode braille natively
- **T5-small cannot work** — its SentencePiece vocabulary doesn't include braille characters (all → `<unk>`)
- **ByT5 requires bf16 or fp32** — fp16 overflows in attention softmax on long byte sequences
- **A100 with bf16 is the sweet spot** — 10 epochs in ~3 hours, only ~10GB of 80GB used
- **Synthetic data works** — model trained on Liblouis-generated data generalizes to real human-transcribed braille
- Optimal confidence threshold: 0.15-0.25 (Stage 1 detection)
- CLAHE preprocessing helps with low-contrast images
- Cell codes (0-63) are the best canonical representation — BRF and dot patterns both convert trivially
- 5 public domain books yield ~28K training sentences with 2.16M braille cells
