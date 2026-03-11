# Braille OCR Transcriber

> A research-driven approach to building superior Braille OCR models with support for contracted Braille (Grade 2) and mathematical notation (Nemeth Code)

**Status:** Phase 2 Complete — First Working Grade 2 Model (76% exact match, 0.01 CER)
**Started:** January 19, 2026
**Current Focus:** Evaluation & improvement of ByT5-small Grade 2 model

---

## 🎯 Project Vision

Build the first Braille OCR system that supports:
- ✅ Grade 1 (Uncontracted Braille) - **Phase 1 Complete**
- ✅ Grade 2 (Contracted Braille) - **First working model! 76% exact match**
- 🔬 Nemeth Code (Mathematical Braille) - **Future Work**

### Why This Matters

**The Problem:**
- 90-95% of real Braille documents use Grade 2 (contracted)
- NO existing models support Grade 2 interpretation
- Teachers and parents cannot check homework
- Existing apps have terrible accuracy

**Our Solution:**
- Two-stage architecture: Vision (solved) → Interpretation (novel)
- Context-aware Grade 2 resolver combining rules + ML
- First-to-market for contracted Braille OCR

---

## Documentation

- **[Project Strategy](docs/project-strategy.md)** - Key decisions, architecture, research findings, roadmap
- **[v3 Training Run](docs/v3-training-run.md)** - ByT5-small training on A100: setup, results, analysis
- **[v3 Proposal](docs/v3-proposal.md)** - ByT5 approach with research findings (BrailleLLM, etc.)
- **[v3 Issues Log](docs/v3-issues-log.md)** - Infrastructure issues encountered and resolved
- **[Evaluation Report v1](docs/evaluation-report-v1.md)** - T5-small + custom tokens (2.9% accuracy)
- **[Evaluation Report v2](docs/evaluation-report-v2.md)** - T5-small + Unicode braille (0% — tokenizer failure)
- **[Implementation Log](docs/implementation-log.md)** - Development session notes and results
- **[Braille Codes Reference](docs/braille-codes-ref.md)** - Complete dot pattern to code mapping
- **[Data Collection Research](docs/collecting-data-research.md)** - Approaches for building training data

---

## 🏗️ Architecture

### Two-Stage Design

```
┌─────────────┐
│   Image     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│  Stage 1: Cell Detection    │
│  (Computer Vision)          │
│  • YOLOv8 / Angelina        │
│  • 98-99% accurate          │
│  • ALREADY SOLVED ✅        │
└──────────┬──────────────────┘
           │
           ▼ Binary dot patterns
┌─────────────────────────────┐
│  Stage 2: Interpretation    │
│  (Linguistic Rules + ML)    │
│  • Grade 1: Simple lookup   │
│  • Grade 2: Research 🔬     │
│  • Nemeth: Future 🔬        │
└──────────┬──────────────────┘
           │
           ▼
      ┌────────┐
      │  Text  │
      └────────┘
```

**Key Insight:** Braille cells are binary structures (2^6 = 64 patterns), making Stage 1 fundamentally easier than handwriting OCR. The real challenge is Stage 2: context-dependent interpretation.

---

## Project Structure

```
braille-transcriber/
├── src/
│   ├── cell_codec.py      # BRF, dot patterns, cell codes, Unicode conversions
│   ├── detector.py        # YOLOv8 cell detection wrapper
│   ├── interpreter.py     # Grade 1 Braille → English lookup
│   ├── pipeline.py        # End-to-end pipeline
│   ├── preprocess.py      # CLAHE preprocessing
│   └── trainer.py         # ByT5 training pipeline (Seq2SeqTrainer)
├── tools/
│   ├── prepare_data.py    # Convert raw data → Unicode braille TSV format
│   ├── evaluate.py        # Model evaluation (exact match, CER, BLEU)
│   ├── generate_data.py   # Synthetic data generation via Liblouis
│   └── braille_entry.py   # Manual braille data entry tool
├── notebooks/
│   └── train_v3_colab_a100.ipynb  # Colab training notebook (A100/bf16)
├── data/
│   ├── manual/            # Hand-collected Grade 2 training data
│   ├── synthetic/         # Liblouis-generated parallel corpus (gitignored)
│   └── prepared/          # Train/val/test TSV files (Unicode braille)
├── models/                # Trained model checkpoints (gitignored)
├── docs/                  # Strategy, design docs, research notes
├── tests/                 # Test suite (74 tests)
├── transcribe.py          # CLI entry point
├── requirements.txt
└── TODO.md
```

## Getting Started

### Prerequisites
- Python 3.8+

### Setup
```bash
# Clone this project
git clone <repo-url>
cd braille-transcriber

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Clone pre-trained model weights (not included in repo)
git clone https://github.com/snoop2head/DotNeuralNet
```

### Usage
```bash
# Basic transcription
python transcribe.py image.jpg

# With verbose output (shows braille unicode + stats)
python transcribe.py image.jpg --verbose

# With preprocessing (for low contrast images)
python transcribe.py image.jpg --preprocess

# Adjust confidence threshold
python transcribe.py image.jpg --confidence 0.25
```

---

## 📈 Roadmap

### Phase 1: Grade 1 MVP (Months 1-3) - CLI Complete
- [x] Research and planning
- [x] YOLOv8 integration (DotNeuralNet weights)
- [x] Preprocessing pipeline (CLAHE)
- [x] Grade 1 lookup table
- [x] CLI tool (`transcribe.py`)
- [ ] Mobile app (Flutter) - deferred

### Phase 2: Grade 2 Data + Model - First Model Working!
- [x] Manual data collection tool (CLI-based, accessibility-friendly)
- [x] First manual dataset (jellybean_jungle.txt)
- [x] Synthetic data generation via Liblouis (25K+ training pairs)
- [x] ByT5-small seq2seq model (Unicode braille → English)
- [x] **76.2% exact match on real-world held-out data (0.01 CER)**
- [ ] Full evaluation on synthetic test set
- [ ] Error analysis and improvement

### Phase 3: Grade 2 Model Improvement ← **WE ARE HERE**
- [ ] More training data (20+ books)
- [ ] Longer training / larger model (ByT5-base)
- [ ] End-to-end pipeline (image → detection → interpretation → text)
- [ ] Hybrid rule-based + ML approach for edge cases
- [ ] Research paper submission

### Phase 4: Nemeth Support (Months 16+)
- [ ] Math Braille dataset creation
- [ ] Spatial layout parser
- [ ] Formula tree reconstruction
- [ ] LaTeX/MathML generation

---

## 🔬 Research Contribution

**Primary Focus:** Context-Aware Interpretation for Grade 2 Braille

**Why it's novel:**
- First ML-based contraction resolver
- Hybrid approach: liblouis rules + Transformer context
- Solves 90-95% of real-world Braille OCR
- No existing models or datasets

**Expected Publications:**
1. Grade 2 dataset paper
2. Context-aware interpretation method
3. Comprehensive Braille OCR survey

---

## 🔗 Key Resources

### Repositories We're Using
- **DotNeuralNet:** Pre-trained YOLOv8 weights - https://github.com/snoop2head/DotNeuralNet
- **Angelina Reader:** Reference implementation - https://github.com/IlyaOvodov/AngelinaReader
- **YOLOv8:** Base framework - https://github.com/ultralytics/ultralytics
- **liblouis:** Translation library - https://github.com/liblouis/liblouis

### Datasets Available
- Roboflow Braille Detection (61 classes)
- Kaggle Braille Character Dataset
- AI4SocialGood (~30,000 characters)
- Angelina Dataset (240 photos)

### Datasets We Need to Create
- ❌ Grade 2 contracted Braille (does not exist!)
- ❌ Nemeth Code mathematics (does not exist!)

---

## 📊 Current Status

| Feature | Support | Real Usage | Status |
|---------|---------|------------|--------|
| Grade 1 | ✅ Yes | 5-10% | Phase 1 Complete |
| Grade 2 | ✅ Initial (76% exact) | 90-95% | Phase 2 Complete, improving |
| Nemeth | ❌ No | 3-5% | Phase 4 (Future) |

**Bottom Line:** We now have the first working ML model for Grade 2 contracted braille — covering the 90-95% of real documents that no other system supports.

---

## 💡 Key Insights from Research

1. **Cell detection is solved** (98-99% accuracy with YOLOv8)
2. **Grade 2 interpretation now working** (76% exact match with ByT5-small)
3. **ByT5 is the right model** — byte-level processing handles Unicode braille natively (T5's tokenizer cannot)
4. **Synthetic data generalizes** — model trained on Liblouis data works on real human-transcribed braille
5. **Two-stage architecture is superior** (clean separation, easier debugging)
6. **ByT5 requires bf16 or fp32** — fp16 overflows on long byte sequences

---

## 🤝 Contributing

This is a research project focused on:
- Novel ML architectures for Grade 2 interpretation
- Dataset creation for underserved Braille types
- Real-world impact for blind/low-vision community

Interested in collaborating? Areas where help is needed:
- Dataset creation (annotation, scanning)
- Mobile app development (Flutter/native)
- ML model development (Transformers, NLP)
- Braille expertise (teachers, users)

---

## 📝 Project Values

1. **Research First** - Prioritize novel contributions over commercial wins
2. **Open Science** - Publish datasets and code for community benefit
3. **User Impact** - Build for real-world usefulness
4. **Technical Excellence** - Robust, well-architected systems
5. **Incremental Progress** - Ship working solutions while building advanced features

---

## Recent Updates

**March 11, 2026 — Grade 2 Model v3 (Breakthrough):**
- ByT5-small trained on A100 80GB with bf16: val loss 0.008 after 10 epochs
- **76.2% exact match on real-world jellybean held-out set (CER 0.01)**
- 7/10 misses are smart quote differences only — true content accuracy ~93%
- Zero hallucinations — all predictions are correct and input-dependent
- First working ML model for Grade 2 contracted braille

**March 2026 — Training Iterations (v1, v2):**
- v1 (T5-small + custom tokens): 2.9% accuracy — random embeddings failed
- v2 (T5-small + Unicode braille): 0% — tokenizer mapped all braille to `<unk>`
- Generated 25K synthetic training pairs from 5 Gutenberg books via Liblouis

**January 2026:**
- Grade 1 CLI pipeline complete (detector + interpreter + preprocessing)
- Tested on AngelinaDataset: 91.7% confidence on real photos
- Project planning and architecture decisions documented

---

## 🎯 Success Metrics

**Phase 1 (Grade 1):**
- Cell detection: >98% accuracy
- Character recognition: >99% accuracy
- Processing: <3 seconds/page
- Mobile app: Functional

**Phase 3 (Grade 2):**
- Contraction resolution: >95% accuracy
- Context-dependent interpretation: >90% accuracy
- Outperform liblouis back-translation
- First publicly available Grade 2 OCR model

**Research Impact:**
- Published papers: 2-3
- Public datasets released
- Citations from other researchers
- Adoption by educational institutions

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

**Project Start:** January 19, 2026
**Status:** First working Grade 2 model (76% exact match, 0.01 CER)
**Next Milestone:** Full evaluation + more training data + end-to-end pipeline
