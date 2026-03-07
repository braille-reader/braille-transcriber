# Braille OCR Transcriber

> A research-driven approach to building superior Braille OCR models with support for contracted Braille (Grade 2) and mathematical notation (Nemeth Code)

**Status:** Phase 1 Complete (CLI) → Grade 2 Data Collection
**Started:** January 19, 2026
**Current Focus:** Synthetic data generation for Grade 2

---

## 🎯 Project Vision

Build the first Braille OCR system that supports:
- ✅ Grade 1 (Uncontracted Braille) - **Phase 1**
- 🔬 Grade 2 (Contracted Braille) - **Primary Research Contribution**
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
- **[Implementation Log](docs/implementation-log.md)** - Development session notes and results
- **[Braille Codes Reference](docs/braille-codes-ref.md)** - Complete dot pattern to code mapping
- **[Data Collection Research](docs/collecting-data-research.md)** - Approaches for building training data
- **[Annotation App Design](docs/annotation-app.md)** - Data collection tool design

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
│   ├── detector.py        # YOLOv8 cell detection wrapper
│   ├── interpreter.py     # Grade 1 Braille → English lookup
│   ├── pipeline.py        # End-to-end pipeline
│   └── preprocess.py      # CLAHE preprocessing
├── data/
│   ├── manual/            # Hand-collected Grade 2 training data
│   └── synthetic/         # Liblouis-generated parallel corpus (gitignored)
├── docs/                  # Strategy, design docs, research notes
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

### Phase 2: Grade 2 Data + Model ← **WE ARE HERE**
- [x] Manual data collection tool (CLI-based, accessibility-friendly)
- [x] First manual dataset (jellybean_jungle.txt)
- [ ] Synthetic data generation via Liblouis
- [ ] Seq2Seq translation model (dot patterns → English)
- [ ] Evaluation against manual test data

### Phase 3: Grade 2 Model (Months 10-15) ⭐ **MAIN CONTRIBUTION**
- [ ] Context-aware resolver architecture
- [ ] Transformer-based context encoder
- [ ] Hybrid rule-based + ML approach
- [ ] Training and evaluation
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

## 📊 Current Limitations

| Feature | Support | Real Usage | Status |
|---------|---------|------------|--------|
| Grade 1 | ✅ Yes | 5-10% | Phase 1 |
| Grade 2 | ❌ No | 90-95% | Phase 3 (Research) |
| Nemeth | ❌ No | 3-5% | Phase 4 |

**Bottom Line:** Existing models only handle 5-10% of real Braille documents. We're building support for the other 90-95%.

---

## 💡 Key Insights from Research

1. **Cell detection is solved** (98-99% accuracy with YOLOv8)
2. **Context-dependent interpretation is not** (no Grade 2 support)
3. **Two-stage architecture is superior** (clean separation, easier debugging)
4. **Braille dots are binary** (simpler than handwriting recognition)
5. **Real challenge is linguistics, not vision**

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

**March 2026:**
- Researching synthetic data generation via Liblouis for Grade 2 training
- Collecting manual Grade 2 data with visually impaired partner

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
**Status:** Grade 2 Data Collection + Model Development
**Next Milestone:** Synthetic data pipeline + Seq2Seq model
