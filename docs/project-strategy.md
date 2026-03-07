# Braille OCR Transcriber - Project Strategy & Decisions

**Project Start Date:** January 19, 2026  
**Focus:** Research-driven approach to building superior Braille OCR models  
**Secondary Goal:** Commercial viability and real-world impact

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Key Decisions](#key-decisions)
3. [Technology Stack](#technology-stack)
4. [Architecture](#architecture)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Research Findings](#research-findings)
7. [Critical Limitations Identified](#critical-limitations-identified)
8. [Resources & Links](#resources--links)
9. [Market Analysis](#market-analysis)
10. [Future Research Directions](#future-research-directions)

---

## Executive Summary

### Project Goal
Build a state-of-the-art Braille OCR system using a two-stage architecture:
- **Stage 1:** Image → Binary Cell Structure (computer vision)
- **Stage 2:** Cell Structure → Text Interpretation (linguistic rules + ML)

### Research Focus
The primary research contribution will be in **Stage 2: Context-Aware Interpretation**, particularly:
- Grade 2 (contracted Braille) contraction resolution
- Nemeth Code (mathematical Braille) spatial parsing
- Hybrid rule-based + ML approach

### Why Two-Stage?
- Braille cells are **well-defined binary structures** (2^6 = 64 combinations)
- Clean separation: Computer Vision (Stage 1) vs. Linguistic Interpretation (Stage 2)
- Stage 1 already solved (98-99% accuracy with existing models)
- **Stage 2 is the unsolved research problem**

---

## Key Decisions

### 1. Architecture Decision: Two-Stage Approach ✅

**Decision:** Implement clear separation between cell detection and interpretation

**Rationale:**
- Braille dots are binary (present/absent) - fundamentally different from handwriting OCR
- Existing models already achieve 98-99% cell detection accuracy
- Real challenge is **context-dependent interpretation**, not dot detection
- Allows independent improvement of each stage
- Easier debugging and testing
- Can update interpretation rules without retraining vision model

**Stage 1:** Use existing SOTA (YOLOv8/Angelina Reader)  
**Stage 2:** Build novel context-aware interpreter (OUR RESEARCH CONTRIBUTION)

---

### 2. Starting Point: Grade 1 with Path to Grade 2 ✅

**Phase 1 (Months 1-3): Launch Grade 1**
- Use existing pre-trained models (YOLOv8, Angelina Reader)
- Support uncontracted Braille only
- Simple lookup table for Stage 2
- Target: Braille beginners, computer Braille users

**Phase 2 (Months 4-9): Grade 2 Dataset Creation**
- Scan Grade 2 Braille books
- Manual annotation of contractions in context
- Build training dataset for context-aware resolver
- Partner with blind schools for data access

**Phase 3 (Months 10-15): Grade 2 Model Development**
- Implement context-aware contraction resolver
- Hybrid approach: Rule-based (liblouis) + ML (Transformer)
- **This is the primary research contribution**
- Target: Educational institutions, mainstream Braille users

**Phase 4 (Months 16+): Nemeth Code Support**
- Spatial layout parsing for mathematical notation
- 2D structure reconstruction
- LaTeX/MathML output

---

### 3. Model Selection: YOLOv8 as Starting Point ✅

**Chosen Model:** YOLOv8 for Stage 1 (cell detection)

**Why YOLOv8:**
- State-of-the-art performance (mAP50: 0.98)
- Real-time processing capability
- Mobile-friendly (TensorFlow Lite, CoreML export)
- Excellent documentation and active community
- Pre-trained weights available (DotNeuralNet)

**Alternatives Considered:**
- Angelina Reader (good but less mobile-friendly)
- Custom CNN (unnecessary - YOLOv8 already excellent)
- Semantic segmentation approaches (overkill for binary dots)

---

### 4. Dataset Strategy ✅

**For Stage 1 (Cell Detection):**
Use existing datasets:
- DotNeuralNet combined dataset (6,013+ images)
- Roboflow Braille Detection (61 classes)
- DSBI (double-sided Braille images)
- Angelina Dataset (240 photos)

**For Stage 2 (Grade 2 Interpretation):**
**Must create custom dataset** - none exists!
- Scan Grade 2 books (target: 1,000+ pages)
- Annotate each cell with:
  - Dot pattern
  - Contextual interpretation
  - Contraction type
  - Position in word
- Estimated effort: 6-12 months
- Estimated cost: $20,000-50,000 (if outsourcing annotation)

---

### 5. Technology Not Supported Initially ❌

**Nemeth Code (Mathematics):**
- No existing datasets
- No current models support it
- Complex 2D spatial structure
- Will be Phase 4 (future work)

**Grade 2 Contractions:**
- No existing models support it
- 90-95% of real-world Braille documents
- Will be Phase 3 (primary research contribution)

---

## Technology Stack

### Stage 1: Cell Detection

**Model Framework:**
```bash
# Primary
YOLOv8 (Ultralytics)
- Model: yolov8m.pt or yolov8_braille.pt (pre-trained)
- Export: TensorFlow Lite, CoreML, ONNX

# Backup/Comparison
Angelina Reader (PyTorch)
```

**Installation:**
```bash
pip install ultralytics opencv-python pillow
pip install torch torchvision  # For Angelina Reader
```

**Image Processing:**
```bash
pip install opencv-python
# For: edge detection, perspective correction, CLAHE, noise removal
```

---

### Stage 2: Interpretation

**For Grade 1 (Simple):**
```python
# Simple lookup dictionary - no external dependencies
GRADE_1_MAP = {64 braille patterns}
```

**For Grade 2 (Research):**
```bash
# Rule-based component
pip install louis  # liblouis Python bindings
# Translation tables: en-ueb-g2.ctb

# ML component (to be developed)
pip install transformers torch
# Transformer-based context encoder
```

**For Nemeth (Future):**
```bash
pip install louis  # Nemeth tables
pip install sympy  # For LaTeX generation
# Custom spatial parser (to be developed)
```

---

### Mobile Development

**Cross-Platform:**
```bash
# Flutter (recommended for MVP)
flutter create braille_ocr

# Dependencies
- tflite_flutter (for TensorFlow Lite models)
- camera plugin
- image processing
```

**Native (Higher Performance):**
```swift
// iOS: Swift + CoreML
import CoreML
import Vision

// Android: Kotlin + TensorFlow Lite
import org.tensorflow.lite.Interpreter
```

---

## Architecture

### Two-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE                          │
│                  (Photo of Braille page)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      PREPROCESSING                          │
│  • Edge detection (Canny)                                   │
│  • Perspective correction                                   │
│  • Lighting normalization (CLAHE)                           │
│  • Noise reduction                                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 1: CELL DETECTION                        │
│                                                             │
│  Model: YOLOv8 / Angelina Reader                           │
│  Input: Preprocessed image                                 │
│  Output: List of cells with:                               │
│    • Position (x, y)                                       │
│    • Binary dot pattern [1,0,1,0,1,0]                     │
│    • Confidence score                                      │
│                                                             │
│  Performance: 98-99% accuracy (SOTA)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 2: INTERPRETATION                        │
│                                                             │
│  ┌──────────────────────────────────────────────┐          │
│  │  Grade 1 (Uncontracted) - SIMPLE             │          │
│  │  • Lookup table: dot pattern → character     │          │
│  │  • O(1) complexity                           │          │
│  └──────────────────────────────────────────────┘          │
│                                                             │
│  ┌──────────────────────────────────────────────┐          │
│  │  Grade 2 (Contracted) - COMPLEX              │          │
│  │  ┌────────────────────────────────────┐      │          │
│  │  │ Rule-Based Component               │      │          │
│  │  │ • liblouis back-translation        │      │          │
│  │  │ • 180+ contraction rules           │      │          │
│  │  │ • Context validation               │      │          │
│  │  └────────────────────────────────────┘      │          │
│  │           ▼                                   │          │
│  │  ┌────────────────────────────────────┐      │          │
│  │  │ ML-Based Component (NOVEL)         │      │          │
│  │  │ • Transformer context encoder      │      │          │
│  │  │ • Contraction classifier           │      │          │
│  │  │ • Handles ambiguous cases          │      │          │
│  │  └────────────────────────────────────┘      │          │
│  │           ▼                                   │          │
│  │  ┌────────────────────────────────────┐      │          │
│  │  │ Hybrid Decision                    │      │          │
│  │  │ • Compare rule vs ML output        │      │          │
│  │  │ • Confidence scoring               │      │          │
│  │  │ • Return best interpretation       │      │          │
│  │  └────────────────────────────────────┘      │          │
│  └──────────────────────────────────────────────┘          │
│                                                             │
│  ┌──────────────────────────────────────────────┐          │
│  │  Nemeth (Mathematics) - FUTURE               │          │
│  │  • Spatial layout parser                     │          │
│  │  • 2D structure reconstruction               │          │
│  │  • Formula tree building                     │          │
│  │  • LaTeX/MathML generation                   │          │
│  └──────────────────────────────────────────────┘          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT TEXT                              │
│               + Confidence Scores                           │
│               + Alternative Interpretations                 │
└─────────────────────────────────────────────────────────────┘
```

---

### Stage 2 Detail: Context-Aware Resolver (Research Contribution)

```python
class HybridBrailleInterpreter:
    """
    Novel approach combining rule-based and ML methods
    for Grade 2 Braille interpretation
    """
    
    def __init__(self):
        # Rule-based component
        self.liblouis = LibLouisBackTranslator('en-ueb-g2.ctb')
        
        # ML component (our research contribution)
        self.context_encoder = TransformerEncoder(
            input_dim=64,  # 64 possible cell patterns
            hidden_dim=128,
            num_layers=6,
            num_heads=8
        )
        self.contraction_classifier = ContractionClassifier()
        
        # Validation
        self.validator = ConsistencyChecker()
    
    def interpret(self, cell_sequence):
        """
        Input: List of cells [(x, y, [dots], conf), ...]
        Output: Text with confidence scores
        """
        
        # Method 1: Rule-based (liblouis)
        rule_result = self.liblouis.back_translate(
            self.cells_to_unicode(cell_sequence)
        )
        rule_confidence = self.validator.check(rule_result, cell_sequence)
        
        # Method 2: ML-based (context-aware)
        # Embed cells and capture context
        embedded = self.context_encoder(cell_sequence)
        ml_result = self.contraction_classifier(embedded)
        ml_confidence = self.compute_confidence(ml_result)
        
        # Hybrid decision
        if rule_confidence > 0.9:
            return rule_result
        elif ml_confidence > rule_confidence:
            return ml_result
        else:
            # Return both with confidence scores
            return {
                'primary': rule_result,
                'alternative': ml_result,
                'confidence': max(rule_confidence, ml_confidence)
            }
```

---

## Implementation Roadmap

### Phase 1: Grade 1 MVP (Months 1-3)

**Goal:** Working app for uncontracted Braille

**Tasks:**
1. **Setup & Infrastructure** (Week 1-2)
   - Clone DotNeuralNet repository
   - Setup development environment
   - Test pre-trained YOLOv8 model
   - Verify model accuracy on test images

2. **Stage 1 Implementation** (Week 3-4)
   - Integrate YOLOv8 model
   - Implement preprocessing pipeline:
     - Edge detection (Canny algorithm)
     - Perspective correction
     - CLAHE lighting normalization
     - Noise reduction
   - Cell detection with confidence thresholding (conf=0.15)
   - Output: Binary dot patterns

3. **Stage 2 Implementation** (Week 5-6)
   - Create Grade 1 lookup table (64 patterns)
   - Implement simple cell → character mapping
   - Handle special cases:
     - Capital indicator
     - Number indicator
     - Punctuation
   - Text assembly from detected cells

4. **Mobile App Development** (Week 7-10)
   - Flutter app setup
   - Camera interface with auto-capture
   - Document detection (OpenCV)
   - Model integration (TensorFlow Lite)
   - Real-time preview
   - Confidence visualization

5. **Testing & Refinement** (Week 11-12)
   - Test on diverse images (lighting, angles, quality)
   - Measure accuracy on benchmark datasets
   - Optimize performance (speed, memory)
   - User testing with Braille users
   - Bug fixes

**Deliverable:** Mobile app supporting Grade 1 Braille OCR

---

### Phase 2: Grade 2 Dataset Creation (Months 4-9)

**Goal:** High-quality annotated dataset for contracted Braille

**Tasks:**
1. **Data Acquisition** (Month 4-5)
   - Partner with blind schools/libraries
   - Scan Grade 2 Braille books (target: 1,000+ pages)
   - Collect diverse sources:
     - Children's books
     - Textbooks
     - Literature
     - Various embossers/quality levels
   - Ensure coverage of all 180+ contractions

2. **Annotation Pipeline** (Month 5-7)
   - Build annotation tool:
     - Display scanned image
     - Show detected cells (from Stage 1)
     - Allow annotator to mark:
       - Cell interpretation
       - Contraction type
       - Position in word (start/middle/end)
       - Context dependencies
   - Hire annotators (ideally Braille experts)
   - Quality control process

3. **Dataset Validation** (Month 8)
   - Inter-annotator agreement testing
   - Cross-validation with liblouis output
   - Error analysis and correction
   - Statistical analysis of contraction usage

4. **Dataset Release** (Month 9)
   - Documentation
   - Publish to research community
   - Paper submission to dataset track

**Deliverable:** Annotated Grade 2 Braille dataset (public release)

---

### Phase 3: Grade 2 Model Development (Months 10-15)

**Goal:** Context-aware contraction resolver

**Tasks:**
1. **Model Architecture** (Month 10-11)
   - Design Transformer-based context encoder
   - Implement contraction classifier
   - Build hybrid decision module
   - Integration with liblouis

2. **Training** (Month 12-13)
   - Train context encoder on dataset
   - Fine-tune classification head
   - Hyperparameter optimization
   - Augmentation strategies

3. **Evaluation** (Month 13-14)
   - Benchmark against liblouis alone
   - Measure accuracy on test set
   - Error analysis (which contractions fail?)
   - Ablation studies (rule vs ML vs hybrid)

4. **Integration & Deployment** (Month 14-15)
   - Integrate into mobile app
   - Model optimization (quantization, pruning)
   - A/B testing with users
   - Performance tuning

**Deliverable:** Grade 2 support in app + research paper

---

### Phase 4: Nemeth Code Support (Months 16+)

**Goal:** Mathematical Braille OCR

**Tasks:**
1. **Dataset Creation**
   - Scan math textbooks in Braille
   - Annotate formulas with LaTeX ground truth
   - Include spatial layout information

2. **Spatial Parser Development**
   - 2D layout reconstruction
   - Formula tree building
   - Integration with liblouis Nemeth tables

3. **LaTeX/MathML Generation**
   - Convert formula trees to LaTeX
   - Validate mathematical correctness
   - Rendering verification

**Deliverable:** Complete Braille OCR system (literary + math)

---

## Research Findings

### Current State of Braille OCR

#### What Works (Stage 1: Cell Detection)
- **YOLOv8 approach:** mAP50 = 0.98 (98% detection accuracy)
- **Fly-LeNet:** 99.77% and 99.80% accuracy on two datasets
- **Angelina Reader:** 99%+ on clean images
- **Arabic Braille:** 99% accuracy, 32.6 seconds/page

**Conclusion:** Cell detection is a **solved problem** with existing SOTA models

---

#### What Doesn't Work (Stage 2: Interpretation)

**Grade 1 Support:**
- ✅ All models support uncontracted Braille
- ✅ Simple lookup table (64 patterns)
- ✅ Works perfectly

**Grade 2 Support:**
- ❌ **NO existing models support contracted Braille**
- ❌ Liblouis back-translation has errors
- ❌ No datasets exist
- 📊 **But 90-95% of real Braille is Grade 2!**

**Nemeth Support:**
- ❌ **NO existing models support math Braille**
- ❌ No datasets exist
- ✅ liblouis has Nemeth tables (text → Braille)
- ❌ But no image → Braille for Nemeth

---

### Market Gap Analysis

| Content Type | % of Real Usage | Current Support | Opportunity |
|--------------|----------------|-----------------|-------------|
| Grade 1 (Uncontracted) | 5-10% | ✅ Fully supported | Low |
| Grade 2 (Contracted) | 90-95% | ❌ Not supported | **HUGE** |
| Nemeth (Math) | 3-5% | ❌ Not supported | **HIGH** |
| Music Braille | <1% | ❌ Not supported | Low |

**Key Insight:** Current models only serve 5-10% of real-world use cases!

---

### Technical Challenges Identified

#### For Grade 2 Contractions:

1. **Context-Dependent Interpretation**
   - Same cell = different meanings based on position/context
   - Example: "⠃" = "b" (letter) OR "but" (word) OR part of "bl" (contraction)

2. **180+ Contraction Rules**
   - Complex positional rules (start/middle/end of word)
   - Cannot use "can" contraction inside "Duncan"
   - Pronunciation-based rules ("th" in "think" ✓, not in "pothole" ✗)

3. **No Training Data**
   - All existing datasets are Grade 1 only
   - Need to create custom annotated dataset
   - Estimated 10,000+ annotated images needed

#### For Nemeth Code:

1. **2D Spatial Structure**
   - Fractions require vertical layout parsing
   - Matrices, summations have complex 2D structure
   - Cannot be solved with 1D sequence processing

2. **Context-Dependent Symbols**
   - Same symbol = different meaning in different contexts
   - Requires mathematical syntax understanding

3. **No Existing Datasets**
   - Must create from scratch
   - Requires math expertise for annotation

---

### User Pain Points (from Research)

**From app store reviews of existing apps:**
- "Eight of us scanned 28 documents multiple times...not one word was translated"
- "The app cannot read well embossed paper from a photo"
- "Horror end user experience rife with multiple annoying ads"
- Apps take excessive time or fail completely

**Primary Use Case:**
- Teachers/parents checking student homework (can't read Braille themselves)
- **Problem:** Most homework is in Grade 2, which no app supports!

**Market Opportunity:**
- Disconnect between academic research (95-99% accuracy) and commercial apps (often complete failures)
- Users desperately need reliable Braille OCR

---

## Critical Limitations Identified

### 1. Grade 2 (Contracted Braille) - Major Gap

**Status:** ❌ Not supported by any existing model

**Impact:** 
- 90-95% of real Braille documents cannot be read
- Primary use case (homework checking) broken
- Books, magazines all use Grade 2

**Why it matters:**
- Almost all books/magazines: <cite>"Almost all books and magazines are printed in contracted braille"</cite>
- Universal standard for fluent readers
- Grade 1 only used by beginners (first few months)

**Example of failure:**
```
Student writes: "The children will go shopping" (Grade 2)
Current models read: "t ch w g shop" (interpret contractions as letters)
Result: COMPLETELY WRONG ❌
```

**Solution:** Our Stage 2 research contribution

---

### 2. Nemeth Code (Mathematics) - Not Supported

**Status:** ❌ No datasets, no models

**What's needed:**
- Specialized symbols (∞, ∑, ∫, etc.)
- 2D spatial layout parsing
- Formula tree reconstruction
- LaTeX/MathML generation

**Market impact:**
- STEM education for blind students underserved
- Math teachers can't grade homework
- Scientific document digitization impossible

**Solution:** Future work (Phase 4)

---

### 3. Translation Tools Exist, but Only One Direction

**Text → Braille (Works):**
- ✅ liblouis supports Grade 2, Nemeth, etc.
- ✅ Used in screen readers (NVDA, JAWS)
- ✅ MathJax + SRE for Nemeth generation

**Braille Images → Text (Broken):**
- ❌ No OCR for Grade 2
- ❌ No OCR for Nemeth
- ❌ liblouis back-translation has errors

**The Gap:** Can generate Braille, but can't read it from images!

---

## Resources & Links

### Code Repositories

#### Primary Resources (Use These)

**DotNeuralNet** (START HERE - has pre-trained weights)
- GitHub: https://github.com/snoop2head/DotNeuralNet
- Contains: yolov8_braille.pt and yolov5_braille.pt
- Dataset: Combined AngelinaDataset, DSBI, braille_natural
- 6,013+ annotated images

**Angelina Reader** (Reference implementation)
- GitHub: https://github.com/IlyaOvodov/AngelinaReader
- Web service: http://angelina-reader.ru
- Created by neuroscientist for his blind daughter
- Handles double-sided Braille

**Ultralytics YOLOv8** (Base framework)
- GitHub: https://github.com/ultralytics/ultralytics
- Docs: https://docs.ultralytics.com
- Best documentation and community support

**liblouis** (Braille translation library)
- GitHub: https://github.com/liblouis/liblouis
- Supports: Grade 2, Nemeth, multiple languages
- Used in: NVDA, JAWS, Orca screen readers

---

### Datasets

#### Available Now (Grade 1 Only)

**Roboflow Braille Detection**
- URL: https://universe.roboflow.com/braille-lq5eh/braille-detection
- Content: 61 classes (A-Z, numbers, symbols)
- Format: YOLO-ready

**Kaggle Braille Character Dataset**
- URL: https://www.kaggle.com/datasets/shanks0465/braille-character-dataset
- Content: Alphabet characters

**AI4SocialGood Dataset**
- GitHub: https://github.com/HelenGezahegn/aeye-alliance
- Content: ~30,000 Braille characters
- Includes: Letters, numbers, punctuation

**Angelina Braille Images Dataset**
- GitHub: https://github.com/IlyaOvodov/AngelinaDataset
- Content: 240 annotated photos
- Types: Book backgrounds, double-sided, natural scenes

**DSBI (Double-Sided Braille Images)**
- Paper: https://dl.acm.org/doi/10.1145/3301506.3301532
- Focus: Double-sided Braille detection challenges

---

#### To Be Created (Grade 2, Nemeth)

**Grade 2 Contracted Braille Dataset**
- Status: MUST CREATE - does not exist
- Required: 1,000+ scanned pages with annotations
- Annotations needed:
  - Cell dot patterns
  - Contextual interpretation
  - Contraction type
  - Position markers

**Nemeth Mathematics Dataset**
- Status: MUST CREATE - does not exist
- Required: Math textbooks in Braille
- Annotations needed:
  - Spatial layout
  - Formula trees
  - LaTeX ground truth

---

### Research Papers (Key References)

**YOLOv8 for Braille Detection**
- "Optical Braille Recognition Using Object Detection Neural Network"
- Ilya G. Ovodov, 2021
- ArXiv: https://arxiv.org/abs/2012.12412

**Fly-LeNet Approach**
- "Fly-LeNet: A deep learning-based framework for converting multilingual braille images"
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10882029/
- Accuracy: 99.77% and 99.80%

**DSBI Dataset Paper**
- "DSBI: Double-Sided Braille Image Dataset and Algorithm Evaluation"
- ACM DL: https://dl.acm.org/doi/10.1145/3301506.3301532

**Automated Nemeth Translation**
- "Math that Feels Good: Enabling Sighted and Blind People to Share the Mathematical Experience"
- NFB: https://nfb.org/images/nfb/publications/bm/bm20/bm2005/bm200507.htm
- PreTeXt + liblouis + MathJax approach

---

### Documentation & Standards

**Nemeth Code 2022 Revision**
- PDF: https://www.brailleauthority.org/sites/default/files/2024-02/Nemeth_2022.pdf
- Authority: Braille Authority of North America (BANA)

**UEB (Unified English Braille) Specifications**
- Website: https://www.brailleauthority.org/
- Includes: Grade 2 contraction rules

**Braille Code Standards**
- BANA: https://www.brailleauthority.org/
- International Council on English Braille specifications

---

### Tools & Libraries

**Python Libraries:**
```bash
# Computer Vision
pip install ultralytics          # YOLOv8
pip install opencv-python        # Image processing
pip install torch torchvision    # PyTorch

# Braille Translation
pip install louis               # liblouis bindings

# ML/NLP
pip install transformers        # For context encoder
pip install numpy pandas        # Data processing
```

**Mobile Development:**
```bash
# Flutter
flutter pub add tflite_flutter
flutter pub add camera
flutter pub add image_picker

# iOS Native
# Use CoreML framework

# Android Native  
# Use TensorFlow Lite
```

---

### Educational Resources

**Nemeth Tutorial**
- Interactive: https://nemeth.aphtech.org/
- Learn Nemeth Code symbols and rules

**UEB Online Training**
- Website: https://uebonline.org/
- Grade 2 contractions reference

**Braille Brain (APH)**
- URL: https://braillebrain.aphtech.org/
- Interactive Braille learning

---

## Market Analysis

### Target Users

**Primary (Phase 1-2):**
1. **Teachers of blind students**
   - Cannot read Braille themselves
   - Need to check homework
   - Currently: ~50,000 monthly users on BrailleTranslators.com

2. **Parents of blind children**
   - Want to support learning
   - Need homework checking
   - Communication with child

3. **Braille learners (beginners)**
   - First 3-6 months (Grade 1 only)
   - Educational institutions
   - Self-learners

**Secondary (Phase 3+):**
1. **Educational institutions**
   - Schools for the blind
   - Mainstream schools with blind students
   - Will pay premium for Grade 2/Nemeth support

2. **Libraries and archives**
   - Document preservation
   - Digitization projects
   - Historical Braille documents

3. **Publishers**
   - Converting Braille books to digital
   - Quality control for Braille production

---

### Competitive Analysis

**Existing Solutions:**

1. **Braille Scanner (iOS)**
   - Free, on-device ML
   - Supports: UEB Grade 1 only
   - Limitations: No Grade 2, no Nemeth

2. **Angelina Reader (Web)**
   - Free service
   - Supports: Grade 1, double-sided
   - Limitations: No Grade 2, no Nemeth, web-only

3. **Google Play "Braille Recognition" apps**
   - User reviews: Mostly failures
   - "Not one word was translated" (common complaint)
   - Poor quality control

**Market Gap:**
- NO app supports Grade 2 (90-95% of real Braille)
- NO app supports Nemeth (math/science)
- Huge disconnect: Academic research (99% accuracy) vs. Commercial apps (often fail)

**Our Competitive Advantage:**
- First to support Grade 2 contractions
- Research-driven approach (higher quality)
- Two-stage architecture (better debugging/accuracy)
- Eventual Nemeth support (STEM education market)

---

### Market Size

**Global Statistics:**
- 285 million visually impaired people worldwide (WHO)
- ~39 million blind, ~246 million low vision
- Braille literacy rates vary: 4% (UK) to higher in developing countries

**US Market:**
- ~10 million blind/low vision
- ~57,000 students in special education for blindness
- Federal Quota funding for educational materials

**Pain Points:**
- Braille books expensive ($100-300 each)
- Limited availability
- Teachers often can't read Braille
- Parents need tools to support children

**Willingness to Pay:**
- Educational institutions: High (federal quota funds)
- Individual users: Moderate ($5-20/month subscription)
- Premium features (Grade 2, Nemeth): Higher tier ($20-50/month)

---

### User Feedback (from existing tools)

**Positive Reception:**
- "AI tools provide independence - feel like less of a burden"
- Meta AI Glasses recognized by National Federation of the Blind
- BrailleTranslators.com: 50,000+ monthly users

**Critical Pain Points:**
- Existing apps have terrible accuracy
- Time-consuming (some take minutes per page)
- Ads and poor UX
- Only work with perfect images
- Don't support Grade 2 (what users actually need!)

**Unmet Needs:**
1. Reliable Grade 2 recognition
2. Fast processing (<3 seconds/page)
3. Works with imperfect images
4. Offline capability
5. Math support (Nemeth)
6. Batch processing

---

## Future Research Directions

### Short-term (Next 6 months)

1. **Optimize Stage 1 for Mobile**
   - Model quantization
   - Reduce inference time
   - Memory optimization
   - Battery efficiency

2. **Improve Preprocessing**
   - Better perspective correction
   - Adaptive lighting normalization
   - Handle worn/damaged Braille
   - Double-sided interference reduction

3. **Build Grade 2 Dataset**
   - Partnership development
   - Annotation pipeline
   - Quality control processes

---

### Medium-term (6-18 months)

1. **Context-Aware Resolver (Main Research)**
   - Transformer architecture design
   - Training methodology
   - Hybrid rule-based + ML approach
   - Benchmark against liblouis

2. **Multi-language Support**
   - Spanish Braille
   - French Braille
   - German Braille (complex Grade 2 rules)
   - Transfer learning approach

3. **Real-time Processing**
   - Video stream processing
   - Live feedback during scanning
   - Guide user to optimal angle/lighting

---

### Long-term (18+ months)

1. **Nemeth Code Support**
   - Spatial layout parsing
   - Formula tree reconstruction
   - LaTeX/MathML generation
   - Integration with math rendering

2. **Advanced Features**
   - Music Braille recognition
   - Chemical notation (chemistry Nemeth)
   - Tactile graphics interpretation
   - Multi-modal output (TTS + Braille display)

3. **Research Publications**
   - Grade 2 interpretation paper
   - Dataset release paper
   - Nemeth spatial parsing paper
   - Survey paper on Braille OCR

---

### Open Research Questions

1. **How to best combine rule-based and ML approaches for Grade 2?**
   - Pure ML might not learn all 180+ rules correctly
   - Pure rules (liblouis) have errors
   - Optimal hybrid strategy?

2. **Can we use semi-supervised learning for dataset creation?**
   - Use liblouis to generate pseudo-labels
   - Correct with human annotations
   - Active learning for ambiguous cases?

3. **Transfer learning from text NLP models?**
   - Can BERT/GPT-style pre-training help?
   - Braille as a "language" with contractions
   - Few-shot learning for rare contractions?

4. **How to handle double-sided Braille interference?**
   - Dots from reverse side appear in image
   - Current solutions: separate models for recto/verso
   - Can single model learn to separate?

5. **Optimal architecture for spatial (2D) Braille?**
   - Nemeth formulas have tree structure
   - Graph neural networks?
   - Recursive parsing?

---

## Success Metrics

### Technical Metrics

**Stage 1 (Cell Detection):**
- Cell detection accuracy: >98%
- False positive rate: <2%
- Processing speed: <3 seconds/page
- Works in varied lighting: >90% success rate

**Stage 2 (Grade 1):**
- Character recognition: >99%
- Full page accuracy: >95%
- Handles punctuation/numbers: >97%

**Stage 2 (Grade 2 - Future):**
- Contraction recognition: >95%
- Context-dependent accuracy: >90%
- Outperforms liblouis back-translation: Yes

---

### Research Metrics

- Papers published: 2-3 (dataset, Grade 2 method, survey)
- Dataset citations: Track usage by other researchers
- Model downloads: Public release on GitHub
- Benchmark performance: SOTA on Grade 2 (first!)

---

### Product Metrics (Secondary)

- User accuracy satisfaction: >4.5/5 stars
- Processing speed: <3 seconds/page
- Works offline: Yes
- Daily active users: Track growth
- Retention rate: >60% monthly

---

## Project Values

1. **Research First:** Prioritize novel contributions over quick commercial wins
2. **Open Science:** Publish datasets and code for community benefit
3. **User Impact:** Real-world usefulness for blind/low-vision community
4. **Technical Excellence:** Build robust, well-architected systems
5. **Incremental Progress:** Ship working solutions while building advanced features

---

---

**Last Updated:** March 2026
**Status:** Phase 1 Complete (CLI) → Grade 2 Data Collection
