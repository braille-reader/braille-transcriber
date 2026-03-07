# Braille Annotation App - Design Document

**Created:** January 24, 2026
**Purpose:** Collect training data for Grade 2 braille transcription model
**Status:** Design Phase

---

## Overview

A web application to collect paired training data from braille experts:
- **Input:** Braille dot patterns
- **Output:** English text translation

This data will train a sequence-to-sequence model for Stage 2 of the braille OCR pipeline.

---

## Key Design Decisions

### 1. Input Format: Dot Notation (Not Numerical Codes)

**Decision:** Use human-readable dot notation instead of computed numerical codes.

**Format:**
```
Dots per cell separated by commas, cells separated by pipe (|)

Example: 1,2,5 | 1,5 | 1,2,3 | 1,2,3 | 1,3,5
Meaning: h     | e   | l     | l     | o
```

**Rationale:**
- Braille experts think in terms of dots (1-6), not computed numbers
- More intuitive and less error-prone for data entry
- Easier to verify correctness visually
- Standard notation in braille education

**Conversion to numerical code:**
```
dots_to_code("1,2,5") = 1 + 2 + 16 = 19

Formula: code = sum(2^(dot-1) for each dot)
  - Dot 1 = 2^0 = 1
  - Dot 2 = 2^1 = 2
  - Dot 3 = 2^2 = 4
  - Dot 4 = 2^3 = 8
  - Dot 5 = 2^4 = 16
  - Dot 6 = 2^5 = 32
```

### 2. Cell Separator: Pipe Character (|)

**Decision:** Use `|` to separate braille cells.

**Rationale:**
- Visually distinct from dot numbers
- Cannot be confused with dots (which are 1-6)
- Easy to type on standard keyboards
- Clear visual boundary between cells

### 3. Space Handling

**Decision:** Empty cell or `0` represents space.

**Options:**
```
"hello world" can be entered as:
  1,2,5|1,5|1,2,3|1,2,3|1,3,5 | | 2,4,5,6|1,3,5|1,2,3,5|1,2,3|1,4,5
  or
  1,2,5|1,5|1,2,3|1,2,3|1,3,5|0|2,4,5,6|1,3,5|1,2,3,5|1,2,3|1,4,5
```

### 4. Data Storage Format

**Decision:** Store in JSON Lines format (.jsonl)

**Schema:**
```json
{
  "id": "uuid",
  "braille_dots": "1,2,5|1,5|1,2,3|1,2,3|1,3,5",
  "braille_codes": [19, 17, 7, 7, 21],
  "english": "hello",
  "grade": 2,
  "contributor": "expert_id",
  "timestamp": "2026-01-24T10:30:00Z",
  "verified": false,
  "notes": ""
}
```

**Rationale:**
- JSONL is easy to append (one record per line)
- Human readable for debugging
- Easy to convert to other formats (CSV, Parquet)
- Supports metadata fields

---

## App Requirements

### Core Features (MVP)

1. **Two Text Areas**
   - Left: Braille dot notation input
   - Right: English translation input

2. **Live Preview**
   - Show braille unicode characters as user types
   - Validate dot notation syntax
   - Show computed numerical codes

3. **Save Entry**
   - Store to local file or database
   - Auto-generate ID and timestamp

4. **Basic Validation**
   - Check dot values are 1-6 only
   - Check for empty fields
   - Warn on unusual patterns

### Future Features

- Review/edit existing entries
- Bulk import from text files
- Export to various formats
- Contributor accounts
- Verification workflow (second expert verifies)
- Statistics dashboard
- Keyboard shortcuts for common patterns

---

## User Interface Mockup

```
┌─────────────────────────────────────────────────────────────────┐
│                 Braille Training Data Collector                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Braille (dot notation):                                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 1,2,5 | 1,5 | 1,2,3 | 1,2,3 | 1,3,5                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Preview: ⠓⠑⠇⠇⠕                                                 │
│  Codes:   19, 17, 7, 7, 21                                      │
│                                                                 │
│  English translation:                                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ hello                                                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Grade: [Grade 1 ▼]  [Grade 2 ▼]                               │
│                                                                 │
│  Notes (optional):                                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  [ Save Entry ]  [ Clear ]  [ View History ]                    │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│  Entries today: 47  |  Total entries: 1,234                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
Expert Input                    Storage                    Model Training
─────────────                   ───────                    ──────────────

┌──────────────┐
│ Dot Notation │ ──┐
│ 1,2,5|1,5|...│   │
└──────────────┘   │
                   │    ┌─────────────┐     ┌─────────────┐
                   ├───▶│  JSONL File │────▶│ Training    │
                   │    │  or DB      │     │ Pipeline    │
┌──────────────┐   │    └─────────────┘     └─────────────┘
│ English Text │ ──┘           │
│ "hello"      │               │
└──────────────┘               ▼
                        ┌─────────────┐
                        │ Conversion  │
                        │ dots → codes│
                        └─────────────┘
```

---

## Conversion Functions

### Dot Notation → Numerical Code

```python
def dots_to_code(dots_str: str) -> int:
    """Convert dot notation to numerical code.

    Args:
        dots_str: Comma-separated dot numbers, e.g., "1,2,5"

    Returns:
        Integer code (0-63)

    Example:
        dots_to_code("1,2,5") → 19
        dots_to_code("1,3,5") → 21
        dots_to_code("") → 0 (space)
    """
    if not dots_str.strip():
        return 0

    dots = [int(d.strip()) for d in dots_str.split(",")]
    code = sum(2 ** (dot - 1) for dot in dots)
    return code
```

### Full Sequence Conversion

```python
def parse_braille_input(input_str: str) -> list[int]:
    """Parse full braille input to code sequence.

    Args:
        input_str: Pipe-separated cells, e.g., "1,2,5|1,5|1,2,3"

    Returns:
        List of integer codes

    Example:
        parse_braille_input("1,2,5|1,5|1,2,3|1,2,3|1,3,5")
        → [19, 17, 7, 7, 21]
    """
    cells = input_str.split("|")
    return [dots_to_code(cell) for cell in cells]
```

### Code → Braille Unicode

```python
def code_to_unicode(code: int) -> str:
    """Convert numerical code to braille unicode.

    Args:
        code: Integer 0-63

    Returns:
        Single braille unicode character

    Example:
        code_to_unicode(19) → "⠓"
        code_to_unicode(0) → "⠀" (blank braille)
    """
    return chr(0x2800 + code)
```

---

## Technology Considerations

### Option A: Simple Static Web App
- HTML + JavaScript (no backend)
- Store data in localStorage or download as file
- **Pros:** Simple, no hosting needed, works offline
- **Cons:** Limited storage, no collaboration

### Option B: Web App with Backend
- Frontend: React/Vue/Svelte
- Backend: Python (FastAPI) or Node.js
- Database: SQLite or PostgreSQL
- **Pros:** Collaboration, unlimited storage, verification workflow
- **Cons:** More complex, needs hosting

### Option C: Google Sheets Integration
- Simple web form that writes to Google Sheet
- **Pros:** Very simple, built-in collaboration, easy export
- **Cons:** Limited validation, dependent on Google

### Recommendation
Start with **Option A** (simple static app) for initial data collection. Migrate to Option B if we need collaboration features.

---

## Training Data Requirements

### Key Decisions

| Question | Decision |
|----------|----------|
| Sentences vs words? | **Sentences** - provides context for contractions |
| Support Nemeth (math)? | **Yes** - separate category in dataset |
| Audio feedback? | **Default browser behavior** - sufficient for accessibility |

### Why Braille→English Needs Less Data Than Typical NLP

| Factor | Impact |
|--------|--------|
| Same language (not cross-lingual) | Reduces complexity |
| Deterministic mapping | One correct output per input |
| Small input vocabulary (64 symbols) | Easier to learn patterns |
| Rule-based structure | Contractions follow patterns |

### Comparison to Similar Tasks

| Task | Typical Data Size | Notes |
|------|-------------------|-------|
| Machine Translation | 1M-100M pairs | Cross-lingual, complex |
| Low-resource Translation | 10K-100K pairs | Limited language pairs |
| Spelling Correction | 10K-50K pairs | Similar to our task |
| OCR Post-correction | 50K-100K pairs | Similar to our task |

### Dataset Size Tiers

| Tier | Sentences | Purpose | Estimated Accuracy |
|------|-----------|---------|-------------------|
| **MVP** | 5,000-10,000 | Prove concept works | ~80% |
| **Functional** | 25,000-50,000 | Good accuracy on common text | ~90% |
| **Production** | 60,000-100,000 | High accuracy, edge cases | ~95%+ |

### Coverage Requirements

**Grade 2 Contractions (~180 rules):**
```
Each contraction needs multiple contexts:
- 180 contractions × 50-100 examples = 9,000-18,000 sentences
- Common contractions (the, and, for) need more examples
- Rare contractions need fewer
```

**Context-Dependent Patterns (~50 ambiguous cases):**
```
Example: "⠮" (code 46) can mean:
- "the" (standalone word)
- Part of a larger word
- Different meaning in Nemeth mode

Each ambiguous pattern × 100 examples = 5,000 sentences
```

**Sentence Length Distribution:**
```
- Short sentences (3-5 words):   2,000
- Medium sentences (6-15 words): 5,000
- Long sentences (16-30 words):  3,000
```

### Recommended Dataset Composition

| Category | Sentences | Purpose |
|----------|-----------|---------|
| Grade 2 contraction coverage | 15,000 | All 180 contractions in context |
| Ambiguous/context-dependent | 5,000 | Patterns that need context |
| General sentences | 20,000 | Variety and natural language |
| Numbers and punctuation | 5,000 | Dates, times, lists |
| Nemeth (mathematics) | 10,000 | Equations, formulas |
| Edge cases | 5,000 | Unusual patterns, errors |
| **Total** | **~60,000** | Production quality |

### Phased Data Collection Plan

| Phase | Target | Cumulative | Purpose |
|-------|--------|------------|---------|
| Phase 1 (MVP) | 5,000 | 5,000 | Validate approach |
| Phase 2 (Beta) | 20,000 | 25,000 | Usable accuracy |
| Phase 3 (Release) | 35,000 | 60,000 | Production quality |
| Phase 4 (Nemeth) | 20,000 | 80,000 | Math support |

### Time Estimates

**Assumption:** Expert can enter ~50 sentences per hour

| Dataset Size | Hours Required | With 1 Person | With 3 People |
|--------------|----------------|---------------|---------------|
| 5,000 (MVP) | 100 hours | 2.5 weeks | ~1 week |
| 25,000 (Beta) | 500 hours | 12 weeks | ~4 weeks |
| 60,000 (Release) | 1,200 hours | 30 weeks | ~10 weeks |
| 80,000 (+ Nemeth) | 1,600 hours | 40 weeks | ~14 weeks |

*Assumes 8 hours/day, 5 days/week*

### Data Categories

1. **Grade 1 (Uncontracted)**
   - Letters, numbers, punctuation
   - Simple words without contractions
   - Baseline for model training

2. **Grade 2 (Contracted)**
   - Single-cell contractions (the, and, for, etc.)
   - Multi-cell contractions
   - Context-dependent patterns
   - **Primary focus of data collection**

3. **Nemeth (Mathematical)**
   - Basic arithmetic expressions
   - Algebraic equations
   - Fractions, exponents, roots
   - Greek letters and symbols
   - **Separate collection phase**

4. **Mixed Content**
   - Sentences combining text and numbers
   - Real-world documents
   - Edge cases and unusual patterns

### Quality Guidelines

- No typos in English translation
- Correct braille notation (verified by preview)
- Include context notes for ambiguous cases
- Mark Grade level (1, 2, or Nemeth) for each entry
- 10% of entries verified by second expert

---

## Next Steps

1. [ ] Build MVP annotation app (static web)
2. [ ] Test with braille experts
3. [ ] Iterate on UX based on feedback
4. [ ] Begin data collection
5. [ ] Set up verification workflow
6. [ ] Export data for model training

---

## Resolved Questions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Sentences or individual words? | **Sentences** | Provides context for contractions |
| Support Nemeth (math)? | **Yes** | Separate phase after Grade 2 |
| Audio feedback for accessibility? | **Browser default** | Screen readers work with standard HTML |

## Open Questions

1. Should we support braille keyboard input directly?
2. Do we need to handle literary vs computer braille?
3. What's the best model architecture for seq2seq (RNN, Transformer, etc.)?
4. Should we use pre-trained language models as a starting point?

---

## References

- [BRAILLE_CODES.md](docs/BRAILLE_CODES.md) - Complete code reference table
- [PROJECT_STRATEGY.md](docs/PROJECT_STRATEGY.md) - Overall project strategy
- [IMPLEMENTATION_LOG.md](docs/IMPLEMENTATION_LOG.md) - Development progress

---

**Document Version:** 1.1
**Last Updated:** January 24, 2026

### Changelog
- v1.1: Added detailed training data size estimates, resolved open questions
- v1.0: Initial design document
