# Data Pipeline Design - Grade 2 Training Data

**Created:** March 2026
**Status:** Implementation

---

## Goal

Build a pipeline to prepare training data for a Grade 2 (contracted) braille-to-English translation model.

## Key Design Decision: Cell Codes as Canonical Representation

### The Problem

We have training data in multiple formats:
- **BRF files** (Braille ASCII) — from Bookshare, professional transcriptions
- **Dot notation** (e.g., `1,2,5|1,5`) — from manual data collection tool
- **Stage 1 output** (dot patterns from YOLOv8) — from the OCR pipeline

### The Insight

BRF characters, dot patterns, and numeric cell codes are all **lossless 1:1 mappings** of the same 64 braille cells. Converting between them is a trivial lookup — not a learned task.

### The Decision

Use **cell codes (0-63)** as the single canonical internal representation.

```
Cell code = sum of 2^(dot-1) for each raised dot

Examples:
  dots [1]       → 2^0           = 1
  dots [1,2]     → 2^0 + 2^1    = 3
  dots [1,2,5]   → 2^0+2^1+2^4  = 19
  dots [2,3,4,6] → 2+4+8+32     = 46
  no dots        → 0 (space)
```

This is the same encoding used by Unicode braille (U+2800 + code = braille character).

### Why Not BRF in the Inference Pipeline?

BRF adds no information the model can't get from cell codes directly. It's just a remapping of the same 64 symbols. Including it in inference would add an unnecessary conversion step.

```
INFERENCE (final):
  image → YOLOv8 → dot patterns → cell codes → model → English

BRF is NOT in this path. It's a data source only.
```

## Data Flow

```
DATA SOURCES                    CANONICAL FORMAT         TRAINING

BRF files ──→ brf_to_codes() ──┐
                                ├──→ cell code sequences ──→ paired with
Dot notation → dots_to_codes()──┤    (integers 0-63)        English text
                                │                            for model
Liblouis ────→ brf_to_codes() ──┘
```

## Conversion Functions

### dots_to_code(dots) → int
Converts a list of dot numbers to a cell code.
```
[1,2,5] → 19
[2,3,4,6] → 46
[] → 0
```

### code_to_dots(code) → list
Reverse of above.
```
19 → [1,2,5]
46 → [2,3,4,6]
```

### brf_char_to_code(char) → int
Converts a BRF ASCII character to a cell code using the standard Braille ASCII table (ASCII 32-95 mapped to 64 braille cells).

### dot_notation_to_codes(line) → list[int]
Parses the pipe-separated dot notation from manual data collection.
```
"1,2,5|1,5|1,2,3" → [19, 17, 7]
"1,2,5| |1,5"     → [19, 0, 17]  (space represented as empty between pipes)
```

### brf_line_to_codes(line) → list[int]
Converts a line of BRF text to cell codes.
```
"HE" → [19, 17]
",ALICE" → [32, 1, 7, 10, 9, 17]
```

## BRF ASCII Mapping

Standard North American Braille ASCII. Each printable ASCII character (32-95) maps to one of the 64 braille cells:

```
Space → code 0  (no dots)
A-Z   → standard braille alphabet
0-9   → lower cell patterns (dots from row 2,3 only)
!     → dots 2,3,4,6 (code 46) — "the" contraction
#     → dots 3,4,5,6 (code 60) — number indicator
&     → dots 1,2,3,4,6 (code 47) — "and" contraction
...etc (full table in src/cell_codec.py)
```

## Training Data Format

Output format for the model (one sample per line pair):

```
Line 1: cell codes as space-separated integers
Line 2: English text
(blank line separator)
```

Example:
```
32 1 7 10 9 17 0 58 1 14 0 3 17 27 10 29 29 10 29 27
Alice was beginning
```

## Data Sources

### 1. BRF Files (Real Contracted Braille)
- Source: Bookshare via partners
- Books: Public domain classics (Pride and Prejudice, Alice in Wonderland, etc.)
- Paired with: Gutenberg plain text
- Value: Professional human transcription, real-world formatting

### 2. Manual Collection (Real Contracted Braille)
- Source: Visually impaired partner using braille_entry.py tool
- Format: Dot notation (pipe-separated)
- Value: Ground truth validation data

### 3. Liblouis Synthetic (Unlimited)
- Source: Any English text run through liblouis with en-ueb-g2.ctb
- Output: BRF ASCII (convert to cell codes)
- Value: Scale — can generate millions of training pairs
