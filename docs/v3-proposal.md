# v3 Training Proposal

**Date:** March 10, 2026
**Context:** v1 (custom tokens, no warmup) → 2.9% accuracy. v2 (Unicode braille) → 0% accuracy (tokens collapsed to `<unk>`).

---

## Problem Statement

We need a seq2seq model that translates braille cell sequences (64 possible cells) to English text. The core challenge: how to represent 64 braille cells as model input without breaking pre-trained weights.

| Approach | Result | Failure Mode |
|----------|--------|-------------|
| v1: Custom tokens in T5-small | 2.9% exact match | Random embeddings destabilized pre-trained weights |
| v2: Unicode braille in T5-small | 0% exact match | All braille chars → single `<unk>` token |

---

## Research Findings

### 1. BrailleLLM (EMNLP 2025) — Most Directly Relevant

Paper: [BrailleLLM: Facilitating LLMs to Master the Braille Domain](https://aclanthology.org/2025.emnlp-main.1454/)

Researchers instruction-tuned LLMs for braille tasks using **BKFT** (Braille Knowledge-Based Fine-Tuning):
- Initialized braille token embeddings from **semantically equivalent word embeddings** (not random)
- Built a mapping knowledge base for rational segmentation of braille sequences
- Results: Chinese braille-to-text CER of 0.054; English braille QA BERTScore of 0.9543
- Surpassed GPT-4 and Claude 3.5 on braille domain tasks
- Released datasets: EBMD (English) and CBMD (Chinese)

**Relevance to us:** Confirms that random token initialization (our v1 approach) is a known problem. Semantic initialization is the fix if staying with subword models. However, this work focused on Chinese braille and general braille QA, not Grade 2 English contraction resolution.

### 2. ByT5 on Small-Alphabet-to-Text Tasks

Paper: [ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models](https://arxiv.org/abs/2105.13626)

ByT5 has strong results on problems structurally similar to ours:

- **Grapheme-to-phoneme (G2P):** ByT5 significantly outperforms token-based mT5 across ~100 languages ([Zhu et al., 2022](https://arxiv.org/abs/2204.03067)). G2P is a small fixed alphabet → variable output task, closely analogous to braille → English.
- **SMILES chemistry notation:** ByT5 matches or exceeds FlanT5 on organic reaction prediction from arbitrary symbolic strings (Pang et al., 2024).
- **Noise robustness:** Only ~1.5 point degradation on noisy input vs ~25 points for mT5. Relevant since OCR-detected braille may have cell detection errors.
- **Multilingual semantic parsing:** ByT5-Base beats mT5-Base by 10-20 exact match points zero-shot on MASSIVE benchmark (51 languages).

**Key insight from G2P literature:** The smaller your input vocabulary, the more ByT5 makes sense. Subword tokenization adds no value when the alphabet is already small and fixed (64 cells).

### 3. Custom Token Embedding Best Practices

If using T5/mT5 with new tokens, research shows:

- **Random init** (what we did in v1) is known to underperform
- **Mean of existing embeddings, Xavier-normalized** is the standard recommended approach ([arxiv 2407.12514](https://arxiv.org/abs/2407.12514))
- **Token Distillation** ([arxiv 2505.20133](https://arxiv.org/html/2505.20133v2)) — attention-aware initialization for new tokens; more sophisticated
- **Two-stage freeze/unfreeze** is standard when adding tokens to pre-trained models
- **Critical T5-specific finding:** Raw pre-trained embeddings perform poorly unless standardized to Xavier-scale range

### 4. Chinese Braille Translation Work

A parallel line of research on Chinese braille:
- [Pre-training for low-resource Chinese-Braille translation](https://www.sciencedirect.com/science/article/abs/pii/S0141938223001397) (Yu et al., 2023)
- [Non-autoregressive Chinese-Braille with CTC loss](https://www.sciencedirect.com/science/article/pii/S0957417424032238) (Yu et al., 2025) — addresses long-sequence problem in braille
- [mT5-Small with braille special tokens](https://arxiv.org/html/2407.06048v1), evaluated with BLEU scores

**No existing ML models for Grade 2 English contracted braille-to-text were found.** This confirms our project's novelty.

### 5. ByT5 Practical Characteristics

| Aspect | Details |
|--------|---------|
| Max sequence length | 1024 bytes (pre-training) |
| Our input range | ~200-500 bytes — fits comfortably |
| Training speed | 4-6x slower per step than T5 (longer sequences) |
| Model size | Small: 300M, Base: 580M |
| Memory | Higher activation memory; Small fits on consumer GPUs |
| Embedding params | Only 0.3% in embeddings (vs 80-85% in subword models) |
| Fine-tuning recipe | AdamW, lr=1e-3, 2-10 epochs, dropout=0.1 |

---

## Options for v3

### Option A: T5-small + Two-Stage Training

Keep custom tokens (`c0`-`c63`) but fix the embedding problem with staged training.

- **Stage 1 (2-3 epochs):** Freeze all weights except input embeddings. Train at 5e-4 so embeddings learn meaningful positions.
- **Stage 2 (7-8 epochs):** Unfreeze everything. Train at 1e-4 with warmup + cosine decay.
- **Enhancement (from BrailleLLM):** Initialize embeddings from mean of existing T5 embeddings, Xavier-normalized, instead of random.

**Pros:**
- Minimal code changes — add freeze/unfreeze logic + embedding init to existing pipeline
- 60M params, fastest training
- Well-supported by literature on custom token initialization

**Cons:**
- Two-stage is a workaround — model still needs to learn 64 new embeddings from limited signal
- Third attempt on same base architecture
- Uncertain if our 25K samples provide enough gradient signal for convergence

**Estimated accuracy:** 40-60% synthetic, 15-30% jellybean

---

### Option B: ByT5-small (Recommended)

Switch base model to ByT5, which operates on raw UTF-8 bytes instead of SentencePiece tokens.

**Input representation:** Braille Unicode characters (⠁⠃⠉) encoded as raw UTF-8 bytes. Each braille character = 3 UTF-8 bytes. ByT5 processes all 256 byte values natively — no tokenizer issues.

```
Input:  "translate Braille to English: ⠠⠺⠁⠽"
ByT5:   processes each byte individually — no <unk>, no custom tokens
```

**Pros:**
- Completely eliminates the tokenizer/embedding problem (root cause of v1 and v2)
- No custom tokens needed — ByT5 has pre-trained embeddings for all 256 bytes
- Proven on analogous tasks (G2P, SMILES) where it outperforms token-based models
- Same HuggingFace API — Seq2SeqTrainer code barely changes
- Robust to noisy input (relevant for OCR pipeline)

**Cons:**
- 4-6x slower per training step (byte-level = longer sequences)
- 300M params vs T5-small's 60M — but fits on T4 (15GB)
- Higher learning rate needed (1e-3 vs 1e-4)

**Code changes:**
- Change `MODEL_NAME` to `google/byt5-small`
- Use `AutoTokenizer` instead of `T5Tokenizer`
- Remove custom token logic
- Adjust LR to 1e-3
- Keep everything else (dataset, Seq2SeqTrainer, data format from v2)

**Estimated accuracy:** 50-70% synthetic, 20-40% jellybean

---

### Option C: T5-base (220M params)

Larger T5 with custom tokens + two-stage training.

**Pros:** More capacity to absorb new embeddings
**Cons:** Doesn't fix the fundamental embedding problem. 4x slower, more memory. Overkill if issue is representation, not capacity.

---

### Option D: Custom Encoder-Decoder from Scratch

Small Transformer or LSTM trained entirely from scratch.

**Pros:** Full control, no pre-trained weight conflicts, tiny 64-symbol embedding table
**Cons:** Loses English language knowledge, needs much more data, major rewrite

---

## Recommendation

**Option B (ByT5-small)** is the strongest choice:

1. It directly fixes the root cause — tokenizer vocabulary mismatch — rather than working around it
2. Proven on the closest analogous task (grapheme-to-phoneme), where it outperforms token-based models
3. Minimal code changes since it uses the same HuggingFace Seq2SeqTrainer API
4. Pre-trained byte-level embeddings mean the model processes braille from the first training step
5. If ByT5 also fails, it tells us the problem is data/task difficulty, not representation

**Fallback:** If ByT5 is too slow or underperforms, try Option A (T5-small two-stage with Xavier-initialized embeddings).

---

## Implementation Plan

### Step 1: Update trainer for ByT5

- Change `MODEL_NAME` to `google/byt5-small`
- Use `AutoTokenizer` / `AutoModelForSeq2SeqLM`
- Remove all custom token logic (`CELL_TOKENS`, `add_tokens`, `resize_token_embeddings`)
- Adjust default LR to 1e-3 (ByT5 fine-tuning recipe)
- Increase `max_source_len` to 1024 (byte sequences are longer)

### Step 2: Keep v2 data format

The prepared TSV files from v2 already have the correct format:
```
translate Braille to English: ⠠⠺⠁⠽⠀⠙⠪⠝	Way down south...
```
ByT5 will tokenize the braille Unicode as raw bytes — no `<unk>` problem.

### Step 3: Update Colab notebook

- Change model name
- Adjust batch size (300M model needs more memory than 60M)
- Adjust LR to 1e-3
- Keep 10 epochs

### Step 4: Train on Colab T4

Estimated training time: 2-4 hours (4-6x slower than T5-small, but T4 is fast).

### Step 5: Evaluate

- Run evaluation with CER, BLEU, exact match
- Compare against v1 and v2
- If successful, document results and plan next improvements

### Fallback: Option A

If ByT5 results are poor or training is impractical:
- Revert to T5-small with custom tokens
- Initialize embeddings as mean of existing embeddings (Xavier-normalized)
- Two-stage freeze/unfreeze training
- Same Colab setup
