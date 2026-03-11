# v3 Training Run: ByT5-small on A100

**Date:** March 11, 2026
**Status:** Training in progress

---

## Setup

| Parameter | Value |
|-----------|-------|
| Model | `google/byt5-small` (300M params) |
| Hardware | Colab Pro — A100 80GB, 167GB system RAM |
| Precision | bf16 |
| Batch size | 8 |
| Grad accumulation | 4 |
| Effective batch | 32 |
| Learning rate | 1e-4 |
| LR schedule | 10% warmup + cosine decay |
| Gradient clipping | max_norm=1.0 |
| Gradient checkpointing | Yes |
| Epochs | 10 |
| Train samples | 25,138 |
| Val samples | 1,396 |
| Max source len | 1024 bytes |
| Max target len | 256 bytes |

**GPU memory usage:** ~9.6 / 80 GB (well within limits — could increase batch size in future runs)

---

## Why ByT5

ByT5 processes raw UTF-8 bytes with a 259-token vocabulary (256 bytes + 3 special tokens). This solves the tokenizer problem that broke v1 and v2:

- **v1 (T5-small + custom tokens):** 64 randomly initialized embeddings produced noisy gradients → 2.9% accuracy
- **v2 (T5-small + Unicode braille):** All braille characters collapsed to a single `<unk>` token → 0% accuracy
- **v3 (ByT5-small + Unicode braille):** Each braille char becomes 3 UTF-8 bytes, all with pre-trained embeddings → no `<unk>`, no random init

---

## Issues Encountered Before This Run

### Issue 1: OOM on T4 with batch=8, fp16
ByT5's byte-level sequences are ~3x longer than subword sequences. batch=8 exceeded T4's 15GB. Fixed by reducing batch.

### Issue 2: gradient_checkpointing + fp16 incompatibility on T4
Known bug — different tensor counts during forward vs recomputation. Removed gradient checkpointing on T4.

### Issue 3: fp16 numerical overflow (critical)
Training loss was 1e17, val loss nan. fp16's max value (~65504) overflows in softmax over ByT5's long attention matrices. This is fundamental — ByT5 cannot train in fp16. Needed bf16 (requires A100+) or fp32 (too slow on T4).

### Issue 4: OOM on A100 40GB with batch=16
First A100 attempt used batch=16 without gradient checkpointing — used all 40GB. Fixed by reducing to batch=8 + gradient checkpointing.

### Resolution
Upgraded to Colab Pro with High RAM enabled → got A100 80GB. Current config (batch=8, bf16, gradient checkpointing) uses only ~10GB. Stable training.

---

## Input Data Format

```
translate Braille to English: ⠠⠺⠁⠽⠀⠙⠪⠝\tWay down
```

- Task prefix: `"translate Braille to English: "` (activates T5's seq2seq mode)
- Braille: Unicode characters U+2800-U+283F (each = 3 UTF-8 bytes in ByT5)
- Target: plain English text
- Source: synthetic parallel data from 5 Gutenberg books via Liblouis Grade 2

---

## What to Check After Training

### 1. Training loss curve
- Is loss decreasing across epochs? (v1 and v2 did not decrease meaningfully)
- Final training loss should be well below 1.0 for meaningful learning

### 2. Validation loss
- Should decrease and not be nan
- Watch for overfitting (val loss increasing while train loss decreases)

### 3. Evaluation metrics
Run on both test sets:
- **Synthetic test** (1,396 samples): expect 50-70% exact match if model learned
- **Jellybean held-out** (42 samples): expect 20-40% exact match (real-world data, harder)
- Also track CER (character error rate) and BLEU

### 4. Sample predictions
Check if predictions are:
- Related to the input (not memorized phrases from corpus)
- Varied across different inputs (not a single repeated output)
- Partially correct even when not exact match

### 5. Comparison with previous versions

| Metric | v1 | v2 | v3 (expected) |
|--------|----|----|---------------|
| Synthetic exact match | 2.9% | 0% | 50-70% |
| Jellybean exact match | 0% | 0% | 20-40% |
| Training loss | 3.58 | N/A | < 0.5 |
| Validation loss | 3.51 | N/A | < 0.5 |
| Prediction quality | Varied hallucinations | Single repeated phrase | Input-dependent |

---

## Next Steps Based on Results

### If results are good (>30% exact match on synthetic)
- Document results in evaluation-report-v3.md
- Run full evaluation with CER and BLEU
- Consider more training data or more epochs
- Test on real-world braille images through full pipeline

### If results are mediocre (10-30% exact match)
- Analyze error patterns — which contractions fail?
- Consider curriculum learning (Grade 1 first, then Grade 2)
- Add more training data or augmentation
- Try longer training (20 epochs)

### If results are poor (<10% exact match)
- Check if model is learning at all (is loss curve decreasing?)
- Consider T5-small + BKFT embedding initialization as fallback
- May need fundamentally different approach (custom encoder-decoder)
