# Evaluation Report: T5-small v2 (Unicode Braille + Seq2SeqTrainer)

**Date:** March 10, 2026
**Model:** T5-small fine-tuned on synthetic Grade 2 braille data
**Training:** 10 epochs, batch_size=16, grad_accum=2, lr=1e-4, warmup=10%, cosine decay, fp16
**Hardware:** Google Colab T4 GPU (15GB VRAM)

---

## What We Changed from v1

| Change | v1 | v2 |
|--------|----|----|
| Input representation | Custom tokens (`c0`-`c63`) | Unicode braille (U+2800-U+283F) |
| Task prefix | None | `"translate Braille to English: "` |
| Training loop | Manual PyTorch loop | HuggingFace `Seq2SeqTrainer` |
| Padding | Static (pad to max_length) | Dynamic (`DataCollatorForSeq2Seq`) |
| LR schedule | Constant 3e-4 | 1e-4 with 10% warmup + cosine decay |
| Gradient clipping | None | max_norm=1.0 |
| Epochs | 5 | 10 |
| Batch size | 4 (local M1) | 16 (Colab T4) |
| Effective batch | 32 (4×8) | 32 (16×2) |
| Precision | fp32 | fp16 |

---

## Results

| Test Set | v1 Exact Match | v2 Exact Match |
|----------|---------------|---------------|
| Synthetic test (1,396 samples) | 40/1396 (2.9%) | 0/1396 (0.0%) |
| Jellybean held-out (42 samples) | 0/42 (0.0%) | 0/42 (0.0%) |

**v2 is worse than v1.** The model outputs a single memorized phrase for every input:

```
"It is a very good thing to be able to do it."
```

v1 at least varied its hallucinations across training corpus phrases. v2 collapsed to a single output regardless of input.

### Sample Predictions (v2)

Every prediction is identical:

| # | Expected | Predicted |
|---|----------|-----------|
| 1 | Mr. | "It is a very good thing to be able to do it." |
| 2 | Gardiner's hope of Lydia's being soon married... | "It is a very good thing to be able to do it." |
| 3 | Way down south where the jungle grows, | "It is a very good thing to be able to do it." |
| 4 | farther and deeper than anyone goes-- | "It is a very good thing to be able to do it." |

---

## Root Cause: Unicode Braille Not in T5 Vocabulary

The key assumption behind v2 — that T5 would tokenize Unicode braille characters using existing embeddings — was **wrong**.

### Evidence

```python
from transformers import T5Tokenizer
tok = T5Tokenizer.from_pretrained('t5-small')
text = 'translate Braille to English: ⠠⠺⠁⠽⠀⠙⠪⠝'
tokens = tok.convert_ids_to_tokens(tok(text)['input_ids'])
# Result: ['▁translate', '▁Br', 'aille', '▁to', '▁English', ':', '▁', '<unk>', '</s>']
```

**All braille characters (⠠⠺⠁⠽⠀⠙⠪⠝) collapsed into a single `<unk>` token.**

T5's SentencePiece vocabulary was trained on English/multilingual web text. Unicode braille (U+2800-U+283F) is not represented. The tokenizer maps the entire braille input to one unknown token, destroying all information.

### Why This Made Things Worse Than v1

- **v1 custom tokens** (`c0`-`c63`): Each braille cell was a distinct token with its own (random) embedding. The model received 64 distinguishable input signals, even though the embeddings were random.
- **v2 Unicode braille**: All braille cells mapped to the same `<unk>` token. The model received zero information about the input — every sample looked identical.

With no input signal at all, the model learned to output the single most "average" English sentence from the training distribution, explaining the collapse to one repeated phrase.

---

## What Worked

Despite the failed result, several v2 changes are improvements worth keeping:

1. **Task prefix** (`"translate Braille to English: "`) — correct practice for T5, activates seq2seq mode
2. **Seq2SeqTrainer** — cleaner code, automatic dynamic padding, warmup, cosine decay, gradient clipping
3. **fp16 on T4** — much faster training than M1 MPS
4. **Dynamic padding** — eliminates wasted computation on short sequences
5. **Colab workflow** — viable for our dataset size, training completes in reasonable time

---

## Key Lesson

**Always verify tokenization before training.** A quick check like this would have caught the issue before a full training run:

```python
tok = T5Tokenizer.from_pretrained('t5-small')
sample = 'translate Braille to English: ⠁⠃⠉'
tokens = tok.convert_ids_to_tokens(tok(sample)['input_ids'])
print(tokens)  # Shows <unk> — problem is immediately visible
```

---

## Revised Plan for v3

The Unicode approach is ruled out for T5. We must use custom tokens. The correct strategy combines:

1. **Custom tokens (`c0`-`c63`)** — the only way to give T5 distinguishable braille input
2. **Task prefix** — keep from v2
3. **Seq2SeqTrainer** — keep from v2 (dynamic padding, warmup, cosine decay, grad clipping)
4. **Two-stage training** — new, addresses the random embedding problem from v1:
   - Stage 1 (2-3 epochs): Freeze all model weights except input embeddings, train at higher LR (5e-4) so embeddings move into meaningful space
   - Stage 2 (7-8 epochs): Unfreeze everything, train at lower LR (1e-4) with warmup + cosine decay

### Why Two-Stage Should Work

The v1 failure was caused by random embeddings producing noisy gradients that destabilized pre-trained weights. Two-stage training solves this:

- **Stage 1** lets the 64 new embeddings learn meaningful positions in T5's vector space without disturbing the pre-trained encoder/decoder
- **Stage 2** fine-tunes the full model starting from stable embeddings, so gradients are coherent from the start

### Expected Outcome

| Metric | v1 | v2 | v3 (estimated) |
|--------|----|----|---------------|
| Synthetic exact match | 2.9% | 0.0% | 40-60% |
| Jellybean exact match | 0.0% | 0.0% | 15-30% |

---

## Version History

| Version | Input Format | Training | Result | Failure Mode |
|---------|-------------|----------|--------|-------------|
| v1 | `c0`-`c63` custom tokens | Manual loop, 5 epochs, constant LR 3e-4 | 2.9% / 0.0% | Hallucinated varied corpus phrases |
| v2 | Unicode braille (U+2800+) | Seq2SeqTrainer, 10 epochs, warmup+cosine 1e-4 | 0.0% / 0.0% | Single repeated phrase (input lost to `<unk>`) |
| v3 | `c0`-`c63` + task prefix | Seq2SeqTrainer, two-stage freeze/unfreeze | TBD | — |
