# Evaluation Report: T5-small v1 (First Training Run)

**Date:** March 8, 2026
**Model:** T5-small fine-tuned on synthetic Grade 2 braille data
**Training:** 5 epochs, batch_size=4, grad_accum=8, lr=3e-4, no warmup/scheduler

---

## Results Summary

| Test Set | Exact Match | Samples |
|----------|-------------|---------|
| Synthetic test (5% hold-out) | 40/1396 (2.9%) | Liblouis-generated data |
| Jellybean (held-out real) | 0/42 (0.0%) | Human-transcribed data |

**Final training loss:** 3.5834
**Final validation loss:** 3.5109

**Verdict:** The model failed to learn the braille→English translation task.

---

## Sample Predictions

### Synthetic Test Set

| # | Expected | Predicted |
|---|----------|-----------|
| 1 | Mr. | Mr. |
| 2 | Gardiner's hope of Lydia's being soon married, her joy burst forth, and every following sentence added to its exuberance. | I have not seen any of them before, but I have not seen any of them before, and I have not seen any of them. |
| 3 | "Stern all!" The oarsmen backed water; the same moment something went hot and hissing along every one of their wrists. | "It is a very good thing to be sure," said the Hatter, "that I have not a very good idea of it. |
| 4 | We were seated at breakfast one morning, my wife and I, when the maid brought in a telegram. | "I am sure that I am not a man of a good quality, but a man of a good quality. |
| 5 | Happiness in marriage is entirely a matter of chance. | I have not a doubt of your being a victim of such a resentment. |

### Jellybean (Real-World)

| # | Expected | Predicted |
|---|----------|-----------|
| 1 | Way down south where the jungle grows, | "It is a very good thing to be sure! |
| 2 | farther and deeper than anyone goes-- | 'I am sure you will be able to do it. |
| 3 | a jellybean jungle stands all alone. | 'I am afraid,' said Scrooge. |
| 4 | Only I have been there, | Is it not a sailor?' |
| 5 | Deep inside the jungle | I am a sailor! |
| 6 | and share the jellybeans with me! | Is it not a sailor's sailor! |

---

## Failure Analysis

The model outputs memorized English phrases from the training corpus (Dickens, Melville, Austen) rather than translating the braille input. Common output patterns:

- "I am a sailor, a sailor!" (Moby Dick)
- "'I am afraid,' said Scrooge." (A Christmas Carol)
- "It is a very good thing to be sure" (Pride and Prejudice / Alice)

This indicates the model learned to generate plausible English from the training distribution but did **not** learn to attend to the braille cell token input.

The 40 exact matches on the synthetic test set (2.9%) are likely very short/trivial strings (e.g., "Mr.", single words) that can be guessed without understanding the input.

### Root Causes

1. **No learning rate warmup or scheduler.** T5 fine-tuning requires warmup to stabilize early training, especially when new token embeddings are being learned from scratch alongside pretrained weights.

2. **Static padding to max_length.** Every sample was padded to 128 (input) and 256 (output) tokens regardless of actual length. Short sentences (5 tokens) get overwhelmed by padding, diluting the gradient signal.

3. **Learning rate too high (3e-4).** Without a scheduler, this may destabilize pretrained weights before the new braille embeddings have learned meaningful representations.

4. **Insufficient epochs (5).** The 64 custom braille tokens start with random embeddings. They need more training steps to learn meaningful representations while the rest of the model adapts.

---

## Fixes for v2

| Change | v1 (current) | v2 (planned) |
|--------|-------------|-------------|
| Learning rate | 3e-4, constant | 1e-4 with warmup + cosine decay |
| LR warmup | None | 10% of total steps |
| Padding | Static (max_length) | Dynamic (longest in batch) |
| Epochs | 5 | 10 |

### Why these fixes should help

- **Warmup** lets the new braille embeddings gradually learn before the pretrained weights shift too far from their initialization.
- **Cosine decay** reduces the learning rate as training progresses, allowing fine-grained convergence.
- **Dynamic padding** eliminates wasted computation on padding tokens and produces cleaner gradients, especially for short sentences.
- **Lower base LR** prevents catastrophic forgetting of pretrained English knowledge.
- **More epochs** give the model additional passes to learn the braille→English mapping.

### Expected training command (v2)

```bash
python src/trainer.py \
  --train data/prepared/train.tsv \
  --val data/prepared/val.tsv \
  --output models/braille-t5-v2 \
  --epochs 10 \
  --batch-size 4 \
  --grad-accum 8 \
  --lr 1e-4
```

(Warmup, scheduler, and dynamic padding to be implemented in code.)
