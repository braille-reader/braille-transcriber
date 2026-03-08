# Training Guide - Braille → English T5 Model

Quick reference for training the Grade 2 braille-to-English translation model.

---

## Prerequisites

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Ensure synthetic data has been generated (see `data/README.md`).

---

## Step 1: Prepare Data

```bash
python tools/prepare_data.py
```

Reads from `data/synthetic/*.train` and `data/manual/jellybean_jungle.txt`.

**Output** (in `data/prepared/`):

| File | Rows | Purpose |
|------|------|---------|
| train.tsv | 25,138 | Training set (90% of synthetic) |
| val.tsv | 1,396 | Validation set (5% of synthetic) |
| test.tsv | 1,396 | Test set (5% of synthetic) |
| jellybean.tsv | 42 | Held-out real-world test (human-transcribed) |

**Format:** `c32 c1 c7<TAB>Alice` — T5 custom tokens separated by spaces, tab, then English text.

---

## Step 2: Train

```bash
python src/trainer.py \
  --train data/prepared/train.tsv \
  --val data/prepared/val.tsv \
  --output models/braille-t5 \
  --epochs 5 \
  --batch-size 4 \
  --grad-accum 8 \
  --lr 3e-4 \
  --max-source-len 128 \
  --max-target-len 256
```

### Parameters

| Flag | Default | Notes |
|------|---------|-------|
| `--epochs` | 5 | 3-5 is typical for fine-tuning |
| `--batch-size` | 4 | Reduce to 2 if OOM on 8GB machines |
| `--grad-accum` | 8 | Effective batch = batch-size x grad-accum |
| `--lr` | 3e-4 | Standard T5 fine-tuning rate |
| `--max-source-len` | 128 | Covers ~95% of inputs |
| `--max-target-len` | 256 | Covers long English output |

### Device Selection

Automatic: MPS (Apple Silicon) > CUDA > CPU.

### Expected Output

```
Device: mps
Train: 25138 samples, Val: 1396 samples
Batch size: 4, Grad accum: 8, Effective batch: 32
Epochs: 5
  Epoch 1 step 100/6285 loss=X.XXXX
  ...
Epoch 1/5: train_loss=X.XXXX val_loss=X.XXXX
  Saved best model (val_loss=X.XXXX)
...
Training complete. Models saved to models/braille-t5/
```

### Estimated Training Time

| Hardware | Estimate |
|----------|----------|
| M1 Air 8GB (MPS) | 4-8 hours |
| M1 Air 8GB (CPU) | 12-20 hours |
| CUDA GPU (e.g. T4) | 1-2 hours |

### If Out of Memory

```bash
# Reduce batch size, increase gradient accumulation to compensate
python src/trainer.py --batch-size 2 --grad-accum 16
```

### Output Files

```
models/braille-t5/
  best/       # Lowest validation loss checkpoint
  final/      # End-of-training checkpoint
```

Each contains the T5 model weights + tokenizer (with custom c0-c63 tokens).

---

## Step 3: Evaluate (After Training)

TODO — evaluation script to be built after first training run.

Will include:
- Load best model
- Run inference on test.tsv and jellybean.tsv
- Compare against liblouis back-translation baseline
- Per-sentence accuracy metrics

---

## Architecture Summary

**Model:** T5-small (60M params) fine-tuned for braille→English

**Input:** Custom tokens `c0`-`c63` representing the 64 braille cell patterns

**Output:** English text

**Why T5-small:**
- Encoder-decoder architecture suits translation tasks
- Already knows English (grammar, vocabulary, spelling)
- Only needs to learn the 64-symbol braille mapping
- 60M params fits 8GB memory; 28K examples is sufficient for fine-tuning

**Why custom tokens (not raw numbers):**
- Each braille cell = exactly one token (no subword splitting)
- Model learns dedicated embeddings for each cell pattern
- Clean 1:1 mapping between braille cells and input tokens

See `docs/data-pipeline-design.md` for full design rationale.
