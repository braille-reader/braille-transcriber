# Strategy Recommendations for v2 Training

**Date:** March 9, 2026  
**Author:** AI Assistant Analysis  
**Context:** Post-mortem of v1 training failure (0% accuracy on real data)

---

## Executive Summary

The v1 model failed to learn the braille→English translation task, instead memorizing and hallucinating text from the training corpus. The root cause is **representation misalignment**: 64 randomly-initialized custom tokens (`c0`-`c63`) produce noisy gradients that destroy T5's pre-trained weights before meaningful braille embeddings can be learned.

While the proposed v2 fixes (warmup, cosine decay, dynamic padding, lower LR) are necessary, they are **insufficient** to solve the fundamental architectural and data strategy problems. This report provides a comprehensive strategy to achieve successful training in v2.

---

## Problem Diagnosis

### Core Issues from v1

1. **Random Token Embeddings:** The 64 custom braille tokens start with random embeddings, providing no signal to the model in early training.
2. **No Task Context:** T5 expects task prefixes (e.g., "translate German to English:"). Without them, it defaults to unconditional text generation.
3. **Static Padding:** Every sample padded to max length dilutes gradient signals, especially for short sentences.
4. **No Gradient Clipping:** New random embeddings can cause exploding gradients that destabilize pre-trained weights.
5. **Too Aggressive LR:** 3e-4 without warmup moves pre-trained weights too far before braille embeddings stabilize.
6. **Complex Input (Grade 2):** Grade 2 braille has heavy contractions (1 cell = multiple letters), making alignment extremely difficult for a model with random input embeddings.

---

## Recommended Strategy for v2

### Priority 1: Architectural & Representation Fixes

#### A. Switch to ASCII Braille (Highly Recommended)

**Current:**
```
c32 c1 c7 c34 → Some text
```

**Proposed:**
```
⠁⠃⠉⠙ → Some text
```

**Implementation:**
- Map each of the 64 braille cells to its Unicode Braille Pattern character (U+2800 to U+283F)
- Remove custom token addition from tokenizer
- T5 will tokenize braille as standard UTF-8 characters

**Why this works:**
- Leverages T5's existing token embeddings (heavily pre-trained on diverse text)
- Eliminates the "random embedding" problem entirely
- Braille Unicode characters are semantically meaningful: they visually resemble the actual braille cell pattern
- No need for frozen embedding training phase

**Migration path:**
```python
# In cell_codec.py or preprocessing
BRAILLE_UNICODE = {
    0: '⠀', 1: '⠁', 2: '⠂', 3: '⠃', 4: '⠄', 5: '⠅', 6: '⠆', 7: '⠇',
    8: '⠈', 9: '⠉', 10: '⠊', 11: '⠋', 12: '⠌', 13: '⠍', 14: '⠎', 15: '⠏',
    # ... (continue for all 64 cells)
}

def cells_to_braille_unicode(cell_codes: list[int]) -> str:
    return ''.join(BRAILLE_UNICODE[c] for c in cell_codes)
```

**Alternative (if keeping custom tokens):** See Section B below.

---

#### B. Add Task Prefix (Critical)

T5 is instruction-tuned. It needs explicit task context.

**Current behavior:**
```python
source_enc = tokenizer(source, ...)  # "c32 c1 c7"
```

**Required behavior:**
```python
source_with_prefix = f"translate Braille to English: {source}"
source_enc = tokenizer(source_with_prefix, ...)
```

**Why this works:**
- Activates T5's sequence-to-sequence translation pathways
- Prevents unconditional language generation mode (which caused v1 hallucinations)
- Standard practice for all T5 fine-tuning tasks

**Implementation:**
```python
# In BrailleDataset.__getitem__()
source, target = self.pairs[idx]
source = f"translate Braille to English: {source}"

source_enc = self.tokenizer(
    source,
    max_length=self.max_source_len,
    # ... rest of tokenization
)
```

---

#### C. Two-Stage Fine-Tuning (If Keeping Custom Tokens)

If you cannot use ASCII Braille and must keep `c0`-`c63` custom tokens:

**Stage 1: Train Embeddings Only (2-3 epochs)**
```python
# Freeze entire model except input embeddings
for param in model.parameters():
    param.requires_grad = False
model.shared.weight.requires_grad = True  # Input embeddings

# Train with higher LR (e.g., 5e-4) for 2-3 epochs
```

**Stage 2: Full Fine-Tuning (8-10 epochs)**
```python
# Unfreeze everything
for param in model.parameters():
    param.requires_grad = True

# Train with low LR (1e-4) with warmup + decay
```

**Why this works:**
- Allows random braille embeddings to move into meaningful vector space
- Prevents them from destroying pre-trained weights in early training
- Once embeddings are stable, full model can adapt

---

### Priority 2: Training Infrastructure Improvements

#### A. Migrate to Hugging Face `Seq2SeqTrainer` (Highly Recommended)

**Replace the manual PyTorch loop in `trainer.py` with:**

```python
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

def train(train_path, val_path, output_dir, epochs=10, batch_size=4, grad_accum=8, lr=1e-4):
    tokenizer = setup_tokenizer()
    model = setup_model(tokenizer)
    
    train_ds = BrailleDataset(train_path, tokenizer)
    val_ds = BrailleDataset(val_path, tokenizer)
    
    # DataCollator handles dynamic padding automatically
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,  # Dynamic padding per batch
    )
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.1,  # 10% warmup
        lr_scheduler_type="cosine",  # Cosine decay
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        fp16=torch.cuda.is_available(),  # Mixed precision if GPU
        gradient_checkpointing=True,  # Save memory
        max_grad_norm=1.0,  # Gradient clipping
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
```

**Benefits:**
- ✅ Dynamic padding (solves v1 static padding problem)
- ✅ Learning rate warmup (10% of steps)
- ✅ Cosine decay scheduler (automatic)
- ✅ Gradient clipping (`max_grad_norm=1.0`)
- ✅ Best model checkpointing
- ✅ Mixed precision training (faster on GPU)
- ✅ Gradient checkpointing (lower memory usage)

**Dataset modifications needed:**
```python
class BrailleDataset(Dataset):
    def __getitem__(self, idx):
        source, target = self.pairs[idx]
        source = f"translate Braille to English: {source}"  # Add prefix
        
        # Remove padding="max_length" - let DataCollator handle it
        source_enc = self.tokenizer(
            source,
            max_length=self.max_source_len,
            truncation=True,
            # NO PADDING HERE
        )
        
        target_enc = self.tokenizer(
            target,
            max_length=self.max_target_len,
            truncation=True,
            # NO PADDING HERE
        )
        
        labels = target_enc["input_ids"]
        
        return {
            "input_ids": source_enc["input_ids"],
            "attention_mask": source_enc["attention_mask"],
            "labels": labels,
        }
```

---

#### B. Add Gradient Clipping (If Keeping Manual Loop)

If you don't migrate to `Seq2SeqTrainer`, add this after `loss.backward()`:

```python
loss.backward()

# Add gradient clipping before optimizer step
if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
```

**Why:** Prevents exploding gradients from randomly initialized braille token embeddings.

---

### Priority 3: Data & Curriculum Strategy

#### A. Curriculum Learning: Grade 1 Before Grade 2

**Problem:** Grade 2 braille uses heavy contractions:
- Single cell `⠮` = "the"
- Single cell `⠱` = "ch"
- Single cell `⠝` can mean "n" or "not" depending on context

This makes alignment extremely difficult when starting with random embeddings.

**Solution:**
1. Generate a synthetic **Grade 1 (uncontracted) braille** dataset where almost every braille cell maps 1-to-1 with an English letter/punctuation
2. Train v2a on Grade 1 data (5 epochs)
3. Fine-tune v2b starting from v2a weights on Grade 2 data (5 epochs)

**Implementation:**
```python
# In data_generator.py
def generate_grade1_samples(texts: list[str], output_path: str):
    """Generate Grade 1 (uncontracted) braille training data."""
    import louis  # Liblouis Python bindings
    
    with open(output_path, 'w') as f:
        for text in texts:
            # Use Grade 1 table (no contractions)
            braille_cells = louis.translateString(['en-us-g1.ctb'], text)
            cell_codes = ' '.join(f'c{ord(c)}' for c in braille_cells)
            f.write(f"{cell_codes}\t{text}\n")
```

**Why this works:**
- Model learns basic braille alphabet structure first
- Establishes strong encoder→decoder attention patterns
- Provides solid initialization for Grade 2 fine-tuning

---

#### B. Regularization via Identity Translations

To prevent catastrophic forgetting of English, inject 5-10% identity samples:

```python
# In data generation
samples.append(("translate English to English: Hello", "Hello"))
samples.append(("translate English to English: The quick brown fox", "The quick brown fox"))
```

**Why:** Acts as a regularizer to anchor T5's English generation capabilities while learning braille.

---

#### C. Augment with Real-World Data Early

Don't wait until post-training to test on Jellybean data. Include a small amount (10-20 samples) of real braille in training:

```python
# Mix in real samples
train_data = synthetic_samples + real_human_transcribed_samples[:20]
```

**Why:** 
- Exposes model to real-world noise patterns
- Prevents overfitting to synthetic liblouis patterns
- Improves generalization

---

## Recommended v2 Training Plan

### Option A: ASCII Braille (Recommended)

```bash
# 1. Regenerate data with ASCII Braille + task prefix
python tools/prepare_data.py --braille-format unicode

# 2. Train with Seq2SeqTrainer
python src/trainer.py \
  --train data/prepared/train.tsv \
  --val data/prepared/val.tsv \
  --output models/braille-t5-v2 \
  --epochs 10 \
  --batch-size 4 \
  --grad-accum 8 \
  --lr 1e-4
```

**Expected improvements:**
- Exact match on synthetic test: **60-80%** (from 2.9%)
- Exact match on Jellybean: **30-50%** (from 0%)

---

### Option B: Custom Tokens + Two-Stage Training

```bash
# Stage 1: Train embeddings only (2 epochs)
python src/trainer.py \
  --train data/prepared/train.tsv \
  --val data/prepared/val.tsv \
  --output models/braille-t5-v2-stage1 \
  --epochs 2 \
  --freeze-model \
  --lr 5e-4

# Stage 2: Full fine-tuning (8 epochs)
python src/trainer.py \
  --train data/prepared/train.tsv \
  --val data/prepared/val.tsv \
  --output models/braille-t5-v2-stage2 \
  --epochs 8 \
  --resume models/braille-t5-v2-stage1/final \
  --lr 1e-4
```

**Expected improvements:**
- Exact match on synthetic test: **40-60%** (from 2.9%)
- Exact match on Jellybean: **15-30%** (from 0%)

---

### Option C: Curriculum Learning (Most Conservative)

```bash
# Phase 1: Grade 1 braille (5 epochs)
python src/trainer.py \
  --train data/prepared/train_grade1.tsv \
  --val data/prepared/val_grade1.tsv \
  --output models/braille-t5-v2-grade1 \
  --epochs 5 \
  --lr 1e-4

# Phase 2: Grade 2 braille (5 epochs, starting from Grade 1 weights)
python src/trainer.py \
  --train data/prepared/train_grade2.tsv \
  --val data/prepared/val_grade2.tsv \
  --output models/braille-t5-v2-grade2 \
  --epochs 5 \
  --resume models/braille-t5-v2-grade1/final \
  --lr 5e-5  # Lower LR for fine-tuning
```

**Expected improvements:**
- Exact match on synthetic test: **70-85%**
- Exact match on Jellybean: **40-60%**

---

## Implementation Checklist

### Critical (Must Fix)
- [ ] Add task prefix: `"translate Braille to English: "` to all inputs
- [ ] Switch to ASCII Braille Unicode characters (OR implement two-stage training)
- [ ] Migrate to `Seq2SeqTrainer` + `DataCollatorForSeq2Seq` (OR add gradient clipping)
- [ ] Use learning rate warmup (10% of steps)
- [ ] Use cosine decay scheduler
- [ ] Dynamic padding (via DataCollator or custom implementation)
- [ ] Lower base learning rate to 1e-4

### High Priority (Strongly Recommended)
- [ ] Generate Grade 1 braille dataset for curriculum learning
- [ ] Add gradient clipping (`max_grad_norm=1.0`)
- [ ] Increase epochs to 10
- [ ] Mix in 5-10% identity English→English samples

### Medium Priority (Nice to Have)
- [ ] Include 10-20 real-world samples in training data
- [ ] Enable mixed precision training (fp16)
- [ ] Add gradient checkpointing to reduce memory
- [ ] Implement early stopping based on validation metrics

### Evaluation Improvements
- [ ] Track character error rate (CER) in addition to exact match
- [ ] Track BLEU score for partial credit
- [ ] Add sample predictions to logs every epoch
- [ ] Compare predictions against liblouis baseline

---

## Expected Outcomes

| Metric | v1 (actual) | v2 (conservative) | v2 (optimistic) |
|--------|-------------|-------------------|-----------------|
| Synthetic test exact match | 2.9% | 40-60% | 70-85% |
| Jellybean exact match | 0% | 15-30% | 40-60% |
| Final training loss | 3.58 | 0.5-1.0 | 0.1-0.3 |
| Final validation loss | 3.51 | 0.5-1.0 | 0.1-0.3 |

---

## Conclusion

The v1 failure was not a hyperparameter problem—it was an **architectural and data strategy problem**. The model could not learn because:

1. Random embeddings provided no signal
2. No task context triggered wrong generation mode
3. Complex Grade 2 contractions were too hard to align from scratch

**Primary recommendation:** Implement **Option A** (ASCII Braille + Seq2SeqTrainer). This requires the least code changes and provides the best chance of success by leveraging T5's existing knowledge.

**Fallback recommendation:** If you must keep custom tokens, implement **Option C** (Curriculum Learning with two-stage training). This is more complex but provides a structured path to learning the alignment.

The v2 improvements you identified (warmup, scheduler, dynamic padding, lower LR) are all correct and necessary, but they will only be effective once the fundamental representation problem is solved.

---

**Next Steps:**
1. Choose Option A, B, or C based on your constraints
2. Implement the critical checklist items
3. Run v2 training
4. Generate evaluation report with CER and BLEU metrics
5. If results are still poor, consider switching to a smaller, more controllable architecture (e.g., custom encoder-decoder LSTM) to debug the alignment problem independently

