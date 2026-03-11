# v3 Issues Log: ByT5-small on Colab T4

**Date:** March 11, 2026
**Model:** google/byt5-small (300M params)
**Hardware:** Google Colab free tier, T4 GPU (15GB VRAM)
**Data:** Same v2 prepared TSVs — Unicode braille with task prefix (25K train, 1.4K val)

---

## Summary

We switched from T5-small to ByT5-small to solve the tokenizer problem (v2's braille chars collapsed to `<unk>`). ByT5 correctly tokenizes braille as UTF-8 bytes — no `<unk>`, no custom tokens. However, we hit three infrastructure issues on Colab T4 before getting a stable training run.

---

## Issue 1: CUDA Out of Memory

**Config:** batch_size=8, grad_accum=4, fp16=True

**Error:**
```
OutOfMemoryError: CUDA out of memory. Tried to allocate 192.00 MiB.
GPU 0 has a total capacity of 14.56 GiB of which 169.81 MiB is free.
```

**Root cause:** ByT5 is 300M params (5x T5-small's 60M) and byte-level tokenization produces much longer sequences (each braille Unicode char = 3 UTF-8 bytes). A batch of 8 with ~500-byte sequences exceeds T4's 15GB VRAM.

**Fix:** Reduced batch_size to 2, increased grad_accum to 16 (kept effective batch at 32).

---

## Issue 2: Gradient Checkpointing + fp16 Incompatibility

**Config:** batch_size=2, grad_accum=16, fp16=True, gradient_checkpointing=True

**Error:**
```
CheckpointError: torch.utils.checkpoint: A different number of tensors was saved
during the original forward and recomputation.
Number of tensors saved during forward: 108
Number of tensors saved during recomputation: 41.
```

**Root cause:** Known bug in some versions of `transformers` + PyTorch where gradient checkpointing and fp16 autocast produce different tensor counts during forward vs recomputation passes. The mixed-precision context changes which intermediate tensors get saved.

**Fix:** Removed `gradient_checkpointing=True`. With batch_size=2 the model fit in memory without it.

---

## Issue 3: fp16 Numerical Overflow (Critical)

**Config:** batch_size=2, grad_accum=16, lr=1e-3, fp16=True

**Symptom:**
```
Epoch  Training Loss                    Validation Loss
1      105565670050457288704.000000      nan
2      120273146060075266211184.000000   nan
```

Training loss was `1e17` from epoch 1 and growing. Validation loss was `nan`. Model produced garbage output.

**Lowered LR to 1e-4:** Same result. Loss still `1e17`, val still `nan`. This confirmed the issue was not learning rate but fp16 precision.

**Root cause:** ByT5's byte-level architecture creates very long attention matrices (sequence length ~500+ bytes). fp16 half-precision has a max representable value of ~65,504. The softmax computation over these long sequences overflows fp16 range, producing `inf` values that propagate as `nan` through the network. This corrupts all gradients from step 1 — no amount of LR tuning can fix it.

This is a **known issue with ByT5 + fp16**. The ByT5 paper trained in fp32/bf16, and T4 GPUs do not support bf16.

**Fix:** Set `fp16=False` (use fp32). To fit in T4 memory with fp32, reduced batch_size to 1 and increased grad_accum to 32.

---

## Corrected Configuration

```python
Seq2SeqTrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,   # effective batch = 32
    learning_rate=1e-4,
    fp16=False,                       # CRITICAL: ByT5 needs fp32
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    max_grad_norm=1.0,
    num_train_epochs=10,
)
```

**Status:** Not yet trained with this config. The Colab session disconnected before a corrected run.

---

## Training Time Estimates on Colab T4

| Config | Batch | Precision | Est. time for 10 epochs |
|--------|-------|-----------|------------------------|
| batch=8, fp16 | 8 | fp16 | OOM |
| batch=2, fp16 | 2 | fp16 | ~10 hrs (but nan loss) |
| batch=1, fp32 | 1 | fp32 | ~15-20 hrs (too long for free Colab) |

**Problem:** With batch_size=1 and fp32, training 10 epochs on 25K samples will take ~15-20 hours. Colab free tier has a 12-hour limit and disconnects on inactivity.

---

## Lessons Learned

1. **Always verify numerical stability before long runs.** A 1-epoch test run would have caught the fp16 overflow in ~1 hour instead of wasting overnight.

2. **ByT5 is incompatible with fp16 on long sequences.** This is fundamental to the architecture, not a bug. bf16 would work but requires A100/newer GPUs.

3. **Byte-level models are memory-hungry.** The longer sequences (3x for UTF-8 braille) combined with the larger model (5x T5-small) mean ByT5 needs ~10x more memory per sample than T5-small.

4. **Colab free tier is marginal for ByT5.** T4's 15GB VRAM barely fits batch_size=1 in fp32, and the 12-hour session limit is tight for 10 epochs.

---

## Options Going Forward

### Option 1: ByT5 on Colab Pro ($10/month)
- A100 GPU (40/80GB VRAM) supports bf16 — solves the precision issue
- Batch size 8-16 in bf16, training would be ~2-3 hours
- Longer session limits (24hr+)

### Option 2: ByT5 on Colab Free with Reduced Scope
- Train 3-5 epochs instead of 10 (fit in 12hr window)
- Use fp32, batch=1, grad_accum=32
- May be enough to see if the model is learning at all

### Option 3: Go Back to T5-small with Improved Embedding Strategy
- Return to the faster, smaller model (60M params)
- Fix the root cause from v1: use smart embedding initialization (BrailleLLM's BKFT approach) + two-stage freeze/unfreeze
- Trains in ~1 hour on T4 with fp16 (no overflow issues with T5-small)
- Custom tokens are the only option for T5, but we now know how to initialize them properly

### Option 4: Hybrid — Quick ByT5 Validation, then T5-small
- Run ByT5 for 2-3 epochs in fp32 on Colab free (fits in ~6-8 hrs)
- Check if loss is decreasing and predictions improve
- If yes: invest in Colab Pro for full run
- If no: pivot to T5-small with BKFT embedding init (Option 3)

---

## Version History

| Version | Model | Input | Precision | Result | Issue |
|---------|-------|-------|-----------|--------|-------|
| v1 | T5-small | c0-c63 custom tokens | fp32 | 2.9% exact | Random embeddings, no warmup |
| v2 | T5-small | Unicode braille | fp16 | 0% exact | All braille → `<unk>` token |
| v3a | ByT5-small | Unicode braille | fp16 | nan loss | fp16 overflow in attention softmax |
| v3b | ByT5-small | Unicode braille | fp32 | Not run yet | Needs batch=1, ~15-20hr on T4 |
