"""
ByT5-small fine-tuning pipeline for braille → English translation.

Uses Seq2SeqTrainer with:
  - ByT5 byte-level model (no tokenizer vocabulary issues)
  - Unicode braille input processed as raw UTF-8 bytes
  - Task prefix: "translate Braille to English: "
  - Dynamic padding via DataCollatorForSeq2Seq
  - LR warmup (10%) + cosine decay
  - Gradient clipping (max_norm=1.0)

Expects TSV input: "translate Braille to English: ⠁⠃⠉"<TAB>English text
"""

import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

MODEL_NAME = "google/byt5-small"


def setup_tokenizer(model_name: str = MODEL_NAME):
    """Load ByT5 tokenizer. Processes raw UTF-8 bytes — all Unicode handled natively."""
    return AutoTokenizer.from_pretrained(model_name)


def setup_model(tokenizer, model_name: str = MODEL_NAME):
    """Load ByT5 model."""
    return AutoModelForSeq2SeqLM.from_pretrained(model_name)


class BrailleDataset(Dataset):
    """Dataset for braille→English translation from TSV files.

    Returns un-padded tokenized sequences; padding is handled
    per-batch by DataCollatorForSeq2Seq.
    """

    def __init__(self, tsv_path: str, tokenizer,
                 max_source_len: int = 1024, max_target_len: int = 256):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.pairs = []

        with open(tsv_path) as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    source, target = line.split('\t', 1)
                    self.pairs.append((source, target))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source, target = self.pairs[idx]

        source_enc = self.tokenizer(
            source,
            max_length=self.max_source_len,
            truncation=True,
        )

        target_enc = self.tokenizer(
            target,
            max_length=self.max_target_len,
            truncation=True,
        )

        return {
            "input_ids": source_enc["input_ids"],
            "attention_mask": source_enc["attention_mask"],
            "labels": target_enc["input_ids"],
        }


def decode_predictions(token_ids: torch.Tensor, tokenizer) -> list[str]:
    """Decode model output token IDs to strings."""
    return tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def get_device() -> torch.device:
    """Pick best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(
    train_path: str,
    val_path: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 4,
    grad_accum_steps: int = 8,
    lr: float = 1e-3,
    max_source_len: int = 1024,
    max_target_len: int = 256,
):
    """Fine-tune ByT5-small on braille→English data using Seq2SeqTrainer."""
    device = get_device()
    print(f"Device: {device}")

    tokenizer = setup_tokenizer()
    model = setup_model(tokenizer)

    train_ds = BrailleDataset(train_path, tokenizer, max_source_len, max_target_len)
    val_ds = BrailleDataset(val_path, tokenizer, max_source_len, max_target_len)

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    print(f"Batch size: {batch_size}, Grad accum: {grad_accum_steps}, Effective batch: {batch_size * grad_accum_steps}")
    print(f"Epochs: {epochs}, LR: {lr}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    use_fp16 = torch.cuda.is_available()

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        fp16=use_fp16,
        max_grad_norm=1.0,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    trainer.train()

    # Save final model
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Training complete. Best model saved to {final_dir}/")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train braille→English ByT5 model")
    parser.add_argument("--train", default="data/prepared/train.tsv")
    parser.add_argument("--val", default="data/prepared/val.tsv")
    parser.add_argument("--output", default="models/braille-byt5-v3")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-source-len", type=int, default=1024)
    parser.add_argument("--max-target-len", type=int, default=256)
    args = parser.parse_args()

    train(
        train_path=args.train,
        val_path=args.val,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
    )
