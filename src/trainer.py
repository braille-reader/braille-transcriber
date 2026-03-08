"""
T5-small fine-tuning pipeline for braille cell codes → English translation.

Expects TSV input files with format: c32 c1 c7<TAB>English text
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer


CELL_TOKENS = [f"c{i}" for i in range(64)]
MODEL_NAME = "t5-small"


def setup_tokenizer(model_name: str = MODEL_NAME) -> T5Tokenizer:
    """Load T5 tokenizer and add 64 custom braille cell tokens."""
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(CELL_TOKENS)
    return tokenizer


def setup_model(tokenizer: T5Tokenizer, model_name: str = MODEL_NAME) -> T5ForConditionalGeneration:
    """Load T5 model and resize embeddings for custom tokens."""
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    return model


class BrailleDataset(Dataset):
    """Dataset for braille→English translation from TSV files."""

    def __init__(self, tsv_path: str, tokenizer: T5Tokenizer,
                 max_source_len: int = 128, max_target_len: int = 256):
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
            padding="max_length",
            return_tensors="pt",
        )

        target_enc = self.tokenizer(
            target,
            max_length=self.max_target_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels = target_enc["input_ids"].squeeze()
        # Replace padding token ids with -100 so they're ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_enc["input_ids"].squeeze(),
            "attention_mask": source_enc["attention_mask"].squeeze(),
            "labels": labels,
        }


def decode_predictions(token_ids: torch.Tensor, tokenizer: T5Tokenizer) -> list[str]:
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
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum_steps: int = 8,
    lr: float = 3e-4,
    max_source_len: int = 128,
    max_target_len: int = 256,
):
    """Fine-tune T5-small on braille→English data."""
    device = get_device()
    print(f"Device: {device}")

    tokenizer = setup_tokenizer()
    model = setup_model(tokenizer)
    model.to(device)

    train_ds = BrailleDataset(train_path, tokenizer, max_source_len, max_target_len)
    val_ds = BrailleDataset(val_path, tokenizer, max_source_len, max_target_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    print(f"Batch size: {batch_size}, Grad accum: {grad_accum_steps}, Effective batch: {batch_size * grad_accum_steps}")
    print(f"Epochs: {epochs}")

    best_val_loss = float('inf')
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += outputs.loss.item()

            if (step + 1) % 100 == 0:
                avg = total_loss / (step + 1)
                print(f"  Epoch {epoch+1} step {step+1}/{len(train_loader)} loss={avg:.4f}")

        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(os.path.join(output_dir, "best"))
            tokenizer.save_pretrained(os.path.join(output_dir, "best"))
            print(f"  Saved best model (val_loss={val_loss:.4f})")

    # Save final model
    model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print(f"Training complete. Models saved to {output_dir}/")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train braille→English T5 model")
    parser.add_argument("--train", default="data/prepared/train.tsv")
    parser.add_argument("--val", default="data/prepared/val.tsv")
    parser.add_argument("--output", default="models/braille-t5")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-source-len", type=int, default=128)
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
