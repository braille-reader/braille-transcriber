"""
Evaluate trained braille→English T5 model on test data.

Loads the best checkpoint and runs inference on test and jellybean sets.
Prints sample predictions side-by-side with ground truth.
"""

import os
import sys
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def load_tsv(path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                source, target = line.split('\t', 1)
                pairs.append((source, target))
    return pairs


def evaluate(model_dir: str, test_files: list[str], num_samples: int = 10):
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    for test_file in test_files:
        name = os.path.basename(test_file)
        pairs = load_tsv(test_file)
        print(f"\n{'='*60}")
        print(f"  {name} ({len(pairs)} samples)")
        print(f"{'='*60}")

        # Show samples
        show = min(num_samples, len(pairs))
        correct = 0
        total = len(pairs)

        for i, (source, expected) in enumerate(pairs):
            input_enc = tokenizer(source, return_tensors="pt", max_length=128,
                                  truncation=True).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_enc["input_ids"],
                    attention_mask=input_enc["attention_mask"],
                    max_length=256,
                )

            predicted = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            if predicted.strip() == expected.strip():
                correct += 1

            if i < show:
                match = "OK" if predicted.strip() == expected.strip() else "MISS"
                print(f"\n[{i+1}] {match}")
                print(f"  Expected:  {expected}")
                print(f"  Predicted: {predicted}")

        accuracy = correct / total * 100 if total > 0 else 0
        print(f"\n--- {name}: {correct}/{total} exact match ({accuracy:.1f}%) ---")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate braille→English model")
    parser.add_argument("--model", default="models/braille-t5/best")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of sample predictions to display per file")
    parser.add_argument("files", nargs="*",
                        default=["data/prepared/test.tsv", "data/prepared/jellybean.tsv"])
    args = parser.parse_args()

    evaluate(args.model, args.files, args.samples)
