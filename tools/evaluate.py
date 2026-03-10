"""
Evaluate trained braille→English T5 model on test data.

Loads the best checkpoint and runs inference on test and jellybean sets.
Reports exact match, character error rate (CER), and BLEU score.
"""

import os
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


def char_error_rate(predicted: str, expected: str) -> float:
    """Compute character error rate using edit distance."""
    n = len(expected)
    if n == 0:
        return 0.0 if len(predicted) == 0 else 1.0

    m = len(predicted)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            cost = 0 if expected[i - 1] == predicted[j - 1] else 1
            dp[j] = min(prev[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[m] / n


def bleu_score(predicted: str, expected: str) -> float:
    """Simple sentence-level BLEU (4-gram) with brevity penalty."""
    import math

    pred_tokens = predicted.split()
    ref_tokens = expected.split()

    if len(pred_tokens) == 0:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens))) if len(pred_tokens) > 0 else 0.0

    # N-gram precisions (1-4)
    log_avg = 0.0
    for n in range(1, 5):
        pred_ngrams = {}
        for i in range(len(pred_tokens) - n + 1):
            ng = tuple(pred_tokens[i:i + n])
            pred_ngrams[ng] = pred_ngrams.get(ng, 0) + 1

        ref_ngrams = {}
        for i in range(len(ref_tokens) - n + 1):
            ng = tuple(ref_tokens[i:i + n])
            ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1

        clipped = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in pred_ngrams.items())
        total = sum(pred_ngrams.values())

        if total == 0 or clipped == 0:
            return 0.0
        log_avg += math.log(clipped / total) / 4

    return bp * math.exp(log_avg)


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

        show = min(num_samples, len(pairs))
        correct = 0
        total = len(pairs)
        total_cer = 0.0
        total_bleu = 0.0

        for i, (source, expected) in enumerate(pairs):
            input_enc = tokenizer(source, return_tensors="pt", max_length=512,
                                  truncation=True).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_enc["input_ids"],
                    attention_mask=input_enc["attention_mask"],
                    max_length=256,
                )

            predicted = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            is_match = predicted.strip() == expected.strip()
            if is_match:
                correct += 1

            cer = char_error_rate(predicted.strip(), expected.strip())
            bl = bleu_score(predicted.strip(), expected.strip())
            total_cer += cer
            total_bleu += bl

            if i < show:
                match = "OK" if is_match else "MISS"
                print(f"\n[{i+1}] {match}  (CER={cer:.2f}, BLEU={bl:.2f})")
                print(f"  Expected:  {expected}")
                print(f"  Predicted: {predicted}")

        accuracy = correct / total * 100 if total > 0 else 0
        avg_cer = total_cer / total if total > 0 else 0
        avg_bleu = total_bleu / total if total > 0 else 0

        print(f"\n--- {name} ---")
        print(f"  Exact match: {correct}/{total} ({accuracy:.1f}%)")
        print(f"  Avg CER:     {avg_cer:.3f}")
        print(f"  Avg BLEU:    {avg_bleu:.3f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate braille→English model")
    parser.add_argument("--model", default="models/braille-t5-v2/final")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of sample predictions to display per file")
    parser.add_argument("files", nargs="*",
                        default=["data/prepared/test.tsv", "data/prepared/jellybean.tsv"])
    args = parser.parse_args()

    evaluate(args.model, args.files, args.samples)
