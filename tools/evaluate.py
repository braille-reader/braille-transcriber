"""
Evaluate braille→English model predictions.

Reads prediction TSVs (from Colab GPU inference) and computes:
- Raw and normalized exact match
- Character error rate (CER)
- BLEU score
- Liblouis back-translation baseline comparison
- Error analysis (severity, patterns, length correlation)

Usage:
    python tools/evaluate.py data/predictions_test.tsv data/predictions_jellybean.tsv
    python tools/evaluate.py data/predictions_jellybean.tsv --samples 42
"""

import os
import sys
import math
import argparse
import importlib.util

# Import cell_codec directly to avoid src/__init__.py triggering detector import
_codec_spec = importlib.util.spec_from_file_location(
    'cell_codec',
    os.path.join(os.path.dirname(__file__), '..', 'src', 'cell_codec.py')
)
cell_codec = importlib.util.module_from_spec(_codec_spec)
_codec_spec.loader.exec_module(cell_codec)


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_quotes(text):
    """Normalize smart/curly quotes and dashes to ASCII equivalents."""
    replacements = {
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201C': '"',   # left double quote
        '\u201D': '"',   # right double quote
        '\u2013': '-',   # en dash
        '\u2014': '--',  # em dash
        '\u2026': '...', # ellipsis
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def char_error_rate(predicted, expected):
    """Character error rate via edit distance."""
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


def bleu_score(predicted, expected):
    """Sentence-level BLEU (4-gram) with brevity penalty."""
    pred_tokens = predicted.split()
    ref_tokens = expected.split()
    if len(pred_tokens) == 0:
        return 0.0
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens)))
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
        clipped = sum(min(c, ref_ngrams.get(ng, 0)) for ng, c in pred_ngrams.items())
        total = sum(pred_ngrams.values())
        if total == 0 or clipped == 0:
            return 0.0
        log_avg += math.log(clipped / total) / 4
    return bp * math.exp(log_avg)


# ---------------------------------------------------------------------------
# Liblouis baseline
# ---------------------------------------------------------------------------

def _setup_liblouis():
    """Try to import liblouis. Returns (back_translate_fn, available_bool)."""
    try:
        import louis
        tables = ['en-ueb-g2.ctb']
        # Quick test
        louis.backTranslateString(tables, ',MR4')

        def back_translate(braille_unicode):
            try:
                codes = [ord(ch) - 0x2800 for ch in braille_unicode
                         if 0 <= ord(ch) - 0x2800 <= 63]
                brf = ''.join(cell_codec.code_to_brf_char(c) for c in codes)
                return louis.backTranslateString(tables, brf)
            except Exception:
                return None

        return back_translate, True
    except Exception as e:
        print(f"  Liblouis not available ({e}) — skipping baseline.")
        return lambda x: None, False


# ---------------------------------------------------------------------------
# Load predictions
# ---------------------------------------------------------------------------

def load_predictions(path):
    """Load prediction TSV: braille<TAB>expected<TAB>predicted"""
    rows = []
    with open(path) as f:
        header = f.readline()
        for line in f:
            line = line.rstrip('\n')
            parts = line.split('\t', 2)
            if len(parts) == 3:
                rows.append({
                    'braille': parts[0],
                    'expected': parts[1],
                    'predicted': parts[2],
                })
    return rows


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_file(path, louis_fn, has_louis, num_samples=10):
    """Run full evaluation on a predictions file."""
    rows = load_predictions(path)
    name = os.path.basename(path)
    total = len(rows)

    if total == 0:
        print(f"\n{name}: No predictions found.")
        return []

    print(f"\n{'='*70}")
    print(f"  {name} ({total} samples)")
    print(f"{'='*70}")

    results = []
    for r in rows:
        exp = r['expected'].strip()
        pred = r['predicted'].strip()
        exp_norm = normalize_quotes(exp)
        pred_norm = normalize_quotes(pred)

        louis_pred = louis_fn(r['braille'])
        louis_norm = normalize_quotes(louis_pred.strip()) if louis_pred else None

        result = {
            'idx': len(results),
            'expected': exp,
            'predicted': pred,
            'expected_norm': exp_norm,
            'predicted_norm': pred_norm,
            'raw_match': pred == exp,
            'norm_match': pred_norm == exp_norm,
            'cer': char_error_rate(pred_norm, exp_norm),
            'bleu': bleu_score(pred_norm, exp_norm),
            'louis_pred': louis_pred,
            'louis_norm_match': (louis_norm.lower() == exp_norm.lower()) if louis_norm else None,
            'louis_cer': char_error_rate(louis_norm.lower(), exp_norm.lower()) if louis_norm else None,
        }
        results.append(result)

    # Show sample misses
    misses = [r for r in results if not r['norm_match']]
    show = min(num_samples, len(misses))
    if show > 0:
        print(f"\n--- Misses (showing {show}/{len(misses)}) ---")
        for r in misses[:show]:
            print(f"\n  [{r['idx']+1}] CER={r['cer']:.3f}  BLEU={r['bleu']:.2f}")
            print(f"    Expected:  {r['expected']}")
            print(f"    Predicted: {r['predicted']}")
            if r['louis_pred']:
                print(f"    Liblouis:  {r['louis_pred']}")

    # Summary
    raw_correct = sum(1 for r in results if r['raw_match'])
    norm_correct = sum(1 for r in results if r['norm_match'])
    avg_cer = sum(r['cer'] for r in results) / total
    avg_bleu = sum(r['bleu'] for r in results) / total

    print(f"\n  ByT5 model:")
    print(f"    Raw exact match:     {raw_correct}/{total} ({raw_correct/total*100:.1f}%)")
    print(f"    Normalized match:    {norm_correct}/{total} ({norm_correct/total*100:.1f}%)")
    print(f"    Avg CER (norm):      {avg_cer:.4f}")
    print(f"    Avg BLEU (norm):     {avg_bleu:.3f}")

    if has_louis:
        louis_results = [r for r in results if r['louis_norm_match'] is not None]
        if louis_results:
            louis_correct = sum(1 for r in louis_results if r['louis_norm_match'])
            louis_avg_cer = sum(r['louis_cer'] for r in louis_results) / len(louis_results)
            print(f"\n  Liblouis baseline (case-insensitive):")
            print(f"    Exact match:         {louis_correct}/{len(louis_results)} ({louis_correct/len(louis_results)*100:.1f}%)")
            print(f"    Avg CER:             {louis_avg_cer:.4f}")

    return results


def error_analysis(results, name):
    """Categorize and analyze errors."""
    misses = [r for r in results if not r['norm_match']]
    total = len(results)

    if not misses:
        print(f"\n{name}: Perfect — no errors after normalization!")
        return

    print(f"\n{'='*70}")
    print(f"  ERROR ANALYSIS: {name}")
    print(f"  {len(misses)} errors out of {total} samples")
    print(f"{'='*70}")

    # Severity buckets
    minor = [r for r in misses if r['cer'] < 0.1]
    moderate = [r for r in misses if 0.1 <= r['cer'] < 0.3]
    severe = [r for r in misses if r['cer'] >= 0.3]

    print(f"\n  By severity:")
    print(f"    Minor   (CER < 0.1):   {len(minor)}")
    print(f"    Moderate (0.1-0.3):    {len(moderate)}")
    print(f"    Severe  (CER >= 0.3):  {len(severe)}")

    # CER distribution
    cers = sorted([r['cer'] for r in misses])
    print(f"\n  CER distribution of misses:")
    print(f"    Min:    {cers[0]:.4f}")
    print(f"    Median: {cers[len(cers)//2]:.4f}")
    print(f"    Max:    {cers[-1]:.4f}")
    print(f"    Mean:   {sum(cers)/len(cers):.4f}")

    # Severe errors
    if severe:
        print(f"\n  --- Severe errors (CER >= 0.3) ---")
        for r in severe:
            print(f"\n  [{r['idx']+1}] CER={r['cer']:.3f}")
            print(f"    Expected:  {r['expected'][:100]}")
            print(f"    Predicted: {r['predicted'][:100]}")

    # Common character-level differences
    print(f"\n  --- Common character differences ---")
    char_diffs = {}
    for r in misses:
        for ec, pc in zip(r['expected_norm'], r['predicted_norm']):
            if ec != pc:
                key = f"'{ec}'->'{pc}'"
                char_diffs[key] = char_diffs.get(key, 0) + 1
    for diff, count in sorted(char_diffs.items(), key=lambda x: -x[1])[:15]:
        print(f"    {diff}: {count}x")

    # Length correlation
    miss_lens = [len(r['expected']) for r in misses]
    all_lens = [len(r['expected']) for r in results]
    print(f"\n  --- Length analysis ---")
    print(f"    Avg length (all):    {sum(all_lens)/len(all_lens):.0f} chars")
    print(f"    Avg length (misses): {sum(miss_lens)/len(miss_lens):.0f} chars")


def print_summary(all_results):
    """Print final summary table across all test sets."""
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY — ByT5-small v3 Evaluation")
    print(f"{'='*70}\n")

    print(f"  {'Dataset':<25} {'N':>5} {'Raw Match':>12} {'Norm Match':>12} {'CER':>8} {'BLEU':>8}")
    print(f"  {'-'*25} {'-'*5} {'-'*12} {'-'*12} {'-'*8} {'-'*8}")

    for name, results in all_results:
        t = len(results)
        raw = sum(1 for r in results if r['raw_match'])
        norm = sum(1 for r in results if r['norm_match'])
        cer = sum(r['cer'] for r in results) / t
        bleu = sum(r['bleu'] for r in results) / t
        print(f"  {name:<25} {t:>5} "
              f"{raw:>4}/{t} ({raw/t*100:4.1f}%) "
              f"{norm:>4}/{t} ({norm/t*100:4.1f}%) "
              f"{cer:>7.4f} {bleu:>7.3f}")

    # Liblouis baseline table
    has_any_louis = any(
        any(r['louis_norm_match'] is not None for r in results)
        for _, results in all_results
    )
    if has_any_louis:
        print(f"\n  {'Dataset':<25} {'N':>5} {'Louis Match':>14} {'Louis CER':>12}")
        print(f"  {'-'*25} {'-'*5} {'-'*14} {'-'*12}")
        for name, results in all_results:
            lr = [r for r in results if r['louis_norm_match'] is not None]
            if lr:
                lc = sum(1 for r in lr if r['louis_norm_match'])
                lcer = sum(r['louis_cer'] for r in lr) / len(lr)
                print(f"  {name:<25} {len(lr):>5} "
                      f"{lc:>4}/{len(lr)} ({lc/len(lr)*100:4.1f}%) "
                      f"{lcer:>11.4f}")
        print(f"\n  Note: Liblouis comparison is case-insensitive (it outputs uppercase).")

    print(f"\n  'Normalized' = after smart quote/dash normalization to ASCII.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate braille model predictions (from Colab) with liblouis baseline"
    )
    parser.add_argument("files", nargs="+",
                        help="Prediction TSV files (braille<TAB>expected<TAB>predicted)")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of sample misses to display per file")
    args = parser.parse_args()

    louis_fn, has_louis = _setup_liblouis()
    if has_louis:
        print("  Liblouis ready for baseline comparison.")

    all_results = []
    for path in args.files:
        results = evaluate_file(path, louis_fn, has_louis, args.samples)
        if results:
            error_analysis(results, os.path.basename(path))
            all_results.append((os.path.basename(path), results))

    if len(all_results) > 1:
        print_summary(all_results)


if __name__ == '__main__':
    main()
