"""
Prepare training data for T5 fine-tuning.

Converts raw data sources into T5-ready format:
  - Jellybean manual data (dot notation + English) → cell codes + English
  - Synthetic .train files (cell codes + English) → train/val/test splits
  - Output: T5 token format (c0 c46 c1 ...) paired with English text
"""

import os
import random
import importlib.util

# Import cell_codec directly to avoid src/__init__.py
_codec_spec = importlib.util.spec_from_file_location(
    "cell_codec",
    os.path.join(os.path.dirname(__file__), '..', 'src', 'cell_codec.py'),
)
_codec = importlib.util.module_from_spec(_codec_spec)
_codec_spec.loader.exec_module(_codec)
dot_notation_to_codes = _codec.dot_notation_to_codes


def parse_jellybean(text: str) -> list[tuple[list[int], str]]:
    """Parse jellybean-format text: alternating dot-notation / English lines.

    Format:
        1,2,5|1,5|1,2,3
        hello
        (blank line)
        ...
    """
    lines = text.split('\n')
    pairs = []
    i = 0
    while i < len(lines):
        # Skip blank lines
        if not lines[i].strip():
            i += 1
            continue
        # Expect dot notation line followed by English line
        dot_line = lines[i].strip()
        if i + 1 < len(lines):
            english = lines[i + 1].strip()
            codes = dot_notation_to_codes(dot_line)
            pairs.append((codes, english))
            i += 2
        else:
            break
    return pairs


def load_synthetic(filepath: str) -> list[tuple[list[int], str]]:
    """Load a .train file with space-separated cell codes and English text.

    Format:
        32 1 7 10 9 17
        Alice
        (blank line)
        ...
    """
    pairs = []
    with open(filepath) as f:
        lines = f.read().split('\n')

    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        code_line = lines[i].strip()
        # Verify this looks like cell codes (all digits and spaces)
        parts = code_line.split()
        try:
            codes = [int(p) for p in parts]
        except ValueError:
            i += 1
            continue
        if i + 1 < len(lines) and lines[i + 1].strip():
            english = lines[i + 1].strip()
            pairs.append((codes, english))
            i += 2
        else:
            i += 1
    return pairs


def split_data(
    pairs: list[tuple[list[int], str]],
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Split pairs into train/val/test sets."""
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return train, val, test


def to_t5_format(pairs: list[tuple[list[int], str]]) -> list[tuple[str, str]]:
    """Convert (cell_codes, english) pairs to T5 input/output strings.

    Input format: "c32 c1 c7"
    Output format: "Alice"
    """
    rows = []
    for codes, english in pairs:
        input_str = ' '.join(f'c{c}' for c in codes)
        rows.append((input_str, english))
    return rows


def write_t5_file(rows: list[tuple[str, str]], filepath: str):
    """Write T5-format pairs to a TSV file (input<tab>output)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for input_str, output_str in rows:
            f.write(f"{input_str}\t{output_str}\n")


if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    synthetic_dir = os.path.join(base_dir, 'data', 'synthetic')
    manual_file = os.path.join(base_dir, 'data', 'manual', 'jellybean_jungle.txt')
    out_dir = os.path.join(base_dir, 'data', 'prepared')

    # Load synthetic data
    synthetic_pairs = []
    for fname in sorted(os.listdir(synthetic_dir)):
        if fname.endswith('.train'):
            path = os.path.join(synthetic_dir, fname)
            pairs = load_synthetic(path)
            print(f"  {fname}: {len(pairs)} pairs")
            synthetic_pairs.extend(pairs)
    print(f"Total synthetic: {len(synthetic_pairs)} pairs")

    # Split synthetic into train/val/test
    train, val, test = split_data(synthetic_pairs, val_ratio=0.05, test_ratio=0.05, seed=42)
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # Load jellybean as held-out real-world test
    with open(manual_file) as f:
        jellybean_pairs = parse_jellybean(f.read())
    print(f"Jellybean (held-out): {len(jellybean_pairs)} pairs")

    # Convert to T5 format and write
    write_t5_file(to_t5_format(train), os.path.join(out_dir, 'train.tsv'))
    write_t5_file(to_t5_format(val), os.path.join(out_dir, 'val.tsv'))
    write_t5_file(to_t5_format(test), os.path.join(out_dir, 'test.tsv'))
    write_t5_file(to_t5_format(jellybean_pairs), os.path.join(out_dir, 'jellybean.tsv'))

    print(f"\nFiles written to {out_dir}/")
    for name in ['train.tsv', 'val.tsv', 'test.tsv', 'jellybean.tsv']:
        path = os.path.join(out_dir, name)
        lines = sum(1 for _ in open(path))
        print(f"  {name}: {lines} rows")
