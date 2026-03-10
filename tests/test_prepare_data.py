import importlib.util
import os
import tempfile
import pytest

# Import modules directly to avoid src/__init__.py
_codec_spec = importlib.util.spec_from_file_location(
    "cell_codec",
    os.path.join(os.path.dirname(__file__), '..', 'src', 'cell_codec.py'),
)
_codec = importlib.util.module_from_spec(_codec_spec)
_codec_spec.loader.exec_module(_codec)

_prep_spec = importlib.util.spec_from_file_location(
    "prepare_data",
    os.path.join(os.path.dirname(__file__), '..', 'tools', 'prepare_data.py'),
)
_prep = importlib.util.module_from_spec(_prep_spec)
_prep_spec.loader.exec_module(_prep)

parse_jellybean = _prep.parse_jellybean
load_synthetic = _prep.load_synthetic
split_data = _prep.split_data
to_t5_format = _prep.to_t5_format
TASK_PREFIX = _prep.TASK_PREFIX


class TestParseJellybean:
    """Parse alternating dot-notation / English lines from manual data."""

    def test_single_pair(self):
        text = "1,2,5|1,5|1,2,3|1,2,3|1,3,5\nhello\n"
        pairs = parse_jellybean(text)
        assert len(pairs) == 1
        codes, english = pairs[0]
        assert codes == [19, 17, 7, 7, 21]
        assert english == "hello"

    def test_multiple_pairs(self):
        text = (
            "6|2,4,5,6|1|1,3,4,5,6| |1,4,5|2,4,6|1,3,4,5| |2,3,4|1,2,5,6|1,4,5,6| |5|1,5,6| |2,3,4,6| |2,4,5|1,3,6|1,3,4,5|1,2,4,5|1,2,3|1,5| |1,2,4,5|1,2,3,5|2,4,6|2,3,4|2\n"
            "Way down south where the jungle grows,\n"
            "\n"
            "1,2,4|3,4,5|2,3,4,6|1,2,3,5| |1,2,3,4,6| |1,4,5|1,5|1,5|1,2,3,4|1,2,4,5,6| |1,4,5,6|1|1,3,4,5| |1|1,3,4,5|1,3,4,5,6|5|1,3,5| |1,2,4,5|1,3,5|1,5|2,3,4|3,6|3,6\n"
            "farther and deeper than anyone goes--\n"
        )
        pairs = parse_jellybean(text)
        assert len(pairs) == 2
        assert pairs[0][1] == "Way down south where the jungle grows,"
        assert pairs[1][1] == "farther and deeper than anyone goes--"

    def test_skips_blank_lines(self):
        text = "\n\n1|2|3\nabc\n\n\n"
        pairs = parse_jellybean(text)
        assert len(pairs) == 1

    def test_cell_codes_are_ints(self):
        text = "1,2,5|1,5\nhello\n"
        pairs = parse_jellybean(text)
        codes = pairs[0][0]
        assert all(isinstance(c, int) for c in codes)
        assert all(0 <= c <= 63 for c in codes)


class TestLoadSynthetic:
    """Load existing .train files (space-separated cell codes + English)."""

    def test_loads_pairs(self):
        content = "32 1 7 10 9 17\nAlice\n\n19 17 7 7 21\nhello\n\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.train', delete=False) as f:
            f.write(content)
            f.flush()
            try:
                pairs = load_synthetic(f.name)
                assert len(pairs) == 2
                assert pairs[0] == ([32, 1, 7, 10, 9, 17], "Alice")
                assert pairs[1] == ([19, 17, 7, 7, 21], "hello")
            finally:
                os.unlink(f.name)

    def test_skips_empty_codes(self):
        content = "\norphan line\n\n32 1\nAlice\n\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.train', delete=False) as f:
            f.write(content)
            f.flush()
            try:
                pairs = load_synthetic(f.name)
                assert len(pairs) == 1
                assert pairs[0] == ([32, 1], "Alice")
            finally:
                os.unlink(f.name)


class TestSplitData:
    """Split pairs into train/val/test sets."""

    def test_split_ratios(self):
        pairs = [([], f"s{i}") for i in range(100)]
        train, val, test = split_data(pairs, val_ratio=0.05, test_ratio=0.05, seed=42)
        assert len(train) == 90
        assert len(val) == 5
        assert len(test) == 5

    def test_no_overlap(self):
        pairs = [([i], f"s{i}") for i in range(100)]
        train, val, test = split_data(pairs, val_ratio=0.05, test_ratio=0.05, seed=42)
        train_texts = {t for _, t in train}
        val_texts = {t for _, t in val}
        test_texts = {t for _, t in test}
        assert len(train_texts & val_texts) == 0
        assert len(train_texts & test_texts) == 0
        assert len(val_texts & test_texts) == 0

    def test_deterministic(self):
        pairs = [([i], f"s{i}") for i in range(100)]
        t1, v1, te1 = split_data(pairs, val_ratio=0.1, test_ratio=0.1, seed=7)
        t2, v2, te2 = split_data(pairs, val_ratio=0.1, test_ratio=0.1, seed=7)
        assert t1 == t2
        assert v1 == v2
        assert te1 == te2

    def test_all_data_preserved(self):
        pairs = [([i], f"s{i}") for i in range(50)]
        train, val, test = split_data(pairs, val_ratio=0.1, test_ratio=0.1, seed=42)
        assert len(train) + len(val) + len(test) == 50


class TestToT5Format:
    """Convert (cell_codes, english) pairs to T5 input/output strings."""

    def test_format_uses_unicode_braille(self):
        pairs = [([1, 3, 9], "abc")]
        rows = to_t5_format(pairs)
        assert len(rows) == 1
        source, target = rows[0]
        # Cell code 1 = U+2801, 3 = U+2803, 9 = U+2809
        assert source == f"{TASK_PREFIX}\u2801\u2803\u2809"
        assert target == "abc"

    def test_space_cell(self):
        pairs = [([19, 0, 17], "h e")]
        rows = to_t5_format(pairs)
        source = rows[0][0]
        # Cell code 0 = U+2800 (blank braille)
        assert source == f"{TASK_PREFIX}\u2813\u2800\u2811"

    def test_task_prefix_present(self):
        pairs = [([1], "a")]
        rows = to_t5_format(pairs)
        assert rows[0][0].startswith("translate Braille to English: ")

    def test_multiple(self):
        pairs = [([1], "a"), ([3], "b")]
        rows = to_t5_format(pairs)
        assert len(rows) == 2
