import importlib.util
import os
import tempfile
import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "trainer",
    os.path.join(os.path.dirname(__file__), '..', 'src', 'trainer.py'),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

BrailleDataset = _mod.BrailleDataset
setup_tokenizer = _mod.setup_tokenizer
decode_predictions = _mod.decode_predictions

# --- Fixtures ---

SAMPLE_TSV = "c32 c1 c7\tAlice\nc19 c0 c17\th e\nc46 c0 c9 c1 c30\tthe cat\n"


@pytest.fixture
def tsv_file(tmp_path):
    f = tmp_path / "test.tsv"
    f.write_text(SAMPLE_TSV)
    return str(f)


@pytest.fixture
def tokenizer():
    return setup_tokenizer()


# --- Tests ---


class TestSetupTokenizer:
    def test_adds_custom_tokens(self, tokenizer):
        # All 64 cell tokens should be in vocab
        for i in range(64):
            token = f"c{i}"
            token_id = tokenizer.convert_tokens_to_ids(token)
            assert token_id != tokenizer.unk_token_id, f"{token} not in vocab"

    def test_tokenizes_braille_input(self, tokenizer):
        encoded = tokenizer("c32 c1 c7", return_tensors="pt")
        # Should produce token IDs (not all UNK)
        ids = encoded["input_ids"][0].tolist()
        unk_id = tokenizer.unk_token_id
        # Filter out special tokens (EOS etc) — at least some should be non-UNK
        non_special = [i for i in ids if i != tokenizer.eos_token_id and i != tokenizer.pad_token_id]
        assert all(i != unk_id for i in non_special)

    def test_can_encode_and_decode_english(self, tokenizer):
        text = "Alice was happy."
        encoded = tokenizer(text, return_tensors="pt")
        decoded = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
        assert decoded == text


class TestBrailleDataset:
    def test_length(self, tsv_file, tokenizer):
        ds = BrailleDataset(tsv_file, tokenizer, max_source_len=64, max_target_len=64)
        assert len(ds) == 3

    def test_item_shape(self, tsv_file, tokenizer):
        ds = BrailleDataset(tsv_file, tokenizer, max_source_len=64, max_target_len=64)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert item["input_ids"].dim() == 1
        assert item["attention_mask"].dim() == 1
        assert item["labels"].dim() == 1

    def test_max_length_respected(self, tsv_file, tokenizer):
        ds = BrailleDataset(tsv_file, tokenizer, max_source_len=16, max_target_len=16)
        item = ds[0]
        assert item["input_ids"].shape[0] <= 16
        assert item["labels"].shape[0] <= 16

    def test_labels_ignore_padding(self, tsv_file, tokenizer):
        ds = BrailleDataset(tsv_file, tokenizer, max_source_len=64, max_target_len=64)
        item = ds[0]
        labels = item["labels"]
        # Padding positions should be -100 (ignored in loss)
        # At minimum, non-padding positions should exist
        assert (labels != -100).any()

    def test_all_items_loadable(self, tsv_file, tokenizer):
        ds = BrailleDataset(tsv_file, tokenizer, max_source_len=64, max_target_len=64)
        for i in range(len(ds)):
            item = ds[i]
            assert item["input_ids"].dtype == torch.long


class TestDecodePredictions:
    def test_decode(self, tokenizer):
        # Encode some text, then decode it back
        text = "the cat"
        encoded = tokenizer(text, return_tensors="pt")
        token_ids = encoded["input_ids"]
        decoded = decode_predictions(token_ids, tokenizer)
        assert len(decoded) == 1
        assert decoded[0] == text

    def test_batch_decode(self, tokenizer):
        texts = ["hello", "world"]
        encoded = tokenizer(texts, padding=True, return_tensors="pt")
        decoded = decode_predictions(encoded["input_ids"], tokenizer)
        assert decoded == texts
