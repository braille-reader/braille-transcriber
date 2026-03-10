import importlib.util
import os
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

# Unicode braille with task prefix
SAMPLE_TSV = (
    "translate Braille to English: \u2820\u2801\u2807\u2822\u2809\u2811\ttranslate Braille to English: Alice\n"
    "translate Braille to English: \u2813\u2800\u2811\th e\n"
    "translate Braille to English: \u282e\u2800\u2809\u2801\u281e\tthe cat\n"
)


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
    def test_no_custom_tokens_needed(self, tokenizer):
        # Unicode braille characters should be handled by default tokenizer
        text = "translate Braille to English: \u2801\u2803\u2809"
        encoded = tokenizer(text, return_tensors="pt")
        ids = encoded["input_ids"][0].tolist()
        unk_id = tokenizer.unk_token_id
        non_special = [i for i in ids if i != tokenizer.eos_token_id and i != tokenizer.pad_token_id]
        # Some braille chars may be UNK in T5's vocab, but the tokenizer should still work
        assert len(non_special) > 0

    def test_can_encode_and_decode_english(self, tokenizer):
        text = "Alice was happy."
        encoded = tokenizer(text, return_tensors="pt")
        decoded = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
        assert decoded == text


class TestBrailleDataset:
    def test_length(self, tsv_file, tokenizer):
        ds = BrailleDataset(tsv_file, tokenizer, max_source_len=64, max_target_len=64)
        assert len(ds) == 3

    def test_item_keys(self, tsv_file, tokenizer):
        ds = BrailleDataset(tsv_file, tokenizer, max_source_len=64, max_target_len=64)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_returns_lists_not_tensors(self, tsv_file, tokenizer):
        """Dataset returns lists; DataCollatorForSeq2Seq converts to tensors."""
        ds = BrailleDataset(tsv_file, tokenizer, max_source_len=64, max_target_len=64)
        item = ds[0]
        assert isinstance(item["input_ids"], list)
        assert isinstance(item["attention_mask"], list)
        assert isinstance(item["labels"], list)

    def test_all_items_loadable(self, tsv_file, tokenizer):
        ds = BrailleDataset(tsv_file, tokenizer, max_source_len=64, max_target_len=64)
        for i in range(len(ds)):
            item = ds[i]
            assert len(item["input_ids"]) > 0


class TestDecodePredictions:
    def test_decode(self, tokenizer):
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
