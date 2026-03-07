import importlib.util
import os
import tempfile
import pytest

_spec = importlib.util.spec_from_file_location(
    "data_generator",
    os.path.join(os.path.dirname(__file__), '..', 'src', 'data_generator.py'),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

english_to_cell_codes = _mod.english_to_cell_codes
strip_gutenberg = _mod.strip_gutenberg
split_sentences = _mod.split_sentences
generate_training_pairs = _mod.generate_training_pairs


class TestEnglishToCellCodes:
    def test_simple_word(self):
        codes = english_to_cell_codes("the")
        # "the" should contract to a single cell (dots 2,3,4,6 = code 46)
        assert 46 in codes

    def test_returns_list_of_ints(self):
        codes = english_to_cell_codes("Hello world")
        assert isinstance(codes, list)
        assert all(isinstance(c, int) for c in codes)
        assert all(0 <= c <= 63 for c in codes)

    def test_space_preserved(self):
        codes = english_to_cell_codes("a b")
        assert 0 in codes  # space = code 0


class TestStripGutenberg:
    def test_strips_header_and_footer(self):
        text = (
            "The Project Gutenberg eBook of Test\n"
            "some header stuff\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            "\n"
            "This is the actual content.\n"
            "More content here.\n"
            "\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            "some footer stuff\n"
        )
        result = strip_gutenberg(text)
        assert "This is the actual content." in result
        assert "More content here." in result
        assert "Project Gutenberg" not in result
        assert "footer stuff" not in result

    def test_no_markers_returns_original(self):
        text = "Just some plain text.\nWith multiple lines.\n"
        result = strip_gutenberg(text)
        assert result.strip() == text.strip()


class TestSplitSentences:
    def test_simple_sentences(self):
        text = "Hello world. How are you? I am fine."
        sentences = split_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "Hello world."
        assert sentences[1] == "How are you?"
        assert sentences[2] == "I am fine."

    def test_multiline_paragraph(self):
        text = "This is a sentence\nthat wraps across lines. And another one."
        sentences = split_sentences(text)
        assert len(sentences) == 2
        assert "that wraps across lines." in sentences[0]

    def test_skips_empty(self):
        text = "First.   \n\n   Second."
        sentences = split_sentences(text)
        assert len(sentences) == 2


class TestGenerateTrainingPairs:
    def test_produces_pairs(self):
        text = "The cat sat. The dog ran."
        pairs = generate_training_pairs(text)
        assert len(pairs) == 2
        for codes, english in pairs:
            assert isinstance(codes, list)
            assert all(0 <= c <= 63 for c in codes)
            assert isinstance(english, str)
            assert len(english) > 0

    def test_english_preserved(self):
        text = "Hello world."
        pairs = generate_training_pairs(text)
        assert pairs[0][1] == "Hello world."

    def test_full_pipeline_with_file(self):
        """Write a small Gutenberg-style file, generate training pairs."""
        content = (
            "Header stuff\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            "\n"
            "Alice was happy. She ran home.\n"
            "\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            "Footer\n"
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            f.flush()
            try:
                text = strip_gutenberg(open(f.name).read())
                pairs = generate_training_pairs(text)
                assert len(pairs) == 2
                assert pairs[0][1] == "Alice was happy."
                assert pairs[1][1] == "She ran home."
            finally:
                os.unlink(f.name)
