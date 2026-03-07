"""
Synthetic training data generator.

Takes English text, translates to contracted braille via liblouis,
converts to cell codes, and outputs paired training data.

Pipeline: English text → liblouis (Grade 2 BRF) → cell codes (0-63)
"""

import re
import os
import importlib.util
import louis

# Import cell_codec directly to avoid src/__init__.py triggering detector import
_codec_spec = importlib.util.spec_from_file_location(
    "cell_codec",
    os.path.join(os.path.dirname(__file__), 'cell_codec.py'),
)
_codec = importlib.util.module_from_spec(_codec_spec)
_codec_spec.loader.exec_module(_codec)
brf_char_to_code = _codec.brf_char_to_code


_TABLES = ["en-ueb-g2.ctb"]


def english_to_cell_codes(text: str) -> list[int]:
    """Translate English text to contracted braille cell codes via liblouis."""
    brf = louis.translateString(_TABLES, text)
    return [brf_char_to_code(ch) for ch in brf]


def strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header and footer from a text file."""
    lines = text.split('\n')

    start = 0
    end = len(lines)

    for i, line in enumerate(lines):
        if re.match(r'\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG', line, re.IGNORECASE):
            start = i + 1
            break

    for i in range(len(lines) - 1, -1, -1):
        if re.match(r'\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG', lines[i], re.IGNORECASE):
            end = i
            break

    return '\n'.join(lines[start:end]).strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Joins lines within paragraphs first."""
    # Collapse line breaks within paragraphs (keep paragraph breaks)
    text = re.sub(r'\n{2,}', '\n\n', text)
    paragraphs = text.split('\n\n')

    sentences = []
    for para in paragraphs:
        # Join wrapped lines within a paragraph
        para = ' '.join(para.split())
        if not para:
            continue
        # Split on sentence-ending punctuation followed by space or end
        parts = re.split(r'(?<=[.!?])\s+', para)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

    return sentences


def generate_training_pairs(text: str) -> list[tuple[list[int], str]]:
    """Generate (cell_codes, english) training pairs from English text."""
    sentences = split_sentences(text)
    pairs = []
    for sentence in sentences:
        codes = english_to_cell_codes(sentence)
        pairs.append((codes, sentence))
    return pairs
