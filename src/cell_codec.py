"""
Cell codec: conversions between braille representations.

Three equivalent representations of the same 64 braille cells:
  - Dot patterns: list of dot numbers [1,2,5]
  - Cell codes: integer 0-63 (sum of 2^(dot-1))
  - BRF ASCII: single character from the Braille ASCII standard

All conversions are lossless 1:1 mappings.
"""

# Standard North American Braille ASCII mapping.
# Maps ASCII characters (space through underscore) to their dot patterns.
# Reference: https://en.wikipedia.org/wiki/Braille_ASCII
_BRF_TO_DOTS = {
    ' ': [],
    '!': [2, 3, 4, 6],
    '"': [5],
    '#': [3, 4, 5, 6],
    '$': [1, 2, 4, 6],
    '%': [1, 4, 6],
    '&': [1, 2, 3, 4, 6],
    "'": [3],
    '(': [1, 2, 3, 5, 6],
    ')': [2, 3, 4, 5, 6],
    '*': [1, 6],
    '+': [3, 4, 6],
    ',': [6],
    '-': [3, 6],
    '.': [4, 6],
    '/': [3, 4],
    '0': [3, 5, 6],
    '1': [2],
    '2': [2, 3],
    '3': [2, 5],
    '4': [2, 5, 6],
    '5': [2, 6],
    '6': [2, 3, 5],
    '7': [2, 3, 5, 6],
    '8': [2, 3, 6],
    '9': [3, 5],
    ':': [1, 5, 6],
    ';': [5, 6],
    '<': [1, 2, 6],
    '=': [1, 2, 3, 4, 5, 6],
    '>': [3, 4, 5],
    '?': [1, 4, 5, 6],
    '@': [4],
    'A': [1],
    'B': [1, 2],
    'C': [1, 4],
    'D': [1, 4, 5],
    'E': [1, 5],
    'F': [1, 2, 4],
    'G': [1, 2, 4, 5],
    'H': [1, 2, 5],
    'I': [2, 4],
    'J': [2, 4, 5],
    'K': [1, 3],
    'L': [1, 2, 3],
    'M': [1, 3, 4],
    'N': [1, 3, 4, 5],
    'O': [1, 3, 5],
    'P': [1, 2, 3, 4],
    'Q': [1, 2, 3, 4, 5],
    'R': [1, 2, 3, 5],
    'S': [2, 3, 4],
    'T': [2, 3, 4, 5],
    'U': [1, 3, 6],
    'V': [1, 2, 3, 6],
    'W': [2, 4, 5, 6],
    'X': [1, 3, 4, 6],
    'Y': [1, 3, 4, 5, 6],
    'Z': [1, 3, 5, 6],
    '[': [2, 4, 6],
    '\\': [1, 2, 5, 6],
    ']': [1, 2, 4, 5, 6],
    '^': [4, 5],
    '_': [4, 5, 6],
}

# Build reverse lookup: cell code → BRF character
_CODE_TO_BRF = {}
_BRF_TO_CODE = {}
for _char, _dots in _BRF_TO_DOTS.items():
    _code = sum(1 << (d - 1) for d in _dots)
    _CODE_TO_BRF[_code] = _char
    _BRF_TO_CODE[_char] = _code


def dots_to_code(dots: list[int]) -> int:
    """Convert dot numbers to cell code. e.g. [1,2,5] → 19"""
    return sum(1 << (d - 1) for d in dots)


def code_to_dots(code: int) -> list[int]:
    """Convert cell code to dot numbers. e.g. 19 → [1,2,5]"""
    return [d + 1 for d in range(6) if code & (1 << d)]


# Liblouis outputs lowercase BRF where `{|}~ map to the same cells as @[\]^
_LOWERCASE_BRF_MAP = {'`': '@', '{': '[', '|': '\\', '}': ']', '~': '^'}


def brf_char_to_code(char: str) -> int:
    """Convert a BRF ASCII character to cell code. Handles both upper and lowercase BRF."""
    char = _LOWERCASE_BRF_MAP.get(char, char).upper()
    return _BRF_TO_CODE[char]


def code_to_brf_char(code: int) -> str:
    """Convert cell code to BRF ASCII character."""
    return _CODE_TO_BRF[code]


def dot_notation_to_codes(line: str) -> list[int]:
    """Parse pipe-separated dot notation to cell codes.

    Format: "1,2,5|1,5| |1,2,3"
    Space between pipes represents an empty cell (space = code 0).
    """
    codes = []
    cells = line.split('|')
    for cell in cells:
        cell = cell.strip()
        if cell == '' or cell == ' ':
            codes.append(0)
        else:
            dots = [int(d.strip()) for d in cell.split(',')]
            codes.append(dots_to_code(dots))
    return codes


def brf_line_to_codes(line: str) -> list[int]:
    """Convert a line of BRF text to cell codes."""
    return [brf_char_to_code(ch) for ch in line]


def code_to_unicode(code: int) -> str:
    """Convert cell code (0-63) to Unicode Braille character (U+2800-U+283F)."""
    return chr(0x2800 + code)


def codes_to_unicode(codes: list[int]) -> str:
    """Convert list of cell codes to Unicode Braille string."""
    return ''.join(code_to_unicode(c) for c in codes)
