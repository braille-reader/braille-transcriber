"""Grade 1 Braille interpreter: Braille Unicode → English text."""

# Braille Unicode block starts at U+2800
# Each character encodes 6 dots as bits:
# Dot 1 = bit 0, Dot 2 = bit 1, Dot 3 = bit 2
# Dot 4 = bit 3, Dot 5 = bit 4, Dot 6 = bit 5

# Grade 1 mapping: dot pattern → character
# Dots are numbered:
#   1 4
#   2 5
#   3 6

GRADE1_MAP = {
    # Letters a-j (dots 1,2,4,5 only - top 4 dots)
    '100000': 'a',  # dot 1
    '110000': 'b',  # dots 1,2
    '100100': 'c',  # dots 1,4
    '100110': 'd',  # dots 1,4,5
    '100010': 'e',  # dots 1,5
    '110100': 'f',  # dots 1,2,4
    '110110': 'g',  # dots 1,2,4,5
    '110010': 'h',  # dots 1,2,5
    '010100': 'i',  # dots 2,4
    '010110': 'j',  # dots 2,4,5

    # Letters k-t (a-j + dot 3)
    '101000': 'k',  # dots 1,3
    '111000': 'l',  # dots 1,2,3
    '101100': 'm',  # dots 1,3,4
    '101110': 'n',  # dots 1,3,4,5
    '101010': 'o',  # dots 1,3,5
    '111100': 'p',  # dots 1,2,3,4
    '111110': 'q',  # dots 1,2,3,4,5
    '111010': 'r',  # dots 1,2,3,5
    '011100': 's',  # dots 2,3,4
    '011110': 't',  # dots 2,3,4,5

    # Letters u-z (k-o + dot 6, with exceptions)
    '101001': 'u',  # dots 1,3,6
    '111001': 'v',  # dots 1,2,3,6
    '010111': 'w',  # dots 2,4,5,6
    '101101': 'x',  # dots 1,3,4,6
    '101111': 'y',  # dots 1,3,4,5,6
    '101011': 'z',  # dots 1,3,5,6

    # Numbers use number indicator + a-j
    # Number indicator: dots 3,4,5,6
    '001111': '#',  # number indicator

    # Space (empty cell - no dots)
    '000000': ' ',

    # Common punctuation
    '010000': ',',   # dot 2
    '011000': ';',   # dots 2,3
    '010010': ':',   # dots 2,5
    '011010': '.',   # dots 2,3,5 (period, full stop)
    '011011': '!',   # dots 2,3,5,6
    '010011': '?',   # dots 2,5,6
    '001000': "'",   # dot 3 (apostrophe)
    '001100': '-',   # dots 3,4 (hyphen)

    # Capital indicator: dot 6
    '000001': '^',   # capital indicator (we'll use ^ as marker)

    # Opening/closing quotes, parentheses
    '011001': '(',   # dots 2,3,6 (opening)
    '001011': ')',   # dots 3,5,6 (closing)
    '001001': '"',   # dots 3,6 (quote)
}

# Letters that follow number indicator become digits
LETTER_TO_DIGIT = {
    'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5',
    'f': '6', 'g': '7', 'h': '8', 'i': '9', 'j': '0',
}


def braille_unicode_to_dots(char: str) -> str:
    """Convert a Braille Unicode character to a 6-digit dot pattern string.

    Args:
        char: A single Braille Unicode character (U+2800 to U+283F)

    Returns:
        6-digit string of 0s and 1s representing dots 1-6
    """
    code = ord(char) - 0x2800
    if code < 0 or code > 63:
        return None

    # Extract bits for each dot
    dots = ''
    for i in range(6):
        dots += '1' if (code & (1 << i)) else '0'
    return dots


def interpret_grade1(braille_text: str) -> str:
    """Interpret Grade 1 Braille Unicode text to English.

    Args:
        braille_text: String of Braille Unicode characters

    Returns:
        English text interpretation
    """
    result = []
    number_mode = False
    capital_next = False

    for char in braille_text:
        dots = braille_unicode_to_dots(char)
        if dots is None:
            # Not a braille character, pass through
            result.append(char)
            continue

        if dots == '000000':
            # Space - reset number mode
            result.append(' ')
            number_mode = False
            continue

        if dots == '001111':
            # Number indicator
            number_mode = True
            continue

        if dots == '000001':
            # Capital indicator
            capital_next = True
            continue

        # Look up the character
        letter = GRADE1_MAP.get(dots)
        if letter is None:
            result.append('?')  # Unknown pattern
            continue

        # Handle number mode
        if number_mode and letter in LETTER_TO_DIGIT:
            result.append(LETTER_TO_DIGIT[letter])
        elif capital_next:
            result.append(letter.upper())
            capital_next = False
        else:
            result.append(letter)

    return ''.join(result)


def interpret_lines(lines: list[str]) -> str:
    """Interpret multiple lines of Braille Unicode.

    Args:
        lines: List of Braille Unicode strings, one per line

    Returns:
        Multi-line English text
    """
    return '\n'.join(interpret_grade1(line) for line in lines)
