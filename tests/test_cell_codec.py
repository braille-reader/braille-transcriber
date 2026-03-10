import importlib.util
import os
import pytest

# Import cell_codec directly to avoid src/__init__.py (which imports detector needing model files)
_spec = importlib.util.spec_from_file_location(
    "cell_codec",
    os.path.join(os.path.dirname(__file__), '..', 'src', 'cell_codec.py'),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

dots_to_code = _mod.dots_to_code
code_to_dots = _mod.code_to_dots
brf_char_to_code = _mod.brf_char_to_code
code_to_brf_char = _mod.code_to_brf_char
dot_notation_to_codes = _mod.dot_notation_to_codes
brf_line_to_codes = _mod.brf_line_to_codes
code_to_unicode = _mod.code_to_unicode
codes_to_unicode = _mod.codes_to_unicode


class TestDotsToCode:
    def test_single_dot(self):
        assert dots_to_code([1]) == 1
        assert dots_to_code([2]) == 2
        assert dots_to_code([3]) == 4
        assert dots_to_code([4]) == 8
        assert dots_to_code([5]) == 16
        assert dots_to_code([6]) == 32

    def test_multiple_dots(self):
        assert dots_to_code([1, 2]) == 3
        assert dots_to_code([1, 2, 5]) == 19
        assert dots_to_code([2, 3, 4, 6]) == 46

    def test_all_dots(self):
        assert dots_to_code([1, 2, 3, 4, 5, 6]) == 63

    def test_empty(self):
        assert dots_to_code([]) == 0

    def test_order_independent(self):
        assert dots_to_code([5, 2, 1]) == dots_to_code([1, 2, 5])


class TestCodeToDots:
    def test_single_dots(self):
        assert code_to_dots(1) == [1]
        assert code_to_dots(2) == [2]
        assert code_to_dots(32) == [6]

    def test_multiple_dots(self):
        assert code_to_dots(3) == [1, 2]
        assert code_to_dots(19) == [1, 2, 5]
        assert code_to_dots(46) == [2, 3, 4, 6]

    def test_all_dots(self):
        assert code_to_dots(63) == [1, 2, 3, 4, 5, 6]

    def test_empty(self):
        assert code_to_dots(0) == []

    def test_roundtrip_all_codes(self):
        for code in range(64):
            assert dots_to_code(code_to_dots(code)) == code


class TestBrfCharToCode:
    """Verify BRF ASCII mapping against known braille letters."""

    def test_space(self):
        assert brf_char_to_code(' ') == 0

    def test_letters(self):
        # A = dot 1
        assert brf_char_to_code('A') == 1
        # B = dots 1,2
        assert brf_char_to_code('B') == 3
        # C = dots 1,4
        assert brf_char_to_code('C') == 9
        # H = dots 1,2,5
        assert brf_char_to_code('H') == 19
        # Z = dots 1,3,5,6
        assert brf_char_to_code('Z') == 53

    def test_common_contractions(self):
        # ! = dots 2,3,4,6 = "the"
        assert brf_char_to_code('!') == 46
        # & = dots 1,2,3,4,6 = "and"
        assert brf_char_to_code('&') == 47
        # + = dots 3,4,6 = "ing"
        assert brf_char_to_code('+') == 44
        # $ = dots 1,2,4,6 = "ed"
        assert brf_char_to_code('$') == 43
        # ] = dots 1,2,4,5,6 = "er"
        assert brf_char_to_code(']') == 59

    def test_indicators(self):
        # , = dot 6 = capital indicator
        assert brf_char_to_code(',') == 32
        # # = dots 3,4,5,6 = number indicator
        assert brf_char_to_code('#') == 60

    def test_case_insensitive(self):
        assert brf_char_to_code('a') == brf_char_to_code('A')

    def test_liblouis_lowercase_brf(self):
        # Liblouis outputs {|}~ instead of [\]^
        assert brf_char_to_code('{') == brf_char_to_code('[')   # dots 2,4,6 = "ow"
        assert brf_char_to_code('|') == brf_char_to_code('\\')  # dots 1,2,5,6 = "ou"
        assert brf_char_to_code('}') == brf_char_to_code(']')   # dots 1,2,4,5,6 = "er"
        assert brf_char_to_code('~') == brf_char_to_code('^')   # dots 4,5
        assert brf_char_to_code('z') == brf_char_to_code('Z')


class TestCodeToBrfChar:
    def test_roundtrip(self):
        for code in range(64):
            char = code_to_brf_char(code)
            assert brf_char_to_code(char) == code


class TestDotNotationToCodes:
    """Test parsing of pipe-separated dot notation from manual data."""

    def test_simple(self):
        # "hello" = h|e|l|l|o
        assert dot_notation_to_codes("1,2,5|1,5|1,2,3|1,2,3|1,3,5") == [19, 17, 7, 7, 21]

    def test_with_spaces(self):
        # space represented as empty between pipes
        assert dot_notation_to_codes("1,2,5| |1,5") == [19, 0, 17]

    def test_single_dot(self):
        # dot 6 = capital indicator
        assert dot_notation_to_codes("6|1") == [32, 1]

    def test_jellybean_first_word(self):
        # ",WAY" = capital W-A-Y
        # 6 = cap(32), 2,4,5,6 = W(58), 1 = A(1), 1,3,4,5,6 = Y(61)
        codes = dot_notation_to_codes("6|2,4,5,6|1|1,3,4,5,6")
        assert codes == [32, 58, 1, 61]


class TestBrfLineToCodes:
    def test_simple_word(self):
        # "HE" = dots 1,2,5 + dots 1,5
        assert brf_line_to_codes("HE") == [19, 17]

    def test_with_space(self):
        assert brf_line_to_codes("A B") == [1, 0, 3]

    def test_contracted(self):
        # ",ALICE" = cap + A + L + I + C + E
        assert brf_line_to_codes(",ALICE") == [32, 1, 7, 10, 9, 17]


class TestBrfAndDotsProduceSameCodes:
    """Cross-verify: BRF and dot notation for the same braille produce identical cell codes."""

    def test_the_contraction(self):
        # "the" = BRF '!' = dots 2,3,4,6
        assert brf_char_to_code('!') == dots_to_code([2, 3, 4, 6])

    def test_and_contraction(self):
        # "and" = BRF '&' = dots 1,2,3,4,6
        assert brf_char_to_code('&') == dots_to_code([1, 2, 3, 4, 6])

    def test_ing_contraction(self):
        # "ing" = BRF '+' = dots 3,4,6
        assert brf_char_to_code('+') == dots_to_code([3, 4, 6])

    def test_capital_indicator(self):
        # capital = BRF ',' = dot 6
        assert brf_char_to_code(',') == dots_to_code([6])

    def test_all_letters(self):
        """Verify all 26 letters produce consistent codes."""
        letter_dots = {
            'A': [1], 'B': [1, 2], 'C': [1, 4], 'D': [1, 4, 5],
            'E': [1, 5], 'F': [1, 2, 4], 'G': [1, 2, 4, 5],
            'H': [1, 2, 5], 'I': [2, 4], 'J': [2, 4, 5],
            'K': [1, 3], 'L': [1, 2, 3], 'M': [1, 3, 4],
            'N': [1, 3, 4, 5], 'O': [1, 3, 5], 'P': [1, 2, 3, 4],
            'Q': [1, 2, 3, 4, 5], 'R': [1, 2, 3, 5], 'S': [2, 3, 4],
            'T': [2, 3, 4, 5], 'U': [1, 3, 6], 'V': [1, 2, 3, 6],
            'W': [2, 4, 5, 6], 'X': [1, 3, 4, 6], 'Y': [1, 3, 4, 5, 6],
            'Z': [1, 3, 5, 6],
        }
        for letter, dots in letter_dots.items():
            assert brf_char_to_code(letter) == dots_to_code(dots), f"Mismatch for letter {letter}"


class TestJellybeanConversion:
    """Verify dot notation from jellybean_jungle.txt converts to valid BRF."""

    def test_first_line_converts_to_brf(self):
        """First line dot patterns should produce the same cell codes as BRF representation."""
        dot_line = "6|2,4,5,6|1|1,3,4,5,6| |1,4,5|2,4,6|1,3,4,5| |2,3,4|1,2,5,6|1,4,5,6| |5|1,5,6| |2,3,4,6| |2,4,5|1,3,6|1,3,4,5|1,2,4,5|1,2,3|1,5| |1,2,4,5|1,3,5|2,4,6|2,3,4|2"
        # English: "Way down south where the jungle grows,"

        dot_codes = dot_notation_to_codes(dot_line)

        # Same sequence via BRF: ,WAY D[N S\\? "W ! JUNGLE GO[S1
        # (where , = cap, [ = ow, \\ = ou, ? = th, " = contraction prefix,
        #  ! = the, 1 = comma in Grade 2)
        # Let's just verify the first word ",WAY" = codes [32, 58, 1, 61]
        assert dot_codes[0] == 32   # dot 6 = capital indicator
        assert dot_codes[1] == 58   # W
        assert dot_codes[2] == 1    # A
        assert dot_codes[3] == 61   # Y
        assert dot_codes[4] == 0    # space

    def test_second_line_farther_and_deeper(self):
        """Verify 'farther and deeper' contractions."""
        dot_line = "1,2,4|3,4,5|2,3,4,6|1,2,3,5| |1,2,3,4,6| |1,4,5|1,5|1,5|1,2,3,4|1,2,4,5,6"
        # English: "farther and deeper"
        # BRF: F>!R & DEEP]
        # f=11, ar(>)=28, the(!)=46, r=23, space, and(&)=47, space, d=25, e=17, e=17, p=15, er(])=59

        codes = dot_notation_to_codes(dot_line)
        expected = [11, 28, 46, 23, 0, 47, 0, 25, 17, 17, 15, 59]
        assert codes == expected


class TestAliceBrfConversion:
    """Verify BRF from Alice in Wonderland converts correctly."""

    def test_chapter_heading(self):
        """',,*APT] #A' = CHAPTER 1 heading."""
        codes = brf_line_to_codes(",,*APT] #A")
        # ,, = two capital indicators (32, 32)
        # * = ch (33), A=1, P=15, T=30, ] = er (59)
        # space
        # # = number indicator (60), A = 1
        assert codes[0] == 32   # first cap indicator
        assert codes[1] == 32   # second cap indicator (word caps)
        assert codes[2] == 33   # * = ch
        assert codes[-2] == 60  # # = number indicator
        assert codes[-1] == 1   # A (= digit 1 after number indicator)

    def test_down_the_rabbit_hole(self):
        """,D[N ! ,RA2IT-,HOLE"""
        codes = brf_line_to_codes(",D[N ! ,RA2IT-,HOLE")
        # ,D[N = cap-D-ow-N = "Down"
        assert codes[0] == 32   # cap
        assert codes[1] == 25   # D
        assert codes[2] == 42   # [ = ow
        assert codes[3] == 29   # N
        assert codes[4] == 0    # space
        assert codes[5] == 46   # ! = the


class TestCodeToUnicode:
    """Test cell code to Unicode braille character conversion."""

    def test_empty_cell(self):
        assert code_to_unicode(0) == '\u2800'  # Braille blank

    def test_dot_1(self):
        assert code_to_unicode(1) == '\u2801'  # ⠁

    def test_all_dots(self):
        assert code_to_unicode(63) == '\u283f'  # ⠿

    def test_known_patterns(self):
        # A = dot 1 = code 1 = U+2801
        assert code_to_unicode(1) == '⠁'
        # B = dots 1,2 = code 3 = U+2803
        assert code_to_unicode(3) == '⠃'
        # "the" contraction = dots 2,3,4,6 = code 46 = U+282E
        assert code_to_unicode(46) == '⠮'


class TestCodesToUnicode:
    def test_hello(self):
        # h=19, e=17, l=7, l=7, o=21
        result = codes_to_unicode([19, 17, 7, 7, 21])
        assert len(result) == 5
        assert all(0x2800 <= ord(c) <= 0x283f for c in result)

    def test_with_space(self):
        result = codes_to_unicode([1, 0, 3])
        assert result[1] == '\u2800'  # space = blank braille

    def test_empty(self):
        assert codes_to_unicode([]) == ''
