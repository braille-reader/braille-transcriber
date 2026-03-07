# Braille Cell Codes Reference

This table maps braille cells to their numerical codes.

## Encoding System

Each braille cell has 6 dots arranged as:
```
1 4
2 5
3 6
```

The numerical code is calculated by treating each dot as a bit:
- Dot 1 = 1
- Dot 2 = 2
- Dot 3 = 4
- Dot 4 = 8
- Dot 5 = 16
- Dot 6 = 32

**Formula:** `code = (dot1×1) + (dot2×2) + (dot3×4) + (dot4×8) + (dot5×16) + (dot6×32)`

---

## Complete Code Table (0-63)

| Code | Braille | Dots | Grade 1 Meaning |
|------|---------|------|-----------------|
| 0 | ⠀ | (none) | space |
| 1 | ⠁ | 1 | a |
| 2 | ⠂ | 2 | , (comma) |
| 3 | ⠃ | 1,2 | b |
| 4 | ⠄ | 3 | ' (apostrophe) |
| 5 | ⠅ | 1,3 | k |
| 6 | ⠆ | 2,3 | ; (semicolon) |
| 7 | ⠇ | 1,2,3 | l |
| 8 | ⠈ | 4 | @ (accent mark) |
| 9 | ⠉ | 1,4 | c |
| 10 | ⠊ | 2,4 | i |
| 11 | ⠋ | 1,2,4 | f |
| 12 | ⠌ | 3,4 | / (slash) |
| 13 | ⠍ | 1,3,4 | m |
| 14 | ⠎ | 2,3,4 | s |
| 15 | ⠏ | 1,2,3,4 | p |
| 16 | ⠐ | 5 | (prefix) |
| 17 | ⠑ | 1,5 | e |
| 18 | ⠒ | 2,5 | : (colon) |
| 19 | ⠓ | 1,2,5 | h |
| 20 | ⠔ | 3,5 | * (asterisk) |
| 21 | ⠕ | 1,3,5 | o |
| 22 | ⠖ | 2,3,5 | ! (exclamation) |
| 23 | ⠗ | 1,2,3,5 | r |
| 24 | ⠘ | 4,5 | (prefix) |
| 25 | ⠙ | 1,4,5 | d |
| 26 | ⠚ | 2,4,5 | j |
| 27 | ⠛ | 1,2,4,5 | g |
| 28 | ⠜ | 3,4,5 | (prefix) |
| 29 | ⠝ | 1,3,4,5 | n |
| 30 | ⠞ | 2,3,4,5 | t |
| 31 | ⠟ | 1,2,3,4,5 | q |
| 32 | ⠠ | 6 | ^ (capital indicator) |
| 33 | ⠡ | 1,6 | * (Grade 2: ch) |
| 34 | ⠢ | 2,6 | ? (question in some contexts) |
| 35 | ⠣ | 1,2,6 | ( (open paren) or Grade 2: gh |
| 36 | ⠤ | 3,6 | - (hyphen) |
| 37 | ⠥ | 1,3,6 | u |
| 38 | ⠦ | 2,3,6 | " (open quote) |
| 39 | ⠧ | 1,2,3,6 | v |
| 40 | ⠨ | 4,6 | (prefix) |
| 41 | ⠩ | 1,4,6 | * (Grade 2: sh) |
| 42 | ⠪ | 2,4,6 | * (Grade 2: ow) |
| 43 | ⠫ | 1,2,4,6 | * (Grade 2: ed) |
| 44 | ⠬ | 3,4,6 | * (Grade 2: er) |
| 45 | ⠭ | 1,3,4,6 | x |
| 46 | ⠮ | 2,3,4,6 | * (Grade 2: the) |
| 47 | ⠯ | 1,2,3,4,6 | * (Grade 2: and) |
| 48 | ⠰ | 5,6 | (letter indicator) |
| 49 | ⠱ | 1,5,6 | * (Grade 2: wh) |
| 50 | ⠲ | 2,5,6 | . (period) |
| 51 | ⠳ | 1,2,5,6 | * (Grade 2: ou) |
| 52 | ⠴ | 3,5,6 | " (close quote) |
| 53 | ⠵ | 1,3,5,6 | z |
| 54 | ⠶ | 2,3,5,6 | " (quote) or ( |
| 55 | ⠷ | 1,2,3,5,6 | ) (close paren) or Grade 2: of |
| 56 | ⠸ | 4,5,6 | (prefix) |
| 57 | ⠹ | 1,4,5,6 | * (Grade 2: th) |
| 58 | ⠺ | 2,4,5,6 | w |
| 59 | ⠻ | 1,2,4,5,6 | * (Grade 2: ing) |
| 60 | ⠼ | 3,4,5,6 | # (number indicator) |
| 61 | ⠽ | 1,3,4,5,6 | y |
| 62 | ⠾ | 2,3,4,5,6 | ) or Grade 2: with |
| 63 | ⠿ | 1,2,3,4,5,6 | * (Grade 2: for) |

---

## Letters (Grade 1)

| Letter | Code | Braille | Dots |
|--------|------|---------|------|
| a | 1 | ⠁ | 1 |
| b | 3 | ⠃ | 1,2 |
| c | 9 | ⠉ | 1,4 |
| d | 25 | ⠙ | 1,4,5 |
| e | 17 | ⠑ | 1,5 |
| f | 11 | ⠋ | 1,2,4 |
| g | 27 | ⠛ | 1,2,4,5 |
| h | 19 | ⠓ | 1,2,5 |
| i | 10 | ⠊ | 2,4 |
| j | 26 | ⠚ | 2,4,5 |
| k | 5 | ⠅ | 1,3 |
| l | 7 | ⠇ | 1,2,3 |
| m | 13 | ⠍ | 1,3,4 |
| n | 29 | ⠝ | 1,3,4,5 |
| o | 21 | ⠕ | 1,3,5 |
| p | 15 | ⠏ | 1,2,3,4 |
| q | 31 | ⠟ | 1,2,3,4,5 |
| r | 23 | ⠗ | 1,2,3,5 |
| s | 14 | ⠎ | 2,3,4 |
| t | 30 | ⠞ | 2,3,4,5 |
| u | 37 | ⠥ | 1,3,6 |
| v | 39 | ⠧ | 1,2,3,6 |
| w | 58 | ⠺ | 2,4,5,6 |
| x | 45 | ⠭ | 1,3,4,6 |
| y | 61 | ⠽ | 1,3,4,5,6 |
| z | 53 | ⠵ | 1,3,5,6 |

---

## Numbers

Numbers use the **number indicator** (code 60: ⠼) followed by letters a-j:

| Number | Sequence | Braille |
|--------|----------|---------|
| 1 | 60,1 | ⠼⠁ |
| 2 | 60,3 | ⠼⠃ |
| 3 | 60,9 | ⠼⠉ |
| 4 | 60,25 | ⠼⠙ |
| 5 | 60,17 | ⠼⠑ |
| 6 | 60,11 | ⠼⠋ |
| 7 | 60,27 | ⠼⠛ |
| 8 | 60,19 | ⠼⠓ |
| 9 | 60,10 | ⠼⠊ |
| 0 | 60,26 | ⠼⠚ |

---

## Common Punctuation

| Symbol | Code | Braille | Dots |
|--------|------|---------|------|
| space | 0 | ⠀ | (none) |
| , | 2 | ⠂ | 2 |
| ; | 6 | ⠆ | 2,3 |
| : | 18 | ⠒ | 2,5 |
| . | 50 | ⠲ | 2,5,6 |
| ! | 22 | ⠖ | 2,3,5 |
| ? | 34 | ⠢ | 2,6 |
| ' | 4 | ⠄ | 3 |
| - | 36 | ⠤ | 3,6 |
| " (open) | 38 | ⠦ | 2,3,6 |
| " (close) | 52 | ⠴ | 3,5,6 |
| ( | 35 | ⠣ | 1,2,6 |
| ) | 55 | ⠷ | 1,2,3,5,6 |

---

## Special Indicators

| Indicator | Code | Braille | Usage |
|-----------|------|---------|-------|
| Capital | 32 | ⠠ | Next letter is uppercase |
| Number | 60 | ⠼ | Following are numbers (until space) |
| Letter | 48 | ⠰ | Return to letter mode |

---

## Grade 2 Contractions (Common)

| Contraction | Code | Braille | Meaning |
|-------------|------|---------|---------|
| the | 46 | ⠮ | "the" |
| and | 47 | ⠯ | "and" |
| for | 63 | ⠿ | "for" |
| of | 55 | ⠷ | "of" |
| with | 62 | ⠾ | "with" |
| ch | 33 | ⠡ | "ch" sound |
| sh | 41 | ⠩ | "sh" sound |
| th | 57 | ⠹ | "th" sound |
| wh | 49 | ⠱ | "wh" sound |
| gh | 35 | ⠣ | "gh" sound |
| ed | 43 | ⠫ | "-ed" ending |
| er | 44 | ⠬ | "-er" ending |
| ou | 51 | ⠳ | "ou" sound |
| ow | 42 | ⠪ | "ow" sound |
| ing | 59 | ⠻ | "-ing" ending |

---

## Example Encodings

| English | Braille | Code Sequence |
|---------|---------|---------------|
| hello | ⠓⠑⠇⠇⠕ | 19,17,7,7,21 |
| world | ⠺⠕⠗⠇⠙ | 58,21,23,7,25 |
| the | ⠮ | 46 |
| Hello | ⠠⠓⠑⠇⠇⠕ | 32,19,17,7,7,21 |
| 123 | ⠼⠁⠃⠉ | 60,1,3,9 |
| and | ⠯ | 47 |
| for | ⠿ | 63 |

---

## Notes for Annotation

When entering code sequences:
- Separate codes with commas: `19,17,7,7,21`
- Space in braille is code `0`
- Example: "hello world" = `19,17,7,7,21,0,58,21,23,7,25`
