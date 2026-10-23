# units of a tokenised comment: the pieces whose effect is measured, each removed at all of its
# positions in the comment at once. Only content positions (1 .. len(tokens) - 2) belong to a
# unit, <s> and </s> never do.
# A unit type maps the comment's token strings to its unit occurrences, each (unit string,
# positions), in position order. N-grams of any unit type are n consecutive units (comment_units
# with n >= 2).

import codecs
import unicodedata


def subtoken_units(tokens):
    # every content token is its own unit, named by its token string (Ġ = leading space)
    return [(tokens[position], [position]) for position in range(1, len(tokens) - 1)]


def _byte_decoder():
    # the byte-level BPE alphabet of the LUAR (RoBERTa) tokenizer: every byte is shown as one
    # printable character (space as Ġ, line break as Ċ, ...); this maps those characters back to bytes
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars = printable[:]
    shift = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            chars.append(256 + shift)
            shift += 1
    return {chr(char): byte for byte, char in zip(printable, chars)}


BYTE_DECODER = _byte_decoder()


def decoded_pieces(tokens):
    # the comment's content tokens decoded to their real characters, as [(text, positions)] in
    # order. A character whose bytes are split over several tokens is decoded with the token that
    # completes it, so that piece then covers the earlier tokens' positions as well; every content
    # position is in exactly one piece.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pieces = []
    pending = []
    for position in range(1, len(tokens) - 1):
        pending.append(position)
        text = decoder.decode(bytes(BYTE_DECODER[char] for char in tokens[position]))
        if text:
            pieces.append((text, pending))
            pending = []
    if pending:
        pieces.append((decoder.decode(b"", final=True), pending))
    return pieces


def piece_kind(text):
    # "word" if the text has a letter or digit, "whitespace" if it is only whitespace, else "symbol"
    if any(unicodedata.category(char)[0] in "LN" for char in text):
        return "word"
    if text.isspace():
        return "whitespace"
    return "symbol"


def word_units_with_text(tokens):
    # [(kind, decoded text, positions)] of the word units in order: a word piece starts a new word
    # if its text begins with whitespace or the previous unit is not a word, otherwise it is
    # appended to the current word; whitespace and symbol pieces are units of their own
    units = []
    for text, positions in decoded_pieces(tokens):
        kind = piece_kind(text)
        if kind == "word" and units and units[-1][0] == "word" and not text[0].isspace():
            _, word_text, word_positions = units[-1]
            units[-1] = ("word", word_text + text, word_positions + positions)
        else:
            units.append((kind, text, positions))
    return units


def word_units(tokens):
    # identity: the decoded text without leading and trailing whitespace, case kept; whitespace
    # units keep their whitespace
    return [
        (text if kind == "whitespace" else text.strip(), positions)
        for kind, text, positions in word_units_with_text(tokens)
    ]


UNIT_TYPES = {
    "subtoken": subtoken_units,
    "word": word_units,
}

# n-gram sizes: n = 1 is the single unit, n >= 2 are n consecutive units of the unit type
NGRAM_SIZES = [1, 2, 3]


def ngrams(units, n):
    # [(identity, positions)] of every n consecutive units, in order; identity = the tuple of the n
    # units' identities (a list in the JSON files), positions = all positions of the n units.
    # Symbol and whitespace units count as units.
    return [
        (tuple(name for name, _ in units[i:i + n]), [p for _, positions in units[i:i + n] for p in positions])
        for i in range(len(units) - n + 1)
    ]


def comment_units(tokens, unit, n=1):
    # the comment's unit occurrences (n = 1) or n-gram occurrences (n >= 2)
    units = UNIT_TYPES[unit](tokens)
    return units if n == 1 else ngrams(units, n)


def unit_key(name):
    # a unit's identity as a dict key: n-grams come back from JSON as lists, made tuples again
    return tuple(name) if isinstance(name, list) else name


# how whitespace characters are shown in tables
WHITESPACE_DISPLAY = {" ": "␣", "\n": "\\n", "\t": "\\t", "\r": "\\r"}


def display_unit(name, unit):
    # a unit's identity as shown in tables: word whitespace units made visible, everything else as
    # is; an n-gram as its units joined by a single space
    if isinstance(name, (list, tuple)):
        return " ".join(display_unit(part, unit) for part in name)
    if unit == "word" and name.isspace():
        return "".join(WHITESPACE_DISPLAY.get(char, f"\\u{ord(char):04x}") for char in name)
    return name
