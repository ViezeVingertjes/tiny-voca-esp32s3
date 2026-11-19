"""The tiny-voca checkout these tools use (model code, text front end) and its weights, plus the
plain lexicon phonemizer of quant_eval.py and speak_s3.py.

TINY_VOCA_DIR: a tiny-voca checkout (default: this repo's parent folder).
TINY_VOCA_WEIGHTS: its weights (default: $TINY_VOCA_DIR/weights, else $TINY_VOCA_DIR/../models/tiny-voca).
"""
import os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]                     # this repo; outputs go to REPO / "out"
TINY_VOCA = Path(os.environ.get("TINY_VOCA_DIR", REPO.parent)).resolve()
WEIGHTS = Path(os.environ.get("TINY_VOCA_WEIGHTS",
                              TINY_VOCA / "weights" if (TINY_VOCA / "weights").exists() else TINY_VOCA.parent / "models/tiny-voca"))
sys.path.insert(0, str(TINY_VOCA / "src"))
from tiny_voca.text import chunks, normalize_text, tokenize, parse_espeak  # noqa: E402


def load_lexicon(path):
    lexicon = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, phones = line.rstrip("\n").split("\t")
            lexicon.setdefault(word, phones.replace(" ", "_"))
    return lexicon


def phonemize(text, lexicon, missing=None):
    """Tokens from a plain lexicon lookup. A word not in the lexicon raises KeyError, or is skipped
    and appended to `missing` when that list is given."""
    pieces = []
    for punctuation, piece in chunks(normalize_text(text)):
        if punctuation:
            pieces.append((True, piece))
            continue
        words = []
        for word in piece.lower().split():
            if word in lexicon:
                words += parse_espeak(lexicon[word])
            elif missing is None:
                raise KeyError(f"{word!r} is not in the lexicon; pick another word")
            else:
                missing.append(word)
        pieces.append((False, words))
    return tokenize(pieces)
