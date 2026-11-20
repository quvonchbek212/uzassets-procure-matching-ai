"""Text normalization for Uzbek Latin, Uzbek Cyrillic and Russian.

The catalog is written by people, in three scripts, inconsistently. "Мой",
"moy", "moyi", "МОЙ" and "mo'y" must all meet somewhere before any lexical
comparison can work. This module produces that meeting point — a lowercase,
transliterated, apostrophe-folded form used **only** for scoring and matching.

The original strings are never replaced. The brief is explicit that model
numbers, dimensions, grades and standards must survive, and they only survive
in the untouched value, which is what reaches the prompt, the evidence list and
the fingerprint. Normalization exists to find candidates, not to describe them.

Cyrillic is transliterated to Latin rather than the other way round because
Uzbek Latin is the catalog's dominant script and the mapping in that direction
is the less lossy one (щ -> sh loses the distinction from ш, but no Uzbek term
in the catalog depends on it).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# Longest sequences first: 'ch' must win before 'c', 'ю' before 'у'.
_CYRILLIC_MAP: tuple[tuple[str, str], ...] = (
    ("ьо", "yo"),
    ("ъе", "ye"),
    ("щ", "sh"),
    ("ш", "sh"),
    ("ч", "ch"),
    ("ц", "ts"),
    ("ю", "yu"),
    ("я", "ya"),
    ("ё", "yo"),
    ("ж", "j"),
    ("х", "x"),
    ("ҳ", "h"),
    ("ғ", "g"),
    ("қ", "q"),
    ("ў", "o"),
    ("э", "e"),
    ("ы", "i"),
    ("а", "a"),
    ("б", "b"),
    ("в", "v"),
    ("г", "g"),
    ("д", "d"),
    ("е", "e"),
    ("з", "z"),
    ("и", "i"),
    ("й", "y"),
    ("к", "k"),
    ("л", "l"),
    ("м", "m"),
    ("н", "n"),
    ("о", "o"),
    ("п", "p"),
    ("р", "r"),
    ("с", "s"),
    ("т", "t"),
    ("у", "u"),
    ("ф", "f"),
    ("ъ", ""),
    ("ь", ""),
)

# Every apostrophe variant Uzbek Latin is typed with (oʻ, o', o‘, o`, o´).
_APOSTROPHES = "‘’ʻʼʽ′´`'‵"
_APOSTROPHE_RE = re.compile(f"[{re.escape(_APOSTROPHES)}]")

_NON_ALNUM_RE = re.compile(r"[^0-9a-z\s]+")
_WHITESPACE_RE = re.compile(r"\s+")

# Words that carry no discriminating signal in this catalog. Dropping them stops
# "Xizmat ko'rsatish" from scoring against every services article equally.
STOPWORDS = frozenset(
    {
        "va", "uchun", "bilan", "hamda", "yoki", "ning", "lar", "boyicha", "buyicha",
        "и", "для", "с", "по", "или", "на", "от", "в",
        "and", "for", "with", "or", "the", "of",
        "turli", "boshqa", "prochie", "prochee", "drugie", "other",
        "xil", "xildagi", "razn", "raznye",
    }
)


def transliterate(text: str) -> str:
    """Cyrillic -> Latin, leaving Latin text untouched."""
    out = text
    for cyr, lat in _CYRILLIC_MAP:
        if cyr in out:
            out = out.replace(cyr, lat)
    return out


@lru_cache(maxsize=20_000)
def normalize(text: str | None) -> str:
    """Lowercase, transliterated, punctuation-free form used for comparison."""
    if not text:
        return ""
    # NFKC first: it folds full-width and composed forms so 'ў' and 'у'+breve
    # become one character before the map runs.
    folded = unicodedata.normalize("NFKC", str(text)).lower()
    folded = _APOSTROPHE_RE.sub("", folded)
    folded = transliterate(folded)
    folded = _NON_ALNUM_RE.sub(" ", folded)
    return _WHITESPACE_RE.sub(" ", folded).strip()


@lru_cache(maxsize=20_000)
def tokenize(text: str | None) -> tuple[str, ...]:
    """Content tokens: normalized, stopword-free, single characters dropped."""
    normalized = normalize(text)
    if not normalized:
        return ()
    return tuple(
        token
        for token in normalized.split(" ")
        if len(token) > 1 and token not in STOPWORDS
    )


def stem(token: str) -> str:
    """Crude suffix stripping so 'moylash' and 'moy' share a root.

    Uzbek agglutinates heavily and there is no light stemmer packaged for it;
    this trims the handful of endings that actually appear on catalog nouns.
    Anything shorter than five characters is left alone — over-stemming short
    roots ("moy" -> "m") is far more damaging than under-stemming.
    """
    if len(token) < 5:
        return token
    for suffix in ("lashtirish", "lantirish", "lash", "lari", "ning", "lar", "ini", "ida", "iga", "dan", "ga", "i"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def stems(text: str | None) -> tuple[str, ...]:
    return tuple(stem(token) for token in tokenize(text))


def token_set(text: str | None) -> frozenset[str]:
    return frozenset(stems(text))


def contains_phrase(haystack: str | None, phrase: str) -> bool:
    """Whole-word phrase test on the normalized forms of both sides."""
    normalized_hay = normalize(haystack)
    normalized_phrase = normalize(phrase)
    if not normalized_hay or not normalized_phrase:
        return False
    return re.search(rf"(?<![0-9a-z]){re.escape(normalized_phrase)}(?![0-9a-z])", normalized_hay) is not None


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / len(left | right)


def coverage(query: frozenset[str], target: frozenset[str]) -> float:
    """Share of `query` tokens present in `target`.

    Asymmetric on purpose: a short article name ("Yoqilg'i") fully covered by a
    long product description is a strong signal, while Jaccard would punish it
    for the length difference alone.
    """
    if not query:
        return 0.0
    return len(query & target) / len(query)


__all__ = [
    "STOPWORDS",
    "contains_phrase",
    "coverage",
    "jaccard",
    "normalize",
    "stem",
    "stems",
    "token_set",
    "tokenize",
    "transliterate",
]
