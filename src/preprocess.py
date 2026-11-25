"""Text cleaning shared by training and serving.

The same function must run in both stages: if serving cleans text differently
from training, every prediction silently degrades.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer

from src.config import EXTRA_STOPWORDS, TFIDF_PARAMS

_URL_RE = re.compile(r"https?://\S+")
# Keep letters, digits and the programming-language tokens c++ / c#.
_JUNK_RE = re.compile(r"[^a-z0-9\s+#]")
_WS_RE = re.compile(r"\s+")
# "Show HN:", "Ask HN:", "Tell HN:", "Launch HN:" say nothing about the topic.
_HN_PREFIX_RE = re.compile(r"^\s*(show|ask|tell|launch)\s+hn\b\s*[:\-\u2013]?\s*", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

MAX_TITLE_CHARS = 400


def clean_title(text: object) -> str:
    """Lowercase, strip URLs, HN prefixes, years and punctuation.

    Returns an empty string for anything that is not usable text, which the
    caller must treat as "no content" rather than as a valid document.
    """
    if not isinstance(text, str):
        return ""
    t = text[:MAX_TITLE_CHARS]
    t = _URL_RE.sub(" ", t)
    t = _HN_PREFIX_RE.sub(" ", t)
    t = t.lower()
    t = _JUNK_RE.sub(" ", t)
    t = _YEAR_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def clean_titles(values: Iterable[object]) -> list[str]:
    return [clean_title(v) for v in values]


def build_vectorizer() -> TfidfVectorizer:
    """A fresh, unfitted vectorizer with the agreed-upon settings."""
    return TfidfVectorizer(
        stop_words=sorted(ENGLISH_STOP_WORDS | EXTRA_STOPWORDS),
        **TFIDF_PARAMS,
    )


def build_count_vectorizer(vocabulary: dict) -> CountVectorizer:
    """Counts over exactly the vocabulary the TF-IDF stage learned.

    Used for c-TF-IDF topic labelling, where raw term frequency inside a
    cluster is the meaningful quantity.
    """
    params = {k: v for k, v in TFIDF_PARAMS.items() if k != "sublinear_tf"}
    return CountVectorizer(
        vocabulary=vocabulary,
        stop_words=sorted(ENGLISH_STOP_WORDS | EXTRA_STOPWORDS),
        **params,
    )


def non_empty_pairs(texts: Sequence[str], ids: Sequence[str] | None = None):
    """Drop rows whose cleaned text is empty, keeping ids aligned."""
    if ids is None:
        return [t for t in texts if t]
    kept_texts, kept_ids = [], []
    for i, t in enumerate(texts):
        if t:
            kept_texts.append(t)
            kept_ids.append(ids[i])
    return kept_texts, kept_ids
