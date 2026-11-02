"""Representasi fitur TF-IDF.

Tiga varian diuji sebagai faktor eksperimen yang sama untuk semua model:
- word      : unigram + bigram kata
- char      : n-gram karakter 2-5 (char_wb), tahan terhadap slang & salah ketik
- word+char : gabungan keduanya
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from .preprocess import TOKEN_PATTERN

FEATURE_KINDS = ("word", "char", "word+char")


def _word(max_features: int | None) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word", token_pattern=TOKEN_PATTERN, ngram_range=(1, 2),
        min_df=2, sublinear_tf=True, lowercase=False, max_features=max_features,
    )


def _char(max_features: int | None) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5),
        min_df=2, sublinear_tf=True, lowercase=False, max_features=max_features,
    )


def build_vectorizer(kind: str, max_features: int | None = None):
    if kind == "word":
        return _word(max_features)
    if kind == "char":
        return _char(max_features)
    if kind == "word+char":
        return FeatureUnion([("word", _word(max_features)), ("char", _char(max_features))])
    raise ValueError(f"kind harus salah satu dari {FEATURE_KINDS}, bukan {kind!r}")
