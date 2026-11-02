"""Normalisasi teks komentar Instagram.

Sengaja ringan:
- Tanpa stopword removal: kata seperti "gak", "tidak", "kok", "apa" membawa sinyal
  untuk kelas KRITIK/TANYA.
- Tanpa stemming: sebagian besar token adalah slang/singkatan yang tidak dikenali
  stemmer baku (Sastrawi), dan variasi ejaan ditangani oleh fitur n-gram karakter.
- Tanda "?" dan "!" serta emoji dipertahankan sebagai token terpisah.
"""

from __future__ import annotations

import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[\w.]+")
HASHTAG_RE = re.compile(r"#(\w+)")
ELONGATION_RE = re.compile(r"(.)\1{2,}")  # "cairrrrr" -> "cairr", "wkwkwk" tetap
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U0001F900-\U0001F9FF]"
)
PUNCT_RUN_RE = re.compile(r"([?!.,])\1+")  # "???" -> "?"
SPACE_RE = re.compile(r"\s+")

# Token: kata (huruf/angka) ATAU satu karakter non-spasi non-kata (?, !, emoji, dll.)
TOKEN_PATTERN = r"(?u)\w+|[^\w\s]"


def clean_text(text: str) -> str:
    s = str(text).lower()
    s = URL_RE.sub(" url ", s)
    s = MENTION_RE.sub(" user ", s)
    s = HASHTAG_RE.sub(r" \1 ", s)
    s = ELONGATION_RE.sub(r"\1\1", s)
    s = PUNCT_RUN_RE.sub(r"\1", s)
    s = EMOJI_RE.sub(lambda m: f" {m.group(0)} ", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s
