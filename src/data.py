"""Memuat dataset berlabel dan membuat split train/test yang tetap."""

from __future__ import annotations

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import LABELS, ROOT, load_config
from .preprocess import clean_text


def load_dataset(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    d = cfg["data"]
    df = pd.read_csv(ROOT / d["path"], encoding="utf-8-sig")
    df = df.rename(columns={d["text_col"]: "comment", d["label_col"]: "label"})
    unknown = set(df["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"Label tidak dikenal: {unknown}")
    df["text"] = df["comment"].map(clean_text)
    return df


def file_md5(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    return hashlib.md5((ROOT / cfg["data"]["path"]).read_bytes()).hexdigest()


def train_test(cfg: dict | None = None):
    """Split stratified 80/20 dengan seed tetap. Mengembalikan (train_df, test_df)."""
    cfg = cfg or load_config()
    df = load_dataset(cfg)
    train_df, test_df = train_test_split(
        df,
        test_size=cfg["split"]["test_size"],
        stratify=df["label"],
        random_state=cfg["seed"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
