"""Path proyek dan pemuat konfigurasi."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"

RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

# Urutan tetap dipakai di semua tabel dan confusion matrix (dari kelas terbesar).
LABELS = ["TUDUH", "KRITIK", "TANYA", "REAKSI", "LAIN", "DUKUNG"]


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
