"""Sambungan ke server MLflow bersama.

Di skrip training / notebook:

    from src.tracking import setup_mlflow
    import mlflow

    setup_mlflow()                      # baca .env, pilih experiment
    with mlflow.start_run(run_name="svm-tfidf"):
        mlflow.log_params({...})
        mlflow.log_metric("f1_macro", f1)

Kredensial dibaca dari .env (MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME,
MLFLOW_TRACKING_PASSWORD). Host server mendapatkannya otomatis dari
scripts/serve.py; anggota lain menyalin dari .env.example dan mengisinya.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = "klasifikasi-komentar-ig"


def load_env(path: Path = ROOT / ".env") -> None:
    """Muat .env ke os.environ tanpa menimpa variabel yang sudah ada."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def setup_mlflow(experiment: str = DEFAULT_EXPERIMENT) -> str:
    """Arahkan MLflow ke server bersama dan pilih experiment. Mengembalikan URI."""
    import mlflow

    # MLflow mencetak emoji; konsol Windows (cp1252) bisa crash tanpa ini.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    load_env()
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI belum diisi. Host: jalankan `python scripts/serve.py`. "
            "Anggota lain: salin .env.example menjadi .env lalu isi dari link yang dibagikan."
        )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri
