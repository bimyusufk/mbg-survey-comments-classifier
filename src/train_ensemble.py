"""Ensemble tree: XGBoost (gradient boosted trees) di atas TF-IDF.

Jumlah fitur dibatasi (max_features per vectorizer) karena pohon keputusan
lambat dan cenderung overfit pada ratusan ribu fitur n-gram yang jarang.
Ketidakseimbangan kelas ditangani dengan sample_weight "balanced" (opsional,
diuji sebagai hyperparameter). Memakai GPU bila tersedia.

    python -m src.train_ensemble
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .config import load_config
from .features import build_vectorizer
from .utils import run_experiment


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = _device()
MAX_FEATURES = 20000

GRID = {
    "features": ["word", "char", "word+char"],
    "max_depth": [4, 8],
    "weighting": [None, "balanced"],
}


def build(p: dict) -> Pipeline:
    return Pipeline([
        ("tfidf", build_vectorizer(p["features"], max_features=MAX_FEATURES)),
        ("clf", XGBClassifier(
            n_estimators=400, learning_rate=0.1, max_depth=p["max_depth"],
            subsample=0.8, colsample_bytree=0.5, tree_method="hist",
            device=DEVICE, random_state=load_config()["seed"], n_jobs=-1,
        )),
    ])


def fit_params(y: np.ndarray, p: dict) -> dict:
    if p["weighting"] == "balanced":
        return {"clf__sample_weight": compute_sample_weight("balanced", y)}
    return {}


if __name__ == "__main__":
    print("Perangkat XGBoost:", DEVICE)
    # n_jobs=1: fold dijalankan berurutan karena semuanya berbagi satu GPU.
    run_experiment("xgboost", "XGBoost", build, GRID, fit_params=fit_params,
                   n_jobs=1 if DEVICE == "cuda" else -1)
