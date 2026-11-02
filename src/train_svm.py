"""SVM linear (LinearSVC) di atas TF-IDF.

    python -m src.train_svm
"""

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import load_config
from .features import FEATURE_KINDS, build_vectorizer
from .utils import run_experiment

GRID = {
    "features": list(FEATURE_KINDS),
    "C": [0.1, 0.3, 1.0, 3.0],
    "class_weight": [None, "balanced"],
}


def build(p: dict) -> Pipeline:
    return Pipeline([
        ("tfidf", build_vectorizer(p["features"])),
        ("clf", LinearSVC(C=p["C"], class_weight=p["class_weight"],
                          random_state=load_config()["seed"], max_iter=5000)),
    ])


if __name__ == "__main__":
    run_experiment("svm", "SVM (LinearSVC)", build, GRID)
