"""Naive Bayes (Multinomial vs Complement) di atas TF-IDF.

ComplementNB dirancang untuk data teks yang tidak seimbang (Rennie et al., 2003),
jadi varian ini diuji berdampingan dengan MultinomialNB standar.

    python -m src.train_bayes
"""

from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline

from .features import FEATURE_KINDS, build_vectorizer
from .utils import run_experiment

GRID = {
    "features": list(FEATURE_KINDS),
    "variant": ["multinomial", "complement"],
    # Diperluas ke bawah setelah run pertama: optimum ada di batas grid (0.03).
    "alpha": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
}


def build(p: dict) -> Pipeline:
    nb = ComplementNB if p["variant"] == "complement" else MultinomialNB
    return Pipeline([
        ("tfidf", build_vectorizer(p["features"])),
        ("clf", nb(alpha=p["alpha"])),
    ])


if __name__ == "__main__":
    run_experiment("naive_bayes", "Naive Bayes", build, GRID)
