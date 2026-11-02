"""Metrik, confusion matrix, dan ringkasan perbandingan antar model.

    python -m src.evaluate      # gabungkan hasil semua model -> summary.csv + grafik
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .config import FIGURES_DIR, LABELS, METRICS_DIR

# --- gaya grafik (palet referensi dataviz, mode terang) -----------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Warna mengikuti entitas model (urutan slot tetap), bukan peringkat.
MODEL_STYLE = {
    "svm": ("SVM (LinearSVC)", "#2a78d6"),
    "naive_bayes": ("Naive Bayes", "#eb6834"),
    "xgboost": ("XGBoost", "#1baf7a"),
}
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# --- metrik ----------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "precision_macro": float(np.mean(p)),
        "recall_macro": float(np.mean(r)),
    }
    for i, lab in enumerate(LABELS):
        out[f"f1_{lab}"] = float(f[i])
        out[f"precision_{lab}"] = float(p[i])
        out[f"recall_{lab}"] = float(r[i])
        out[f"support_{lab}"] = int(s[i])
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def plot_confusion(y_true, y_pred, title: str, path) -> None:
    """Confusion matrix ternormalisasi per baris (recall), dengan jumlah absolut."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor=SURFACE)
    im = ax.imshow(norm, cmap=CMAP, vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            v = norm[i, j]
            ax.text(j, i, f"{v:.0%}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                    color="#ffffff" if v >= 0.5 else INK)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=35, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Prediksi", color=INK_2)
    ax.set_ylabel("Label sebenarnya", color=INK_2)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.tick_params(colors=INK_2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, format=lambda x, _: f"{x:.0%}")
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# --- ringkasan antar model ----------------------------------------------------
def summarize() -> pd.DataFrame:
    rows = []
    for key in MODEL_STYLE:
        f = METRICS_DIR / f"{key}_test.json"
        if f.exists():
            rows.append({"model": key, **json.loads(f.read_text(encoding="utf-8"))})
    if not rows:
        raise SystemExit("Belum ada hasil. Jalankan python -m src.train_svm (dst.) dulu.")
    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(METRICS_DIR / "summary.csv")

    # Grafik F1 per kelas, dikelompokkan per kelas, satu batang per model.
    models = list(df.index)
    x = np.arange(len(LABELS))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8.2, 4.2), facecolor=SURFACE)
    _style(ax)
    for k, m in enumerate(models):
        name, color = MODEL_STYLE[m]
        vals = [df.loc[m, f"f1_{lab}"] for lab in LABELS]
        ax.bar(x - 0.4 + width * (k + 0.5), vals, width - 0.02, color=color,
               label=f"{name} (macro-F1 {df.loc[m, 'f1_macro']:.3f})",
               edgecolor=SURFACE, linewidth=1)
    support = [int(df.iloc[0][f"support_{lab}"]) for lab in LABELS]
    ax.set_xticks(x, [f"{lab}\nn={n}" for lab, n in zip(LABELS, support)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 (data uji)", color=INK_2)
    ax.set_title("F1 per kelas pada data uji", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp_f1_per_kelas.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return df


if __name__ == "__main__":
    df = summarize()
    cols = ["f1_macro", "accuracy", "f1_weighted", "precision_macro", "recall_macro",
            "fit_seconds", "predict_ms_per_1k"] + [f"f1_{lab}" for lab in LABELS]
    print(df[[c for c in cols if c in df.columns]].T.to_string())
