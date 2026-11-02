"""Runner eksperimen bersama untuk semua model.

Alur (sama untuk setiap algoritma, supaya perbandingannya adil):
1. Split stratified 80/20 dengan seed tetap (src.data.train_test).
2. Untuk setiap kombinasi hyperparameter: stratified 5-fold CV di data latih,
   skor utama macro-F1. Setiap kombinasi dicatat sebagai *child run* MLflow,
   jadi progresnya bisa dipantau langsung dari dashboard.
3. Kombinasi terbaik dilatih ulang di seluruh data latih, lalu dievaluasi SEKALI
   di data uji. Hasil disimpan ke results/ dan parent run MLflow.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from .config import FIGURES_DIR, LABELS, METRICS_DIR, MODELS_DIR, ROOT, load_config
from .data import file_md5, train_test
from .evaluate import compute_metrics, plot_confusion

LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}

BuildFn = Callable[[dict], Pipeline]
FitParamsFn = Callable[[np.ndarray, dict], dict]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def run_experiment(
    key: str,
    display_name: str,
    build: BuildFn,
    grid: list[dict] | dict,
    fit_params: FitParamsFn | None = None,
    n_jobs: int = -1,
) -> dict:
    import mlflow

    from .tracking import setup_mlflow

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    cfg = load_config()
    for d in (METRICS_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test(cfg)
    X_tr, X_te = train_df["text"].to_numpy(), test_df["text"].to_numpy()
    y_tr = train_df["label"].map(LABEL2ID).to_numpy()
    y_te = test_df["label"].to_numpy()
    skf = StratifiedKFold(cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["seed"])
    candidates = list(ParameterGrid(grid))
    fit_params = fit_params or (lambda y, p: {})

    setup_mlflow(cfg["mlflow"]["experiment"])
    print(f"\n=== {display_name}: {len(candidates)} kombinasi x {cfg['cv']['n_splits']}-fold CV ===")

    with mlflow.start_run(run_name=display_name) as parent:
        mlflow.set_tags({"model": key, "stage": "grid+test", "git_commit": _git_commit()})
        mlflow.log_params({
            "seed": cfg["seed"], "test_size": cfg["split"]["test_size"],
            "cv_folds": cfg["cv"]["n_splits"], "n_train": len(train_df),
            "n_test": len(test_df), "n_candidates": len(candidates),
            "data_md5": file_md5(cfg),
        })

        rows = []
        for i, params in enumerate(candidates, 1):
            t0 = time.perf_counter()
            cv = cross_validate(
                build(params), X_tr, y_tr, cv=skf, n_jobs=n_jobs,
                scoring=["f1_macro", "accuracy"], params=fit_params(y_tr, params),
            )
            elapsed = time.perf_counter() - t0
            row = {
                **{f"param_{k}": str(v) for k, v in params.items()},
                "cv_f1_macro_mean": cv["test_f1_macro"].mean(),
                "cv_f1_macro_std": cv["test_f1_macro"].std(),
                "cv_accuracy_mean": cv["test_accuracy"].mean(),
                "fit_seconds_mean": cv["fit_time"].mean(),
                **{f"f1_fold{j}": s for j, s in enumerate(cv["test_f1_macro"])},
            }
            rows.append((params, row))
            with mlflow.start_run(run_name=f"{key}-{i:02d}", nested=True):
                mlflow.set_tags({"model": key, "stage": "cv"})
                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_metrics({k: v for k, v in row.items() if not k.startswith("param_")})
            print(f"[{i:2d}/{len(candidates)}] {params} -> macro-F1 "
                  f"{row['cv_f1_macro_mean']:.4f} ± {row['cv_f1_macro_std']:.4f} ({elapsed:.0f} s)",
                  flush=True)

        cv_df = pd.DataFrame([r for _, r in rows]).sort_values("cv_f1_macro_mean", ascending=False)
        cv_path = METRICS_DIR / f"{key}_cv.csv"
        cv_df.to_csv(cv_path, index=False)
        best_params, best_row = max(rows, key=lambda pr: pr[1]["cv_f1_macro_mean"])

        # --- latih ulang konfigurasi terbaik di seluruh data latih, uji sekali
        model = build(best_params)
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr, **fit_params(y_tr, best_params))
        fit_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred_ids = model.predict(X_te)
        predict_ms = (time.perf_counter() - t0) * 1000 / len(X_te) * 1000
        y_pred = np.array(LABELS)[np.asarray(pred_ids, dtype=int)]

        metrics = compute_metrics(y_te, y_pred)
        metrics.update({
            "cv_f1_macro_mean": round(best_row["cv_f1_macro_mean"], 4),
            "cv_f1_macro_std": round(best_row["cv_f1_macro_std"], 4),
            "fit_seconds": round(fit_seconds, 2),
            "predict_ms_per_1k": round(predict_ms, 1),
        })
        result = {"best_params": {k: str(v) for k, v in best_params.items()}, **metrics}

        (METRICS_DIR / f"{key}_test.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        pd.DataFrame({
            "comment_id": test_df[cfg["data"]["id_col"]].astype(str),
            "post_id": test_df[cfg["data"]["group_col"]],
            "comment": test_df["comment"], "label": y_te, "pred": y_pred,
        }).to_csv(METRICS_DIR / f"{key}_test_pred.csv", index=False)
        cm_path = FIGURES_DIR / f"exp_cm_{key}.png"
        plot_confusion(y_te, y_pred, f"{display_name}: confusion matrix data uji", cm_path)
        joblib.dump(model, MODELS_DIR / f"{key}.joblib")

        mlflow.log_params({f"best_{k}": str(v) for k, v in best_params.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()
                            if isinstance(v, (int, float)) and not k.startswith("support_")})
        mlflow.log_artifact(str(cv_path))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(METRICS_DIR / f"{key}_test.json"))

    print(f"\nTerbaik: {best_params}")
    print(f"CV macro-F1 {metrics['cv_f1_macro_mean']:.4f} | TEST macro-F1 {metrics['f1_macro']:.4f} "
          f"| akurasi {metrics['accuracy']:.4f}")
    print("F1 per kelas:", {lab: metrics[f"f1_{lab}"] for lab in LABELS})
    return result
