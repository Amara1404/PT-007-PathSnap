"""Shared experiment runner: grouped CV on the 80 training trips + scoring on the
10 student-test trips, with results appended to results/*.csv.

Data rules (from the hackathon guidelines):
  * fit and cross-validate on training trips only, folds from data/cv_folds.csv
  * compare models on student-test trips
  * learned preprocessing (scalers) lives INSIDE the sklearn pipeline, so it is
    re-fitted on each training fold only
  * only rows with usable == 1 (reliable reference labels) are used

Subsampling: GPS is 1 Hz, so neighbouring observations are near-duplicates.
Training uses every `stride`-th observation of each trip (all candidates of that
observation kept). Validation folds and the student-test set are scored on ALL
usable observations.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from src.eval.metrics import observation_top1, route_continuity, row_metrics, top1_choice

PROCESSED = Path("data/processed")
RESULTS = Path("results")
ARTIFACTS = Path("artifacts")
FEATURES: list[str] = json.loads((PROCESSED / "feature_columns.json").read_text())
KEY_COLS = ["trip_id", "obs_idx", "arc_id", "label", "dist_m"]
SEED = 42


def load_candidates(partition: str, stride: int = 1) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / f"candidates_{partition}.parquet",
                         columns=list(dict.fromkeys(KEY_COLS + ["usable"] + FEATURES)))
    df = df[df.usable == 1]
    if stride > 1:
        df = df[df.obs_idx % stride == 0]
    return df.reset_index(drop=True)


_ARCS = None


def load_arc_nodes() -> pd.DataFrame:
    global _ARCS
    if _ARCS is None:
        _ARCS = pd.read_parquet(PROCESSED / "arcs.parquet", columns=["trip_id", "arc_id", "from_node", "to_node"])
    return _ARCS


@dataclass
class Experiment:
    name: str                       # short id, e.g. "random_forest"
    family: str                     # algorithm family for the report table
    config: str                     # key configuration in words
    make_model: Callable            # () -> unfitted sklearn estimator with predict_proba
    stride: int = 5
    notes: dict = field(default_factory=dict)


def _score(df: pd.DataFrame, prob: np.ndarray) -> dict:
    d = df[["trip_id", "obs_idx", "arc_id", "label", "dist_m"]].copy()
    d["prob"] = prob
    out = row_metrics(d.label, d.prob)
    out.update(observation_top1(d))
    out["route_continuity"] = route_continuity(top1_choice(d), load_arc_nodes())
    return out


def _predict(model, X):
    return model.predict_proba(X)[:, 1]


def run_experiment(exp: Experiment, folds_csv: Path = Path("data/cv_folds.csv"), verbose: bool = True) -> dict:
    t_all = time.time()
    folds = pd.read_csv(folds_csv, dtype={"trip_id": str})
    full = load_candidates("train", stride=1)
    full["fold"] = full.trip_id.map(folds.set_index("trip_id").fold)
    train_sub = full[full.obs_idx % exp.stride == 0]

    fold_rows = []
    for k in sorted(folds.fold.unique()):
        tr = train_sub[train_sub.fold != k]
        va = full[full.fold == k]
        model = exp.make_model()
        t0 = time.time(); model.fit(tr[FEATURES], tr.label); fit_s = time.time() - t0
        va_s = _score(va, _predict(model, va[FEATURES]))
        # training-fold score (same subsample it was fitted on) -> over/underfitting evidence
        tr_s = row_metrics(tr.label, _predict(model, tr[FEATURES]))
        fold_rows.append({"model": exp.name, "fold": int(k), "fit_s": fit_s,
                          **{f"val_{m}": v for m, v in va_s.items()},
                          **{f"train_{m}": v for m, v in tr_s.items()}})
        if verbose:
            print(f"  [{exp.name}] fold {k}: val macroF1={va_s['macro_f1']:.4f} top1={va_s['top1_acc']:.4f} "
                  f"train macroF1={tr_s['macro_f1']:.4f} ({fit_s:.1f}s)")
    cv = pd.DataFrame(fold_rows)

    # final fit on all (subsampled) training rows -> student-test evaluation
    model = exp.make_model()
    t0 = time.time(); model.fit(train_sub[FEATURES], train_sub.label); fit_s = time.time() - t0
    st = load_candidates("student_test")
    t0 = time.time(); prob = _predict(model, st[FEATURES]); inf_s = time.time() - t0
    st_s = _score(st, prob)

    row = {
        "model": exp.name, "family": exp.family, "config": exp.config,
        "train_rows": len(train_sub), "stride": exp.stride,
        "cv_macro_f1_mean": cv.val_macro_f1.mean(), "cv_macro_f1_std": cv.val_macro_f1.std(),
        "cv_top1_mean": cv.val_top1_acc.mean(), "cv_top1_std": cv.val_top1_acc.std(),
        "cv_log_loss_mean": cv.val_log_loss.mean(),
        "cv_route_continuity_mean": cv.val_route_continuity.mean(),
        "train_macro_f1_mean": cv.train_macro_f1.mean(),
        "st_top1_acc": st_s["top1_acc"], "st_macro_f1": st_s["macro_f1"],
        "st_precision_pos": st_s["precision_pos"], "st_recall_pos": st_s["recall_pos"],
        "st_log_loss": st_s["log_loss"], "st_route_continuity": st_s["route_continuity"],
        "st_candidate_miss_rate": st_s["candidate_miss_rate"], "st_n_obs": st_s["n_obs"],
        "fit_time_s": fit_s, "inference_s": inf_s,
        "inference_ms_per_1k_obs": inf_s / st_s["n_obs"] * 1e6,
        "total_time_s": time.time() - t_all,
    }
    if exp.notes.get("log_loss_na"):
        # hard 0/1 scores: log loss only measures the clipping constant, report N/A
        row["cv_log_loss_mean"] = row["st_log_loss"] = float("nan")

    # ---- persist ----
    for d in (RESULTS / "preds", RESULTS / "history", ARTIFACTS / "models"):
        d.mkdir(parents=True, exist_ok=True)
    _upsert(RESULTS / "model_comparison.csv", pd.DataFrame([row]), "model", exp.name)
    _upsert(RESULTS / "cv_fold_scores.csv", cv, "model", exp.name)
    out = st[["trip_id", "obs_idx", "arc_id", "label", "dist_m"]].copy()
    out["prob"] = prob
    out.to_parquet(RESULTS / "preds" / f"{exp.name}_student_test.parquet", index=False)
    joblib.dump(model, ARTIFACTS / "models" / f"{exp.name}.joblib", compress=3)
    hist = _history(model)
    if hist is not None:
        hist.to_csv(RESULTS / "history" / f"{exp.name}_history.csv", index=False)
    if verbose:
        print(f"  [{exp.name}] CV macroF1={row['cv_macro_f1_mean']:.4f}±{row['cv_macro_f1_std']:.4f} | "
              f"student-test top1={row['st_top1_acc']:.4f} macroF1={row['st_macro_f1']:.4f} "
              f"logloss={row['st_log_loss']:.4f} | {row['total_time_s']:.0f}s total")
    return row


def _history(model):
    """Epoch-wise history for models that produce one (MLP); None otherwise."""
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "loss_curve_"):
        h = pd.DataFrame({"epoch": np.arange(1, len(est.loss_curve_) + 1), "train_loss": est.loss_curve_})
        if getattr(est, "validation_scores_", None) is not None:
            h["val_accuracy"] = est.validation_scores_
        return h
    if hasattr(est, "train_score_") and getattr(est, "validation_score_", None) is not None:
        # HistGradientBoosting: scores are negative log loss per iteration
        return pd.DataFrame({"iteration": np.arange(len(est.train_score_)),
                             "train_loss": -est.train_score_, "val_loss": -est.validation_score_})
    return None


def _upsert(path: Path, new: pd.DataFrame, key: str, value: str):
    if path.exists():
        old = pd.read_csv(path)
        old = old[old[key] != value]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
