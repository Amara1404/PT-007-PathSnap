"""Bayesian hyperparameter optimisation of HistGradientBoosting (report section 4).

    python -m src.models.tune_hgb --trials 40

Search: Optuna TPE sampler (a Bayesian method: it models which regions of the search
space gave good scores and samples there more), seed 42. Trial 0 is the untuned
default configuration, so the history starts from the baseline.

Objective (maximised): mean observation-level Top-1 accuracy over the same 5
trip-grouped folds as the model comparison, training on every 5th point, validating
on all usable points of the held-out fold. Macro F1 and log loss are recorded per trial.
Trials that are clearly worse than the median after the first folds are pruned.

Outputs:
  results/optuna_trials.csv         every trial: parameters, fold scores, time, state
  results/tuning_comparison.csv     untuned vs tuned on the same metrics and partitions
  results/figures/tuning_history.png, tuning_params.png, tuned_vs_untuned.png
  artifacts/models/hist_gb_tuned.joblib, artifacts/hist_gb_tuned_params.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.eval.metrics import observation_top1, row_metrics

from .common import ARTIFACTS, FEATURES, RESULTS, SEED, Experiment, load_candidates, run_experiment

DEFAULTS = {"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 1e-6,
            "limit_depth": False, "max_features": 1.0}
STRIDE = 5


def make_hgb(p: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=p["learning_rate"], max_iter=1000, max_leaf_nodes=p["max_leaf_nodes"],
        min_samples_leaf=p["min_samples_leaf"], l2_regularization=p["l2_regularization"],
        max_depth=p.get("max_depth") if p.get("limit_depth") else None, max_features=p["max_features"],
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=SEED)


def suggest(trial: optuna.Trial) -> dict:
    p = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 500, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
        "limit_depth": trial.suggest_categorical("limit_depth", [False, True]),
        "max_features": trial.suggest_float("max_features", 0.5, 1.0),
    }
    if p["limit_depth"]:
        p["max_depth"] = trial.suggest_int("max_depth", 3, 12)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    a = ap.parse_args(argv)

    folds = pd.read_csv("data/cv_folds.csv", dtype={"trip_id": str}).set_index("trip_id").fold
    full = load_candidates("train", stride=1)
    full["fold"] = full.trip_id.map(folds)
    sub = full[full.obs_idx % STRIDE == 0]
    fold_ids = sorted(folds.unique())

    def objective(trial):
        p = suggest(trial)
        top1s, f1s, lls, iters = [], [], [], []
        for step, k in enumerate(fold_ids):
            tr, va = sub[sub.fold != k], full[full.fold == k]
            m = make_hgb(p).fit(tr[FEATURES], tr.label)
            prob = m.predict_proba(va[FEATURES])[:, 1]
            top1s.append(observation_top1(va.assign(prob=prob))["top1_acc"])
            rm = row_metrics(va.label, prob)
            f1s.append(rm["macro_f1"]); lls.append(rm["log_loss"]); iters.append(m.n_iter_)
            trial.report(float(np.mean(top1s)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_top1", top1s)
        trial.set_user_attr("cv_macro_f1", float(np.mean(f1s)))
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("mean_boosting_iters", float(np.mean(iters)))
        return float(np.mean(top1s))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="hist_gb_top1",
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    study.enqueue_trial({k: v for k, v in DEFAULTS.items()})   # trial 0 = untuned defaults
    t0 = time.time()

    def log(st, tr):
        best = st.best_value if st.best_trial else float("nan")
        val = f"{tr.value:.4f}" if tr.value is not None else "pruned"
        print(f"  trial {tr.number:2d}: {val:>7} | best {best:.4f} | {tr.duration.total_seconds():.0f}s", flush=True)

    study.optimize(objective, n_trials=a.trials, callbacks=[log])
    search_s = time.time() - t0

    # ---- trial log ----
    df = study.trials_dataframe(attrs=("number", "value", "state", "duration", "params", "user_attrs"))
    df["duration_s"] = df.pop("duration").dt.total_seconds()
    df.to_csv(RESULTS / "optuna_trials.csv", index=False)

    best = dict(study.best_params)
    print(f"\nBest CV Top-1 {study.best_value:.4f} (trial {study.best_trial.number}); search took {search_s / 60:.1f} min")
    print("Best parameters:", json.dumps(best))

    # ---- refit tuned model with the standard runner (same folds, same student-test scoring) ----
    cfg = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in best.items())
    tuned = Experiment("hist_gb_tuned", "Boosting (tuned)", f"HistGB tuned: {cfg}, max_iter=1000, early stopping",
                       lambda: make_hgb(best), stride=STRIDE,
                       notes={"search_trials": a.trials, "search_minutes": search_s / 60})
    run_experiment(tuned)
    (ARTIFACTS / "hist_gb_tuned_params.json").write_text(json.dumps(
        {"model": "hist_gb_tuned", "params": best, "cv_top1": study.best_value,
         "source": "src/models/tune_hgb.py, results/optuna_trials.csv"}, indent=2))
    # The API model (artifacts/model.joblib) is chosen separately, from the tuned vs untuned comparison.

    # ---- untuned vs tuned table ----
    cmp = pd.read_csv(RESULTS / "model_comparison.csv").set_index("model").loc[["hist_gb", "hist_gb_tuned"]]
    cols = ["cv_macro_f1_mean", "cv_top1_mean", "cv_log_loss_mean", "st_top1_acc", "st_macro_f1",
            "st_log_loss", "st_route_continuity", "fit_time_s", "inference_ms_per_1k_obs"]
    out = cmp[cols].T
    out["change"] = out.hist_gb_tuned - out.hist_gb
    out.to_csv(RESULTS / "tuning_comparison.csv")
    print(out.round(4).to_string())
    pd.Series({"trials": a.trials, "pruned": int((df.state == "PRUNED").sum()),
               "search_minutes": round(search_s / 60, 2), "best_trial": study.best_trial.number}) \
        .to_csv(RESULTS / "tuning_cost.csv", header=["value"])


if __name__ == "__main__":
    main()
