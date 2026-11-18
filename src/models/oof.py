"""Out-of-fold (OOF) candidate probabilities on the 80 training trips.

For each fold k the model is fitted on the other folds (same rows and settings as
run_experiment) and predicts EVERY candidate row of fold k's trips, including
low-confidence points, so the HMM can decode complete trips. A model never
scores a trip it was trained on, so these probabilities are as honest as
student-test probabilities - use them to tune HMM parameters.

    python -m src.models.oof --model hist_gb
    -> results/oof/<model>_train_oof.parquet  (trip_id, obs_idx, arc_id, prob)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .common import FEATURES, PROCESSED, RESULTS, load_candidates
from .train_models import ALIASES, EXPERIMENTS


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="hist_gb")
    p.add_argument("--folds", type=Path, default=Path("data/cv_folds.csv"))
    a = p.parse_args(argv)
    exp = EXPERIMENTS[ALIASES.get(a.model, a.model)]

    folds = pd.read_csv(a.folds, dtype={"trip_id": str}).set_index("trip_id").fold
    train = load_candidates("train", stride=1)                       # usable rows (training data)
    train = train[train.obs_idx % exp.stride == 0]
    train["fold"] = train.trip_id.map(folds)
    every = pd.read_parquet(PROCESSED / "candidates_train.parquet", columns=["trip_id", "obs_idx", "arc_id"] + FEATURES)
    every["fold"] = every.trip_id.map(folds)

    out = []
    t0 = time.time()
    for k in sorted(folds.unique()):
        tr = train[train.fold != k]
        model = exp.make_model().fit(tr[FEATURES], tr.label)
        va = every[every.fold == k]
        out.append(va[["trip_id", "obs_idx", "arc_id"]].assign(prob=model.predict_proba(va[FEATURES])[:, 1]))
        print(f"  fold {k}: {va.trip_id.nunique()} trips, {len(va):,} rows")
    oof = pd.concat(out, ignore_index=True)
    path = RESULTS / "oof" / f"{exp.name}_train_oof.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(path, index=False)
    print(f"wrote {path} ({len(oof):,} rows, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
