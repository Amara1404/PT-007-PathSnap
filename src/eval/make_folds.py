"""Create the shared, trip-grouped cross-validation folds (data/cv_folds.csv).

All observations of a trip stay in one fold (no leakage between correlated 1 Hz
points). Trip sizes vary 60x, so trips are assigned greedily - largest first, to
the fold with the fewest usable observations so far - giving balanced folds.
A seeded shuffle breaks ties so the assignment is reproducible.

    python -m src.eval.make_folds
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("data/cv_folds.csv"))
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    obs = pd.read_parquet(a.processed / "observations_train.parquet", columns=["trip_id", "usable"])
    size = obs.groupby("trip_id").usable.sum()
    rng = np.random.default_rng(a.seed)
    order = size.sample(frac=1, random_state=int(rng.integers(1 << 31))).sort_values(ascending=False, kind="stable")
    load = np.zeros(a.k)
    fold = {}
    for trip, n in order.items():
        f = int(np.argmin(load))
        fold[trip] = f
        load[f] += n
    out = pd.DataFrame({"trip_id": list(fold), "fold": list(fold.values())}).sort_values("trip_id")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.groupby("fold").trip_id.count().rename("trips").to_frame()
          .join(pd.Series(load, name="usable_obs").astype(int)))


if __name__ == "__main__":
    main()
