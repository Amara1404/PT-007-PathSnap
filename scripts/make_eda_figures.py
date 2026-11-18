"""Regenerate every EDA figure and table:
    python scripts/make_eda_figures.py --raw "Student folder/train" --proc data/processed --out results/figures
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.viz import eda  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("Student folder/train"))
    ap.add_argument("--proc", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    ap.add_argument("--case-trips", nargs="+", default=["00000033", "00000054", "00000010"])
    a = ap.parse_args()
    inv, net, iss, both = eda.load_tables(a.proc)
    steps = eda.raw_dt_speed(a.raw)
    steps.groupby("trip_id").agg(n_steps=("dt", "size"), dt_max=("dt", "max"), speed_max_kmh=("speed_kmh", "max"),
                                 n_speed_gt_250=("speed_kmh", lambda s: int((s > 250).sum()))).to_csv(a.out.parent / "eda_raw_step_stats.csv")
    for p in (eda.fig_inventory(inv, a.out), eda.fig_time_gaps(inv, steps, a.out), eda.fig_speed(inv, steps, a.out),
              eda.fig_network(both, a.out), eda.fig_issues(iss, a.out), eda.fig_label_quality(inv, a.out),
              eda.fig_extent(inv, a.out)):
        print("wrote", p)
    for tid in a.case_trips:
        d = next(iter(a.raw.rglob(tid)), None)
        if d is not None and d.is_dir():
            print("wrote", eda.fig_trip_overview(d, a.out))
