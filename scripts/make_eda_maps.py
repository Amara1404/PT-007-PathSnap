"""Interactive Folium maps for EDA (GPS + road network + reference route).

    python scripts/make_eda_maps.py --raw "Student folder/train" --out results/maps
    python scripts/make_eda_maps.py --raw "Student folder/train" --trips 00000033 00000054 00000010

Trips 33 and 54 are the data-quality case studies (reference route far from the GPS).
Uses partition train only; no model is involved in Phase 1 maps.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.viz.maps import build_trip_map          # noqa: E402
from src.viz.trip_layers import load_layers      # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("Student folder/train"))
    ap.add_argument("--out", type=Path, default=Path("results/maps"))
    ap.add_argument("--trips", nargs="+", default=["00000033", "00000054", "00000010"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for tid in a.trips:
        d = next(iter(a.raw.rglob(tid)), None)
        if d is None or not d.is_dir():
            print(f"[skip] {tid}: folder not found under {a.raw}")
            continue
        L = load_layers(d)
        m = build_trip_map(L["gps"], network=L["network"], reference=L["reference"], title=f"Trip {tid} (train)")
        path = a.out / f"eda_trip_{tid}.html"
        m.save(str(path))
        print("wrote", path)
