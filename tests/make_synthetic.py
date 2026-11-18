"""Generate a tiny fake dataset in the raw PathSnap format, ONLY to smoke-test
the processing pipeline before the real data is available. Never train on it.

    python tests/make_synthetic.py --out tests/synthetic_raw --trips 12
"""
import argparse
from pathlib import Path

import numpy as np


def make_trip(rng, out: Path, name: str):
    lon0, lat0 = rng.uniform(-10, 30), rng.uniform(35, 60)
    n, spacing = 8, 120.0  # 8x8 grid of two-way streets, 120 m blocks
    kx = 111_320 * np.cos(np.deg2rad(lat0)); ky = 110_540
    ij = [(i, j) for i in range(n) for j in range(n)]
    nid = {p: k for k, p in enumerate(ij)}
    nodes = np.array([[lon0 + i * spacing / kx, lat0 + j * spacing / ky] for i, j in ij])
    arcs = []
    for i, j in ij:
        for di, dj in ((1, 0), (0, 1)):
            if i + di < n and j + dj < n:
                arcs += [(nid[(i, j)], nid[(i + di, j + dj)]), (nid[(i + di, j + dj)], nid[(i, j)])]
    arcs = np.array(arcs)
    arc_of = {(a, b): k for k, (a, b) in enumerate(arcs)}

    # random monotone-ish walk on the grid
    cur = (rng.integers(0, 3), rng.integers(0, 3)); route = []
    for _ in range(12):
        di, dj = [(1, 0), (0, 1)][rng.integers(0, 2)]
        nxt = (cur[0] + di, cur[1] + dj)
        if nxt[0] >= n or nxt[1] >= n:
            continue
        route.append(arc_of[(nid[cur], nid[nxt])]); cur = nxt

    # drive the route at ~10 m/s, 1 Hz, with noise and a couple of gaps
    pts, t = [], 1_000_000.0
    for a in route:
        A, B = nodes[arcs[a, 0]], nodes[arcs[a, 1]]
        for f in np.arange(0, 1, 10 / spacing):
            p = A + f * (B - A)
            noise = rng.normal(0, 6, 2) / [kx, ky]
            pts.append([*(p + noise), t]); t += 1 if rng.random() > 0.02 else rng.integers(5, 20)
    pts = np.array(pts)

    d = out / name; d.mkdir(parents=True, exist_ok=True)
    np.savetxt(d / f"{name}.track", pts, fmt=["%.7f", "%.7f", "%.1f"])
    np.savetxt(d / f"{name}.nodes", nodes, fmt="%.7f")
    np.savetxt(d / f"{name}.arcs", arcs, fmt="%d")
    np.savetxt(d / f"{name}.route", np.array(route), fmt="%d")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("tests/synthetic_raw"))
    ap.add_argument("--trips", type=int, default=12)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for k in range(a.trips):
        make_trip(rng, a.out, f"{k:08d}")
    print(f"wrote {a.trips} synthetic trips to {a.out}")
