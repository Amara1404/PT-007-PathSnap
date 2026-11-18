"""EDA figures for the PathSnap dataset (Phase 1). Every figure is drawn from files in
data/processed/*.csv or from the raw trip files, and saved to results/figures/.

Partition note: the CSVs cover train + student_test; raw-file plots cover whichever
partitions are present under --raw (the evaluator partition is never used here).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.geo import LocalProjection

COL = {"train": "#1f77b4", "student_test": "#ff7f0e", "eval": "#7f7f7f"}
CASE = ("00000033", "00000054")


def load_tables(proc: Path):
    rd = lambda n: pd.read_csv(proc / n, dtype={"trip_id": str})
    inv, net, iss = rd("inventory_trips.csv"), rd("network_stats.csv"), rd("data_quality_issues.csv")
    return inv, net, iss, inv.merge(net, on="trip_id")


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    return out / name


def fig_inventory(inv, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    c = inv.partition.value_counts()
    ax[0].bar(c.index, c.values, color=[COL.get(p) for p in c.index])
    for i, v in enumerate(c.values):
        ax[0].text(i, v + 1, str(v), ha="center")
    ax[0].set(title="Trips per partition", ylabel="trips")
    for p, g in inv.groupby("partition"):
        ax[1].hist(g.n_obs, bins=np.logspace(np.log10(inv.n_obs.min()), np.log10(inv.n_obs.max()), 20),
                   alpha=0.7, label=p, color=COL.get(p))
    ax[1].set(xscale="log", title="GPS points per trip", xlabel="points (log)", ylabel="trips")
    ax[1].legend()
    return _save(fig, out, "eda_01_trips_and_points.png")


def raw_dt_speed(raw: Path) -> pd.DataFrame:
    """Per-step dt (s) and speed (km/h) for every trip under raw (local metric projection)."""
    rows = []
    for f in sorted(raw.rglob("*.track")):
        tr = pd.read_csv(f, sep=r"\s+", header=None, names=["lon", "lat", "t"])
        pr = LocalProjection(tr.lon.mean(), tr.lat.mean())
        x, y = pr.to_xy(tr.lon, tr.lat)
        dt = np.diff(tr.t.to_numpy())
        rows.append(pd.DataFrame({"trip_id": f.stem, "dt": dt, "speed_kmh": np.hypot(np.diff(x), np.diff(y)) / dt * 3.6}))
    return pd.concat(rows, ignore_index=True)


def fig_time_gaps(inv, steps, out):
    bins = [0, 1, 2, 3, 5, 10, 60, np.inf]
    labels = ["1 s", "2 s", "3 s", "4-5 s", "6-10 s", "11-60 s", ">60 s"]
    cnt = pd.cut(steps.dt, bins=bins, labels=labels, right=True).value_counts().reindex(labels)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(labels, cnt.values, color="#1f77b4")
    ax[0].set(yscale="log", title=f"Time between consecutive GPS points (train raw, {steps.trip_id.nunique()} trips)", ylabel="steps (log)")
    for i, v in enumerate(cnt.values):
        ax[0].text(i, v * 1.1, f"{v:,}", ha="center", fontsize=8)
    g = inv.sort_values("n_gaps_gt_5s", ascending=False).head(12)
    ax[1].bar(g.trip_id, g.n_gaps_gt_5s, color=[COL.get(p) for p in g.partition])
    ax[1].set(title="Trips with most gaps > 5 s", ylabel="gaps")
    ax[1].tick_params(axis="x", rotation=60)
    return _save(fig, out, "eda_02_time_gaps.png")


def fig_speed(inv, steps, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    s = steps.speed_kmh[(steps.speed_kmh < 400)]
    ax[0].hist(s, bins=80, color="#2ca02c")
    ax[0].axvline(250, color="r", ls="--", label="250 km/h (implausible)")
    ax[0].set(yscale="log", title="Step speed (train raw, <400 km/h shown)", xlabel="km/h", ylabel="steps (log)")
    ax[0].legend()
    g = inv.sort_values("max_speed_kmh", ascending=False).head(12)
    ax[1].bar(g.trip_id, g.max_speed_kmh, color=[COL.get(p) for p in g.partition])
    ax[1].axhline(250, color="r", ls="--")
    ax[1].set(title="Highest max speed per trip", ylabel="km/h")
    ax[1].tick_params(axis="x", rotation=60)
    return _save(fig, out, "eda_03_speed_outliers.png")


def fig_network(df, out):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for p, g in df.groupby("partition"):
        ax[0].scatter(g.n_nodes, g.n_arcs, s=18, alpha=0.8, label=p, color=COL.get(p))
    ax[0].set(xscale="log", yscale="log", xlabel="nodes", ylabel="arcs", title="Road-network size per trip")
    ax[0].legend()
    ax[1].scatter(df.n_nodes, df.n_weak_components, s=18, c=[COL.get(p) for p in df.partition])
    ax[1].set(xscale="log", yscale="log", xlabel="nodes", ylabel="weak components", title="Disconnected components")
    ax[2].hist(df.largest_weak_component_frac, bins=20, color="#9467bd")
    ax[2].set(xlabel="largest weak component / used nodes", ylabel="trips", title="Share of network in largest component")
    return _save(fig, out, "eda_04_network_size_connectivity.png")


def fig_issues(iss, out):
    c = iss.groupby("issue").trip_id.nunique().sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(c.index, c.values, color="#d62728")
    for i, v in enumerate(c.values):
        ax.text(v + 0.2, i, str(v), va="center")
    ax.set(xlabel="trips affected", title="Data-quality issues found by the build")
    return _save(fig, out, "eda_05_data_quality_issues.png")


def fig_label_quality(inv, out):
    d = inv.sort_values("frac_label_low_conf", ascending=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    top = d.head(10)
    ax[0].bar(top.trip_id, top.frac_label_low_conf, color=["#d62728" if t in CASE else "#1f77b4" for t in top.trip_id])
    ax[0].set(title="Share of GPS points > 30 m from the reference route", ylabel="fraction")
    ax[0].tick_params(axis="x", rotation=60)
    ax[1].scatter(inv.label_dist_median_m, inv.label_dist_p95_m, s=18, c=["#d62728" if t in CASE else "#1f77b4" for t in inv.trip_id])
    for t in CASE:
        r = inv[inv.trip_id == t].iloc[0]
        ax[1].annotate(t, (r.label_dist_median_m, r.label_dist_p95_m))
    ax[1].set(xlabel="median GPS-to-reference distance (m)", ylabel="95th percentile (m)", title="Reference-route alignment per trip")
    return _save(fig, out, "eda_06_reference_alignment.png")


def fig_extent(inv, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for p, g in inv.groupby("partition"):
        ax[0].scatter((g.lon_min + g.lon_max) / 2, (g.lat_min + g.lat_max) / 2, s=22, label=p, color=COL.get(p))
    ax[0].set(xlabel="longitude", ylabel="latitude", title="Trip locations (centre of each track)")
    ax[0].legend()
    ax[1].scatter(inv.extent_x_km, inv.extent_y_km, s=18, c=[COL.get(p) for p in inv.partition])
    ax[1].set(xlabel="east-west extent (km)", ylabel="north-south extent (km)", title="Trip extent (local metric projection)")
    return _save(fig, out, "eda_07_geographic_extent.png")


def fig_trip_overview(trip_dir: Path, out: Path):
    """Static GPS + nearby network + reference route for one trip (matplotlib twin of the Folium map)."""
    from src.viz.trip_layers import load_layers
    L = load_layers(trip_dir)
    fig, ax = plt.subplots(figsize=(7, 7))
    def draw(segs, **kw):
        for s in segs:
            ax.plot([s[0][1], s[1][1]], [s[0][0], s[1][0]], **kw)
    draw(L["network"], color="#bbbbbb", lw=0.6, zorder=1)
    if L["reference"]:
        draw(L["reference"], color="#2ca02c", lw=3, alpha=0.7, zorder=2)
    ax.scatter(L["gps"][:, 1], L["gps"][:, 0], s=3, color="#1f5fbf", zorder=5)
    ax.set(title=f"Trip {L['trip_id']}: GPS (blue), reference route (green), network (grey)", xlabel="lon", ylabel="lat")
    ax.set_aspect(1 / np.cos(np.deg2rad(L["gps"][:, 0].mean())))
    return _save(fig, out, f"eda_trip_{L['trip_id']}.png")
