"""Build all processed datasets for PathSnap from the raw trip files.

Usage
-----
    python -m src.data.build_datasets --raw data/raw --out data/processed
    python -m src.data.build_datasets --raw data/raw --splits data/splits.csv

Partitions come from (in order): --splits CSV (columns trip_id,partition),
folder names in the raw path (train / student_test / eval ...). If neither
gives a partition for every trip, a seeded 80/10/10 split is generated and
written to <out>/splits_generated.csv - replace it with the official split as
soon as you have it.

Outputs (in --out)
------------------
  inventory_trips.csv              per-trip EDA inventory (counts, sampling, gaps, extent)
  network_stats.csv                per-trip road-network size + connectivity
  data_quality_issues.csv          every cleaning action / anomaly found
  arcs.parquet                     projected arc geometry for all trips (for HMM + maps)
  observations_<part>.parquet      cleaned GPS points + point features (+ labels, not eval)
  candidates_<part>.parquet        (observation, candidate arc) rows + features (+ label, not eval)
  candidate_summary.csv            candidate recall / miss rate per partition
  server_only/eval_labels.parquet  evaluator ground truth - API side only, never a model input
  feature_columns.json, build_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import FEATURE_COLUMNS, candidate_features, generate_candidates, observation_features
from .geo import LocalProjection
from .io import PARTITIONS, discover_trips, load_trip
from .labels import align_route
from .network import build_network, network_stats


def assign_partitions(files, seed: int, out: Path):
    if all(f.partition for f in files):
        return
    ids = sorted(f.trip_id for f in files)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_eval = n_st = max(1, round(n * 0.10))
    part = {i: "eval" for i in ids[:n_eval]}
    part.update({i: "student_test" for i in ids[n_eval:n_eval + n_st]})
    part.update({i: "train" for i in ids[n_eval + n_st:]})
    for f in files:
        f.partition = f.partition or part[f.trip_id]
    pd.DataFrame({"trip_id": [f.trip_id for f in files], "partition": [f.partition for f in files]}) \
        .sort_values(["partition", "trip_id"]).to_csv(out / "splits_generated.csv", index=False)
    print(f"[warn] partitions not supplied -> generated seeded split: {out / 'splits_generated.csv'}")


def process_trip(f, args):
    trip = load_trip(f, f.partition)
    tr = trip.track
    if len(tr) == 0:
        raise ValueError("no valid GPS observations after cleaning")
    proj = LocalProjection(tr.lon.mean(), tr.lat.mean())
    x, y = proj.to_xy(tr.lon.to_numpy(), tr.lat.to_numpy())
    t = tr.t.to_numpy(float)
    net = build_network(trip.nodes, trip.arcs, proj)

    obs = observation_features(x, y, t, gap_s=args.gap_s)
    obs.insert(0, "trip_id", trip.trip_id)
    obs.insert(1, "obs_idx", np.arange(len(tr)))
    obs["lon"], obs["lat"], obs["t"], obs["x"], obs["y"] = tr.lon, tr.lat, t, x, y

    # ---- labels (all partitions computed; eval ones are routed to server_only) ----
    has_labels = trip.route is not None and len(trip.route) > 0
    if has_labels:
        ref, pos, ldist = align_route(x, y, trip.route, net.arcs)
        obs["ref_arc"], obs["route_pos"], obs["label_dist_m"] = ref, pos, ldist
        obs["label_low_conf"] = (ldist > args.label_max_dist).astype(np.int8)

    # ---- candidates ----
    cand = generate_candidates(x, y, net, args.radius, args.max_cands, sample_step_m=args.sample_step)
    cand = candidate_features(cand, obs, x, y, net)
    cand.insert(0, "trip_id", trip.trip_id)
    if has_labels:
        cand["label"] = (cand.arc_id.to_numpy() == obs.ref_arc.to_numpy()[cand.obs_idx]).astype(np.int8)
        cand["label_low_conf"] = obs.label_low_conf.to_numpy()[cand.obs_idx]
        hit = cand.groupby("obs_idx").label.max().reindex(obs.obs_idx, fill_value=0).to_numpy()
        obs["ref_in_candidates"] = hit.astype(np.int8)

    # ---- inventory row ----
    dt = np.diff(t)
    step = obs.step_m.to_numpy()[1:]
    inv = {
        "trip_id": trip.trip_id, "partition": f.partition,
        "track_bytes": f.track.stat().st_size,
        "nodes_bytes": f.nodes.stat().st_size if f.nodes else 0,
        "arcs_bytes": f.arcs.stat().st_size if f.arcs else 0,
        "has_route": int(has_labels),
        "route_len_arcs": int(len(trip.route)) if has_labels else 0,
        "n_obs": len(tr),
        "duration_s": float(t[-1] - t[0]),
        "dt_median_s": float(np.median(dt)) if len(dt) else np.nan,
        "dt_frac_exactly_1s": float(np.mean(dt == 1)) if len(dt) else np.nan,
        "dt_max_s": float(dt.max()) if len(dt) else np.nan,
        f"n_gaps_gt_{args.gap_s:g}s": int((dt > args.gap_s).sum()),
        "path_length_km": float(np.nansum(step) / 1000),
        "mean_speed_kmh": float(np.nansum(step) / max(t[-1] - t[0], 1) * 3.6),
        "max_speed_kmh": float(np.nanmax(obs.speed_mps) * 3.6) if len(tr) > 1 else np.nan,
        "n_implausible_speed": int(obs.implausible_speed.sum()),
        "frac_stationary": float(1 - obs.heading_valid.mean()),
        "lon_min": tr.lon.min(), "lon_max": tr.lon.max(), "lat_min": tr.lat.min(), "lat_max": tr.lat.max(),
        "extent_x_km": float((x.max() - x.min()) / 1000), "extent_y_km": float((y.max() - y.min()) / 1000),
        "mean_cands_per_obs": float(cand.groupby("obs_idx").size().mean()),
        "frac_obs_fallback_cands": float(cand.groupby("obs_idx").cand_fallback.max().mean()),
    }
    if has_labels:
        inv.update({
            "label_dist_median_m": float(np.median(obs.label_dist_m)),
            "label_dist_p95_m": float(np.percentile(obs.label_dist_m, 95)),
            "frac_label_low_conf": float(obs.label_low_conf.mean()),
            "candidate_miss_rate": float(1 - obs.ref_in_candidates.mean()),
            "route_arcs_covered_frac": float(len(np.unique(obs.route_pos)) / len(trip.route)),
        })

    arcs_out = net.arcs.reset_index()
    arcs_out.insert(0, "trip_id", trip.trip_id)
    lon_a, lat_a = trip.nodes.lon.to_numpy(), trip.nodes.lat.to_numpy()
    fn, tn = arcs_out.from_node.clip(lower=0), arcs_out.to_node.clip(lower=0)
    arcs_out["from_lon"], arcs_out["from_lat"] = lon_a[fn], lat_a[fn]
    arcs_out["to_lon"], arcs_out["to_lat"] = lon_a[tn], lat_a[tn]
    # shrink: local metres fit float32 (<1 cm error within ~100 km); lon/lat stay float64
    arcs_out = arcs_out.astype({c: "float32" for c in ("ax", "ay", "bx", "by", "length_m", "bearing")})
    arcs_out = arcs_out.astype({c: "int32" for c in ("arc_id", "from_node", "to_node", "to_out_deg", "from_in_deg")})
    return trip, inv, network_stats(trip.trip_id, net), obs, cand, arcs_out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--splits", type=Path, default=None)
    p.add_argument("--radius", type=float, default=50.0, help="candidate search radius (m)")
    p.add_argument("--max-cands", type=int, default=10, help="max candidates per observation")
    p.add_argument("--sample-step", type=float, default=10.0, help="arc densification step (m)")
    p.add_argument("--gap-s", type=float, default=5.0, help="time gap (s) counted as a GPS gap")
    p.add_argument("--label-max-dist", type=float, default=30.0,
                   help="alignment distance (m) above which a label is flagged low-confidence")
    p.add_argument("--max-trip-low-conf", type=float, default=0.5,
                   help="flag a whole trip unusable if more than this fraction of its labels is low-confidence")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                   help="parallel worker processes (1 = sequential)")
    args = p.parse_args(argv)

    out = args.out
    (out / "server_only").mkdir(parents=True, exist_ok=True)
    files = discover_trips(args.raw, args.splits)
    assign_partitions(files, args.seed, out)
    print(f"Found {len(files)} trips: " +
          ", ".join(f"{k}={sum(f.partition == k for f in files)}" for k in PARTITIONS))

    inv_rows, net_rows, issues, arcs_all = [], [], [], []
    obs_by, cand_by = {k: [] for k in PARTITIONS}, {k: [] for k in PARTITIONS}
    t0 = time.time()
    # Trips are independent -> process them in parallel worker processes.
    # Results are consumed in the original trip order so outputs are deterministic.
    workers = max(1, args.workers)
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    futures = [pool.submit(process_trip, f, args) for f in files] if pool else None
    for n, f in enumerate(files, 1):
        try:
            res = futures[n - 1].result() if pool else process_trip(f, args)
            trip, inv, ns, obs, cand, arcs_out = res
        except Exception as e:  # keep going; record the failure
            issues.append({"trip_id": f.trip_id, "issue": "trip_failed", "count": 1, "detail": repr(e)})
            print(f"  [{n}/{len(files)}] {f.trip_id}: FAILED {e!r}")
            continue
        issues += trip.issues
        inv_rows.append(inv); net_rows.append(ns); arcs_all.append(arcs_out)
        obs_by[f.partition].append(obs); cand_by[f.partition].append(cand)
        print(f"  [{n}/{len(files)}] {f.trip_id} ({f.partition}): {len(obs)} obs, {len(cand)} cand rows")
    if pool:
        pool.shutdown()

    pd.DataFrame(inv_rows).to_csv(out / "inventory_trips.csv", index=False)
    pd.DataFrame(net_rows).to_csv(out / "network_stats.csv", index=False)
    pd.concat(arcs_all, ignore_index=True).to_parquet(out / "arcs.parquet", index=False)

    label_cols = ["ref_arc", "route_pos", "label_dist_m", "label_low_conf", "ref_in_candidates"]
    summary = []
    for part in PARTITIONS:
        if not obs_by[part]:
            continue
        obs = pd.concat(obs_by[part], ignore_index=True)
        cand = pd.concat(cand_by[part], ignore_index=True)
        if part != "eval" and "label_low_conf" in obs:
            # Trips whose reference labels are mostly unreliable (road network or route
            # does not cover the GPS track) are kept for EDA but flagged unusable.
            trip_bad = obs.groupby("trip_id").label_low_conf.mean()
            bad = trip_bad[trip_bad > args.max_trip_low_conf]
            for tid, frac in bad.items():
                issues.append({"trip_id": tid, "issue": "trip_labels_unreliable", "count": 1,
                               "detail": f"{frac:.1%} of points >{args.label_max_dist:g} m from reference route; usable=0"})
            obs["usable"] = ((obs.label_low_conf == 0) & ~obs.trip_id.isin(bad.index)).astype(np.int8)
            cand = cand.merge(obs[["trip_id", "obs_idx", "usable"]], on=["trip_id", "obs_idx"], how="left")
        if part == "eval":
            # ground truth stays server-side; model-facing files carry no labels
            keep = [c for c in label_cols if c in obs]
            if keep:
                obs[["trip_id", "obs_idx", *keep]].to_parquet(out / "server_only" / "eval_labels.parquet", index=False)
            obs = obs.drop(columns=keep)
            cand = cand.drop(columns=[c for c in ("label", "label_low_conf") if c in cand])
        obs.to_parquet(out / f"observations_{part}.parquet", index=False)
        cand.to_parquet(out / f"candidates_{part}.parquet", index=False)
        row = {"partition": part, "trips": obs.trip_id.nunique(), "observations": len(obs),
               "candidate_rows": len(cand), "mean_cands_per_obs": round(len(cand) / len(obs), 2)}
        if "label" in cand:
            u = obs.usable == 1
            row.update({"usable_obs_frac": round(u.mean(), 4),
                        "candidate_miss_rate_usable": round(1 - obs.ref_in_candidates[u].mean(), 4),
                        "positive_rate": round(cand.label.mean(), 4),
                        "candidate_miss_rate": round(1 - obs.ref_in_candidates.mean(), 4),
                        "nearest_arc_top1_acc": round(cand[cand.dist_rank == 0].label.mean(), 4),
                        "low_conf_label_frac": round(obs.label_low_conf.mean(), 4)})
        summary.append(row)
    pd.DataFrame(summary).to_csv(out / "candidate_summary.csv", index=False)
    pd.DataFrame(issues, columns=["trip_id", "issue", "count", "detail"]).to_csv(out / "data_quality_issues.csv", index=False)

    (out / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    (out / "build_config.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, indent=2))
    print(f"\nDone in {time.time() - t0:.1f}s -> {out}")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
