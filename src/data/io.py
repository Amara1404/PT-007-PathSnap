"""Raw-file discovery, loading and cleaning.

Raw format (per trip, whitespace separated, no header):
  <trip>.track : lon lat t_seconds
  <trip>.nodes : lon lat               (row index = node_id, 0-based)
  <trip>.arcs  : from_node to_node     (row index = arc_id, 0-based)
  <trip>.route : arc_id                (chronological ground-truth arcs)

Trips are discovered by recursively searching for *.track files, so this works
whether each trip lives in its own folder or all files share one folder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PARTITIONS = ("train", "student_test", "eval")
_PARTITION_ALIASES = {
    "train": "train", "training": "train",
    "student_test": "student_test", "student_testing": "student_test", "studenttest": "student_test",
    "student": "student_test", "dev": "student_test", "test": "student_test",
    "eval": "eval", "evaluator": "eval", "evaluator_test": "eval", "evaluator_testing": "eval",
    "evaluation": "eval", "holdout": "eval",
}


@dataclass
class TripFiles:
    trip_id: str
    track: Path
    nodes: Path | None
    arcs: Path | None
    route: Path | None
    partition: str | None = None


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame            # lon, lat, t  (cleaned, sorted)
    nodes: pd.DataFrame            # lon, lat     (index = node_id)
    arcs: pd.DataFrame             # from_node, to_node (index = arc_id)
    route: np.ndarray | None       # arc ids, or None if not supplied
    issues: list[dict] = field(default_factory=list)


def _sibling(track: Path, ext: str) -> Path | None:
    p = track.with_suffix(ext)
    if p.exists():
        return p
    # also accept e.g. trip_folder/{nodes,arcs}.ext with differently named stems
    cands = list(track.parent.glob(f"*{ext}"))
    return cands[0] if len(cands) == 1 else None


def _partition_from_path(path: Path) -> str | None:
    for part in reversed(path.parts[:-1]):
        key = part.lower().replace("-", "_").replace(" ", "_")
        if key in _PARTITION_ALIASES:
            return _PARTITION_ALIASES[key]
    return None


def discover_trips(raw_dir: Path, splits_csv: Path | None = None) -> list[TripFiles]:
    raw_dir = Path(raw_dir)
    tracks = sorted(raw_dir.rglob("*.track"))
    if not tracks:
        raise FileNotFoundError(f"No *.track files found under {raw_dir}")

    split_map: dict[str, str] = {}
    if splits_csv is not None:
        s = pd.read_csv(splits_csv, dtype=str)
        split_map = {r.trip_id: _PARTITION_ALIASES.get(r.partition.lower(), r.partition)
                     for r in s.itertuples()}

    out = []
    for t in tracks:
        # trip id: folder name if the track sits in a per-trip folder, else file stem
        trip_id = t.stem if t.stem not in ("track", "gps") else t.parent.name
        out.append(TripFiles(
            trip_id=trip_id, track=t,
            nodes=_sibling(t, ".nodes"), arcs=_sibling(t, ".arcs"), route=_sibling(t, ".route"),
            partition=split_map.get(trip_id) or _partition_from_path(t.relative_to(raw_dir)),
        ))
    ids = [f.trip_id for f in out]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate trip ids: {sorted(dupes)[:10]}")
    return out


def _read_ws(path: Path, names: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python", comment="#")
    if df.shape[1] < len(names):
        raise ValueError(f"{path.name}: expected {len(names)} columns, found {df.shape[1]}")
    df = df.iloc[:, : len(names)]
    df.columns = names
    return df


def load_trip(f: TripFiles, partition: str) -> Trip:
    issues: list[dict] = []

    def log(kind, n=1, detail=""):
        issues.append({"trip_id": f.trip_id, "issue": kind, "count": int(n), "detail": detail})

    # ---- track -------------------------------------------------------------
    tr = _read_ws(f.track, ["lon", "lat", "t"]).apply(pd.to_numeric, errors="coerce")
    n_raw = len(tr)
    bad = tr.isna().any(axis=1)
    if bad.any():
        log("track_nan_or_non_numeric", bad.sum())
    tr = tr[~bad]
    # Detect swapped coordinate order (lat in first column) - common pitfall.
    if len(tr) and (tr.lat.abs() > 90).mean() > 0.5:
        log("track_lonlat_swapped", 1, "columns swapped back")
        tr = tr.rename(columns={"lon": "lat", "lat": "lon"})[["lon", "lat", "t"]]
    invalid = (tr.lon.abs() > 180) | (tr.lat.abs() > 90) | ((tr.lon == 0) & (tr.lat == 0))
    if invalid.any():
        log("track_invalid_coordinate", invalid.sum())
    tr = tr[~invalid]
    if not tr.t.is_monotonic_increasing:
        log("track_unsorted_timestamps", (tr.t.diff() < 0).sum())
        tr = tr.sort_values("t", kind="stable")
    dup_t = tr.t.duplicated(keep="first")
    if dup_t.any():
        log("track_duplicate_timestamp", dup_t.sum(), "kept first")
        tr = tr[~dup_t]
    tr = tr.reset_index(drop=True)
    tr["raw_row"] = tr.index  # positional index after cleaning
    if len(tr) < n_raw:
        log("track_rows_dropped", n_raw - len(tr))

    # ---- road network ------------------------------------------------------
    if f.nodes is None or f.arcs is None:
        raise FileNotFoundError(f"{f.trip_id}: missing .nodes or .arcs file")
    nodes = _read_ws(f.nodes, ["lon", "lat"]).apply(pd.to_numeric, errors="coerce")
    if nodes.isna().any(axis=None):
        log("nodes_nan", nodes.isna().any(axis=1).sum())
    arcs = _read_ws(f.arcs, ["from_node", "to_node"]).apply(pd.to_numeric, errors="coerce")
    bad_arc = arcs.isna().any(axis=1) | ~arcs.from_node.between(0, len(nodes) - 1) | \
        ~arcs.to_node.between(0, len(nodes) - 1)
    if bad_arc.any():
        log("arcs_reference_missing_node", bad_arc.sum(), "arc kept but unusable")
    arcs = arcs.fillna(-1).astype(int)
    arcs.index.name, nodes.index.name = "arc_id", "node_id"
    selfloops = (arcs.from_node == arcs.to_node).sum()
    if selfloops:
        log("arcs_self_loop", selfloops)

    # ---- route -------------------------------------------------------------
    route = None
    if f.route is not None:
        r = pd.read_csv(f.route, sep=r"\s+", header=None, engine="python").iloc[:, 0]
        r = pd.to_numeric(r, errors="coerce").dropna().astype(int).to_numpy()
        oob = (r < 0) | (r >= len(arcs))
        if oob.any():
            log("route_arc_out_of_range", oob.sum())
            r = r[~oob]
        # consecutive arcs should connect (to_node of a == from_node of b)
        if len(r) > 1:
            a, b = arcs.loc[r[:-1]], arcs.loc[r[1:]]
            breaks = (a.to_node.to_numpy() != b.from_node.to_numpy()).sum()
            if breaks:
                log("route_disconnected_steps", breaks, "reference route not fully contiguous")
        route = r
    elif partition != "eval":
        log("route_missing", 1)

    return Trip(f.trip_id, partition, tr, nodes, arcs, route, issues)
