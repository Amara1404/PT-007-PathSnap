"""End-to-end inference for one trip: raw files -> features -> classifier -> decoder.

The API calls only `predict_trip` and `evaluate_trip`.

Leakage guard: `predict_trip` loads the trip WITHOUT its .route file, so the
reference route can never reach features, model or decoder. `evaluate_trip`
reads the route afterwards, server-side, only to score the finished prediction.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.build_datasets import process_trip
from src.data.geo import LocalProjection
from src.data.io import TripFiles, load_trip
from src.data.labels import align_route
from src.data.network import build_network
from src.eval.metrics import route_continuity, top1_choice
from src.hmm.viterbi import decode

ARTIFACTS = Path("artifacts")
PROCESSED = Path("data/processed")
FEATURES = json.loads((PROCESSED / "feature_columns.json").read_text())
BUILD_ARGS = argparse.Namespace(**{k.replace("-", "_"): v for k, v in
                                   json.loads((PROCESSED / "build_config.json").read_text()).items()})


class _Nearest:  # fallback until artifacts/model.joblib exists
    def predict_proba(self, X):
        p = (X["dist_rank"].to_numpy() == 0).astype(float)
        return np.column_stack([1 - p, p])


def load_model():
    path = ARTIFACTS / "model.joblib"
    model = joblib.load(path) if path.exists() else _Nearest()
    params_path = ARTIFACTS / "hmm_params.json"
    params = json.loads(params_path.read_text()) if params_path.exists() else {}
    return model, params, (path.name if path.exists() else "nearest-arc stub")


def _trip_files(trip_dir: Path, with_route: bool) -> TripFiles:
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    get = lambda ext: (trip_dir / f"{tid}{ext}") if (trip_dir / f"{tid}{ext}").exists() else None
    return TripFiles(tid, trip_dir / f"{tid}.track", get(".nodes"), get(".arcs"),
                     get(".route") if with_route else None, partition="eval")


def predict_trip(trip_dir: Path, model=None, params=None) -> dict:
    if model is None:
        model, params, _ = load_model()
    t0 = time.time()
    f = _trip_files(trip_dir, with_route=False)
    trip, _, _, obs, cand, arcs = process_trip(f, BUILD_ARGS)
    t1 = time.time()
    cand["prob"] = model.predict_proba(cand[FEATURES])[:, 1]
    classifier_pick = top1_choice(cand, "prob")
    decoded = decode(cand, obs, arcs, params)
    t2 = time.time()
    return {"trip_id": trip.trip_id, "obs": obs, "arcs": arcs, "classifier_pick": classifier_pick,
            "decoded": decoded, "timing_ms": {"features": (t1 - t0) * 1e3, "model_decode": (t2 - t1) * 1e3}}


def evaluate_trip(trip_dir: Path, pred: dict) -> dict:
    """Score a finished prediction against the server-side reference route."""
    trip = load_trip(_trip_files(trip_dir, with_route=True), "eval")
    if trip.route is None:
        return {}
    tr = trip.track
    proj = LocalProjection(tr.lon.mean(), tr.lat.mean())
    x, y = proj.to_xy(tr.lon.to_numpy(), tr.lat.to_numpy())
    ref_arc, _, dist = align_route(x, y, trip.route, build_network(trip.nodes, trip.arcs, proj).arcs)
    ref = pd.DataFrame({"obs_idx": np.arange(len(tr)), "ref_arc": ref_arc,
                        "reliable": dist <= BUILD_ARGS.label_max_dist})

    def acc(p):
        m = p.merge(ref, on="obs_idx")
        m = m[m.reliable]
        return float((m.arc_id == m.ref_arc).mean())

    arcs_nodes = pred["arcs"][["trip_id", "arc_id", "from_node", "to_node"]]
    return {"top1_accuracy": acc(pred["decoded"]),
            "classifier_only_top1": acc(pred["classifier_pick"]),
            "route_continuity": route_continuity(pred["decoded"], arcs_nodes),
            "classifier_only_continuity": route_continuity(pred["classifier_pick"], arcs_nodes),
            "n_obs_scored": int(ref.reliable.sum())}


def route_geometry(pred: dict) -> list:
    """Predicted route as [[lat, lon], [lat, lon]] segments, consecutive duplicates removed."""
    seq = pred["decoded"].sort_values("obs_idx").arc_id.to_numpy()
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]]
    a = pred["arcs"].set_index("arc_id").loc[seq]
    return [[[r.from_lat, r.from_lon], [r.to_lat, r.to_lon]] for r in a.itertuples()]


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "Student folder/student_test/00000000")
    p = predict_trip(d)
    print(p["trip_id"], len(p["obs"]), "obs", p["timing_ms"], evaluate_trip(d, p))
