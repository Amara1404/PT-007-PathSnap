"""SERVER-SIDE reference handling: reads the .route file and turns it into per-observation labels.

This module is the only place the evaluator's ground truth is read at inference time.
It is called by the API after prediction, and its output never goes to the model,
the decoder or the UI (only scores computed from it leave the server).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.labels import align_route

LABEL_MAX_DIST_M = 30.0        # same defaults as src/data/build_datasets.py
MAX_TRIP_LOW_CONF = 0.5


def read_route(route_file: Path, n_arcs: int) -> np.ndarray:
    r = pd.read_csv(route_file, sep=r"\s+", header=None, engine="python").iloc[:, 0]
    r = pd.to_numeric(r, errors="coerce").dropna().astype(int).to_numpy()
    return r[(r >= 0) & (r < n_arcs)]


def usable_mask(label_dist_m: np.ndarray, max_dist: float = LABEL_MAX_DIST_M,
                max_trip_low_conf: float = MAX_TRIP_LOW_CONF) -> np.ndarray:
    """Same rule as the training build: label within max_dist, and trip not mostly unreliable."""
    low = np.asarray(label_dist_m) > max_dist
    if low.mean() > max_trip_low_conf:
        return np.zeros(len(low), bool)
    return ~low


def load_reference(route_file: Path | None, x: np.ndarray, y: np.ndarray, arcs: pd.DataFrame) -> dict | None:
    """Per-observation reference arc, alignment distance and usable flag (None if no route file)."""
    if route_file is None or not Path(route_file).exists():
        return None
    route = read_route(Path(route_file), len(arcs))
    if len(route) == 0:
        return None
    ref, _, dist = align_route(x, y, route, arcs)
    return {"ref_arc": ref, "label_dist_m": dist, "usable": usable_mask(dist), "route": route}
