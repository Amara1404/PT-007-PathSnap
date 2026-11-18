"""Load a raw trip folder into map/plot layers (GPS, nearby network, reference route).

Reference routes are read here ONLY for offline EDA/report maps of train / student-test trips.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.io import TripFiles, load_trip
from src.viz.geometry import nearby_arc_ids, segments_from_arcs


def load_layers(trip_dir, with_reference: bool = True, margin_m: float = 300.0, max_arcs: int = 30000) -> dict:
    d = Path(trip_dir)
    tid = d.name
    f = TripFiles(tid, d / f"{tid}.track", d / f"{tid}.nodes", d / f"{tid}.arcs",
                  (d / f"{tid}.route") if with_reference and (d / f"{tid}.route").exists() else None, "train")
    trip = load_trip(f, "train")
    gps = np.column_stack([trip.track.lat, trip.track.lon])
    fn, tn = trip.arcs.from_node.to_numpy(), trip.arcs.to_node.to_numpy()
    lon, lat = trip.nodes.lon.to_numpy(), trip.nodes.lat.to_numpy()
    near = nearby_arc_ids(fn, tn, lon, lat, gps, margin_m, max_arcs)
    return {
        "trip_id": tid, "gps": gps,
        "network": segments_from_arcs(fn, tn, lon, lat, near, collapse=False),
        "reference": segments_from_arcs(fn, tn, lon, lat, trip.route, collapse=False) if trip.route is not None else None,
        "trip": trip,
    }
