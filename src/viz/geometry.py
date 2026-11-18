"""Folium-free geometry helpers shared by the API, the maps and the EDA plots.

Segment format used everywhere: [[lat, lon], [lat, lon]] (Folium order, not lon/lat).
"""
from __future__ import annotations

import numpy as np


def segments_from_arcs(from_node, to_node, node_lon, node_lat, arc_ids, collapse: bool = True) -> list:
    """Arc ids -> list of [[lat, lon], [lat, lon]] segments (one per arc, in order).

    collapse=True merges consecutive repeats of the same arc (many GPS points sit on one arc).
    """
    arc_ids = np.asarray(arc_ids, dtype=np.int64)
    arc_ids = arc_ids[arc_ids >= 0]
    if collapse and len(arc_ids) > 1:
        arc_ids = arc_ids[np.r_[True, arc_ids[1:] != arc_ids[:-1]]]
    f = np.asarray(from_node)[arc_ids]
    t = np.asarray(to_node)[arc_ids]
    lon, lat = np.asarray(node_lon), np.asarray(node_lat)
    return [[[float(lat[a]), float(lon[a])], [float(lat[b]), float(lon[b])]] for a, b in zip(f, t)]


def nearby_arc_ids(from_node, to_node, node_lon, node_lat, gps_latlon, margin_m: float = 300.0,
                   max_arcs: int = 30000) -> np.ndarray:
    """Arcs with an endpoint inside the GPS bounding box (+ margin); margin shrinks if too many.

    Keeps the map readable and the HTML small: the full network of a trip can have 300k arcs.
    """
    gps = np.asarray(gps_latlon, float)
    lat0, lon0 = gps[:, 0].mean(), gps[:, 1].mean()
    m_per_deg_lat = 111_195.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.deg2rad(lat0))
    lat_lo, lat_hi = gps[:, 0].min(), gps[:, 0].max()
    lon_lo, lon_hi = gps[:, 1].min(), gps[:, 1].max()
    lon, lat = np.asarray(node_lon), np.asarray(node_lat)
    f, t = np.asarray(from_node), np.asarray(to_node)
    ids = np.arange(len(f))
    valid = (f >= 0) & (t >= 0)
    ids, f, t = ids[valid], f[valid], t[valid]
    sel = ids
    while True:
        dlat, dlon = margin_m / m_per_deg_lat, margin_m / m_per_deg_lon
        def inside(n):
            return (lat[n] >= lat_lo - dlat) & (lat[n] <= lat_hi + dlat) & \
                   (lon[n] >= lon_lo - dlon) & (lon[n] <= lon_hi + dlon)
        sel = ids[inside(f) | inside(t)]
        if len(sel) <= max_arcs or margin_m <= 10:
            return sel
        margin_m *= 0.5
