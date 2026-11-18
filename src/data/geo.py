"""Geometry helpers.

Coordinates in the raw files are (longitude, latitude) in degrees. Distances in
degrees are meaningless (1 deg of longitude shrinks with cos(latitude)), so each
trip is projected to a local metric plane (east/north metres) centred on that
trip's own centroid. At city scale (< ~50 km) this equirectangular/ENU
approximation is accurate to well under 0.1 %, and unlike a fixed UTM zone it
works for trips anywhere in the world without per-trip zone lookup.
"""
from __future__ import annotations

import numpy as np

EARTH_R = 6_371_008.8  # mean Earth radius (m)


class LocalProjection:
    """lon/lat (deg) <-> local x/y (m) around a fixed origin."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self._kx = np.deg2rad(1.0) * EARTH_R * np.cos(np.deg2rad(self.lat0))
        self._ky = np.deg2rad(1.0) * EARTH_R

    def to_xy(self, lon, lat):
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        return (lon - self.lon0) * self._kx, (lat - self.lat0) * self._ky

    def to_lonlat(self, x, y):
        return np.asarray(x) / self._kx + self.lon0, np.asarray(y) / self._ky + self.lat0


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def bearing_deg(dx, dy):
    """Compass bearing (0 = north, 90 = east) of a vector in local x/y."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)


def point_segment_distance(px, py, ax, ay, bx, by):
    """Vectorised distance from points P to segments AB (broadcasting).

    Returns (distance_m, t) where t in [0,1] is the projection fraction along AB.
    """
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    seg_len2 = vx * vx + vy * vy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(seg_len2 > 0, (wx * vx + wy * vy) / seg_len2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * vx, ay + t * vy
    return np.hypot(px - cx, py - cy), t
