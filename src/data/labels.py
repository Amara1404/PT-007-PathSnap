"""Per-observation reference labels from the .route file.

The .route file only lists the ordered arcs the vehicle traversed - it does NOT
say which GPS row was on which arc. We recover a per-observation label with a
monotone alignment (dynamic programming):

    choose k_0 <= k_1 <= ... <= k_{N-1}   (indices into the route)
    minimising  sum_i dist(p_i, route_arc[k_i])

Monotonicity preserves the travel order (so a road driven twice, or an out-and-
back on a two-way street, is labelled with the correct direction/pass), and
skipping is allowed because at 1 Hz short arcs can be crossed between samples.

The alignment distance of each point is kept as `label_dist_m`; large values
mark low-confidence labels (GPS far from the reference route, or reference
route limitations) - these should be reported, not silently trusted.
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment_distance


def align_route(px: np.ndarray, py: np.ndarray, route: np.ndarray, arcs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (ref_arc_id per point, route position per point, distance in m)."""
    n, k = len(px), len(route)
    if n == 0 or k == 0:
        return np.full(n, -1), np.full(n, -1), np.full(n, np.nan)

    ax = arcs.ax.to_numpy()[route]; ay = arcs.ay.to_numpy()[route]
    bx = arcs.bx.to_numpy()[route]; by = arcs.by.to_numpy()[route]

    back = np.empty((n, k), dtype=np.int32)
    dist_rows = np.empty((n, k), dtype=np.float32)
    prev = None
    for i in range(n):  # row-by-row keeps memory at O(N*K) int32 only
        d, _ = point_segment_distance(px[i], py[i], ax, ay, bx, by)
        dist_rows[i] = d
        if prev is None:
            cost = d.astype(np.float64)
            back[i] = np.arange(k)
        else:
            # prefix minimum of previous row: best j <= k
            pm_idx = np.zeros(k, dtype=np.int32)
            best = np.minimum.accumulate(prev)
            # index of the prefix minimum
            is_new = np.r_[True, prev[1:] < best[:-1]]
            pm_idx = np.maximum.accumulate(np.where(is_new, np.arange(k), 0)).astype(np.int32)
            cost = d + best
            back[i] = pm_idx
        prev = cost

    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    return route[pos], pos, dist_rows[np.arange(n), pos].astype(float)
