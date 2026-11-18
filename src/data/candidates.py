"""Observation features, candidate generation and candidate features.

Problem framing: for every GPS observation i we generate a small set of nearby
road arcs (candidates). Each (observation, candidate) pair becomes one row; the
classifier predicts P(candidate is the reference arc | features). The per-row
probabilities are later used as HMM emission scores, and arc adjacency gives the
transition model - so every row keeps (trip_id, obs_idx, arc_id).

No feature uses the reference route, so the same code runs on evaluator trips.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import angle_diff_deg, bearing_deg, point_segment_distance
from .network import RoadNetwork, arc_relation


def observation_features(x: np.ndarray, y: np.ndarray, t: np.ndarray, gap_s: float, half_win: int = 2) -> pd.DataFrame:
    n = len(x)
    dt_prev = np.r_[np.nan, np.diff(t)]
    dt_next = np.r_[np.diff(t), np.nan]
    step = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y))]
    speed = step / dt_prev
    # heading from a centred window (robust to 1 Hz jitter)
    i0 = np.clip(np.arange(n) - half_win, 0, n - 1)
    i1 = np.clip(np.arange(n) + half_win, 0, n - 1)
    hdx, hdy = x[i1] - x[i0], y[i1] - y[i0]
    disp = np.hypot(hdx, hdy)
    obs = pd.DataFrame({
        "dt_prev": dt_prev, "dt_next": dt_next,
        "step_m": step, "speed_mps": speed,
        "speed_smooth_mps": pd.Series(speed).rolling(5, center=True, min_periods=1).median().to_numpy(),
        "gps_heading": bearing_deg(hdx, hdy),
        "heading_valid": (disp > 3.0).astype(np.int8),   # stationary -> heading is noise
        "gap_before": (dt_prev > gap_s).astype(np.int8),
        "gap_after": (dt_next > gap_s).astype(np.int8),
    })
    # turn rate: change of heading between consecutive points
    hd = obs.gps_heading.to_numpy()
    obs["turn_deg"] = np.r_[np.nan, angle_diff_deg(hd[1:], hd[:-1])]
    obs["implausible_speed"] = (obs.speed_mps > 70).astype(np.int8)   # > 250 km/h
    return obs


def _arc_samples(arcs: pd.DataFrame, step_m: float):
    """Densify arcs into sample points so a KD-tree can find long arcs too."""
    v = arcs[arcs.valid]
    n_s = np.maximum(2, np.ceil(v.length_m.to_numpy() / step_m).astype(int) + 1)
    ids = np.repeat(v.index.to_numpy(), n_s)
    frac = np.concatenate([np.linspace(0, 1, k) for k in n_s])
    ax = np.repeat(v.ax.to_numpy(), n_s); ay = np.repeat(v.ay.to_numpy(), n_s)
    bx = np.repeat(v.bx.to_numpy(), n_s); by = np.repeat(v.by.to_numpy(), n_s)
    return np.column_stack([ax + frac * (bx - ax), ay + frac * (by - ay)]), ids


def generate_candidates(x, y, net: RoadNetwork, radius_m: float, max_cands: int,
                        sample_step_m: float = 10.0, fallback_k: int = 3) -> pd.DataFrame:
    A = net.arcs
    pts, sample_arc = _arc_samples(A, sample_step_m)
    tree = cKDTree(pts)
    P = np.column_stack([x, y])

    hits = tree.query_ball_point(P, r=radius_m + sample_step_m / 2)
    obs_idx = np.repeat(np.arange(len(P)), [len(h) for h in hits])
    arc_idx = sample_arc[np.concatenate(hits).astype(int)] if len(obs_idx) else np.array([], int)
    pairs = np.unique(np.column_stack([obs_idx, arc_idx]), axis=0) if len(obs_idx) else np.empty((0, 2), int)

    def exact(pairs_):
        o, a = pairs_[:, 0], pairs_[:, 1]
        d, tt = point_segment_distance(x[o], y[o], A.ax.to_numpy()[a], A.ay.to_numpy()[a],
                                       A.bx.to_numpy()[a], A.by.to_numpy()[a])
        return pd.DataFrame({"obs_idx": o, "arc_id": a, "dist_m": d, "proj_t": tt})

    c = exact(pairs)
    c = c[c.dist_m <= radius_m]
    c["cand_fallback"] = np.int8(0)

    # Observations with no candidate inside the radius: take the nearest arcs anyway
    missing = np.setdiff1d(np.arange(len(P)), c.obs_idx.unique())
    if len(missing):
        _, nn = tree.query(P[missing], k=min(60, len(pts)))
        nn = np.atleast_2d(nn)
        fb = np.unique(np.column_stack([np.repeat(missing, nn.shape[1]), sample_arc[nn.ravel()]]), axis=0)
        f = exact(fb).sort_values(["obs_idx", "dist_m"]).groupby("obs_idx").head(fallback_k)
        f["cand_fallback"] = np.int8(1)
        c = pd.concat([c, f], ignore_index=True)

    c = c.sort_values(["obs_idx", "dist_m"], kind="stable")
    c["dist_rank"] = c.groupby("obs_idx").cumcount()
    c = c[c.dist_rank < max_cands].reset_index(drop=True)
    return c


def candidate_features(c: pd.DataFrame, obs: pd.DataFrame, x, y, net: RoadNetwork, half_win: int = 2) -> pd.DataFrame:
    A = net.arcs
    a = c.arc_id.to_numpy()
    o = c.obs_idx.to_numpy()
    g = c.groupby("obs_idx")

    c["n_cands"] = g.arc_id.transform("size").astype(np.int16)
    dmin = g.dist_m.transform("min")
    c["dist_minus_min"] = c.dist_m - dmin
    c["dist_ratio"] = (c.dist_m + 1.0) / (dmin + 1.0)
    c["arc_len_m"] = A.length_m.to_numpy()[a]
    c["dist_to_end_m"] = np.minimum(c.proj_t, 1 - c.proj_t) * c.arc_len_m
    c["at_arc_end"] = (c.dist_to_end_m < 3.0).astype(np.int8)  # projection clamped to a node

    # direction agreement between GPS motion and directed arc
    arc_b = A.bearing.to_numpy()[a]
    gh = obs.gps_heading.to_numpy()[o]
    hv = obs.heading_valid.to_numpy()[o].astype(bool)
    diff = angle_diff_deg(gh, arc_b)
    c["heading_diff"] = np.where(hv, diff, 90.0)            # neutral value when not moving
    c["cos_heading_diff"] = np.where(hv, np.cos(np.deg2rad(diff)), 0.0)
    c["heading_valid"] = hv.astype(np.int8)

    c["is_two_way"] = A.is_two_way.to_numpy()[a].astype(np.int8)
    c["to_out_deg"] = A.to_out_deg.to_numpy()[a]
    c["from_in_deg"] = A.from_in_deg.to_numpy()[a]

    # window distance: how well this arc fits the neighbouring observations too
    n = len(x)
    ax, ay, bx, by = (A[k].to_numpy()[a] for k in ("ax", "ay", "bx", "by"))
    acc = np.zeros(len(c)); cnt = np.zeros(len(c))
    for off in range(-half_win, half_win + 1):
        j = o + off
        ok = (j >= 0) & (j < n)
        jj = np.clip(j, 0, n - 1)
        d, _ = point_segment_distance(x[jj], y[jj], ax, ay, bx, by)
        acc += np.where(ok, d, 0); cnt += ok
    c["win_dist_mean"] = acc / cnt
    c["win_dist_minus_min"] = c.win_dist_mean - c.groupby("obs_idx").win_dist_mean.transform("min")

    # topology w.r.t. the *nearest* candidate of the neighbouring observations
    # (uses only geometry, no labels) - a cheap local continuity signal
    nearest = c[c.dist_rank == 0].set_index("obs_idx").arc_id
    for name, off in (("prev", -1), ("next", 1)):
        nb = pd.Series(o + off).map(nearest).to_numpy()
        has = ~np.isnan(nb)
        rel = np.zeros(len(c), np.int8)
        nb_i = np.where(has, nb, 0).astype(int)
        rel_all = arc_relation(net, nb_i, a) if name == "prev" else arc_relation(net, a, nb_i)
        rel[has] = rel_all[has]
        c[f"{name}_same_arc"] = (rel == 3).astype(np.int8)
        c[f"{name}_direct_link"] = (rel == 2).astype(np.int8)
        c[f"{name}_touch"] = (rel == 1).astype(np.int8)

    # observation-level context copied onto each candidate row
    # first/last observation have no neighbour -> neutral fills (1 s step, no turn, not moving)
    fills = {"speed_smooth_mps": 0.0, "dt_prev": 1.0, "gap_before": 0, "gap_after": 0, "turn_deg": 0.0}
    for col, fill in fills.items():
        c[col] = obs[col].fillna(fill).to_numpy()[o]
    return c


FEATURE_COLUMNS = [
    "dist_m", "dist_rank", "dist_minus_min", "dist_ratio", "proj_t", "dist_to_end_m", "at_arc_end",
    "arc_len_m", "heading_diff", "cos_heading_diff", "heading_valid", "is_two_way", "to_out_deg",
    "from_in_deg", "win_dist_mean", "win_dist_minus_min", "prev_same_arc", "prev_direct_link",
    "prev_touch", "next_same_arc", "next_direct_link", "next_touch", "n_cands", "cand_fallback",
    "speed_smooth_mps", "dt_prev", "gap_before", "gap_after", "turn_deg",
]
