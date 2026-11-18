"""Road-network geometry, connectivity statistics and arc adjacency."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .geo import LocalProjection, bearing_deg


@dataclass
class RoadNetwork:
    arcs: pd.DataFrame        # per arc: from_node,to_node,ax,ay,bx,by,length_m,bearing,valid,is_two_way,...
    node_xy: np.ndarray       # (n_nodes, 2)
    out_arcs: dict            # node_id -> np.array of arc ids leaving the node
    in_arcs: dict             # node_id -> np.array of arc ids entering the node


def build_network(nodes: pd.DataFrame, arcs: pd.DataFrame, proj: LocalProjection) -> RoadNetwork:
    nx_, ny_ = proj.to_xy(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    node_xy = np.column_stack([nx_, ny_])
    a = arcs.copy()
    valid = (a.from_node >= 0) & (a.to_node >= 0)
    f = a.from_node.clip(lower=0).to_numpy()
    t = a.to_node.clip(lower=0).to_numpy()
    a["ax"], a["ay"] = node_xy[f, 0], node_xy[f, 1]
    a["bx"], a["by"] = node_xy[t, 0], node_xy[t, 1]
    a["length_m"] = np.hypot(a.bx - a.ax, a.by - a.ay)
    a["bearing"] = bearing_deg(a.bx - a.ax, a.by - a.ay)
    a["valid"] = valid & np.isfinite(a.length_m)

    pair = set(zip(a.from_node, a.to_node))
    a["is_two_way"] = [(v, u) in pair for u, v in zip(a.from_node, a.to_node)]

    out_deg = a[valid].groupby("from_node").size()
    in_deg = a[valid].groupby("to_node").size()
    a["to_out_deg"] = a.to_node.map(out_deg).fillna(0).astype(int)    # branching at arc end
    a["from_in_deg"] = a.from_node.map(in_deg).fillna(0).astype(int)

    out_arcs = {k: g.index.to_numpy() for k, g in a[valid].groupby("from_node")}
    in_arcs = {k: g.index.to_numpy() for k, g in a[valid].groupby("to_node")}
    return RoadNetwork(a, node_xy, out_arcs, in_arcs)


def network_stats(trip_id: str, net: RoadNetwork) -> dict:
    a = net.arcs[net.arcs.valid]
    n = len(net.node_xy)
    used = np.zeros(n, bool)
    used[a.from_node.to_numpy()] = True
    used[a.to_node.to_numpy()] = True
    g = coo_matrix((np.ones(len(a)), (a.from_node, a.to_node)), shape=(n, n)).tocsr()
    n_weak, lab_w = connected_components(g, directed=True, connection="weak")
    n_strong, lab_s = connected_components(g, directed=True, connection="strong")
    # ignore isolated nodes when sizing components
    largest_w = np.bincount(lab_w[used]).max() / max(used.sum(), 1) if used.any() else 0
    largest_s = np.bincount(lab_s[used]).max() / max(used.sum(), 1) if used.any() else 0
    return {
        "trip_id": trip_id,
        "n_nodes": n,
        "n_arcs": len(net.arcs),
        "n_invalid_arcs": int((~net.arcs.valid).sum()),
        "n_isolated_nodes": int((~used).sum()),
        "n_weak_components": int(len(np.unique(lab_w[used]))),
        "largest_weak_component_frac": round(float(largest_w), 4),
        "n_strong_components": int(len(np.unique(lab_s[used]))),
        "largest_strong_component_frac": round(float(largest_s), 4),
        "two_way_arc_frac": round(float(a.is_two_way.mean()), 4) if len(a) else np.nan,
        "zero_length_arcs": int((a.length_m < 0.01).sum()),
        "arc_len_median_m": round(float(a.length_m.median()), 2) if len(a) else np.nan,
        "arc_len_max_m": round(float(a.length_m.max()), 2) if len(a) else np.nan,
        "total_road_km": round(float(a.length_m.sum() / 1000), 3),
        "mean_out_degree": round(float(len(a) / max(used.sum(), 1)), 3),
    }


def arc_relation(net: RoadNetwork, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Relation of arc a (earlier) to arc b (later), vectorised.

    0 = unrelated, 1 = share a node in any direction, 2 = b directly follows a
    (a.to == b.from), 3 = same arc.
    """
    A = net.arcs
    fa, ta = A.from_node.to_numpy()[a], A.to_node.to_numpy()[a]
    fb, tb = A.from_node.to_numpy()[b], A.to_node.to_numpy()[b]
    rel = np.zeros(len(a), np.int8)
    touch = (fa == fb) | (fa == tb) | (ta == fb) | (ta == tb)
    rel[touch] = 1
    rel[ta == fb] = 2
    rel[a == b] = 3
    return rel
