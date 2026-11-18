"""Hidden Markov Model (HMM) Route Reconstruction with Viterbi Decoding.

Contract:
    decode(cands, obs, arcs, params) -> DataFrame[obs_idx, arc_id] (and trip_id if present)

    cands  : candidate rows for ONE trip, with a `prob` column from the classifier.
    obs    : observation table for that trip (obs_idx, t, x, y, gap_before, dt_prev, step_m, ...)
    arcs   : that trip's arcs (arc_id, from_node, to_node, ax, ay, bx, by, length_m, ...)
    params : dict, loaded from artifacts/hmm_params.json (or default parameters)
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PARAMS: dict[str, Any] = {
    "beta": 10.0,                      # Scale parameter for transition distance mismatch |d_net - d_gps| / beta
    "emission_weight": 1.0,            # Multiplier for log emission probability
    "transition_weight": 1.0,          # Multiplier for log transition probability
    "disconnected_penalty": 1e-5,      # Probability penalty for disconnected transitions in directed graph
    "min_emission": 1e-6,              # Floor for candidate emission probabilities
    "gap_threshold_s": 5.0,            # Time gap threshold (seconds) triggering restart/gap transition
    "gap_disconnected_factor": 4.0,    # Attenuation factor for GPS displacement penalty during large gaps
    "max_search_dist": 300.0,          # Maximum network search distance in meters for shortest path
    "max_speed_mps": 45.0,             # Plausible speed threshold (~162 km/h)
    "speed_penalty_weight": 0.1,       # Weight for exceeding max speed threshold
    "direct_turn_scale": 1.0,          # Scaling for direct junction transitions (v_prev == u_curr)
    "same_arc_factor": 2.0,            # Divisor for same-arc transition penalty
}


class HMMRouteDecoder:
    """Hidden Markov Model for GPS Map-Matching and Route Reconstruction.

    Hidden states: candidate directed road arcs at each observation time t.
    Observations: noisy GPS sequence (x, y, t).
    Emissions: classifier confidence prob, locally normalized in log-space.
    Transitions: directed network reachability, shortest-path road distance vs GPS distance.
    Decoding: dynamic programming Viterbi algorithm with backtracking.
    """

    def __init__(self, params: dict | None = None):
        p = dict(DEFAULT_PARAMS)
        if params:
            p.update(params)
        self.params = p

        self.beta = float(p["beta"])
        self.emission_weight = float(p["emission_weight"])
        self.transition_weight = float(p["transition_weight"])
        self.disconnected_penalty = float(p["disconnected_penalty"])
        self.min_emission = float(p["min_emission"])
        self.gap_threshold_s = float(p["gap_threshold_s"])
        self.gap_disconnected_factor = float(p["gap_disconnected_factor"])
        self.max_search_dist = float(p["max_search_dist"])
        self.max_speed_mps = float(p["max_speed_mps"])
        self.speed_penalty_weight = float(p["speed_penalty_weight"])
        self.direct_turn_scale = float(p["direct_turn_scale"])
        self.same_arc_factor = float(p["same_arc_factor"])

        self.log_disconnected = math.log(max(self.disconnected_penalty, 1e-12))
        self.log_min_emission = math.log(max(self.min_emission, 1e-12))

    def _build_network_graph(self, arcs: pd.DataFrame):
        """Constructs adjacency list and node lookup tables from the trip's arcs."""
        adj = defaultdict(list)
        direct_edges = {}
        arc_info = {}

        for r in arcs.itertuples():
            arc_id = int(getattr(r, "arc_id", r.Index))
            u = int(r.from_node)
            v = int(r.to_node)
            length = float(getattr(r, "length_m", 10.0))
            if not math.isfinite(length) or length < 0:
                length = 1.0

            adj[u].append((v, length, arc_id))
            if (u, v) not in direct_edges or length < direct_edges[(u, v)]:
                direct_edges[(u, v)] = length

            arc_info[arc_id] = (u, v, length)

        return adj, direct_edges, arc_info

    def _shortest_path_dist(self, adj: dict, source: int, target: int, cutoff: float, cache: dict) -> float | None:
        """Finds directed shortest-path distance from source to target up to cutoff."""
        if source == target:
            return 0.0

        key = (source, target)
        if key in cache:
            return cache[key]

        dist = {source: 0.0}
        heap = [(0.0, source)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")):
                continue
            if u == target:
                cache[key] = d
                return d
            if d > cutoff:
                continue

            for v, w, _ in adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, float("inf")) and nd <= cutoff:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))

        cache[key] = None
        return None

    def decode(self, cands: pd.DataFrame, obs: pd.DataFrame | None = None,
               arcs: pd.DataFrame | None = None) -> pd.DataFrame:
        """Performs Viterbi route decoding.

        Returns DataFrame containing:
            obs_idx, arc_id (and trip_id if present)
        """
        # Determine trip_id
        trip_id = None
        if cands is not None and not cands.empty and "trip_id" in cands.columns:
            trip_id = cands["trip_id"].iloc[0]
        elif obs is not None and not obs.empty and "trip_id" in obs.columns:
            trip_id = obs["trip_id"].iloc[0]

        out_cols = ["obs_idx", "arc_id"]
        if trip_id is not None:
            out_cols = ["trip_id", "obs_idx", "arc_id"]

        # Handle empty input
        if cands is None or cands.empty:
            if obs is not None and not obs.empty:
                raise ValueError("No candidate road arcs provided for observations")
            return pd.DataFrame(columns=out_cols)

        # Build network graph
        if arcs is not None and not arcs.empty:
            adj, direct_edges, arc_info = self._build_network_graph(arcs)
        else:
            adj, direct_edges, arc_info = {}, {}, {}

        # Observation tracking
        obs_meta = {}
        if obs is not None and not obs.empty:
            for r in obs.itertuples():
                idx = int(r.obs_idx)
                x = float(getattr(r, "x", 0.0))
                y = float(getattr(r, "y", 0.0))
                t = float(getattr(r, "t", 0.0))
                gap = int(getattr(r, "gap_before", 0))
                dt = float(getattr(r, "dt_prev", 1.0))
                step = float(getattr(r, "step_m", 0.0))
                obs_meta[idx] = (x, y, t, gap, dt, step)

        # Group candidates by observation index
        # Deduplicate candidates per observation, taking highest prob / nearest distance
        clean_cands = cands.copy()
        if "prob" not in clean_cands.columns:
            clean_cands["prob"] = 1.0

        # Replace NaNs or infinities in prob
        clean_cands["prob"] = pd.to_numeric(clean_cands["prob"], errors="coerce").fillna(1e-4).clip(lower=0.0)

        # Sort so highest prob and lowest dist_m come first
        sort_cols = ["obs_idx", "prob"]
        ascending = [True, False]
        if "dist_m" in clean_cands.columns:
            sort_cols.append("dist_m")
            ascending.append(True)
        clean_cands.sort_values(sort_cols, ascending=ascending, inplace=True)
        clean_cands.drop_duplicates(subset=["obs_idx", "arc_id"], keep="first", inplace=True)

        # Determine total observation sequence
        if obs is not None and not obs.empty:
            all_obs_indices = sorted(obs["obs_idx"].unique())
        else:
            all_obs_indices = sorted(clean_cands["obs_idx"].unique())

        T = len(all_obs_indices)
        if T == 0:
            return pd.DataFrame(columns=out_cols)

        # Group clean candidates by obs_idx
        grouped = {k: v for k, v in clean_cands.groupby("obs_idx")}

        # Fallback candidate arc if an observation has no candidate
        fallback_arc = int(clean_cands["arc_id"].iloc[0])

        obs_cands_list = []
        obs_probs_list = []
        obs_cand_meta_list = []

        has_dist_to_end = "dist_to_end_m" in clean_cands.columns
        has_proj_t = "proj_t" in clean_cands.columns

        for idx in all_obs_indices:
            if idx in grouped:
                grp = grouped[idx]
                arc_ids = grp["arc_id"].to_numpy(dtype=np.int32)
                p = grp["prob"].to_numpy(dtype=np.float64)
                if has_dist_to_end:
                    d_end = grp["dist_to_end_m"].to_numpy(dtype=np.float64)
                else:
                    d_end = np.full(len(arc_ids), -1.0, dtype=np.float64)
                if has_proj_t:
                    p_t = grp["proj_t"].to_numpy(dtype=np.float64)
                else:
                    p_t = np.full(len(arc_ids), -1.0, dtype=np.float64)
            else:
                # Missing candidate fallback
                arc_ids = np.array([fallback_arc], dtype=np.int32)
                p = np.array([1.0], dtype=np.float64)
                d_end = np.array([-1.0], dtype=np.float64)
                p_t = np.array([-1.0], dtype=np.float64)

            # Local emission normalization
            p_sum = p.sum()
            if p_sum > 0 and math.isfinite(p_sum):
                p_norm = p / p_sum
            else:
                p_norm = np.full(len(p), 1.0 / max(1, len(p)), dtype=np.float64)

            log_emiss = np.log(np.maximum(p_norm, self.min_emission)) * self.emission_weight
            obs_cands_list.append(arc_ids)
            obs_probs_list.append(log_emiss)
            obs_cand_meta_list.append((d_end, p_t))

        # Handle T == 1 edge case
        if T == 1:
            best_k = int(np.argmax(obs_probs_list[0]))
            best_arc = int(obs_cands_list[0][best_k])
            res = pd.DataFrame({"obs_idx": all_obs_indices, "arc_id": [best_arc]})
            if trip_id is not None:
                res.insert(0, "trip_id", trip_id)
            return res

        # Viterbi tables
        # V[t] = 1D array of size K_t
        # backpointer[t] = 1D array of size K_t (index into t-1 candidates)
        V = [np.zeros(len(obs_cands_list[t]), dtype=np.float64) for t in range(T)]
        backpointer = [np.zeros(len(obs_cands_list[t]), dtype=np.int32) for t in range(T)]

        # Step t = 0: Initialize
        V[0] = obs_probs_list[0].copy()

        # Shortest path query cache for this trip
        sp_cache = {}

        # Recurrence step t = 1 to T-1
        for t in range(1, T):
            idx_curr = all_obs_indices[t]
            idx_prev = all_obs_indices[t - 1]

            # GPS displacement and elapsed time
            if idx_curr in obs_meta and idx_prev in obs_meta:
                x_c, y_c, t_c, gap_c, dt_c, step_c = obs_meta[idx_curr]
                x_p, y_p, t_p, _, _, _ = obs_meta[idx_prev]
                d_gps = step_c if math.isfinite(step_c) and step_c > 0 else math.hypot(x_c - x_p, y_c - y_p)
                dt = dt_c if math.isfinite(dt_c) and dt_c > 0 else max(1e-2, t_c - t_p)
                is_gap = bool(gap_c) or (dt > self.gap_threshold_s)
            else:
                d_gps = 10.0
                dt = 1.0
                is_gap = False

            cands_prev = obs_cands_list[t - 1]
            cands_curr = obs_cands_list[t]
            emiss_curr = obs_probs_list[t]
            meta_prev = obs_cand_meta_list[t - 1]
            meta_curr = obs_cand_meta_list[t]
            n_prev = len(cands_prev)
            n_curr = len(cands_curr)

            V_prev = V[t - 1]

            for k in range(n_curr):
                arc_curr = int(cands_curr[k])
                u_curr, v_curr, len_curr = arc_info.get(arc_curr, (-1, -1, 10.0))
                d_end_curr = meta_curr[0][k]
                pt_curr = meta_curr[1][k]
                d_start_curr = (len_curr * pt_curr) if pt_curr >= 0 else ((len_curr - d_end_curr) if d_end_curr >= 0 else (0.5 * len_curr))

                best_val = -float("inf")
                best_j = 0

                for j in range(n_prev):
                    arc_prev = int(cands_prev[j])
                    u_prev, v_prev, len_prev = arc_info.get(arc_prev, (-2, -2, 10.0))
                    d_end_prev = meta_prev[0][j]
                    pt_prev = meta_prev[1][j]
                    d_to_end_p = d_end_prev if d_end_prev >= 0 else ((1.0 - pt_prev) * len_prev if pt_prev >= 0 else (0.5 * len_prev))

                    # Transition probability calculation
                    if arc_prev == arc_curr:
                        # 1. Exact same directed arc
                        if pt_curr >= 0 and pt_prev >= 0:
                            delta_s = (pt_curr - pt_prev) * len_curr
                            if delta_s >= -2.0:
                                d_net = max(0.0, delta_s)
                                diff = abs(d_net - d_gps)
                                log_trans = - (diff / (self.beta * self.same_arc_factor))
                            else:
                                # Moving backwards against one-way flow
                                log_trans = - ((abs(delta_s) + d_gps + 20.0) / self.beta)
                        else:
                            log_trans = - (d_gps / (self.beta * self.same_arc_factor))
                    elif v_prev >= 0 and u_curr >= 0 and v_prev == u_curr:
                        # 2. Directly connected junction (end node of prev = start node of curr)
                        d_net = max(0.0, d_to_end_p + d_start_curr)
                        diff = abs(d_net - d_gps)
                        log_trans = - (diff / self.beta) * self.direct_turn_scale
                    elif v_prev >= 0 and u_curr >= 0:
                        # 3. Directed network shortest-path search
                        # Direct 1-hop edge fast check
                        if (v_prev, u_curr) in direct_edges:
                            path_len = direct_edges[(v_prev, u_curr)]
                        else:
                            path_len = self._shortest_path_dist(adj, v_prev, u_curr, self.max_search_dist, sp_cache)

                        if path_len is not None:
                            d_net = max(0.0, d_to_end_p + path_len + d_start_curr)
                            diff = abs(d_net - d_gps)
                            log_trans = - (diff / self.beta)
                            # Speed plausibility penalty
                            if dt > 0 and self.max_speed_mps > 0:
                                speed = d_net / dt
                                if speed > self.max_speed_mps:
                                    excess = (speed - self.max_speed_mps) / 10.0
                                    log_trans -= self.speed_penalty_weight * (excess ** 2)
                        else:
                            # Disconnected or large detour in directed road network
                            if is_gap:
                                log_trans = - (d_gps / (self.beta * self.gap_disconnected_factor))
                            else:
                                log_trans = self.log_disconnected
                    else:
                        # Unknown topology fallback
                        log_trans = self.log_disconnected

                    log_trans *= self.transition_weight
                    score = V_prev[j] + log_trans

                    if score > best_val:
                        best_val = score
                        best_j = j

                V[t][k] = best_val + emiss_curr[k]
                backpointer[t][k] = best_j

        # Backtrack optimal candidate sequence
        best_last_k = int(np.argmax(V[T - 1]))

        sequence_indices = [best_last_k]
        for t in range(T - 1, 0, -1):
            prev_k = backpointer[t][sequence_indices[-1]]
            sequence_indices.append(prev_k)

        sequence_indices.reverse()
        best_arcs = [int(obs_cands_list[t][k]) for t, k in enumerate(sequence_indices)]

        res_df = pd.DataFrame({
            "obs_idx": all_obs_indices,
            "arc_id": best_arcs
        })

        if trip_id is not None:
            res_df.insert(0, "trip_id", trip_id)

        return res_df


def decode(cands: pd.DataFrame, obs: pd.DataFrame, arcs: pd.DataFrame,
           params: dict | None = None) -> pd.DataFrame:
    """Route decoding interface for PathSnap.

    Args:
        cands  : candidate rows for ONE trip, with a `prob` column from the classifier.
        obs    : observation table for that trip (obs_idx, x, y, t, gap_before, dt_prev, ...)
        arcs   : that trip's arcs (arc_id, from_node, to_node, ax, ay, bx, by, length_m, ...)
        params : dict, loaded from artifacts/hmm_params.json

    Returns:
        DataFrame containing [obs_idx, arc_id] (and trip_id if available)
    """
    decoder = HMMRouteDecoder(params=params)
    return decoder.decode(cands, obs, arcs)
