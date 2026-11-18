"""Automated Test Suite for HMM & Viterbi Route Reconstruction.

Tests all 16 required HMM properties:
1. Decoder output contract.
2. One candidate per observation.
3. Multiple candidates.
4. Emission scoring.
5. Transition scoring.
6. Directed arc connectivity.
7. Same-arc transitions.
8. Multi-arc paths.
9. Disconnected components.
10. GPS gaps.
11. NaN and zero probabilities.
12. Viterbi backtracking.
13. Long-sequence stability.
14. Deterministic results.
15. Ground-truth leakage prevention.
16. Runtime performance.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.hmm.viterbi import DEFAULT_PARAMS, HMMRouteDecoder, decode


@pytest.fixture
def simple_network():
    """A small linear directed network: (0)->(1)->(2)->(3) with parallel branch (1)->(4)->(3).

    Arc 0: 0 -> 1, len 100
    Arc 1: 1 -> 2, len 100
    Arc 2: 2 -> 3, len 100
    Arc 3: 1 -> 4, len 100
    Arc 4: 4 -> 3, len 100
    Arc 5: 10 -> 11, len 100 (isolated disconnected component)
    """
    arcs = pd.DataFrame({
        "arc_id": [0, 1, 2, 3, 4, 5],
        "from_node": [0, 1, 2, 1, 4, 10],
        "to_node": [1, 2, 3, 4, 3, 11],
        "length_m": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "ax": [0.0, 100.0, 200.0, 100.0, 200.0, 1000.0],
        "ay": [0.0, 0.0, 0.0, 50.0, 50.0, 1000.0],
        "bx": [100.0, 200.0, 300.0, 200.0, 300.0, 1100.0],
        "by": [0.0, 0.0, 0.0, 50.0, 0.0, 1000.0],
    })
    return arcs


def test_1_decoder_output_contract(simple_network):
    """1. Decoder output contract: returns DataFrame with obs_idx, arc_id."""
    cands = pd.DataFrame({
        "trip_id": ["test_trip", "test_trip"],
        "obs_idx": [0, 1],
        "arc_id": [0, 1],
        "prob": [0.9, 0.8],
        "dist_m": [2.0, 3.0]
    })
    obs = pd.DataFrame({
        "trip_id": ["test_trip", "test_trip"],
        "obs_idx": [0, 1],
        "x": [50.0, 150.0],
        "y": [0.0, 0.0],
        "t": [0.0, 1.0],
        "gap_before": [0, 0],
        "dt_prev": [0.0, 1.0],
        "step_m": [0.0, 100.0]
    })
    out = decode(cands, obs, simple_network)
    assert isinstance(out, pd.DataFrame)
    assert "obs_idx" in out.columns
    assert "arc_id" in out.columns
    assert len(out) == 2
    assert list(out["obs_idx"]) == [0, 1]
    assert list(out["arc_id"]) == [0, 1]


def test_2_one_candidate_per_observation(simple_network):
    """2. Exactly one candidate arc per observation, in order."""
    cands = pd.DataFrame({
        "obs_idx": [0, 0, 1, 1, 2, 2],
        "arc_id": [0, 3, 1, 3, 2, 4],
        "prob": [0.6, 0.4, 0.7, 0.3, 0.8, 0.2]
    })
    obs = pd.DataFrame({
        "obs_idx": [0, 1, 2],
        "x": [50.0, 150.0, 250.0],
        "y": [0.0, 0.0, 0.0],
        "t": [0.0, 1.0, 2.0]
    })
    out = decode(cands, obs, simple_network)
    assert len(out) == 3
    assert out["obs_idx"].tolist() == [0, 1, 2]
    assert not out["obs_idx"].duplicated().any()


def test_3_multiple_candidates(simple_network):
    """3. Correctly handles multiple candidate options per GPS observation."""
    cands = pd.DataFrame({
        "obs_idx": [0, 0, 0, 1, 1, 1],
        "arc_id": [0, 3, 5, 1, 4, 5],
        "prob": [0.8, 0.1, 0.1, 0.8, 0.1, 0.1],
        "dist_m": [5.0, 45.0, 900.0, 5.0, 45.0, 900.0]
    })
    obs = pd.DataFrame({
        "obs_idx": [0, 1],
        "x": [50.0, 150.0],
        "y": [0.0, 0.0],
        "t": [0.0, 1.0],
        "step_m": [0.0, 100.0]
    })
    out = decode(cands, obs, simple_network)
    assert out["arc_id"].tolist() == [0, 1]


def test_4_emission_scoring(simple_network):
    """4. Higher emission probability dominates when transition costs are equal."""
    # Observations equidistant, choosing between parallel arcs 1 and 3
    cands = pd.DataFrame({
        "obs_idx": [0, 0],
        "arc_id": [1, 3],
        "prob": [0.95, 0.05]
    })
    obs = pd.DataFrame({"obs_idx": [0], "x": [150.0], "y": [25.0], "t": [0.0]})
    out = decode(cands, obs, simple_network)
    assert out["arc_id"].iloc[0] == 1


def test_5_transition_scoring(simple_network):
    """5. Transition penalty favors route consistency over conflicting classifier noise."""
    # At t=0 we are on Arc 0. At t=1, candidate on Arc 1 connects smoothly (0->1->2),
    # while candidate on Arc 5 (isolated) has a spurious slight edge in raw classifier prob.
    cands = pd.DataFrame({
        "obs_idx": [0, 1, 1],
        "arc_id": [0, 1, 5],
        "prob": [0.9, 0.45, 0.55],
        "dist_m": [2.0, 5.0, 50.0]
    })
    obs = pd.DataFrame({
        "obs_idx": [0, 1],
        "x": [50.0, 150.0],
        "y": [0.0, 0.0],
        "t": [0.0, 1.0],
        "step_m": [0.0, 100.0]
    })
    out = decode(cands, obs, simple_network)
    # The connected Arc 1 must be chosen due to strong transition bonus over disconnected Arc 5
    assert out["arc_id"].tolist() == [0, 1]


def test_6_directed_arc_connectivity(simple_network):
    """6. Respects road direction; reverse traversal along one-way is penalized."""
    # From Arc 1 (1->2), moving to Arc 0 (0->1) is going backwards against the flow.
    cands = pd.DataFrame({
        "obs_idx": [0, 1, 1],
        "arc_id": [1, 2, 0],
        "prob": [0.9, 0.48, 0.52]
    })
    obs = pd.DataFrame({
        "obs_idx": [0, 1],
        "x": [150.0, 250.0],
        "y": [0.0, 0.0],
        "t": [0.0, 1.0],
        "step_m": [0.0, 100.0]
    })
    out = decode(cands, obs, simple_network)
    # Forward directed connection (1 -> 2) must win
    assert out["arc_id"].tolist() == [1, 2]


def test_7_same_arc_transitions(simple_network):
    """7. Low penalty for staying on the same arc consecutively."""
    cands = pd.DataFrame({
        "obs_idx": [0, 1, 2],
        "arc_id": [0, 0, 0],
        "prob": [0.9, 0.9, 0.9]
    })
    obs = pd.DataFrame({
        "obs_idx": [0, 1, 2],
        "x": [20.0, 50.0, 80.0],
        "y": [0.0, 0.0, 0.0],
        "t": [0.0, 1.0, 2.0],
        "step_m": [0.0, 30.0, 30.0]
    })
    out = decode(cands, obs, simple_network)
    assert out["arc_id"].tolist() == [0, 0, 0]


def test_8_multi_arc_paths(simple_network):
    """8. Correctly traverses multi-arc paths via shortest-path search."""
    # Arc 0 (0->1) to Arc 2 (2->3) skipping observation on Arc 1
    cands = pd.DataFrame({
        "obs_idx": [0, 1],
        "arc_id": [0, 2],
        "prob": [0.9, 0.9]
    })
    obs = pd.DataFrame({
        "obs_idx": [0, 1],
        "x": [50.0, 250.0],
        "y": [0.0, 0.0],
        "t": [0.0, 2.0],
        "step_m": [0.0, 200.0]
    })
    out = decode(cands, obs, simple_network)
    assert out["arc_id"].tolist() == [0, 2]


def test_9_disconnected_components(simple_network):
    """9. Disconnected components handled cleanly via documented restart strategy without crashing."""
    cands = pd.DataFrame({
        "obs_idx": [0, 1],
        "arc_id": [0, 5],   # 5 is in a separate disconnected component
        "prob": [0.9, 0.9]
    })
    obs = pd.DataFrame({
        "obs_idx": [0, 1],
        "x": [50.0, 1050.0],
        "y": [0.0, 1000.0],
        "t": [0.0, 10.0]
    })
    out = decode(cands, obs, simple_network)
    assert len(out) == 2
    assert out["arc_id"].tolist() == [0, 5]


def test_10_gps_gaps(simple_network):
    """10. GPS time gaps trigger restart/gap handling without breaking sequence."""
    cands = pd.DataFrame({
        "obs_idx": [0, 1],
        "arc_id": [0, 2],
        "prob": [0.8, 0.8]
    })
    obs = pd.DataFrame({
        "obs_idx": [0, 1],
        "x": [50.0, 250.0],
        "y": [0.0, 0.0],
        "t": [0.0, 30.0],          # 30 second gap
        "gap_before": [0, 1],
        "dt_prev": [0.0, 30.0]
    })
    out = decode(cands, obs, simple_network)
    assert len(out) == 2
    assert out["arc_id"].tolist() == [0, 2]


def test_11_nan_and_zero_probabilities(simple_network):
    """11. NaN, zero, negative, or infinite probabilities do not produce NaNs or errors."""
    cands = pd.DataFrame({
        "obs_idx": [0, 0, 1, 1],
        "arc_id": [0, 1, 1, 2],
        "prob": [np.nan, 0.0, -1.0, 1e-12]
    })
    obs = pd.DataFrame({
        "obs_idx": [0, 1],
        "x": [50.0, 150.0],
        "y": [0.0, 0.0],
        "t": [0.0, 1.0]
    })
    out = decode(cands, obs, simple_network)
    assert len(out) == 2
    assert not out["arc_id"].isna().any()


def test_12_viterbi_backtracking(simple_network):
    """12. Viterbi backtracking chooses global sequence over greedy local choice."""
    # Trellis:
    # t=0: Arc 0 (prob 0.9)
    # t=1: Arc 3 (Branch A, prob 0.6) vs Arc 1 (Branch B, prob 0.4)
    # t=2: Arc 2 (Branch B continuation, prob 0.99) vs Arc 5 (Disconnected, prob 0.1)
    # Arc 3 does not connect directly to Arc 2; Arc 1 does connect to Arc 2.
    cands = pd.DataFrame({
        "obs_idx": [0, 1, 1, 2],
        "arc_id": [0, 3, 1, 2],
        "prob": [0.9, 0.6, 0.4, 0.99]
    })
    obs = pd.DataFrame({
        "obs_idx": [0, 1, 2],
        "x": [50.0, 150.0, 250.0],
        "y": [0.0, 0.0, 0.0],
        "t": [0.0, 1.0, 2.0],
        "step_m": [0.0, 100.0, 100.0]
    })
    out = decode(cands, obs, simple_network)
    # Global Viterbi path must be 0 -> 1 -> 2, despite Arc 3 having higher local classifier prob at t=1
    assert out["arc_id"].tolist() == [0, 1, 2]


def test_13_long_sequence_stability(simple_network):
    """13. Numerical stability and memory boundedness over 5,000 observations."""
    n = 5000
    cands = pd.DataFrame({
        "obs_idx": np.arange(n),
        "arc_id": np.zeros(n, dtype=int),
        "prob": np.full(n, 0.85)
    })
    obs = pd.DataFrame({
        "obs_idx": np.arange(n),
        "x": np.linspace(0, 100, n),
        "y": np.zeros(n),
        "t": np.arange(n, dtype=float),
        "step_m": np.full(n, 100.0 / n)
    })
    out = decode(cands, obs, simple_network)
    assert len(out) == n
    assert not out["arc_id"].isna().any()


def test_14_deterministic_results(simple_network):
    """14. Repeated runs on the same input produce identical decoded routes."""
    cands = pd.DataFrame({
        "obs_idx": [0, 0, 1, 1],
        "arc_id": [0, 3, 1, 4],
        "prob": [0.55, 0.45, 0.6, 0.4]
    })
    obs = pd.DataFrame({
        "obs_idx": [0, 1],
        "x": [50.0, 150.0],
        "y": [0.0, 0.0],
        "t": [0.0, 1.0]
    })
    out1 = decode(cands, obs, simple_network)
    out2 = decode(cands, obs, simple_network)
    pd.testing.assert_frame_equal(out1, out2)


def test_15_ground_truth_leakage_prevention(simple_network):
    """15. Ground-truth labels in inputs are never accessed during decoding."""
    cands = pd.DataFrame({
        "obs_idx": [0, 1],
        "arc_id": [0, 1],
        "prob": [0.8, 0.8],
        "label": [1, 1],          # Ground-truth columns
        "ref_arc": [0, 1]
    })
    obs = pd.DataFrame({
        "obs_idx": [0, 1],
        "x": [50.0, 150.0],
        "y": [0.0, 0.0],
        "t": [0.0, 1.0],
        "ref_arc": [0, 1]
    })
    # Corrupt ground truth to completely misleading numbers
    cands_corrupted = cands.copy()
    cands_corrupted["label"] = 0
    cands_corrupted["ref_arc"] = 99999

    out_clean = decode(cands, obs, simple_network)
    out_corrupted = decode(cands_corrupted, obs, simple_network)
    # Output must be identical since decoder ignores label and ref_arc
    pd.testing.assert_frame_equal(out_clean, out_corrupted)


def test_16_runtime_performance(simple_network):
    """16. Fast runtime performance (< 500ms for 1,000 observations)."""
    n = 1000
    cands_list = []
    for i in range(n):
        cands_list.append({"obs_idx": i, "arc_id": 0 if i % 2 == 0 else 1, "prob": 0.8})
        cands_list.append({"obs_idx": i, "arc_id": 3 if i % 2 == 0 else 4, "prob": 0.2})
    cands = pd.DataFrame(cands_list)
    obs = pd.DataFrame({
        "obs_idx": np.arange(n),
        "x": np.linspace(0, 200, n),
        "y": np.zeros(n),
        "t": np.arange(n, dtype=float),
        "step_m": np.full(n, 200.0 / n)
    })
    t0 = time.time()
    out = decode(cands, obs, simple_network)
    duration_s = time.time() - t0
    assert len(out) == n
    assert duration_s < 0.50, f"Decoding took {duration_s:.3f}s (budget: 0.50s)"
