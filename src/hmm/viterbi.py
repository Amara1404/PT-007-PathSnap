"""Route decoding. STUB - to be replaced by the HMM/Viterbi decoder (Phase 2, ML track).

Contract (keep this signature when implementing the real decoder):

    decode(cands, obs, arcs, params) -> DataFrame[trip_id, obs_idx, arc_id]

    cands  : candidate rows for ONE trip, with a `prob` column from the classifier
    obs    : observation table for that trip (t, x, y, gap flags, ...)
    arcs   : that trip's arcs (arc_id, from_node, to_node, ax, ay, bx, by, length_m, ...)
    params : dict, loaded from artifacts/hmm_params.json

The stub returns the classifier's top-1 candidate per observation (no sequence model).
"""
from __future__ import annotations

import pandas as pd

from src.eval.metrics import top1_choice


def decode(cands: pd.DataFrame, obs: pd.DataFrame, arcs: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return top1_choice(cands, "prob").sort_values("obs_idx").reset_index(drop=True)
