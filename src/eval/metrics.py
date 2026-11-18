"""Shared evaluation metrics. Both the model experiments and the API import these,
so every number in the report is computed the same way.

Two levels:
  * candidate rows  (binary: is this candidate the reference arc?)  -> macro F1, precision,
    recall, log loss, confusion matrix
  * observations    (pick the highest-scoring candidate per GPS point) -> Top-1 accuracy,
    candidate miss rate, route continuity
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score


def row_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.clip(np.asarray(prob, float), 1e-7, 1 - 1e-7)
    pred = (prob >= threshold).astype(int)
    return {
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "precision_pos": precision_score(y_true, pred, zero_division=0),
        "recall_pos": recall_score(y_true, pred, zero_division=0),
        "log_loss": log_loss(y_true, prob, labels=[0, 1]),
    }


def row_confusion(y_true, prob, threshold: float = 0.5) -> np.ndarray:
    return confusion_matrix(np.asarray(y_true).astype(int), (np.asarray(prob) >= threshold).astype(int), labels=[0, 1])


def top1_choice(cands: pd.DataFrame, score_col: str = "prob") -> pd.DataFrame:
    """Highest-scoring candidate per (trip, observation). Ties -> nearest arc."""
    c = cands.sort_values(["trip_id", "obs_idx", score_col, "dist_m"], ascending=[True, True, False, True])
    return c.drop_duplicates(["trip_id", "obs_idx"])[["trip_id", "obs_idx", "arc_id"]]


def observation_top1(cands: pd.DataFrame, score_col: str = "prob") -> dict:
    """Top-1 accuracy over observations. An observation whose reference arc is not
    among its candidates counts as wrong (it cannot be matched) - reported separately
    as the candidate miss rate."""
    pick = top1_choice(cands, score_col)
    has_pos = cands.groupby(["trip_id", "obs_idx"]).label.max()
    chosen = cands.merge(pick, on=["trip_id", "obs_idx", "arc_id"])
    correct = chosen.set_index(["trip_id", "obs_idx"]).label.reindex(has_pos.index, fill_value=0)
    return {"top1_acc": float(correct.mean()),
            "candidate_miss_rate": float(1 - has_pos.mean()),
            "n_obs": int(len(has_pos))}


def route_continuity(pred: pd.DataFrame, arcs: pd.DataFrame) -> float:
    """Share of consecutive observations whose predicted arcs are the same arc or
    directly connected (end node of one = start node of the next).

    pred: trip_id, obs_idx, arc_id      arcs: trip_id, arc_id, from_node, to_node
    """
    p = pred.sort_values(["trip_id", "obs_idx"]).merge(
        arcs[["trip_id", "arc_id", "from_node", "to_node"]], on=["trip_id", "arc_id"], how="left")
    same_trip = p.trip_id.to_numpy()[1:] == p.trip_id.to_numpy()[:-1]
    a, b = p.iloc[:-1], p.iloc[1:]
    ok = (a.arc_id.to_numpy() == b.arc_id.to_numpy()) | (a.to_node.to_numpy() == b.from_node.to_numpy())
    return float(ok[same_trip].mean()) if same_trip.any() else float("nan")


def pointwise_accuracy(pred: pd.DataFrame, ref: pd.DataFrame) -> float:
    """Accuracy of a per-observation arc prediction (e.g. after HMM decoding)
    against reference arcs. ref: trip_id, obs_idx, ref_arc."""
    m = pred.merge(ref, on=["trip_id", "obs_idx"])
    return float((m.arc_id == m.ref_arc).mean())
