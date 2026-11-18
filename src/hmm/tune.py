"""Fast HMM Parameter Tuning on Representative Training Trips using OOF Probabilities."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.hmm.data import get_trip_slice, load_partition_data
from src.hmm.evaluate import evaluate_trip
from src.hmm.viterbi import DEFAULT_PARAMS, HMMRouteDecoder


ARTIFACTS = Path("artifacts")
RESULTS = Path("results/hmm")


# 6 curated, meaningful configurations
CANDIDATE_CONFIGS = [
    {
        "name": "default",
        "beta": 10.0, "emission_weight": 1.0, "transition_weight": 1.0,
        "disconnected_penalty": 1e-5, "same_arc_factor": 2.0, "direct_turn_scale": 1.0,
        "gap_threshold_s": 5.0, "max_search_dist": 300.0, "max_speed_mps": 45.0
    },
    {
        "name": "high_continuity",
        "beta": 8.0, "emission_weight": 0.9, "transition_weight": 1.2,
        "disconnected_penalty": 1e-5, "same_arc_factor": 2.0, "direct_turn_scale": 1.0,
        "gap_threshold_s": 5.0, "max_search_dist": 300.0, "max_speed_mps": 45.0
    },
    {
        "name": "high_smoothing",
        "beta": 12.0, "emission_weight": 0.8, "transition_weight": 1.3,
        "disconnected_penalty": 1e-6, "same_arc_factor": 2.5, "direct_turn_scale": 1.1,
        "gap_threshold_s": 5.0, "max_search_dist": 300.0, "max_speed_mps": 45.0
    },
    {
        "name": "emission_priority",
        "beta": 10.0, "emission_weight": 1.2, "transition_weight": 0.9,
        "disconnected_penalty": 1e-5, "same_arc_factor": 2.0, "direct_turn_scale": 0.9,
        "gap_threshold_s": 5.0, "max_search_dist": 300.0, "max_speed_mps": 45.0
    },
    {
        "name": "tight_distance",
        "beta": 6.0, "emission_weight": 1.0, "transition_weight": 1.1,
        "disconnected_penalty": 1e-5, "same_arc_factor": 1.5, "direct_turn_scale": 1.0,
        "gap_threshold_s": 5.0, "max_search_dist": 250.0, "max_speed_mps": 45.0
    },
    {
        "name": "broad_distance",
        "beta": 14.0, "emission_weight": 1.0, "transition_weight": 1.0,
        "disconnected_penalty": 1e-4, "same_arc_factor": 2.0, "direct_turn_scale": 1.0,
        "gap_threshold_s": 6.0, "max_search_dist": 350.0, "max_speed_mps": 50.0
    },
]


def run_fast_tuning(trips_list: list[str] | None = None) -> dict:
    if trips_list is None:
        # 5 fast, diverse representative training trips
        trips_list = ["00000001", "00000003", "00000006", "00000014", "00000022"]

    print(f"Loading training OOF partition for {len(trips_list)} representative trips...")
    cands_all, obs_all, arcs_all = load_partition_data(partition="train", model_name="hist_gb")

    trip_cache = {}
    for tid in trips_list:
        trip_cache[tid] = get_trip_slice(tid, cands_all, obs_all, arcs_all)

    print(f"Running 6 candidate parameter configurations across trips: {trips_list}")
    trials = []
    best_score = -float("inf")
    best_config = dict(CANDIDATE_CONFIGS[0])

    for i, cfg in enumerate(CANDIDATE_CONFIGS):
        t0 = time.time()
        top1_accs = []
        conts = []
        precs = []
        recs = []
        times = []

        for tid in trips_list:
            c_t, o_t, a_t = trip_cache[tid]
            res = evaluate_trip(tid, c_t, o_t, a_t, params=cfg)
            top1_accs.append(res["hmm_top1_acc"])
            conts.append(res["hmm_continuity"])
            precs.append(res["hmm_precision"])
            recs.append(res["hmm_recall"])
            times.append(res["decode_ms"])

        m_top1 = float(np.mean(top1_accs))
        m_cont = float(np.mean(conts))
        m_prec = float(np.mean(precs))
        m_rec = float(np.mean(recs))
        m_time = float(np.mean(times))
        duration = time.time() - t0

        # Score balances Top-1 Accuracy and Route Continuity
        score = 0.5 * m_top1 + 0.5 * m_cont

        record = {
            "config_name": cfg.get("name", f"cfg_{i}"),
            "beta": cfg["beta"],
            "emission_weight": cfg["emission_weight"],
            "transition_weight": cfg["transition_weight"],
            "disconnected_penalty": cfg["disconnected_penalty"],
            "same_arc_factor": cfg["same_arc_factor"],
            "top1_acc": m_top1,
            "continuity": m_cont,
            "precision": m_prec,
            "recall": m_rec,
            "decode_ms": m_time,
            "score": score,
            "eval_duration_s": duration
        }
        trials.append(record)
        print(f"[{i+1}/6] {record['config_name']:<18}: Top1={m_top1*100:.2f}% Cont={m_cont*100:.2f}% "
              f"Prec={m_prec*100:.2f}% Rec={m_rec*100:.2f}% Score={score:.4f} ({duration:.2f}s)")

        if score > best_score:
            best_score = score
            best_config = dict(cfg)

    RESULTS.mkdir(parents=True, exist_ok=True)
    df_trials = pd.DataFrame(trials).sort_values("score", ascending=False)
    trials_csv = RESULTS / "tuning_trials.csv"
    df_trials.to_csv(trials_csv, index=False)
    print(f"\nSaved trials to {trials_csv}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    params_file = ARTIFACTS / "hmm_params.json"
    clean_params = {k: v for k, v in best_config.items() if k != "name"}
    params_file.write_text(json.dumps(clean_params, indent=2))
    print(f"Saved best tuned parameters ({best_config.get('name')}) to {params_file}")

    return clean_params


if __name__ == "__main__":
    run_fast_tuning()
