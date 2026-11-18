"""Runs complete evaluation on student_test and train partitions for:
- Classifier-Only Baseline
- Untuned HMM
- Tuned HMM

Outputs:
- results/hmm/summary.csv
- results/hmm/student_test_untuned_per_trip.csv
- results/hmm/student_test_tuned_per_trip.csv
- results/hmm/train_untuned_per_trip.csv
- results/hmm/train_tuned_per_trip.csv
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import pandas as pd

from src.hmm.evaluate import evaluate_partition
from src.hmm.viterbi import DEFAULT_PARAMS


ARTIFACTS = Path("artifacts")
RESULTS = Path("results/hmm")


def run_all_evaluations():
    RESULTS.mkdir(parents=True, exist_ok=True)
    tuned_params_file = ARTIFACTS / "hmm_params.json"
    if tuned_params_file.exists():
        tuned_params = json.loads(tuned_params_file.read_text())
    else:
        tuned_params = DEFAULT_PARAMS

    untuned_params = DEFAULT_PARAMS

    print("==================================================================")
    print("STEP 1: Evaluating STUDENT_TEST with UNTUNED HMM...")
    st_untuned_df, st_untuned_summary = evaluate_partition(
        partition="student_test", params=untuned_params, tag="untuned", out_dir=RESULTS
    )

    print("==================================================================")
    print("STEP 2: Evaluating STUDENT_TEST with TUNED HMM (Frozen Params)...")
    st_tuned_df, st_tuned_summary = evaluate_partition(
        partition="student_test", params=tuned_params, tag="tuned", out_dir=RESULTS
    )

    print("==================================================================")
    print("STEP 3: Evaluating TRAINING trips with UNTUNED HMM...")
    tr_untuned_df, tr_untuned_summary = evaluate_partition(
        partition="train", params=untuned_params, tag="untuned", out_dir=RESULTS
    )

    print("==================================================================")
    print("STEP 4: Evaluating TRAINING trips with TUNED HMM...")
    tr_tuned_df, tr_tuned_summary = evaluate_partition(
        partition="train", params=tuned_params, tag="tuned", out_dir=RESULTS
    )

    # Consolidated Summary Table
    summary_rows = [
        # Student test
        {
            "Partition": "student_test",
            "Pipeline Mode": "Classifier-Only Baseline",
            "Top-1 Accuracy": f"{st_tuned_summary['clf_top1_acc_mean']*100:.2f}%",
            "Route Continuity": f"{st_tuned_summary['clf_continuity_mean']*100:.2f}%",
            "Route Precision": f"{st_tuned_summary['clf_precision_mean']*100:.2f}%",
            "Route Recall": f"{st_tuned_summary['clf_recall_mean']*100:.2f}%",
            "Mean Latency (ms)": 0.0,
            "Max Latency (ms)": 0.0,
            "Improved Trips": 0,
            "Worsened Trips": 0,
            "Total Obs": st_tuned_summary["total_obs"]
        },
        {
            "Partition": "student_test",
            "Pipeline Mode": "Untuned HMM Viterbi",
            "Top-1 Accuracy": f"{st_untuned_summary['hmm_top1_acc_mean']*100:.2f}%",
            "Route Continuity": f"{st_untuned_summary['hmm_continuity_mean']*100:.2f}%",
            "Route Precision": f"{st_untuned_summary['hmm_precision_mean']*100:.2f}%",
            "Route Recall": f"{st_untuned_summary['hmm_recall_mean']*100:.2f}%",
            "Mean Latency (ms)": round(st_untuned_summary["mean_decode_ms"], 1),
            "Max Latency (ms)": round(st_untuned_summary["max_decode_ms"], 1),
            "Improved Trips": st_untuned_summary["improved_acc_trips"],
            "Worsened Trips": st_untuned_summary["worsened_acc_trips"],
            "Total Obs": st_untuned_summary["total_obs"]
        },
        {
            "Partition": "student_test",
            "Pipeline Mode": "Tuned HMM Viterbi",
            "Top-1 Accuracy": f"{st_tuned_summary['hmm_top1_acc_mean']*100:.2f}%",
            "Route Continuity": f"{st_tuned_summary['hmm_continuity_mean']*100:.2f}%",
            "Route Precision": f"{st_tuned_summary['hmm_precision_mean']*100:.2f}%",
            "Route Recall": f"{st_tuned_summary['hmm_recall_mean']*100:.2f}%",
            "Mean Latency (ms)": round(st_tuned_summary["mean_decode_ms"], 1),
            "Max Latency (ms)": round(st_tuned_summary["max_decode_ms"], 1),
            "Improved Trips": st_tuned_summary["improved_acc_trips"],
            "Worsened Trips": st_tuned_summary["worsened_acc_trips"],
            "Total Obs": st_tuned_summary["total_obs"]
        },
        # Train
        {
            "Partition": "train",
            "Pipeline Mode": "Classifier-Only Baseline",
            "Top-1 Accuracy": f"{tr_tuned_summary['clf_top1_acc_mean']*100:.2f}%",
            "Route Continuity": f"{tr_tuned_summary['clf_continuity_mean']*100:.2f}%",
            "Route Precision": f"{tr_tuned_summary['clf_precision_mean']*100:.2f}%",
            "Route Recall": f"{tr_tuned_summary['clf_recall_mean']*100:.2f}%",
            "Mean Latency (ms)": 0.0,
            "Max Latency (ms)": 0.0,
            "Improved Trips": 0,
            "Worsened Trips": 0,
            "Total Obs": tr_tuned_summary["total_obs"]
        },
        {
            "Partition": "train",
            "Pipeline Mode": "Untuned HMM Viterbi",
            "Top-1 Accuracy": f"{tr_untuned_summary['hmm_top1_acc_mean']*100:.2f}%",
            "Route Continuity": f"{tr_untuned_summary['hmm_continuity_mean']*100:.2f}%",
            "Route Precision": f"{tr_untuned_summary['hmm_precision_mean']*100:.2f}%",
            "Route Recall": f"{tr_untuned_summary['hmm_recall_mean']*100:.2f}%",
            "Mean Latency (ms)": round(tr_untuned_summary["mean_decode_ms"], 1),
            "Max Latency (ms)": round(tr_untuned_summary["max_decode_ms"], 1),
            "Improved Trips": tr_untuned_summary["improved_acc_trips"],
            "Worsened Trips": tr_untuned_summary["worsened_acc_trips"],
            "Total Obs": tr_untuned_summary["total_obs"]
        },
        {
            "Partition": "train",
            "Pipeline Mode": "Tuned HMM Viterbi",
            "Top-1 Accuracy": f"{tr_tuned_summary['hmm_top1_acc_mean']*100:.2f}%",
            "Route Continuity": f"{tr_tuned_summary['hmm_continuity_mean']*100:.2f}%",
            "Route Precision": f"{tr_tuned_summary['hmm_precision_mean']*100:.2f}%",
            "Route Recall": f"{tr_tuned_summary['hmm_recall_mean']*100:.2f}%",
            "Mean Latency (ms)": round(tr_tuned_summary["mean_decode_ms"], 1),
            "Max Latency (ms)": round(tr_tuned_summary["max_decode_ms"], 1),
            "Improved Trips": tr_tuned_summary["improved_acc_trips"],
            "Worsened Trips": tr_tuned_summary["worsened_acc_trips"],
            "Total Obs": tr_tuned_summary["total_obs"]
        },
    ]

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = RESULTS / "summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print("==================================================================")
    print("ALL EVALUATIONS COMPLETE! CONSOLIDATED RESULTS:")
    print(df_summary.to_string(index=False))
    print("==================================================================")


if __name__ == "__main__":
    run_all_evaluations()
