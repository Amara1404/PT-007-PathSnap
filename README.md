# PathSnap: Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon (team PT-007). Matches noisy GPS tracks to a road
network: a classifier scores candidate road arcs per GPS point, then HMM/Viterbi decoding
reconstructs a connected route. Served by a FastAPI backend (Laptop 1) to a UI (Laptop 2).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Copy the hackathon `Student folder/` (containing `train/` and `student_test/`) into the repo
root. It is git-ignored and never committed.

## Build the processed datasets

```bash
python -m src.data.build_datasets --raw "Student folder" --out data/processed
```

Takes a few minutes. Check `data/processed/candidate_summary.csv`: 80 train and 10
student_test trips.

| Output | Contents |
|---|---|
| `inventory_trips.csv` | Per-trip EDA inventory: points, sampling, gaps, extent, label quality |
| `network_stats.csv` | Road-network size and connectivity per trip |
| `data_quality_issues.csv` | Every cleaning action and anomaly found |
| `candidates_<partition>.parquet` | One row per (GPS point, candidate arc) with features and label |
| `observations_<partition>.parquet` | Cleaned GPS points with point features and reference labels |
| `arcs.parquet` | Projected road geometry for all trips |

Train only on rows with `usable == 1` (trips 33 and 54 and far-off-route points are
flagged unreliable). Feature list: `feature_columns.json`.

## Repository layout

| Folder | Purpose |
|---|---|
| `src/data` | Raw-data loading, cleaning, labels, candidate generation, features |
| `tests` | Synthetic-data generator for smoke-testing the pipeline |

More folders (models, HMM, API, UI, report) are added as the project progresses.
