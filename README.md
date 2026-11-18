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

Trips are processed in parallel (`--workers`, default up to 8); `--workers 1` runs
sequentially and gives identical output.

## Model experiments

```bash
python -m src.eval.make_folds                    # data/cv_folds.csv (already committed)
python -m src.models.train_models                # all models; or --models lr,rf,hgb
python -m src.eval.plots                         # figures -> results/figures/
```

Every model is cross-validated on the 80 training trips with 5 trip-grouped folds
(`data/cv_folds.csv`) and then scored on the 10 student-test trips. Results:

| Output | Contents |
|---|---|
| `results/model_comparison.csv` | One row per model: CV and student-test metrics, timings |
| `results/cv_fold_scores.csv` | Per-fold validation and training scores |
| `results/preds/<model>_student_test.parquet` | Candidate probabilities on student-test trips |
| `results/history/<model>_history.csv` | Epoch-wise training history (MLP, HistGB only) |
| `artifacts/models/<model>.joblib` | Fitted model (final fit on all training trips) |

## Running the system

Laptop 1 (API):

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Set `PATHSNAP_TRIPS_DIR` to the folder of evaluation trips (one sub-folder per trip).
Without it, the API serves `data/eval_trips`, or `Student folder/student_test` for testing.
The API loads `artifacts/model.joblib` (nearest-arc stub if missing) and
`artifacts/hmm_params.json`.

Laptop 2 (UI), on the same network:

```bash
set PATHSNAP_API=http://<laptop-1-ip>:8000     # macOS/Linux: export PATHSNAP_API=...
streamlit run ui/app.py
```

Allow inbound TCP port 8000 through the firewall on Laptop 1.

## Repository layout

| Folder | Purpose |
|---|---|
| `src/data` | Raw-data loading, cleaning, labels, candidate generation, features |
| `src/models` | Experiment runner and model definitions |
| `src/eval` | Shared metrics, CV folds, comparison plots |
| `src/hmm` | Route decoding (`decode()`; stub until the HMM/Viterbi is implemented) |
| `src/viz` | Folium map helper used by EDA, report and UI |
| `src/pipeline.py` | One-trip inference: raw files to predicted route; server-side scoring |
| `api` | FastAPI backend |
| `ui` | Streamlit evaluator interface |
| `tests` | Synthetic-data generator for smoke-testing the pipeline |
