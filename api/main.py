"""PathSnap API (runs on Laptop 1). STARTER - owned by the system track.

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Env:
  PATHSNAP_TRIPS_DIR  folder holding one sub-folder per trip (default: evaluation trips
                      in data/eval_trips, falling back to Student folder/student_test for testing)

Endpoints:
  GET  /health   -> status + which model is loaded
  GET  /trips    -> [{trip_id, n_obs}]
  POST /predict  {"trip_id": "..."} -> GPS points, predicted route, metrics

The reference route is read only inside evaluate_trip() on this machine; the
response carries scores, never reference geometry.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import evaluate_trip, load_model, predict_trip, route_geometry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap")


def _trips_dir() -> Path:
    env = os.environ.get("PATHSNAP_TRIPS_DIR")
    if env:
        return Path(env)
    default = Path("data/eval_trips")
    return default if default.exists() else Path("Student folder/student_test")


TRIPS_DIR = _trips_dir()
MODEL, PARAMS, MODEL_NAME = load_model()
app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
log.info("Loaded model: %s | trips dir: %s", MODEL_NAME, TRIPS_DIR)


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "trips_dir": str(TRIPS_DIR)}


@app.get("/trips")
def trips():
    out = []
    for d in sorted(p for p in TRIPS_DIR.iterdir() if p.is_dir()):
        track = d / f"{d.name}.track"
        if track.exists():
            with open(track) as fh:
                out.append({"trip_id": d.name, "n_obs": sum(1 for _ in fh)})
    return out


@app.post("/predict")
def predict(req: PredictRequest):
    trip_dir = TRIPS_DIR / req.trip_id
    if not (trip_dir / f"{req.trip_id}.track").exists():
        raise HTTPException(404, f"unknown trip_id {req.trip_id}")
    log.info("PREDICT request trip_id=%s", req.trip_id)
    t0 = time.time()
    pred = predict_trip(trip_dir, MODEL, PARAMS)
    inference_ms = (time.time() - t0) * 1e3
    metrics = evaluate_trip(trip_dir, pred)
    metrics.update({"n_obs": len(pred["obs"]), "inference_ms": round(inference_ms, 1)})
    log.info("PREDICT done trip_id=%s n_obs=%d top1=%.4f continuity=%.4f inference_ms=%.0f",
             req.trip_id, metrics["n_obs"], metrics.get("top1_accuracy", float("nan")),
             metrics.get("route_continuity", float("nan")), inference_ms)
    return {"trip_id": req.trip_id, "model": MODEL_NAME,
            "gps": pred["obs"][["lat", "lon"]].round(7).values.tolist(),
            "predicted_route": route_geometry(pred),
            "metrics": metrics}
