"""Phase-1 smoke tests: python tests/test_phase1.py  (or pytest). Needs `Student folder/train` or PATHSNAP_TRIPS_DIR."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.metrics import route_continuity  # noqa: E402
from src.pipeline import predict_trip           # noqa: E402

TRIPS = Path(os.environ.get("PATHSNAP_TRIPS_DIR", "Student folder/train"))


def test_predict_trip_contract(trip_id="00000010"):
    t0 = time.time()
    out = predict_trip(TRIPS / trip_id)
    it = out.pop("_internal")
    assert out["trip_id"] == trip_id and out["mode"] in ("stub", "real")
    assert len(out["gps"]) == out["n_obs"] == len(it["pred_arc"]) > 0
    assert all(len(p) == 2 for p in out["gps"][:50])
    assert out["predicted_route"] and len(out["predicted_route"][0]) == 2 and len(out["predicted_route"][0][0]) == 2
    assert (it["pred_arc"] >= 0).all()
    json.dumps(out)                                   # response must be JSON-serialisable
    assert not any(k.startswith(("ref", "truth")) for k in out)
    net = it["net"].arcs
    rc = route_continuity(it["pred_arc"], net.from_node.to_numpy(), net.to_node.to_numpy(), it["obs"].gap_before.to_numpy())
    print(f"{trip_id}: n_obs={out['n_obs']} segments={len(out['predicted_route'])} continuity={rc:.3f} "
          f"inference_ms={out['inference_ms']} wall={time.time()-t0:.1f}s mode={out['mode']}")
    return out


if __name__ == "__main__":
    for t in ("00000010", "00000033", "00000054"):
        test_predict_trip_contract(t)
    print("ok")
