"""PathSnap evaluator UI (runs on Laptop 2). STARTER - owned by the system track.

    streamlit run ui/app.py
    # API address: set PATHSNAP_API (e.g. http://192.168.1.20:8000) or edit it in the sidebar

Lists the evaluation trips from the API, sends the chosen trip_id to /predict and
shows the GPS points, predicted route and the metrics the API computed.
The UI never sees reference routes - only the scores.
"""
import os
import sys
from pathlib import Path

import requests
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.viz.maps import trip_map  # noqa: E402

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap: route reconstruction")

api = st.sidebar.text_input("API address", os.environ.get("PATHSNAP_API", "http://localhost:8000")).rstrip("/")

try:
    health = requests.get(f"{api}/health", timeout=5).json()
    trips = requests.get(f"{api}/trips", timeout=10).json()
    st.sidebar.success(f"Connected - model: {health['model']}")
except Exception as e:  # noqa: BLE001
    st.sidebar.error(f"Cannot reach API at {api}: {e}")
    st.stop()

labels = {f"{t['trip_id']}  ({t['n_obs']} GPS points)": t["trip_id"] for t in trips}
choice = st.selectbox(f"Evaluation trip ({len(trips)} available)", list(labels))

if st.button("Reconstruct route", type="primary"):
    with st.spinner("Running prediction on the API..."):
        r = requests.post(f"{api}/predict", json={"trip_id": labels[choice]}, timeout=300)
    if r.ok:
        st.session_state["result"] = r.json()
    else:
        st.error(f"API error {r.status_code}: {r.text}")

res = st.session_state.get("result")
if res:
    m = res["metrics"]
    c = st.columns(5)
    c[0].metric("Top-1 accuracy", f"{m.get('top1_accuracy', float('nan')):.1%}")
    c[1].metric("Classifier-only Top-1", f"{m.get('classifier_only_top1', float('nan')):.1%}")
    c[2].metric("Route continuity", f"{m.get('route_continuity', float('nan')):.1%}")
    c[3].metric("GPS points", m.get("n_obs"))
    c[4].metric("Inference", f"{m.get('inference_ms', 0) / 1000:.2f} s")
    st_folium(trip_map(gps=res["gps"], predicted=res["predicted_route"]), height=600, use_container_width=True)
    st.caption(f"Trip {res['trip_id']} | model: {res['model']}")
