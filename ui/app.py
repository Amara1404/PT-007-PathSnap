"""PathSnap Evaluator UI  — Laptop 2
Run:
    streamlit run ui/app.py

Set the API address in the sidebar (default http://localhost:8000) or via the
PATHSNAP_API environment variable.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.viz.maps import trip_map  # noqa: E402

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PathSnap",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* tighten metric cards */
    [data-testid="metric-container"] {
        background: var(--background-color);
        border: 1px solid rgba(100,100,100,.2);
        border-radius: 10px;
        padding: 10px 14px 8px 14px;
    }
    [data-testid="stMetricLabel"] { font-size: .75rem; font-weight: 600; }

    /* header strip */
    .ps-header {
        display: flex; align-items: center; gap: 12px;
        padding: 10px 0 6px 0; margin-bottom: 10px;
        border-bottom: 2px solid #e5e7eb;
    }
    .ps-header h1 { margin: 0; font-size: 1.6rem; }
    .ps-header .ps-badge {
        font-size: .7rem; font-weight: 700; letter-spacing: .06em;
        background: #2563eb; color: #fff;
        border-radius: 4px; padding: 2px 8px; margin-left: 6px;
    }
    .ps-section { margin-top: 1.2rem; }

    /* prediction log box */
    .log-box {
        font-family: monospace; font-size: .78rem;
        background: #0f172a; color: #94a3b8;
        border-radius: 8px; padding: 12px 14px;
        max-height: 220px; overflow-y: auto;
        line-height: 1.5;
    }
    .log-box .ok   { color: #4ade80; }
    .log-box .warn { color: #fbbf24; }
    .log-box .err  { color: #f87171; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="ps-header">'
    '<h1>🗺️ PathSnap</h1>'
    '<span class="ps-badge">PT-007</span>'
    '</div>',
    unsafe_allow_html=True,
)
st.caption("Distributed map-matching & route reconstruction · ML Hackathon UE24CS352A")

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input(
        "API server (Laptop 1)",
        value=os.environ.get("PATHSNAP_API", "http://localhost:8000").rstrip("/"),
        help="URL of the FastAPI server running on Laptop 1",
    )

    st.divider()
    st.markdown("**Layer visibility**")
    show_gps     = st.checkbox("GPS observations",  value=True)
    show_pred    = st.checkbox("Predicted route",   value=True)
    show_network = st.checkbox("Road network",      value=False)

    st.divider()
    st.markdown(
        "<small style='color:grey'>"
        "Ground truth is <b>never</b> sent to this UI.<br>"
        "Scores are computed server-side.</small>",
        unsafe_allow_html=True,
    )

# ── API health check ───────────────────────────────────────────────────────
@st.cache_data(ttl=15, show_spinner=False)
def _health(url: str) -> dict | None:
    try:
        return requests.get(f"{url}/health", timeout=5).json()
    except Exception:
        return None

@st.cache_data(ttl=15, show_spinner=False)
def _trips(url: str) -> list | None:
    try:
        return requests.get(f"{url}/trips", timeout=10).json()
    except Exception:
        return None

health = _health(api_url)
trips  = _trips(api_url)

if health is None or trips is None:
    st.error(
        f"**Cannot reach API** at `{api_url}`  —  "
        "make sure the FastAPI server is running on Laptop 1 "
        "(`uvicorn api.main:app --host 0.0.0.0 --port 8000`) "
        "and that your firewall allows port 8000."
    )
    st.stop()

with st.sidebar:
    st.success(f"✅ Connected  |  model: **{health.get('model','?')}**")

# ── Trip selector ──────────────────────────────────────────────────────────
trip_labels = {
    f"{t['trip_id']}  —  {t['n_obs']} GPS pts": t["trip_id"]
    for t in trips
}

col_sel, col_btn = st.columns([4, 1])
with col_sel:
    choice_label = st.selectbox(
        f"Evaluation trip  ({len(trips)} available)",
        list(trip_labels),
        label_visibility="visible",
    )
with col_btn:
    st.markdown("<div style='margin-top:28px'>", unsafe_allow_html=True)
    run = st.button("▶ Run", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Run prediction ─────────────────────────────────────────────────────────
if run:
    trip_id = trip_labels[choice_label]
    log_lines: list[str] = []

    def _log(msg: str, kind: str = ""):
        log_lines.append(f'<span class="{kind}">{msg}</span>')

    with st.spinner(f"Running map-matching for **{trip_id}** …"):
        t0 = time.time()
        try:
            _log(f"→ POST /predict  trip_id={trip_id}")
            resp = requests.post(
                f"{api_url}/predict",
                json={"trip_id": trip_id},
                timeout=300,
            )
            elapsed = time.time() - t0
            if resp.ok:
                data = resp.json()
                _log(f"← 200 OK  ({elapsed:.2f} s)", "ok")
                m = data.get("metrics", {})
                _log(f"   top-1 accuracy    : {m.get('top1_accuracy', float('nan')):.4f}", "ok")
                _log(f"   clf-only top-1    : {m.get('classifier_only_top1', float('nan')):.4f}")
                _log(f"   route continuity  : {m.get('route_continuity', float('nan')):.4f}", "ok")
                _log(f"   obs scored        : {m.get('n_obs_scored', '?')}")
                _log(f"   inference         : {m.get('inference_ms', 0)/1000:.2f} s")
                st.session_state["result"] = data
                st.session_state["log"]    = log_lines
            else:
                _log(f"← {resp.status_code}  {resp.text[:300]}", "err")
                st.session_state["log"] = log_lines
                st.error(f"API error {resp.status_code}: {resp.text}")
        except Exception as exc:
            _log(f"× request failed: {exc}", "err")
            st.session_state["log"] = log_lines
            st.error(str(exc))

# ── Results ────────────────────────────────────────────────────────────────
res  = st.session_state.get("result")
logs = st.session_state.get("log", [])

if logs:
    with st.expander("📋 Execution log", expanded=False):
        st.markdown(
            '<div class="log-box">' + "<br>".join(logs) + "</div>",
            unsafe_allow_html=True,
        )

if res:
    m = res.get("metrics", {})
    st.markdown('<div class="ps-section"></div>', unsafe_allow_html=True)

    # ── Metric cards ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        v = m.get("top1_accuracy", None)
        st.metric("Top-1 Accuracy (HMM)", f"{v:.1%}" if v is not None else "—",
                  help="Fraction of GPS points matched to the correct arc by HMM/Viterbi")
    with c2:
        v = m.get("classifier_only_top1", None)
        st.metric("Top-1 (Clf only)", f"{v:.1%}" if v is not None else "—",
                  help="Classifier greedy pick, no smoothing")
    with c3:
        v = m.get("route_continuity", None)
        st.metric("Route Continuity", f"{v:.1%}" if v is not None else "—",
                  help="Fraction of consecutive arc pairs that share a node")
    with c4:
        st.metric("GPS Points", m.get("n_obs", res.get("gps") and len(res["gps"])) or "—")
    with c5:
        v = m.get("inference_ms", None)
        st.metric("Inference time", f"{v/1000:.2f} s" if v is not None else "—")

    st.divider()

    # ── Delta vs classifier-only ───────────────────────────────────────────
    hmm_acc = m.get("top1_accuracy")
    clf_acc = m.get("classifier_only_top1")
    if hmm_acc is not None and clf_acc is not None:
        delta = hmm_acc - clf_acc
        sign  = "+" if delta >= 0 else ""
        color = "#16a34a" if delta >= 0 else "#dc2626"
        st.markdown(
            f"<p style='font-size:.85rem;color:{color};margin-top:-8px'>"
            f"HMM/Viterbi vs Classifier-only: "
            f"<b>{sign}{delta:.2%}</b> in top-1 accuracy</p>",
            unsafe_allow_html=True,
        )

    # ── Map ────────────────────────────────────────────────────────────────
    gps_data  = res["gps"]       if show_gps  else None
    pred_data = res.get("predicted_route") if show_pred else None

    with st.spinner("Rendering map …"):
        fmap = trip_map(
            gps       = gps_data,
            predicted = pred_data,
        )

    st.subheader("🗺️ Predicted Route")
    st_folium(fmap, height=600, use_container_width=True)
    st.caption(
        f"Trip **{res['trip_id']}**  ·  model: `{res.get('model','?')}`  "
        f"·  {len(gps_data) if gps_data else '?'} GPS pts  "
        f"·  {len(pred_data) if pred_data else 0} predicted arc segments"
    )

    # ── Raw JSON viewer ────────────────────────────────────────────────────
    with st.expander("🔍 Raw API response (excluding geometry)", expanded=False):
        st.json({k: v for k, v in res.items() if k not in ("gps", "predicted_route")})

else:
    # ── Idle state ─────────────────────────────────────────────────────────
    st.info(
        "Select a trip above and click **▶ Run** to start map-matching.",
        icon="ℹ️",
    )
    with st.expander("How it works", expanded=True):
        st.markdown(
            """
**Pipeline (runs on Laptop 1)**
1. Load GPS track (`.track`) and road network (`.nodes`, `.arcs`)
2. Project to UTM — metric distances, no degree arithmetic
3. Find candidate road arcs within radius R for every GPS ping
4. Extract spatial + kinematic features per (ping, arc) pair
5. Classifier scores each candidate arc
6. HMM / Viterbi smooths the sequence into a connected route
7. Score against ground-truth `.route` (server-side only)

**This UI (Laptop 2)**
- Sends `POST /predict { trip_id }` to Laptop 1
- Receives predicted arc IDs, GPS coordinates, and metrics
- Renders an interactive Folium map with layer controls

| Layer | Colour |
|---|---|
| GPS observations | 🔵 blue |
| Predicted route  | 🔴 red |
| Reference route  | 🟢 green (EDA only) |
            """
        )
