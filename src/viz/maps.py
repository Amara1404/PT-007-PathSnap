"""Folium map helper shared by EDA, the report and the UI.

    m = trip_map(gps=[[lat, lon], ...],
                 predicted=[[[lat, lon], [lat, lon]], ...],     # segments
                 reference=[...], network=[...])
    m.save("results/maps/trip.html")

Each layer is its own FeatureGroup (toggle with the layer control); a fixed legend
names the colours.
"""
from __future__ import annotations

import folium

COLORS = {"gps": "#2563eb", "predicted": "#dc2626", "reference": "#16a34a", "network": "#9ca3af"}


def trip_map(gps=None, predicted=None, reference=None, network=None, zoom_start: int = 15) -> folium.Map:
    pts = list(gps or [])
    seg_pts = [p for s in (predicted or []) + (reference or []) for p in s]
    allp = pts + seg_pts
    center = [sum(p[0] for p in allp) / len(allp), sum(p[1] for p in allp) / len(allp)] if allp else [0, 0]
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap", control_scale=True)

    if network:
        g = folium.FeatureGroup("Road network", show=True)
        for s in network:
            folium.PolyLine(s, color=COLORS["network"], weight=1.5, opacity=0.6).add_to(g)
        g.add_to(m)
    if reference:
        g = folium.FeatureGroup("Reference route")
        for s in reference:
            folium.PolyLine(s, color=COLORS["reference"], weight=6, opacity=0.6).add_to(g)
        g.add_to(m)
    if predicted:
        g = folium.FeatureGroup("Predicted route")
        for s in predicted:
            folium.PolyLine(s, color=COLORS["predicted"], weight=3, opacity=0.9).add_to(g)
        g.add_to(m)
    if pts:
        g = folium.FeatureGroup("GPS observations")
        for p in pts:
            folium.CircleMarker(p, radius=2, color=COLORS["gps"], fill=True, fill_opacity=0.8, weight=0).add_to(g)
        g.add_to(m)

    if allp:
        m.fit_bounds([[min(p[0] for p in allp), min(p[1] for p in allp)],
                      [max(p[0] for p in allp), max(p[1] for p in allp)]])
    folium.LayerControl(collapsed=False).add_to(m)
    rows = "".join(f"<div><span style='display:inline-block;width:14px;height:4px;background:{c};"
                   f"margin-right:6px;vertical-align:middle'></span>{n}</div>"
                   for n, c, on in (("GPS observations", COLORS["gps"], pts), ("Predicted route", COLORS["predicted"], predicted),
                                    ("Reference route", COLORS["reference"], reference), ("Road network", COLORS["network"], network)) if on)
    m.get_root().html.add_child(folium.Element(
        f"<div style='position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 12px;"
        f"border:1px solid #ccc;border-radius:6px;font:13px sans-serif'>{rows}</div>"))
    return m
