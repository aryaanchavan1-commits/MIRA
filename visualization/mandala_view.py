"""Mandala visualization (spec §26).

Plotly polar scatter: angle = sector slice, radius = ring/radial distance.
Shows center, rings, sector labels, edges, importance-sized markers,
memory-type colors, and an optional active retrieval path highlight.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import plotly.graph_objects as go

from core.memory import MemoryFrame

# stable, readable palette for memory types
TYPE_COLORS = {
    "semantic": "#6366f1", "episodic": "#f59e0b", "procedural": "#10b981",
    "working": "#ef4444", "fact": "#3b82f6", "entity": "#a855f7",
    "event": "#f97316", "document": "#64748b", "concept": "#14b8a6",
    "relation": "#d946ef",
}
PATH_COLOR = "#dc2626"


def _angles(frame: MemoryFrame) -> Dict[str, float]:
    """Deterministic angle per node: sector slice, index-spread within."""
    sectors: Dict[str, List[str]] = {}
    for nid, n in frame.nodes.items():
        sectors.setdefault(n.sector or "unassigned", []).append(nid)
    out: Dict[str, float] = {}
    sector_names = sorted(sectors)
    n_sectors = max(1, len(sector_names))
    slice_w = 360.0 / n_sectors
    for si, sname in enumerate(sector_names):
        ids = sectors[sname]
        base = si * slice_w
        step = slice_w / max(1, len(ids))
        for j, nid in enumerate(sorted(ids)):
            out[nid] = base + step * (j + 0.5)
    return out


def figure(frame: MemoryFrame, highlight_path: Optional[List[str]] = None,
           selected_id: Optional[str] = None,
           height: int = 620) -> go.Figure:
    path_ids: Set[str] = set(highlight_path or [])
    angles = _angles(frame)
    max_ring = max([n.ring for n in frame.nodes.values() if n.ring is not None],
                   default=4) or 4

    fig = go.Figure()
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
        polar=dict(radialaxis=dict(visible=True, showticklabels=False,
                                   range=[-0.15, max_ring + 0.6]),
                   angularaxis=dict(visible=False)),
    )

    # ring guide circles
    for r in range(max_ring + 1):
        fig.add_trace(go.Scatterpolar(
            r=[r] * 181, theta=list(range(181)),
            mode="lines", line=dict(width=1, color="#8884"),
            hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatterpolar(
        r=[0], theta=[0], mode="markers+text",
        marker=dict(size=16, color="#111", symbol="diamond"),
        text=["CORE"], textposition="top center",
        name="core", hoverinfo="text",
    ))

    # edges under the nodes
    edge_r, edge_theta = [], []
    path_r, path_theta = [], []
    for e in frame.edges:
        a, b = angles.get(e.source_id), angles.get(e.target_id)
        ra = frame.nodes[e.source_id].ring or 0 if e.source_id in frame.nodes else 0
        rb = frame.nodes[e.target_id].ring or 0 if e.target_id in frame.nodes else 0
        if a is None or b is None:
            continue
        is_path = e.source_id in path_ids and e.target_id in path_ids
        (path_r if is_path else edge_r).extend([ra, rb, None])
        (path_theta if is_path else edge_theta).extend([a, b, None])
    if edge_r:
        fig.add_trace(go.Scatterpolar(
            r=edge_r, theta=edge_theta, mode="lines",
            line=dict(width=1, color="#94a3b855"), hoverinfo="skip",
            showlegend=False,
        ))
    if path_r:
        fig.add_trace(go.Scatterpolar(
            r=path_r, theta=path_theta, mode="lines",
            line=dict(width=3, color=PATH_COLOR), name="retrieval path",
        ))

    # nodes grouped by type so the legend is readable
    by_type: Dict[str, List] = {}
    for nid, n in frame.nodes.items():
        by_type.setdefault(n.memory_type.value, []).append(nid)
    for tname, ids in sorted(by_type.items()):
        fig.add_trace(go.Scatterpolar(
            r=[(frame.nodes[i].ring if frame.nodes[i].ring is not None else 0) for i in ids],
            theta=[angles[i] for i in ids],
            mode="markers",
            marker=dict(
                size=[10 + 18 * (frame.nodes[i].importance or 0.5) for i in ids],
                color=TYPE_COLORS.get(tname, "#64748b"),
                line=dict(width=[2 if i == selected_id or i in path_ids else 0.5 for i in ids],
                          color=[PATH_COLOR if i in path_ids else "#fff" for i in ids]),
                opacity=0.9,
            ),
            name=tname,
            customdata=[[i, frame.nodes[i].concept,
                         frame.nodes[i].ring, frame.nodes[i].sector,
                         round(frame.nodes[i].confidence, 2),
                         round(frame.nodes[i].importance, 2),
                         (frame.nodes[i].summary or "")[:140]] for i in ids],
            hovertemplate=("<b>%{customdata[1]}</b><br>type: " + tname +
                           "<br>ring: %{customdata[2]} · sector: %{customdata[3]}"
                           "<br>conf: %{customdata[4]} · imp: %{customdata[5]}"
                           "<br>%{customdata[6]}<extra>%{customdata[0]}</extra>"),
        ))
    return fig
