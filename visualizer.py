"""
visualizer.py
=============
All chart/graph rendering helpers.
Produces Plotly figures and PyVis interactive HTML graphs.
"""

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import tempfile, os
from typing import Dict, List


# ─── Colour helpers ──────────────────────────────────────────────────────────

PALETTE = [
    "#00d4aa", "#7c3aed", "#ffa502", "#ff4757", "#2ed573",
    "#1e90ff", "#ff6b81", "#eccc68", "#a29bfe", "#fd79a8",
    "#00cec9", "#fdcb6e", "#e17055", "#74b9ff", "#55efc4",
]


def score_to_color(score: float) -> str:
    """Map a [0,1] score to a green→red gradient hex string."""
    r = int(255 * (1 - score))
    g = int(200 * score)
    return f"#{r:02x}{g:02x}55"


# ─── Plotly – Degree Distribution ────────────────────────────────────────────

def plot_degree_distribution(G: nx.Graph) -> go.Figure:
    degrees = [d for _, d in G.degree()]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=degrees, nbinsx=30,
        marker=dict(color="#00d4aa", opacity=0.85,
                    line=dict(color="#0a0a0f", width=0.5)),
        name="Degree"
    ))
    fig.update_layout(
        title="Degree Distribution",
        xaxis_title="Degree (connections)", yaxis_title="Count",
        plot_bgcolor="#0a0a0f", paper_bgcolor="#111128",
        font=dict(color="#aaa", family="Space Grotesk"),
        title_font=dict(color="#fff", size=16),
        xaxis=dict(gridcolor="#1e1e3a"), yaxis=dict(gridcolor="#1e1e3a"),
        showlegend=False, height=320, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ─── Plotly – Centrality Bar Chart ───────────────────────────────────────────

def plot_centrality_bar(centralities: Dict[str, Dict], G: nx.Graph,
                        metric: str, top_k: int = 15) -> go.Figure:
    scores = centralities[metric]
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    nodes  = [G.nodes[n].get("username", f"Node {n}") for n, _ in sorted_items]
    values = [v for _, v in sorted_items]

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(nodes))]

    fig = go.Figure(go.Bar(
        x=values[::-1], y=nodes[::-1], orientation="h",
        marker=dict(color=colors[::-1]),
        text=[f"{v:.4f}" for v in values[::-1]], textposition="outside",
        textfont=dict(size=10, color="#aaa"),
    ))
    fig.update_layout(
        title=f"Top {top_k} Nodes — {metric}",
        xaxis_title="Score", yaxis_title="",
        plot_bgcolor="#0a0a0f", paper_bgcolor="#111128",
        font=dict(color="#aaa", family="Space Grotesk"),
        title_font=dict(color="#fff", size=15),
        xaxis=dict(gridcolor="#1e1e3a"), yaxis=dict(gridcolor="#1e1e3a"),
        height=max(320, top_k * 28),
        margin=dict(l=120, r=60, t=50, b=40),
    )
    return fig


# ─── Plotly – Centrality Comparison Radar ────────────────────────────────────

def plot_radar_comparison(centralities: Dict[str, Dict], nodes: List[int],
                          G: nx.Graph) -> go.Figure:
    metrics = ["Degree Centrality", "PageRank", "Betweenness Centrality",
               "Closeness Centrality", "Eigenvector Centrality"]
    fig = go.Figure()

    for node in nodes:
        label  = G.nodes[node].get("username", f"Node {node}")
        vals   = [centralities[m].get(node, 0) for m in metrics]
        # Normalise each metric so max=1
        mx = [max(centralities[m].values()) or 1 for m in metrics]
        norm_vals = [v / mx[i] for i, v in enumerate(vals)]
        norm_vals.append(norm_vals[0])  # close polygon

        fig.add_trace(go.Scatterpolar(
            r=norm_vals,
            theta=metrics + [metrics[0]],
            fill="toself",
            name=label,
            opacity=0.75,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#111128",
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="#1e1e3a", color="#555"),
            angularaxis=dict(gridcolor="#1e1e3a", color="#aaa"),
        ),
        paper_bgcolor="#111128",
        title="Centrality Radar — Top Influencers",
        title_font=dict(color="#fff", size=15),
        font=dict(color="#aaa", family="Space Grotesk"),
        legend=dict(bgcolor="#0a0a0f", bordercolor="#1e1e3a", borderwidth=1),
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


# ─── Plotly – Community Pie Chart ────────────────────────────────────────────

def plot_community_pie(partition: Dict[int, int]) -> go.Figure:
    from collections import Counter
    counts = Counter(partition.values())
    labels = [f"Community {k}" for k in sorted(counts)]
    values = [counts[k] for k in sorted(counts)]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.45,
        marker=dict(colors=PALETTE[:len(labels)],
                    line=dict(color="#0a0a0f", width=2)),
        textinfo="percent+label",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title="Community Membership Distribution",
        paper_bgcolor="#111128",
        font=dict(color="#aaa", family="Space Grotesk"),
        title_font=dict(color="#fff", size=15),
        showlegend=True,
        legend=dict(bgcolor="#0a0a0f", bordercolor="#1e1e3a", borderwidth=1),
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ─── Plotly – Score vs Degree Scatter ────────────────────────────────────────

def plot_score_scatter(composite: Dict[int, float], G: nx.Graph,
                       partition: Dict[int, int]) -> go.Figure:
    import pandas as pd
    rows = []
    for node in G.nodes():
        rows.append({
            "Degree"       : G.degree(node),
            "Influence (%)" : round(composite.get(node, 0) * 100, 2),
            "Username"     : G.nodes[node].get("username", f"Node {node}"),
            "Community"    : f"Community {partition.get(node, 0)}",
            "Size"         : 6 + G.degree(node) * 0.4,
        })
    df = pd.DataFrame(rows)

    fig = px.scatter(
        df,
        x="Degree",
        y="Influence (%)",
        color="Community",
        size="Size",
        hover_name="Username",
        hover_data={"Size": False, "Degree": True, "Influence (%)": True},
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(
        title="Influence Score vs Degree",
        plot_bgcolor="#0a0a0f", paper_bgcolor="#111128",
        font=dict(color="#aaa", family="Space Grotesk"),
        title_font=dict(color="#fff", size=15),
        xaxis=dict(gridcolor="#1e1e3a"), yaxis=dict(gridcolor="#1e1e3a"),
        legend=dict(title="Community", bgcolor="#0a0a0f",
                    bordercolor="#1e1e3a", borderwidth=1),
        height=380,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


# ─── Plotly – Correlation Heatmap of Centralities ────────────────────────────

def plot_centrality_heatmap(centralities: Dict[str, Dict], G: nx.Graph) -> go.Figure:
    import pandas as pd
    keys = list(centralities.keys())
    data = pd.DataFrame({k: centralities[k] for k in keys})
    corr = data.corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, "#ff4757"], [0.5, "#1e1e3a"], [1, "#00d4aa"]],
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
        zmin=-1, zmax=1,
    ))
    fig.update_layout(
        title="Centrality Measures — Correlation Heatmap",
        paper_bgcolor="#111128",
        plot_bgcolor="#0a0a0f",
        font=dict(color="#aaa", family="Space Grotesk", size=10),
        title_font=dict(color="#fff", size=15),
        height=420,
        margin=dict(l=120, r=20, t=60, b=120),
        xaxis=dict(tickangle=-35),
    )
    return fig


# ─── PyVis – Interactive Network Graph (HTML) ─────────────────────────────────

def build_pyvis_graph(G: nx.Graph, composite: Dict[int, float],
                      partition: Dict[int, int]) -> str:
    """
    Build a PyVis interactive HTML network and return the HTML string.
    Node size ∝ composite influence; colour = community.
    """
    net = Network(height="550px", width="100%", bgcolor="#0a0a0f",
                  font_color="#cccccc", notebook=False)
    net.barnes_hut(gravity=-5000, central_gravity=0.3,
                   spring_length=110, spring_strength=0.04,
                   damping=0.09)

    # Limit to at most 300 nodes to keep it responsive
    nodes_to_show = list(G.nodes())
    if len(nodes_to_show) > 300:
        # Show top-N by composite score
        nodes_to_show = sorted(composite, key=composite.get, reverse=True)[:300]

    sub = G.subgraph(nodes_to_show)

    for node in sub.nodes():
        score  = composite.get(node, 0)
        cid    = partition.get(node, 0) % len(PALETTE)
        color  = PALETTE[cid]
        size   = 8 + score * 40
        label  = G.nodes[node].get("username", str(node))
        title  = (f"<b>{label}</b><br>"
                  f"Influence: {score*100:.1f}%<br>"
                  f"Degree: {G.degree(node)}<br>"
                  f"Community: {partition.get(node, '?')}")
        net.add_node(node, label=label if score > 0.3 else "",
                     color=color, size=size, title=title,
                     font={"size": 10 if score > 0.3 else 0, "color": "#fff"})

    for u, v in sub.edges():
        net.add_edge(u, v, color="#1e1e3a", width=0.8)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html",
                                      dir=tempfile.gettempdir())
    tmp.close()
    net.save_graph(tmp.name)
    with open(tmp.name, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(tmp.name)
    return html
