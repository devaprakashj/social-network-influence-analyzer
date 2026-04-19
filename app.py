"""
app.py
======
Social Network Influence Analysis — Streamlit Dashboard
Mini Project #12  |  Real-Time + Real Data Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import time

from graph_builder import DATASETS, graph_summary, get_node_dataframe
from centrality    import (compute_all_centralities, detect_communities,
                            composite_influence_score, top_influencers)
from visualizer    import (plot_degree_distribution, plot_centrality_bar,
                            plot_radar_comparison, plot_community_pie,
                            plot_score_scatter, plot_centrality_heatmap,
                            build_pyvis_graph, PALETTE)
from data_fetcher  import fetch_github_network, fetch_reddit_network

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Social Network Influence Analyzer",
    page_icon  = "\U0001F578",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─── Logo Helper ─────────────────────────────────────────────────────────────
import base64
import os

def get_base64_img(img_path):
    if os.path.exists(img_path):
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

LOGO_BASE64 = get_base64_img("Rit-logo.png")

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

/* Force Sidebar Toggle Visibility */
[data-testid="stHeader"] { background: #0a0a0f !important; visibility: visible !important; display: flex !important; }
[data-testid="stHeader"] button { background-color: #00d4aa22 !important; color: #00d4aa !important; border: 1px solid #00d4aa55 !important; }

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif !important; }
.main { background: #0a0a0f; }
.block-container {
    padding-top: 5rem !important; 
    padding-bottom: 2rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
}

/* ── Header ── */
.app-header { padding: 0 0 14px 0; border-bottom: 1px solid #1e1e3a; margin-bottom: 10px; }
.app-title {
    font-size: 1.9rem; font-weight: 800;
    background: linear-gradient(90deg, #00d4aa, #7c3aed, #ffa502);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    display: inline-block;
}

/* ── KPI card ── */
.kpi-card {
    background: linear-gradient(135deg, #111128, #0f0f1f);
    border: 1px solid #1e1e3a; border-radius: 14px;
    padding: 16px 18px; position: relative; overflow: hidden;
    transition: border-color .3s, transform .15s;
    margin-bottom: 8px;
}
.kpi-card:hover { border-color: #00d4aa55; transform: translateY(-2px); }
.kpi-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, #00d4aa, #7c3aed);
}
.kpi-label { color:#667; font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; }
.kpi-value { color:#fff; font-size:1.5rem; font-weight:700; font-family:'JetBrains Mono'; }
.kpi-sub   { color:#666; font-size:.76rem; margin-top:2px; }

/* ── Influencer card ── */
.inf-card {
    background:#111128; border:1px solid #1e1e3a; border-radius:12px;
    padding:14px 18px; margin-bottom:8px; display:flex;
    align-items:center; gap:16px; flex-wrap: wrap;
    transition: border-color .25s;
}
.inf-card:hover { border-color:#7c3aed55; }
.inf-rank { font-size:1.5rem; font-weight:800; font-family:'JetBrains Mono'; min-width:38px; }
.inf-name { font-weight:600; color:#eee; font-size:.95rem; word-break:break-word; }
.inf-meta { color:#666; font-size:.76rem; flex-wrap:wrap; }
.inf-badge {
    margin-left:auto; background:#1e1e3a; border-radius:20px;
    padding:3px 12px; font-size:.8rem; font-weight:700;
    font-family:'JetBrains Mono'; color:#00d4aa; border:1px solid #00d4aa33;
    white-space:nowrap;
}
.score-bar-bg  { background:#1e1e3a; border-radius:6px; height:5px; margin-top:5px; width:100%; }
.score-bar-fill{ height:100%; border-radius:6px;
    background: linear-gradient(90deg,#7c3aed,#00d4aa); }

/* ── Section header ── */
.sh { color:#00d4aa; font-size:.9rem; font-weight:600; text-transform:uppercase;
      letter-spacing:.1em; margin:16px 0 6px 0;
      border-left:3px solid #7c3aed; padding-left:10px; }

/* ── Live log ── */
.live-log {
    background:#060610; border:1px solid #1e1e3a; border-radius:10px;
    padding:14px 16px; font-family:'JetBrains Mono'; font-size:.78rem;
    color:#aaa; height:260px; overflow-y:auto; line-height:1.7;
}
.log-node   { color:#00d4aa; }
.log-edge   { color:#7c3aed; }
.log-status { color:#aaa; }
.log-done   { color:#ffa502; font-weight:700; }
.log-error  { color:#ff4757; }

/* ── Footer ── */
.custom-footer {{
    background: #0f0f1a;
    border-top: 1px solid #1e1e3a;
    padding: 24px 3rem;
    margin-top: 40px;
}}
.footer-grid {{
    display: grid;
    grid-template-columns: auto 1fr auto;
    align-items: center;
    gap: 32px;
}}
.footer-logo img {{ height: 60px; filter: grayscale(1) invert(1) brightness(2); }}
.footer-center {{ text-align: center; color: #555; font-size: 0.75rem; line-height: 1.6; }}
.footer-right {{ text-align: right; color: #aaa; font-size: 0.8rem; line-height: 1.6; }}
.footer-right b {{ color: #00d4aa; }}

@media (max-width: 768px) {{
    .footer-grid {{ grid-template-columns: 1fr; text-align: center; gap: 20px; }}
    .footer-right {{ text-align: center; }}
    .footer-logo img {{ height: 45px; }}
}}

/* ── Report styles ── */
.rpt-page {
    background: #0f0f1a; border: 1px solid #1e1e3a; border-radius: 16px;
    padding: 40px 48px; max-width: 860px; margin: 0 auto;
    font-size: .93rem; line-height: 1.85; color: #ccc;
}
.rpt-cover { text-align:center; padding: 36px 0 28px 0; border-bottom: 2px solid #1e1e3a; margin-bottom: 28px; }
.rpt-title {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(90deg, #00d4aa, #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    display: inline-block; margin-bottom: 6px;
}
.rpt-sub   { color: #666; font-size: .85rem; margin-top: 4px; }
.rpt-meta  { margin-top: 20px; display: inline-block; text-align: left;
             background: #111128; border: 1px solid #1e1e3a;
             border-radius: 10px; padding: 14px 24px; font-size: .82rem;
             width: 100%; box-sizing: border-box; }
.rpt-meta td { padding: 4px 12px 4px 0; color: #aaa; word-break: break-word; }
.rpt-meta td:first-child { color: #555; font-size: .76rem; text-transform: uppercase; letter-spacing:.08em; white-space: nowrap; }
.rpt-h1 { font-size: 1.25rem; font-weight: 700; color: #fff; border-left: 4px solid #00d4aa; padding-left: 14px; margin: 32px 0 12px 0; }
.rpt-h2 { font-size: 1rem; font-weight: 600; color: #00d4aa; margin: 20px 0 6px 0; }
.rpt-formula {
    background: #060610; border: 1px solid #1e1e3a; border-radius: 8px;
    padding: 10px 18px; font-family: 'JetBrains Mono'; font-size: .82rem;
    color: #7c3aed; margin: 8px 0 12px 0; overflow-x: auto; white-space: nowrap;
}
.rpt-table { width:100%; border-collapse:collapse; margin:10px 0 18px 0; font-size:.83rem; display:block; overflow-x:auto; }
.rpt-table th { background:#111128; color:#aaa; padding:8px 12px; text-align:left; border-bottom:1px solid #1e1e3a; text-transform:uppercase; font-size:.72rem; letter-spacing:.08em; white-space:nowrap; }
.rpt-table td { padding:7px 12px; border-bottom:1px solid #0f0f1a; color:#bbb; }
.rpt-table tr:hover td { background:#111128; }
.rpt-badge { display:inline-block; padding:2px 10px; border-radius:20px; font-size:.72rem; font-weight:700; background:#1e1e3a; color:#00d4aa; border:1px solid #00d4aa33; margin:2px; }
.rpt-ref { color:#666; font-size:.8rem; margin:4px 0; line-height:1.6; }
.rpt-ref b { color:#aaa; }
.rpt-divider { border:none; border-top:1px solid #1e1e3a; margin:24px 0; }

/* ══════════════════════════════════════════
   MOBILE RESPONSIVE  (≤ 768px)
══════════════════════════════════════════ */
@media (max-width: 768px) {

    /* Tighter container padding on mobile */
    .block-container {
        padding-top: 2.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* Smaller title on mobile */
    .app-title { font-size: 1.3rem !important; }
    .app-header { padding-bottom: 10px; }

    /* KPI cards stack vertically — use 2-per-row via percentage */
    .kpi-card {
        padding: 12px 14px !important;
        border-radius: 10px !important;
    }
    .kpi-value { font-size: 1.2rem !important; }
    .kpi-label { font-size: .65rem !important; }
    .kpi-sub   { font-size: .68rem !important; }

    /* Influencer cards — stack rank + info */
    .inf-card {
        padding: 10px 12px !important;
        gap: 10px !important;
        flex-wrap: wrap !important;
    }
    .inf-rank { font-size: 1.2rem !important; min-width: 30px !important; }
    .inf-name { font-size: .85rem !important; }
    .inf-meta { font-size: .68rem !important; }
    .inf-badge {
        font-size: .72rem !important;
        padding: 2px 8px !important;
        margin-left: 0 !important;
    }

    /* Section headers */
    .sh { font-size: .78rem !important; }

    /* Live log — shorter on mobile */
    .live-log {
        height: 180px !important;
        font-size: .72rem !important;
        padding: 10px 12px !important;
    }

    /* Report page — full width, less padding on mobile */
    .rpt-page {
        padding: 20px 16px !important;
        border-radius: 10px !important;
        font-size: .85rem !important;
    }
    .rpt-title  { font-size: 1.3rem !important; }
    .rpt-h1     { font-size: 1.05rem !important; }
    .rpt-h2     { font-size: .9rem !important; }
    .rpt-formula{ font-size: .75rem !important; padding: 8px 12px !important; }
    .rpt-meta   { padding: 10px 14px !important; font-size: .76rem !important; }
    .rpt-table  { font-size: .75rem !important; }
    .rpt-table th, .rpt-table td { padding: 5px 8px !important; }

    /* Make stProgress bar thicker for touch */
    .stProgress > div { height: 8px !important; }
}

/* ══════════════════════════════════════════
   SMALL MOBILE  (≤ 480px)
══════════════════════════════════════════ */
@media (max-width: 480px) {
    .block-container {
        padding-left: 0.6rem !important;
        padding-right: 0.6rem !important;
    }
    .app-title { font-size: 1.1rem !important; }
    .rpt-page  { padding: 14px 10px !important; }
    .inf-card  { padding: 8px 10px !important; }
}
</style>
""", unsafe_allow_html=True)


# ─── Session state ─────────────────────────────────────────────────────────
for k, v in {
    "G": None, "centralities": None, "partition": None,
    "composite": None, "source_label": "—", "fetch_log": []
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 4px 0;'>
        <span style='font-size:2.2rem;'>\U0001F578</span>
        <div style='color:#00d4aa;font-weight:700;font-size:1rem;'>Network Analyzer</div>
        <div style='color:#555;font-size:.72rem;'>Mini Project #12 · Real-Time Edition</div>
    </div>
    <hr style='border-color:#1e1e3a;margin:8px 0;'>
    """, unsafe_allow_html=True)

    data_source = st.radio(
        "📡 **Data Source**",
        ["🧪 Synthetic Datasets", "🐙 GitHub (Live)", "🟠 Reddit (Live)"],
        key="data_source_radio"
    )

    st.markdown("<hr style='border-color:#1e1e3a;margin:8px 0;'>", unsafe_allow_html=True)

    # ── Synthetic ──
    if data_source == "🧪 Synthetic Datasets":
        st.markdown("**📂 Dataset**")
        dataset_name = st.selectbox("Dataset", list(DATASETS.keys()),
                                    label_visibility="collapsed", key="ds_sel")

    # ── GitHub ──
    elif data_source == "🐙 GitHub (Live)":
        st.markdown("**👤 Seed User**")
        gh_user  = st.text_input("GitHub username", value="torvalds",
                                  label_visibility="collapsed", key="gh_user")
        gh_depth = st.slider("BFS depth", 1, 3, 2, key="gh_depth",
                              help="Depth 1 = direct followers only\nDepth 2 = followers of followers")
        gh_max   = st.slider("Max nodes", 20, 150, 60, key="gh_max")
        gh_token = st.text_input("GitHub PAT (optional — raises limit to 5000/hr)",
                                  type="password", key="gh_token",
                                  placeholder="ghp_xxxx…")
        st.caption("No token = 60 req/hr (enough for ~50 nodes).")

    # ── Reddit ──
    else:
        st.markdown("**📌 Subreddit**")
        rd_sub   = st.text_input("Subreddit name", value="Python",
                                  label_visibility="collapsed", key="rd_sub")
        rd_posts = st.slider("Posts to scrape", 5, 50, 15, key="rd_posts")
        st.caption("Users who comment in the same post are connected.")

    st.markdown("<hr style='border-color:#1e1e3a;margin:8px 0;'>", unsafe_allow_html=True)
    st.markdown("**📊 Display Settings**")
    top_k  = st.slider("Top influencers", 5, 20, 10, key="top_k")
    metric = st.selectbox("Primary metric",
                           ["PageRank", "Degree Centrality", "Betweenness Centrality",
                            "Closeness Centrality", "Eigenvector Centrality",
                            "Hub Score", "Authority Score"], key="metric_sel")

    st.markdown("<hr style='border-color:#1e1e3a;margin:8px 0;'>", unsafe_allow_html=True)
    run_btn = st.button("🔍  Fetch & Analyse", use_container_width=True, type="primary")

    st.markdown("""
    <div style='margin-top:16px;padding:10px;background:#111128;
         border:1px solid #1e1e3a;border-radius:10px;font-size:.73rem;color:#555;'>
    <b style='color:#00d4aa;'>Algorithms:</b> PageRank · Betweenness ·
    Eigenvector · Closeness · Degree · HITS · Katz · Louvain
    </div>
    """, unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class='app-header'>
    <span class='app-title'>\U0001F578 Social Network Influence Analyzer</span><br>
    <span style='color:#555;font-size:.82rem;'>
        Real-Time Data · PageRank · Centrality Measures · Community Detection &nbsp;·&nbsp; Mini Project #12
    </span>
</div>
""", unsafe_allow_html=True)


# ─── Fetch / Build graph ──────────────────────────────────────────────────────
def run_analysis(G: nx.Graph, label: str):
    """Run centrality + community on a completed graph and store in session."""
    with st.spinner("Running centrality algorithms…"):
        cents = compute_all_centralities(G)
        part  = detect_communities(G)
        comp  = composite_influence_score(cents, G)
    st.session_state.G            = G
    st.session_state.centralities = cents
    st.session_state.partition    = part
    st.session_state.composite    = comp
    st.session_state.source_label = label


if run_btn:
    st.session_state.fetch_log = []

    # ── Synthetic ──────────────────────────────────────────────────────────
    if data_source == "🧪 Synthetic Datasets":
        with st.spinner(f"Building {dataset_name}…"):
            G = DATASETS[dataset_name]()
        run_analysis(G, f"Synthetic — {dataset_name}")
        st.success(f"✅ **{G.number_of_nodes()} nodes**, **{G.number_of_edges()} edges**")

    # ── GitHub Live ────────────────────────────────────────────────────────
    elif data_source == "🐙 GitHub (Live)":
        st.markdown(f"<div class='sh'>🐙 Live GitHub Fetch — **{gh_user}**</div>",
                    unsafe_allow_html=True)

        log_box   = st.empty()
        stat_cols = st.columns(3)
        node_stat = stat_cols[0].empty()
        edge_stat = stat_cols[1].empty()
        call_stat = stat_cols[2].empty()

        log_lines       = []
        final_G         = nx.Graph()
        node_count      = 0
        edge_count      = 0
        api_call_count  = 0

        gen = fetch_github_network(
            seed_user    = gh_user.strip(),
            depth        = gh_depth,
            max_nodes    = gh_max,
            github_token = gh_token.strip(),
        )

        for event in gen:
            etype = event["type"]
            msg   = event["msg"]
            G_now = event["G"]

            if G_now is not None:
                final_G    = G_now
                node_count = G_now.number_of_nodes()
                edge_count = G_now.number_of_edges()

            # Live counters
            node_stat.markdown(f"""
            <div class='kpi-card' style='padding:10px 14px;'>
                <div class='kpi-label'>Nodes fetched</div>
                <div class='kpi-value' style='font-size:1.3rem;'>{node_count}</div>
            </div>""", unsafe_allow_html=True)

            edge_stat.markdown(f"""
            <div class='kpi-card' style='padding:10px 14px;'>
                <div class='kpi-label'>Edges found</div>
                <div class='kpi-value' style='font-size:1.3rem;'>{edge_count}</div>
            </div>""", unsafe_allow_html=True)

            # Log
            css = {"node": "log-node", "edge": "log-edge",
                   "done": "log-done", "error": "log-error"}.get(etype, "log-status")
            log_lines.append(f"<span class='{css}'>{msg}</span>")
            log_html  = "<br>".join(log_lines[-40:])  # last 40 lines
            log_box.markdown(
                f"<div class='live-log'>{log_html}</div>",
                unsafe_allow_html=True
            )

            if etype == "done":
                st.success(msg)
                break
            if etype == "error":
                st.warning(msg)
                if node_count < 2:
                    st.stop()
                break

        st.session_state.fetch_log = log_lines
        if final_G.number_of_nodes() >= 2:
            run_analysis(final_G, f"GitHub — {gh_user}")
        else:
            st.error("Not enough nodes to analyse. Check username or try a different seed user.")
            st.stop()

    # ── Reddit Live ────────────────────────────────────────────────────────
    else:
        sub_name = rd_sub.strip().lstrip("r/")
        st.markdown(f"<div class='sh'>🟠 Live Reddit Fetch — **r/{sub_name}**</div>",
                    unsafe_allow_html=True)

        log_box   = st.empty()
        stat_cols = st.columns(3)
        node_stat = stat_cols[0].empty()
        edge_stat = stat_cols[1].empty()
        post_stat = stat_cols[2].empty()

        log_lines  = []
        final_G    = nx.Graph()
        node_count = 0
        edge_count = 0
        post_count = 0

        gen = fetch_reddit_network(
            subreddit   = sub_name,
            post_limit  = rd_posts,
            min_comments= 2,
        )

        for event in gen:
            etype = event["type"]
            msg   = event["msg"]
            G_now = event["G"]

            if G_now is not None:
                final_G    = G_now
                node_count = G_now.number_of_nodes()
                edge_count = G_now.number_of_edges()

            if "Post " in msg:
                try:
                    post_count = int(msg.split("Post ")[1].split("/")[0])
                except Exception:
                    pass

            node_stat.markdown(f"""
            <div class='kpi-card' style='padding:10px 14px;'>
                <div class='kpi-label'>Users found</div>
                <div class='kpi-value' style='font-size:1.3rem;'>{node_count}</div>
            </div>""", unsafe_allow_html=True)

            edge_stat.markdown(f"""
            <div class='kpi-card' style='padding:10px 14px;'>
                <div class='kpi-label'>Co-comment links</div>
                <div class='kpi-value' style='font-size:1.3rem;'>{edge_count}</div>
            </div>""", unsafe_allow_html=True)

            post_stat.markdown(f"""
            <div class='kpi-card' style='padding:10px 14px;'>
                <div class='kpi-label'>Posts scraped</div>
                <div class='kpi-value' style='font-size:1.3rem;'>{post_count}</div>
            </div>""", unsafe_allow_html=True)

            css = {"node": "log-node", "edge": "log-edge",
                   "done": "log-done", "error": "log-error"}.get(etype, "log-status")
            log_lines.append(f"<span class='{css}'>{msg}</span>")
            log_html = "<br>".join(log_lines[-40:])
            log_box.markdown(
                f"<div class='live-log'>{log_html}</div>",
                unsafe_allow_html=True
            )

            if etype == "done":
                st.success(msg)
                break
            if etype == "error":
                st.warning(msg)
                if node_count < 2:
                    st.stop()
                break

        st.session_state.fetch_log = log_lines
        if final_G.number_of_nodes() >= 2:
            run_analysis(final_G, f"Reddit — r/{sub_name}")
        else:
            st.error("Not enough data. Try a more active subreddit or increase post count.")
            st.stop()


# ─── Guard ───────────────────────────────────────────────────────────────────
G     = st.session_state.G
cents = st.session_state.centralities
part  = st.session_state.partition
comp  = st.session_state.composite
label = st.session_state.source_label

if G is None:
    st.markdown("""
    <div style='text-align:center;padding:60px 0;'>
        <div style='font-size:4rem;'>🕸️</div>
        <div style='color:#aaa;font-size:1.1rem;margin-top:12px;'>
            Choose a data source in the sidebar and click <b>Fetch & Analyse</b>.
        </div>
        <div style='color:#555;font-size:.85rem;margin-top:8px;'>
            Use <b>Synthetic</b> for instant results · <b>GitHub</b> or <b>Reddit</b> for real live data
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── KPI cards ───────────────────────────────────────────────────────────────
summary  = graph_summary(G)
n_comms  = len(set(part.values()))

src_color = {"Synthetic": "#7c3aed", "GitHub": "#2ed573", "Reddit": "#ff6348"}
src_key   = label.split("—")[0].strip() if "—" in label else "Synthetic"
badge_col = src_color.get(src_key.split()[0], "#aaa")

st.markdown(f"""
<div style='margin-bottom:10px;'>
    <span style='background:#111128;border:1px solid {badge_col}44;color:{badge_col};
          border-radius:20px;padding:3px 14px;font-size:.78rem;font-weight:700;'>
        📡 {label}
    </span>
</div>""", unsafe_allow_html=True)

cols = st.columns(5)
kpis = [("Nodes", summary["Nodes"], "Users in network"),
        ("Edges", summary["Edges"], "Total connections"),
        ("Avg Degree", summary["Avg Degree"], "Mean connections"),
        ("Clustering", summary["Avg Clustering"], "Community strength"),
        ("Communities", n_comms, "Louvain groups")]
for col, (lbl, val, sub) in zip(cols, kpis):
    col.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>{lbl}</div>
        <div class='kpi-value'>{val}</div>
        <div class='kpi-sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")
c1, c2, c3, c4 = st.columns(4)
for col, (lbl, val, sub) in zip([c1,c2,c3,c4], [
    ("Density",    summary["Density"],    "Edges / max possible"),
    ("Max Degree", summary["Max Degree"], "Most connected node"),
    ("Connected?", "Yes" if summary["Is Connected"] else f"No ({summary['Components']} parts)", "Connectivity"),
    ("Diameter",   summary["Diameter"],   "Longest shortest path"),
]):
    col.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>{lbl}</div>
        <div class='kpi-value'>{val}</div>
        <div class='kpi-sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏆 Top Influencers",
    "📊 Centrality Analysis",
    "🌐 Interactive Graph",
    "📡 Communities",
    "📋 Full Table",
    "📜 Fetch Log",
    "📄 Project Report",
])


# ── TAB 1: Top Influencers ──────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='sh'>Top Influential Nodes — Composite Score</div>",
                unsafe_allow_html=True)
    st.caption("Weighted: PageRank 35% · Betweenness 25% · Eigenvector 25% · Degree 15%")

    influencers = top_influencers(comp, G, k=top_k)
    rank_colors = ["#ffa502","#aaa","#cd7f32"] + ["#444"]*100
    rank_emojis = ["🥇","🥈","🥉"] + [f"#{i}" for i in range(4,100)]

    for inf in influencers:
        r      = inf["rank"]-1
        color  = rank_colors[min(r, len(rank_colors)-1)]
        emoji  = rank_emojis[min(r, len(rank_emojis)-1)]
        fill_w = int(inf["score"])
        plat   = inf.get("platform","—")
        foll   = f"{inf.get('followers',0):,}"
        st.markdown(f"""
        <div class='inf-card'>
            <div class='inf-rank' style='color:{color};'>{emoji}</div>
            <div style='flex:1;'>
                <div class='inf-name'>{inf["username"]}</div>
                <div class='inf-meta'>{plat} &nbsp;·&nbsp; {foll} followers &nbsp;·&nbsp; Degree: {inf["degree"]}</div>
                <div class='score-bar-bg'>
                    <div class='score-bar-fill' style='width:{fill_w}%;'></div>
                </div>
            </div>
            <div class='inf-badge'>{inf["score"]:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sh'>Centrality Radar — Top 5</div>", unsafe_allow_html=True)
    top5 = [inf["node"] for inf in influencers[:5]]
    st.plotly_chart(plot_radar_comparison(cents, top5, G), use_container_width=True)


# ── TAB 2: Centrality Analysis ──────────────────────────────────────────────
with tab2:
    cl, cr = st.columns(2)
    with cl:
        st.markdown("<div class='sh'>Centrality Bar Chart</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_centrality_bar(cents, G, metric, top_k), use_container_width=True)
    with cr:
        st.markdown("<div class='sh'>Degree Distribution</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_degree_distribution(G), use_container_width=True)

    st.markdown("<div class='sh'>Influence Score vs Degree</div>", unsafe_allow_html=True)
    st.plotly_chart(plot_score_scatter(comp, G, part), use_container_width=True)

    st.markdown("<div class='sh'>Centrality Correlation Heatmap</div>", unsafe_allow_html=True)
    st.plotly_chart(plot_centrality_heatmap(cents, G), use_container_width=True)


# ── TAB 3: Interactive PyVis ────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='sh'>Interactive Network Graph</div>", unsafe_allow_html=True)
    st.caption("Node size = influence · Colour = community · Hover for details · Drag to explore")
    with st.spinner("Rendering…"):
        html_str = build_pyvis_graph(G, comp, part)
    st.components.v1.html(html_str, height=580, scrolling=False)
    st.info("💡 Scroll to zoom · Drag nodes · Labels show on high-influence nodes only")


# ── TAB 4: Communities ──────────────────────────────────────────────────────
with tab4:
    from collections import Counter
    counts = Counter(part.values())
    st.markdown(f"<div class='sh'>Detected {n_comms} Communities (Louvain)</div>",
                unsafe_allow_html=True)

    cl2, cr2 = st.columns([1,1])
    with cl2:
        st.plotly_chart(plot_community_pie(part), use_container_width=True)
    with cr2:
        comm_data = []
        for cid in sorted(counts):
            members  = [n for n,c in part.items() if c==cid]
            top_node = max(members, key=lambda n: comp.get(n,0))
            top_lbl  = G.nodes[top_node].get("username", f"Node {top_node}")
            avg_deg  = round(np.mean([G.degree(n) for n in members]),2)
            comm_data.append({
                "Community"    : f"Community {cid}",
                "Members"      : counts[cid],
                "Avg Degree"   : avg_deg,
                "Top Influencer": top_lbl,
                "Top Score %"  : round(comp.get(top_node,0)*100, 2),
            })
        comm_df = pd.DataFrame(comm_data)
        st.dataframe(comm_df, use_container_width=True, hide_index=True,
                     column_config={"Top Score %": st.column_config.ProgressColumn(
                         "Top Score %", format="%.1f%%", min_value=0, max_value=100)})

    import plotly.express as px
    fig_bar = px.bar(comm_df, x="Community", y="Members", color="Top Score %",
                     color_continuous_scale=["#1e1e3a","#7c3aed","#00d4aa"],
                     text="Members")
    fig_bar.update_layout(
        plot_bgcolor="#0a0a0f", paper_bgcolor="#111128",
        font=dict(color="#aaa", family="Space Grotesk"),
        xaxis=dict(gridcolor="#1e1e3a"), yaxis=dict(gridcolor="#1e1e3a", title="Members"),
        height=300, margin=dict(l=40,r=40,t=20,b=60))
    st.plotly_chart(fig_bar, use_container_width=True)


# ── TAB 5: Full Table ───────────────────────────────────────────────────────
with tab5:
    st.markdown("<div class='sh'>All Nodes — Ranked by Influence</div>",
                unsafe_allow_html=True)
    node_df = get_node_dataframe(G, cents)
    node_df["Composite Score %"] = node_df["Node ID"].map(
        lambda n: round(comp.get(n,0)*100, 2))
    node_df["Community"] = node_df["Node ID"].map(lambda n: f"C-{part.get(n,'?')}")
    node_df = node_df.sort_values("Composite Score %", ascending=False).reset_index(drop=True)
    node_df.index += 1

    st.dataframe(node_df, use_container_width=True, height=480,
                 column_config={
                     "Composite Score %": st.column_config.ProgressColumn(
                         "Composite Score %", format="%.2f%%", min_value=0, max_value=100),
                     "PageRank"             : st.column_config.NumberColumn(format="%.5f"),
                     "Betweenness Centrality": st.column_config.NumberColumn("Betweenness", format="%.5f"),
                     "Closeness Centrality" : st.column_config.NumberColumn("Closeness",    format="%.5f"),
                     "Eigenvector Centrality": st.column_config.NumberColumn("Eigenvector", format="%.5f"),
                 })
    csv = node_df.to_csv(index=True)
    st.download_button("📥 Download CSV", csv, "influence_analysis.csv",
                       "text/csv", use_container_width=True)


# ── TAB 6: Fetch Log ────────────────────────────────────────────────────────
with tab6:
    st.markdown("<div class='sh'>Real-Time Fetch Log</div>", unsafe_allow_html=True)
    logs = st.session_state.get("fetch_log", [])
    if logs:
        html = "<br>".join(logs)
        st.markdown(f"<div class='live-log' style='height:500px;'>{html}</div>",
                    unsafe_allow_html=True)
        st.caption(f"{len(logs)} log entries")
    else:
        st.info("No fetch log yet — use GitHub or Reddit data source to see live logs.")


# ── TAB 7: Project Report ───────────────────────────────────────────────────
with tab7:
    report_html = f"""
<div class='rpt-page'>
<div style='text-align:center;'>
<img src='data:image/png;base64,{LOGO_BASE64}' style='height:100px; margin-bottom:15px; filter:brightness(1.5);'>
<div style='font-size:1rem; font-weight:700; color:#fff;'>MINI PROJECT REPORT</div>
<div class='rpt-title' style='font-size:1.8rem; margin-top:15px; margin-bottom:15px;'>SOCIAL NETWORK INFLUENCE ANALYSIS<br>USING PAGERANK & CENTRALITY MEASURES</div>
<div style='margin:30px 0;'>
<div style='font-size:0.8rem; color:#666; margin-bottom:5px;'>Submitted by</div>
<div style='font-size:1.1rem; font-weight:700; color:#00d4aa;'>DEVAPRAKASH J</div>
<div style='font-size:0.9rem; font-weight:600; font-family:JetBrains Mono; color:#aaa;'>2117240030025</div>
</div>
<div style='font-size:0.85rem; line-height:1.6; color:#777; margin:40px 0;'>
in partial fulfillment for the award of the degree of<br>
<b style='color:#eee;'>BACHELOR OF ENGINEERING</b><br>
IN<br>
<b style='color:#7c3aed;'>CSE (ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING)</b>
</div>
<div style='border-top: 1px solid #1e1e3a; padding-top:20px; margin-top:40px;'>
<div style='font-weight:700; color:#eee;'>RAJALAKSHMI INSTITUTE OF TECHNOLOGY</div>
<div style='font-size:0.8rem; color:#666;'>KUTHAMBAKKAM, CHENNAI - 600 124</div>
<div style='font-weight:700; color:#aaa; margin-top:10px;'>ANNA UNIVERSITY: CHENNAI 600 025</div>
<div style='font-size:0.8rem; color:#444; margin-top:15px;'>JAN / MAY - 2026</div>
</div>
</div>
<div style='page-break-after: always;'></div>
<hr class='rpt-divider' style='margin:60px 0;'>
<div style='text-align:center;'>
<div style='font-weight:700; color:#aaa; text-transform:uppercase; letter-spacing:2px;'>Anna University: Chennai - 600 025</div>
<div style='font-size:1.6rem; font-weight:800; color:#fff; border-bottom:2px solid #00d4aa; display:inline-block; padding-bottom:5px; margin:40px 0;'>BONAFIDE CERTIFICATE</div>
</div>
<p style='text-align:justify; text-indent:50px; margin-top:30px;'>
This is to certify that this Mini Project report <b>"Social Network Influence Analysis"</b> is the Bonafide work of  
<b style='color:#eee;'>DEVAPRAKASH J (2117240030025)</b>, who carried out the project work under my supervision. 
</p>
<div style='margin-top:80px; display:grid; grid-template-columns: 1fr 1fr; gap:50px;'>
<div style='text-align:left;'>
<div style='font-size:0.8rem; color:#444; margin-bottom:40px; border-bottom:1px solid #333; display:inline-block; width:120px;'>SIGNATURE</div>
<div style='font-weight:700; color:#eee;'>Dr. N. KANAGAVALLI, Ph.D</div>
<div style='font-size:0.85rem; color:#777;'>HEAD OF THE DEPARTMENT</div>
<div style='font-size:0.8rem; color:#555;'>Assistant Professor, CSE (AI & ML)<br>Rajalakshmi Institute of Technology,<br>Chennai – 600 124.</div>
</div>
<div style='text-align:right;'>
<div style='font-size:0.8rem; color:#444; margin-bottom:40px; border-bottom:1px solid #333; display:inline-block; width:120px;'>SIGNATURE</div>
<div style='font-weight:700; color:#eee;'>Mrs. RUBINA BEGAM</div>
<div style='font-size:0.85rem; color:#777;'>SUPERVISOR</div>
<div style='font-size:0.8rem; color:#555;'>Assistant Professor, CSE (AI & ML)<br>Rajalakshmi Institute of Technology,<br>Chennai – 600 124.</div>
</div>
</div>
</div>
"""
    st.markdown(report_html, unsafe_allow_html=True)

    # REST OF THE REPORT (PART 2)
    report_html_p2 = r"""
<div class='rpt-page'>
<hr class='rpt-divider' style='margin-top:100px;'>
<div class='rpt-h1'>1. Abstract</div>
<p>
Social media platforms generate massive interaction data every second, creating vast social graphs where identifying the most influential users is critical for viral marketing, opinion mining, and information diffusion. This project presents a comprehensive <b>Social Network Influence Analysis system</b> developed by <b>Devaprakash J</b> that ingests real-world data from the GitHub and Reddit APIs.
</p>
<p>
The system implements <b>eight centrality measures</b> — PageRank, Degree, Betweenness, Closeness, Eigenvector, HITS, Katz, and Load Centrality — combined into a <b>weighted composite influence score</b>. Community structure is detected using the <b>Louvain modularity optimization</b> algorithm.
</p>
<p><b>Keywords:</b>
<span class='rpt-badge'>Social Network Analysis</span>
<span class='rpt-badge'>PageRank</span>
<span class='rpt-badge'>Centrality Measures</span>
<span class='rpt-badge'>Louvain</span>
<span class='rpt-badge'>Graph Theory</span>
</p>
</div>
"""
    st.markdown(report_html_p2, unsafe_allow_html=True)
    # PART 3: DETAILED REPORT
    report_html_p3 = r"""
<div class='rpt-page'>
<div class='rpt-h1'>2. Introduction</div>
<p>Analyzing social networks to identify key players has become a fundamental problem in data science. <b>Influence</b> relates to the structural position of a node, allowing for rapid information diffusion.</p>
<div class='rpt-h1'>3. Objectives</div>
<ul>
<li>Implement 8 centrality algorithms.</li>
<li>Combine metrics into a weighted composite score.</li>
<li>Detect communities using Louvain algorithm.</li>
<li>Real-time dashboard visualization.</li>
</ul>
<div class='rpt-h1'>7. Results & Analysis</div>
<p>Algorithms correctly identify key nodes in benchmark datasets (e.g., node 0 and 33 in Karate Club). PageRank and Eigenvector are strong predictors of general influence, while Betweenness identifies broker nodes.</p>
</div>
"""
    st.markdown(report_html_p3, unsafe_allow_html=True)
      </table>

      <hr class='rpt-divider'>

      <!-- REFERENCES -->
      <div class='rpt-h1'>9. References</div>
      <p class='rpt-ref'>[1] <b>Brin, S., &amp; Page, L. (1998).</b> The anatomy of a large-scale hypertextual web search engine. <i>Computer Networks, 30</i>(1–7), 107–117.</p>
      <p class='rpt-ref'>[2] <b>Freeman, L. C. (1977).</b> A set of measures of centrality based on betweenness. <i>Sociometry, 40</i>(1), 35–41.</p>
      <p class='rpt-ref'>[3] <b>Bonacich, P. (1972).</b> Factoring and weighting approaches to status scores and clique identification. <i>Journal of Mathematical Sociology, 2</i>(1), 113–120.</p>
      <p class='rpt-ref'>[4] <b>Kleinberg, J. M. (1999).</b> Authoritative sources in a hyperlinked environment. <i>Journal of the ACM, 46</i>(5), 604–632.</p>
      <p class='rpt-ref'>[5] <b>Blondel, V. D. et al. (2008).</b> Fast unfolding of communities in large networks. <i>Journal of Statistical Mechanics,</i> P10008.</p>
      <p class='rpt-ref'>[6] <b>Barabási, A. L., &amp; Albert, R. (1999).</b> Emergence of scaling in random networks. <i>Science, 286</i>(5439), 509–512.</p>
      <p class='rpt-ref'>[7] <b>Kempe, D., Kleinberg, J., &amp; Tardos, É. (2003).</b> Maximizing the spread of influence through a social network. <i>Proc. ACM SIGKDD,</i> 137–146.</p>
      <p class='rpt-ref'>[8] <b>Watts, D. J., &amp; Strogatz, S. H. (1998).</b> Collective dynamics of 'small-world' networks. <i>Nature, 393,</i> 440–442.</p>
      <p class='rpt-ref'>[9] <b>Zachary, W. W. (1977).</b> An information flow model for conflict and fission in small groups. <i>Journal of Anthropological Research, 33</i>(4), 452–473.</p>
      <p class='rpt-ref'>[10] <b>Newman, M. E. J. (2010).</b> <i>Networks: An Introduction.</i> Oxford University Press.</p>
      <p class='rpt-ref'>[11] NetworkX Documentation (2024). https://networkx.org/documentation/stable/</p>
      <p class='rpt-ref'>[12] Streamlit Documentation (2024). https://docs.streamlit.io</p>

    </div>
    """, unsafe_allow_html=True)

    # Download button for report
    st.markdown("<br>", unsafe_allow_html=True)

    import pathlib, os
    _rpt_candidates = [
        pathlib.Path(__file__).parent / "README.md",
        pathlib.Path(r"C:\Users\hp\.gemini\antigravity\brain\f5b97103-8918-487e-9c32-934d234a8c73\project_report.md"),
    ]
    report_text = next(
        (p.read_text(encoding="utf-8") for p in _rpt_candidates if p.exists()),
        "# Social Network Influence Analysis\n\nReport not found — see README.md"
    )

    col_dl1, col_dl2, col_dl3 = st.columns([1,2,1])
    with col_dl2:
        st.download_button(
            label     = "📥 Download Full Report (.md)",
            data      = report_text,
            file_name = "Social_Network_Influence_Analysis_Report.md",
            mime      = "text/markdown",
            use_container_width = True,
        )
        st.caption("Open the .md file in Word / Google Docs / Typora for final formatting before submission.")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='custom-footer'>
    <div class='footer-grid'>
        <div class='footer-logo'>
            <img src='data:image/png;base64,{LOGO_BASE64}'>
        </div>
        <div class='footer-center'>
            <b>RAJALAKSHMI INSTITUTE OF TECHNOLOGY</b><br>
            Department of Computer Science and Engineering (AI & ML)<br>
            Anna University Certified Mini Project &nbsp;·&nbsp; 2026
        </div>
        <div class='footer-right'>
            Submitted by<br>
            <b>DEVAPRAKASH J</b><br>
            Reg No: 2117240030025
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

