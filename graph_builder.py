"""
graph_builder.py
================
Builds and manages the social network graph.
Supports synthetic datasets and real-world network patterns.
"""

import networkx as nx
import numpy as np
import pandas as pd
import random
from typing import Tuple, Dict

# ─── Seed for reproducibility ────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ─── Dataset generators ──────────────────────────────────────────────────────

def generate_karate_club() -> nx.Graph:
    """Zachary's Karate Club – classic social network benchmark."""
    G = nx.karate_club_graph()
    return G


def generate_social_network(n: int = 100, m: int = 3) -> nx.Graph:
    """
    Barabási–Albert preferential-attachment network.
    Mimics real social media follower graphs (power-law degree distribution).
    """
    G = nx.barabasi_albert_graph(n, m, seed=SEED)
    # Assign synthetic user attributes
    platforms = ["Twitter", "Instagram", "LinkedIn", "TikTok", "Facebook"]
    for node in G.nodes():
        G.nodes[node]["username"] = f"User_{node:04d}"
        G.nodes[node]["platform"] = random.choice(platforms)
        G.nodes[node]["followers"] = random.randint(10, 50_000)
        G.nodes[node]["posts"]     = random.randint(1, 5_000)
    return G


def generate_celebrity_network(n_celebs: int = 10, n_fans: int = 90) -> nx.Graph:
    """Star + community model – celebrities with clusters of fans."""
    G = nx.Graph()
    n_total = n_celebs + n_fans
    # Add celebrity nodes
    for i in range(n_celebs):
        G.add_node(i, username=f"Celebrity_{i}", platform="Instagram",
                   followers=random.randint(100_000, 5_000_000), posts=random.randint(500, 10_000))
    # Add fan nodes
    for i in range(n_celebs, n_total):
        G.add_node(i, username=f"Fan_{i - n_celebs:04d}", platform=random.choice(["Twitter", "TikTok"]),
                   followers=random.randint(10, 5_000), posts=random.randint(1, 500))
        # Each fan connects to 1-3 celebrities
        celebs = random.sample(range(n_celebs), k=min(random.randint(1, 3), n_celebs))
        for c in celebs:
            G.add_edge(i, c)
    # Add cross-celebrity connections
    for i in range(n_celebs):
        for j in range(i + 1, n_celebs):
            if random.random() < 0.6:
                G.add_edge(i, j)
    # Add fan-to-fan connections
    for i in range(n_celebs, n_total):
        for j in range(i + 1, n_total):
            if random.random() < 0.03:
                G.add_edge(i, j)
    return G


def generate_corporate_network() -> nx.Graph:
    """Watts–Strogatz small-world graph, mimicking corporate/academic networks."""
    G = nx.watts_strogatz_graph(80, 6, 0.3, seed=SEED)
    departments = ["Engineering", "Marketing", "Finance", "HR", "Operations", "Research"]
    for node in G.nodes():
        G.add_node(node, username=f"Employee_{node:03d}",
                   department=random.choice(departments),
                   platform="LinkedIn",
                   followers=random.randint(50, 5_000),
                   posts=random.randint(5, 300))
    return G


# ─── Dataset registry ────────────────────────────────────────────────────────

DATASETS = {
    "Karate Club (Classic)"      : generate_karate_club,
    "Social Media Network (BA)"  : lambda: generate_social_network(150, 3),
    "Celebrity Fan Network"      : lambda: generate_celebrity_network(15, 135),
    "Corporate / Small-World"    : generate_corporate_network,
}


# ─── Graph utilities ─────────────────────────────────────────────────────────

def graph_summary(G: nx.Graph) -> Dict:
    """Return basic statistics about the graph."""
    deg_seq = [d for _, d in G.degree()]
    return {
        "Nodes"                : G.number_of_nodes(),
        "Edges"                : G.number_of_edges(),
        "Avg Degree"           : round(np.mean(deg_seq), 3),
        "Max Degree"           : int(np.max(deg_seq)),
        "Density"              : round(nx.density(G), 5),
        "Is Connected"         : nx.is_connected(G),
        "Components"           : nx.number_connected_components(G),
        "Avg Clustering"       : round(nx.average_clustering(G), 4),
        "Diameter"             : _safe_diameter(G),
    }


def _safe_diameter(G: nx.Graph):
    try:
        lcc = G.subgraph(max(nx.connected_components(G), key=len))
        return nx.diameter(lcc)
    except Exception:
        return "N/A"


def get_edge_dataframe(G: nx.Graph) -> pd.DataFrame:
    """Return edges as a DataFrame for display."""
    rows = [{"Source": u, "Target": v} for u, v in G.edges()]
    return pd.DataFrame(rows)


def get_node_dataframe(G: nx.Graph, centralities: Dict[str, Dict]) -> pd.DataFrame:
    """Combine node attributes + centrality scores into a single DataFrame."""
    rows = []
    for node in G.nodes():
        attr = G.nodes[node]
        row  = {"Node ID": node,
                "Username": attr.get("username", str(node)),
                "Degree"  : G.degree(node)}
        for algo, scores in centralities.items():
            row[algo] = round(scores.get(node, 0), 6)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("PageRank" if "PageRank" in df.columns else "Degree",
                          ascending=False).reset_index(drop=True)
