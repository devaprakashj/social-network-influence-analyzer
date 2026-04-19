"""
centrality.py
=============
Computes all centrality / influence metrics for the social network.
"""

import networkx as nx
import numpy as np
from typing import Dict


# ─── Core centrality measures ────────────────────────────────────────────────

def compute_all_centralities(G: nx.Graph) -> Dict[str, Dict]:
    """
    Compute a comprehensive set of centrality measures.
    Returns a dict: { measure_name -> { node -> score } }
    """
    results = {}

    # 1. Degree Centrality — how many direct connections
    results["Degree Centrality"] = nx.degree_centrality(G)

    # 2. PageRank — iterative random-walk importance (Google's algorithm)
    results["PageRank"] = nx.pagerank(G, alpha=0.85, max_iter=200)

    # 3. Betweenness Centrality — how often a node lies on shortest paths
    results["Betweenness Centrality"] = nx.betweenness_centrality(G, normalized=True)

    # 4. Closeness Centrality — average distance to all others
    results["Closeness Centrality"] = nx.closeness_centrality(G)

    # 5. Eigenvector Centrality — influential neighbours amplify score
    try:
        results["Eigenvector Centrality"] = nx.eigenvector_centrality(G, max_iter=500)
    except nx.PowerIterationFailedConvergence:
        results["Eigenvector Centrality"] = {n: 0.0 for n in G.nodes()}

    # 6. HITS — hubs vs. authorities
    hubs, auths = nx.hits(G, max_iter=200)
    results["Hub Score"]       = hubs
    results["Authority Score"] = auths

    # 7. Katz Centrality — generalised adjacency walk
    try:
        results["Katz Centrality"] = nx.katz_centrality(G, alpha=0.005, max_iter=1000)
    except Exception:
        results["Katz Centrality"] = {n: 0.0 for n in G.nodes()}

    # 8. Load Centrality (edge-betweenness variant)
    results["Load Centrality"] = nx.load_centrality(G)

    return results


# ─── Community detection ─────────────────────────────────────────────────────

def detect_communities(G: nx.Graph) -> Dict[int, int]:
    """
    Louvain community detection via python-louvain.
    Returns {node -> community_id}.
    Falls back to greedy modularity if unavailable.
    """
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, random_state=42)
        return partition
    except ImportError:
        pass

    # Fallback: greedy modularity
    communities = nx.community.greedy_modularity_communities(G)
    return {node: cid for cid, comm in enumerate(communities) for node in comm}


# ─── Influence ranking ────────────────────────────────────────────────────────

def composite_influence_score(centralities: Dict[str, Dict], G: nx.Graph) -> Dict[int, float]:
    """
    Weighted composite score combining PageRank, Betweenness,
    Eigenvector Centrality, and Degree Centrality.
    Returns {node -> composite_score} normalised to [0, 1].
    """
    weights = {
        "PageRank"              : 0.35,
        "Betweenness Centrality": 0.25,
        "Eigenvector Centrality": 0.25,
        "Degree Centrality"     : 0.15,
    }
    raw = {}
    for node in G.nodes():
        score = sum(
            w * centralities[metric].get(node, 0)
            for metric, w in weights.items()
            if metric in centralities
        )
        raw[node] = score

    max_score = max(raw.values()) or 1.0
    return {n: v / max_score for n, v in raw.items()}


def top_influencers(composite: Dict[int, float], G: nx.Graph, k: int = 10):
    """Return the top-k influential nodes with their scores and attributes."""
    sorted_nodes = sorted(composite, key=composite.get, reverse=True)[:k]
    result = []
    for node in sorted_nodes:
        attr = G.nodes[node]
        result.append({
            "rank"     : len(result) + 1,
            "node"     : node,
            "username" : attr.get("username", f"Node {node}"),
            "platform" : attr.get("platform", "—"),
            "followers": attr.get("followers", 0),
            "score"    : round(composite[node] * 100, 2),
            "degree"   : G.degree(node),
        })
    return result
