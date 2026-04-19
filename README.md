# 🕸️ Social Network Influence Analysis
### Mini Project #12

Identify the **most influential individuals** in a social network using graph analysis, centrality measures, and community detection.

---

## 📌 Project Objective
> To identify influential individuals in a social network using graph analysis techniques — applying algorithms such as **PageRank** and various **centrality measures** to identify the most influential nodes.

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the dashboard
streamlit run app.py
```

---

## 📂 Project Structure

```
DA MINI/
├── app.py              ← Main Streamlit dashboard
├── graph_builder.py    ← Network construction & dataset generators
├── centrality.py       ← Centrality algorithms & composite scoring
├── visualizer.py       ← Plotly & PyVis charts/graphs
├── requirements.txt    ← Dependencies
└── README.md
```

---

## 🧠 Algorithms Implemented

| Algorithm | Type | Purpose |
|---|---|---|
| **PageRank** | Global | Random-walk based importance (Google's algorithm) |
| **Degree Centrality** | Local | Direct connection count |
| **Betweenness Centrality** | Path | Bridge/broker detection |
| **Closeness Centrality** | Distance | How quickly info spreads |
| **Eigenvector Centrality** | Spectral | Influence from influential neighbours |
| **HITS (Hub & Authority)** | Iterative | Hubs vs. authorities |
| **Katz Centrality** | Walk-based | Generalised adjacency walks |
| **Louvain Community Detection** | Community | Modularity-optimised clustering |

---

## 📊 Datasets

| Dataset | Nodes | Type |
|---|---|---|
| Karate Club (Classic) | 34 | Real benchmark |
| Social Media Network (BA) | 150 | Barabási–Albert synthetic |
| Celebrity Fan Network | 150 | Star + community synthetic |
| Corporate / Small-World | 80 | Watts–Strogatz synthetic |

---

## 📈 Dashboard Features

- **KPI Cards** — Nodes, edges, density, clustering, diameter, communities
- **Top Influencers Panel** — Ranked by composite influence score with animated score bars
- **Centrality Bar Chart** — Top-N nodes for any selected metric
- **Degree Distribution Histogram**
- **Centrality Radar Chart** — Multi-metric comparison for top 5 nodes
- **Interactive PyVis Graph** — Drag, zoom, hover for details; coloured by community
- **Community Detection Table** — Per-community stats and top influencer
- **Centrality Correlation Heatmap** — See which metrics agree/disagree
- **Full Node Table** — All nodes with every centrality score, downloadable as CSV

---

## 🎓 Expected Outcome
Students will determine the **most influential users** in a social network by:
1. Building a graph from a social network dataset
2. Computing multiple centrality measures
3. Combining them into a **composite influence score**
4. Visualising influence patterns and communities interactively
