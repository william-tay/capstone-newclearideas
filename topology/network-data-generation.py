"""
Intelligent Synthetic Network Generator (Centrality + Connection Type Realism + Weighted Scores)
-----------------------------------------------------------------------------------------------
Generates a realistic undirected network where:
- Node roles (router/end-device) are determined by centrality metrics.
- Edge attributes (latency, bandwidth, reliability, etc.) depend on node types & connection tech.
- Each edge gets a composite "weight" score representing connection quality.
"""

import random
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

# ===== Parameters =====
N_NODES = 350
NETWORK_TYPE = "scale_free"  # Options: "scale_free", "small_world"
OUTPUT_FILE = "synthetic-intelligent-network.edges"

# ===== Helper Functions =====
def make_base_graph(n_nodes: int, graph_type: str):
    """Generate base topology with chosen degree distribution."""
    if graph_type == "scale_free":
        G = nx.scale_free_graph(n_nodes)
        G = nx.Graph(G)  # undirected
    elif graph_type == "small_world":
        G = nx.watts_strogatz_graph(n_nodes, k=4, p=0.3)
        G = nx.Graph(G)
    else:
        raise ValueError("Invalid graph type.")
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def random_timestamp():
    """Generate a synthetic timestamp between 2024 and 2025."""
    start = int(time.mktime(time.strptime("2024-01-01", "%Y-%m-%d")))
    end = int(time.mktime(time.strptime("2025-12-31", "%Y-%m-%d")))
    return random.randint(start, end)

def normalize(value, vmin, vmax):
    """Normalize to 0–1 range."""
    return (value - vmin) / (vmax - vmin + 1e-9)

# ===== Create Base Graph =====
G = make_base_graph(N_NODES, NETWORK_TYPE)

# ===== Determine Node Roles Using Centrality =====
deg_cent = nx.degree_centrality(G)
bet_cent = nx.betweenness_centrality(G)

# Combine degree and betweenness to rank influence
combined_cent = {n: 0.6 * deg_cent[n] + 0.4 * bet_cent[n] for n in G.nodes()}
sorted_nodes = sorted(combined_cent.items(), key=lambda x: x[1], reverse=True)

# Top 30% = routers, rest = endpoints
router_cutoff = int(0.3 * N_NODES)
routers = [n for n, _ in sorted_nodes[:router_cutoff]]
endpoints = [n for n, _ in sorted_nodes[router_cutoff:]]

for n in G.nodes():
    G.nodes[n]["type"] = "router" if n in routers else "endpoint"

# ===== Connection Profiles =====
connection_profiles = {
    "fiber": {
        "latency": (1, 10),
        "bandwidth": (700, 1200),
        "reliability": (0.98, 1.0),
        "traffic_load": (20, 70)
    },
    "ethernet": {
        "latency": (5, 25),
        "bandwidth": (200, 800),
        "reliability": (0.9, 0.98),
        "traffic_load": (10, 40)
    },
    "wireless": {
        "latency": (30, 100),
        "bandwidth": (10, 250),
        "reliability": (0.7, 0.95),
        "traffic_load": (1, 15)
    }
}

def choose_connection_type(u_type, v_type):
    """Decide connection type based on node roles."""
    if u_type == "router" and v_type == "router":
        return random.choices(["fiber", "ethernet"], weights=[0.7, 0.3])[0]
    elif (u_type == "router" and v_type == "endpoint") or (u_type == "endpoint" and v_type == "router"):
        return random.choices(["ethernet", "wireless"], weights=[0.6, 0.4])[0]
    else:
        return "wireless"

# ===== Assign Edge Attributes =====
for u, v in G.edges():
    if u == v:
        continue
    u_type = G.nodes[u]["type"]
    v_type = G.nodes[v]["type"]
    conn_type = choose_connection_type(u_type, v_type)
    prof = connection_profiles[conn_type]

    latency = round(random.uniform(*prof["latency"]), 2)
    bandwidth = round(random.uniform(*prof["bandwidth"]), 2)
    reliability = round(random.uniform(*prof["reliability"]), 3)
    traffic_load = round(random.uniform(*prof["traffic_load"]), 2)
    timestamp = random_timestamp()

    G.edges[u, v].update({
        "color": conn_type,
        "latency": latency,
        "bandwidth": bandwidth,
        "traffic_load": traffic_load,
        "reliability": reliability,
        "timestamp": timestamp
    })

# ===== Compute Weighted Edge Scores =====
# Weighting importance for scoring
w_bandwidth = 0.4
w_reliability = 0.3
w_latency = 0.2
w_traffic = 0.1

for u, v, data in G.edges(data=True):
    conn_type = data["color"]
    prof = connection_profiles[conn_type]

    norm_bandwidth = normalize(data["bandwidth"], *prof["bandwidth"])
    norm_reliability = normalize(data["reliability"], *prof["reliability"])
    norm_latency = normalize(data["latency"], *prof["latency"])
    norm_traffic = normalize(data["traffic_load"], *prof["traffic_load"])

    # Higher = better
    weight = (
        w_bandwidth * norm_bandwidth +
        w_reliability * norm_reliability +
        w_latency * (1 - norm_latency) +
        w_traffic * (1 - norm_traffic)
    )

    data["weight"] = round(weight, 3)

# ===== Export to .edges File =====
edge_rows = []
for u, v, data in G.edges(data=True):
    if u != v:
        edge_rows.append([
            u, v,
            data["latency"],
            data["bandwidth"],
            data["traffic_load"],
            data["reliability"],
            data["timestamp"],
            data["color"],
            data["weight"]
        ])

df = pd.DataFrame(edge_rows, columns=[
    "node1", "node2", "latency", "bandwidth", "traffic_load",
    "reliability", "timestamp", "connection_type", "weight"
])
df.to_csv(OUTPUT_FILE, sep=" ", index=False, header=False)

print(f"Intelligent synthetic network saved to {OUTPUT_FILE}")
print(f"Routers: {len(routers)}, Endpoints: {len(endpoints)}, Edges: {len(G.edges())}")

# # ===== Visualization =====
# plt.figure(figsize=(12, 8))
# pos = nx.spring_layout(G, seed=42)

# # Node colors by role
# node_colors = ["red" if G.nodes[n]["type"] == "router" else "skyblue" for n in G.nodes()]
# nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=350, alpha=0.85)

# # Edge colors by connection type
# edge_colors = []
# for _, _, data in G.edges(data=True):
#     if data["color"] == "fiber":
#         edge_colors.append("green")
#     elif data["color"] == "ethernet":
#         edge_colors.append("orange")
#     else:
#         edge_colors.append("purple")

# nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, alpha=0.6)
# plt.title("Intelligent Synthetic Network (Centrality Roles + Connection Type + Weighted Edges)")
# plt.axis("off")
# plt.tight_layout()
# plt.show()
