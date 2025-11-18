import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.bipartite.centrality import degree_centrality

G_all = nx.read_edgelist(
    "../tech-as-topology/tech-as-topology-visual.edges",
    nodetype=int,
    data=True
)

min_degree_of_centrality = min(nx.degree_centrality(G_all).values())
max_degree_of_centrality = max(nx.degree_centrality(G_all).values())
dc = nx.degree_centrality(G_all)
#print(max(G_all.degree()))
#print(min(G_all.degree()))

#max_node, max_degree = max(G_all.degree(), key=lambda x: x[1]) - DEBUG
#print(f'Node with max degree: {max_node}, Degree: {max_degree}') - DEBUG

for node, data in G_all.nodes(data=True):
    degree = G_all.degree(node)
    #print(f'Node: {node}, Degree: {degree}')

#Calculate centrality and degrees for endpoint and router designations
deg_c = nx.degree_centrality(G_all)
bet_c = nx.betweenness_centrality(G_all, normalized=True)

# Combine into classification
for node in G_all.nodes():
    if deg_c[node] < 0.1 and bet_c[node] < 0.02: #Subjective, had to tweak values per dataset for close results? Looks good in viz
        role = "Endpoint"
    else:
        role = "Router"
    G_all.nodes[node]["role"] = role
    #print(f"Node {node}: Degree={deg_c[node]:.3f}, Betweenness={bet_c[node]:.3f}, Role={role}") - DEBUG


print(f'Minimum Centrality: {min_degree_of_centrality}')
print(f'Maximum Centrality: {max_degree_of_centrality} {max(nx.degree_centrality(G_all).keys())}')

#print(f'Before removal: {G_all}') - DEBUG
nx.write_edgelist(G_all, "../tech-as-topology/tech-as-topology-visual.edges", data=True)
#print(f'After removal: {G_all}') - DEBUG

import random
from datetime import datetime, timedelta
import networkx as nx

# 1) Helpers ---------------------------------------------------------------

def normalize(x, lo, hi):
    if hi == lo:
        return 0.5  # neutral if no range
    return (x - lo) / (hi - lo)

def random_timestamp(days_back=30):
    # random time within the last N days
    now = datetime.now()
    dt = now - timedelta(days=random.uniform(0, days_back), seconds=random.uniform(0, 86400))
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def choose_connection_type(u_type, v_type):
    """Decide connection type based on node roles ("router"/"endpoint")."""
    if u_type == "router" and v_type == "router":
        return random.choices(["fiber", "ethernet"], weights=[0.7, 0.3])[0]
    elif (u_type == "router" and v_type == "endpoint") or (u_type == "endpoint" and v_type == "router"):
        return random.choices(["ethernet", "wireless"], weights=[0.6, 0.4])[0]
    else:
        return "wireless"

# Profiles: (min,max) per metric for each connection type
# Tweak these ranges to fit your domain
connection_profiles = {
    "fiber": {
        "latency": (0.2, 5.0),        # ms
        "bandwidth": (10000, 40000),  # Mbps
        "reliability": (0.995, 0.9999),
        "traffic_load": (0.05, 0.6),  # fraction of capacity used
    },
    "ethernet": {
        "latency": (1.0, 12.0),
        "bandwidth": (100, 1000),
        "reliability": (0.97, 0.995),
        "traffic_load": (0.1, 0.8),
    },
    "wireless": {
        "latency": (5.0, 50.0),
        "bandwidth": (30, 300),
        "reliability": (0.9, 0.98),
        "traffic_load": (0.2, 0.95),
    },
}

# 2) Assign edge attributes ------------------------------------------------
for u, v in G_all.edges():
    if u == v:
        continue

    # Your nodes have "role" = "Endpoint"/"Router". Convert to lower for the chooser.
    u_type = G_all.nodes[u].get("role", "Router").lower()
    v_type = G_all.nodes[v].get("role", "Router").lower()

    conn_type = choose_connection_type(u_type, v_type)
    prof = connection_profiles[conn_type]

    latency = round(random.uniform(*prof["latency"]), 2)
    bandwidth = round(random.uniform(*prof["bandwidth"]), 2)
    reliability = round(random.uniform(*prof["reliability"]), 3)
    traffic_load = round(random.uniform(*prof["traffic_load"]), 2)
    timestamp = random_timestamp()

    G_all.edges[u, v].update({
        "color": conn_type,      # you’re using "color" to store type
        "latency": latency,
        "bandwidth": bandwidth,
        "traffic_load": traffic_load,
        "reliability": reliability,
        "timestamp": timestamp
    })

# 3) Compute a composite weight (higher = better) --------------------------
# Tunable weights
w_bandwidth  = 0.4
w_reliability = 0.3
w_latency    = 0.2
w_traffic    = 0.1

for u, v, data in G_all.edges(data=True):
    conn_type = data["color"]
    prof = connection_profiles[conn_type]

    norm_bandwidth  = normalize(data["bandwidth"],  *prof["bandwidth"])
    norm_reliability = normalize(data["reliability"], *prof["reliability"])
    norm_latency    = normalize(data["latency"],    *prof["latency"])
    norm_traffic    = normalize(data["traffic_load"], *prof["traffic_load"])

    weight = (
        w_bandwidth  * norm_bandwidth +
        w_reliability * norm_reliability +
        w_latency    * (1 - norm_latency) +   # lower latency is better
        w_traffic    * (1 - norm_traffic)     # lower load is better
    )
    data["weight"] = round(weight, 3)

# 4) Save with attributes --------------------------------------------------
# a) Back to edgelist (key=value pairs after u v)
nx.write_edgelist(
    G_all,
    "../tech-as-topology/tech-as-topology-visual.edges",
    data=True
)

# b) (Recommended for Gephi) Export a full-featured format:
# nx.write_gexf(G_all, "../tech-as-topology/tech-as-topology-visual.gexf")
# or edges-only CSV if you prefer:
# import pandas as pd
# rows = [(u, v, *d.values()) for u, v, d in G_all.edges(data=True)]
# pd.DataFrame(
#     [(u, v, d) for u, v, d in G_all.edges(data=True)],
# ).to_csv("../tech-as-topology/tech-as-topology-visual-edges.csv", index=False)
