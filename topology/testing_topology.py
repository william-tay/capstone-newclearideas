"""
% sym positive
% This is the network of connections between autonomous systems of the Internet.
% The nodes are autonomous systems (AS), i.e. collections of connected IP routing prefixes controlled by independent network operators.
% Edges are connections between autonomous systems. Multiple edges may connect two nodes, each representing an individual connection in time.
% Edges are annotated with the timepoint of the connection.
"""

import pandas as pd
import networkx as nx
from datetime import datetime
df = pd.read_csv("../tech-as-topology/tech-as-topology.edges",
                 sep=" ",
                 header=None,
                 names=["node1", "node2", "weight", "timestamp"])

# Build the graph
G = nx.from_pandas_edgelist(
    df,
    source="node1",
    target="node2",
    edge_attr=["weight", "timestamp"],
    create_using=nx.Graph()
)

# Example: print one edge’s data
u, v, data = list(G.edges(data=True))[0]
print(u, v, data)

print(df.head())
