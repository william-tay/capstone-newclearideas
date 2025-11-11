import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.bipartite.centrality import degree_centrality

"""
Network X Section:

DO NOT VISUALIZE IT, VERY SLOW

Can we add direction at all?
Different types of physical connections?

NEED TO VISUALIZE (if you want to easily interpret data returned by GNN)

"""

#G = nx.read_edgelist("../tech-as-topology/tech-as-topology.edges")

#Nodes - end device vs router
#bandwidth, latency, randomize weighting
#centrality -ranking types (router vs end device, etc) , used to explain actions
#prune edges with no connections, 2 nodes 1 edge remove.

'''
labels with data and graph
'''

G = nx.read_edgelist(
"../tech-as-topology/tech-as-topology-mini.edges",
   delimiter=' ',  # Adjust if your delimiter is different (e.g., ',')
   data=(('weight', float), ('color', str)), # Example attributes and their types
   create_using=nx.Graph() # Or nx.DiGraph() if your graph is directed
)


#G = nx.read_edgelist("../tech-as-topology/tech-as-topology-mini.edges", nodetype=int)

print(G) # print out with data=true

edge_labels = {}
# this assumes all the edges have the same labels 'marks' and 'cable_name'
for u, v, data in G.edges(data=True):
    edge_labels[u, v] = f"{data['weight']}\n{data['color']}"
    print(f'Edge: {u}-{v}: {edge_labels[u, v]}')

degree_of_centrality = min(nx.degree_centrality(G).values())

#Specify values in dictionary, since there will be many attributes {weight, latency, etc.}
dc = nx.degree_centrality(G)
least_node = min(dc, key=dc.get("weight"))
print(least_node)
least_value = dc[least_node]

print(least_value)
print(degree_of_centrality)

# counter = 0
#
# #CHALLENGE: Parsing through and identifying high centrality nodes (routers) vs edgepoints
# while counter < 10:
#     counter += 1
#     print(nx.degree_centrality(G)[counter])
#
# nx.draw(G)
# plt.show()

"""
df = pd.read_csv(
    "../tech-as-topology/tech-as-topology.edges",
    sep=" ", header=None, comment="%",
    names=["node1", "node2", "weight", "timestamp", "value"]
)
"""


#df.drop(columns=["weight", "timestamp"], inplace=True)

#df["value"] = (np.random.randint(1, 6, size=len(df)))/10


#print(df["value"])

#df.to_csv("../tech-as-topology/tech-as-topology.edges", sep=" ", index=False, header=False)
