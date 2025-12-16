import torch
import torch.nn.functional as F
import ast
import math
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from collections import defaultdict

#parsing
edges = []
features = []

file_path = "tech-as-topology/tech-as-topology-visual.edges"

color_map = {}
next_color_id = 0

with open(file_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        #split first two tokens and keep the rest as a string
        src, dst, datas = line.split(maxsplit=2)

        #convert string dict to dict
        data_dict = ast.literal_eval(datas)

        #extract numeric features
        feats = {
            "weight": float(data_dict["weight"]),
            "latency": float(data_dict["latency"]),
            "bandwidth": float(data_dict["bandwidth"]),
            "traffic_load": float(data_dict["traffic_load"]),
            "reliability": float(data_dict["reliability"]),
        }

        #encode categorical "color"
        color = data_dict["color"]
        if color not in color_map:
            color_map[color] = next_color_id
            next_color_id += 1
        feats["color"] = float(color_map[color])

        edges.append([int(src) - 1, int(dst) - 1])
        features.append(feats)

#markov transition probs
scores = [0.0] * len(edges)
outbound = defaultdict(list)

for i, feats in enumerate(features):
    w = feats["weight"]
    lat = feats["latency"]
    bw = feats["bandwidth"]
    tl = feats["traffic_load"]
    rel = feats["reliability"]

    #score formula (editable)
    score = (w * rel * (1 - tl)) * math.log(bw + 1) / (lat + 1e-6)

    scores[i] = score
    src, _ = edges[i]
    outbound[src].append((i, score))

#normalize to get probabilities
probability = [0.0] * len(edges)
for src, pairs in outbound.items():
    total = sum(score for (_, score) in pairs)
    for edge_idx, score in pairs:
        probability[edge_idx] = 0.0 if total == 0 else score / total

#graph
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(probability, dtype=torch.float)  #markov weight

num_nodes = max(max(src, dst) for src, dst in edges) + 1
x = torch.randn((num_nodes, 3))  #placeholder node features

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_weight  #edge_weight passed as edge_attr
)

print("Graph loaded:")
print(data)
print("Number of edges:", edge_index.size(1))
print("Color map:", color_map)

#gcn model
class TrafficGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x

#train here
model = TrafficGCN(in_channels=3, hidden_channels=16, out_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
target = torch.randn((num_nodes, 1))  #placeholder targets

for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.mse_loss(out, target)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

print("Training done. Final loss:", loss.item())

#predict here
out = model(data.x, data.edge_index, data.edge_attr)
print("\nPredicted node outputs:")
print(out)
