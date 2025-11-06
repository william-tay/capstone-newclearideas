import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

edges = []
weights = []

with open('../tech-as-topology/tech-as-topology.edges') as f:
    for line in f:
        src, dst, w = line.strip().split()
        edges.append([int(src)-1, int(dst)-1]) # 1 base to 0
        weights.append(float(w))

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(1)

num_nodes = max(max(src, dst) for src, dst in edges) + 1 #num nodes from largest index
x = torch.randn((num_nodes, 3)) # =need to implement real features, placeholder for now

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
print(data)


class TrafficGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x

model = TrafficGCN(in_channels=3, hidden_channels=8, out_channels=1)
out = model(data.x, data.edge_index, data.edge_attr)
print(out)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #learning loop temp
target = torch.randn((num_nodes, 1))

for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.mse_loss(out, target)
    loss.backward()
    optimizer.step()

print("Training done. Final loss:", loss.item())

file_path = "../tech-as-topology/tech-as-topology.edges"

edges = []
weights = []

with open(file_path, "r") as f:
    for line in f:
        if not line.strip() or line.startswith("#"):
            continue
        src, dst, w = line.strip().split()
        edges.append([int(src) - 1, int(dst) - 1])
        weights.append(float(w))

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(1)

num_nodes = max(max(src, dst) for src, dst in edges) + 1
x = torch.randn((num_nodes, 3)) #random features for now

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr) #object

print("Graph loaded:")
print(data)
print("Edges:\n", edge_index.t())
print("Edge weights:\n", edge_attr.squeeze())
print("Number of nodes:", num_nodes)

class TrafficGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels) #2 graph layers for convolutional nn

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x

model = TrafficGCN(in_channels=3, hidden_channels=8, out_channels=1)

out = model(data.x, data.edge_index, data.edge_attr)
print("\nPredicted node outputs:")
print(out)