import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import ast
import csv
import os

filePath = "tech-as-topology/tech-as-topology-visual.edges"
outputCsvPath = "optimization_results.csv"

#cost
installCosts = {'fiber': 500.0, 'ethernet': 50.0, 'wireless': 90.0}
typeMap = {'fiber': 0, 'ethernet': 1, 'wireless': 2}

#load
allEdgesRaw = []
allAttrsRaw = []

print(f"loading edges from {filePath}...")
with open(filePath, "r") as f:
    for line in f:
        if not line.strip() or line.startswith("#"):
            continue
        src, dst, attrStr = line.strip().split(maxsplit=2)
        attrDict = ast.literal_eval(attrStr)
        allEdgesRaw.append([int(src)-1, int(dst)-1])
        allAttrsRaw.append(attrDict)

if not allEdgesRaw:
    raise RuntimeError("no valid edges found")

maxLatency = max(d['latency'] for d in allAttrsRaw)
maxBandwidth = max(d['bandwidth'] for d in allAttrsRaw)
maxReliability = max(d.get('reliability', 1.0) for d in allAttrsRaw)

edges = []
edgeFeatures = []
edgeTargets = []

for (src, dst), attr in zip(allEdgesRaw, allAttrsRaw):
    #bidirectional edges
    edges.append([src, dst])
    edges.append([dst, src])

    #normalize feats
    normLatency = attr['latency'] / maxLatency
    normBandwidth = attr['bandwidth'] / maxBandwidth
    normReliability = attr['reliability'] / maxReliability
    connType = float(typeMap[attr['color']])

    #feature vector
    feat = [attr['weight'], normLatency, normBandwidth, normReliability, connType]
    edgeFeatures.append(feat)
    edgeFeatures.append(feat)

    #compute graph "stress" with realistic scaling
    stress = attr['traffic_load'] * (normLatency / (normBandwidth + 1e-6)) / (normReliability + 1e-6)
    stress *= (1 + 5*attr['traffic_load'])  #emphasize high-traffic edges
    edgeTargets.append(stress)
    edgeTargets.append(stress)

edgeIndex = torch.tensor(edges, dtype=torch.long).t().contiguous()
edgeAttr = torch.tensor(edgeFeatures, dtype=torch.float)

#normalize edge features
edgeAttr = (edgeAttr - edgeAttr.mean(dim=0)) / (edgeAttr.std(dim=0) + 1e-6)
edgeY = torch.tensor(edgeTargets, dtype=torch.float).unsqueeze(1)
edgeY = (edgeY - edgeY.mean()) / (edgeY.std() + 1e-6)

numNodes = edgeIndex.max().item() + 1
print(f"graph stats: {numNodes} nodes, {len(allEdgesRaw)} unique edges")

nodeEmbeddingDim = 32
nodeEmbedding = torch.nn.Embedding(numNodes + 10, nodeEmbeddingDim)
x = nodeEmbedding.weight
data = Data(x=x, edge_index=edgeIndex, edge_attr=edgeAttr, y=edgeY)

#gcn
class FlowGCN(torch.nn.Module):
    def __init__(self, numNodes, hiddenDim, edgeFeatureDim=5):
        super().__init__()
        self.nodeEmb = torch.nn.Embedding(numNodes, hiddenDim)
        self.conv1 = GCNConv(hiddenDim, hiddenDim)
        self.conv2 = GCNConv(hiddenDim, hiddenDim)
        self.edgePredictor = torch.nn.Sequential(
            torch.nn.Linear(2*hiddenDim + edgeFeatureDim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, edgeIndex, edgeAttr):
        x = self.nodeEmb.weight
        x = F.relu(self.conv1(x, edgeIndex))
        x = self.conv2(x, edgeIndex)
        srcNodes = x[edgeIndex[0]]
        dstNodes = x[edgeIndex[1]]
        edgeRep = torch.cat([srcNodes, dstNodes, edgeAttr], dim=1)
        out = self.edgePredictor(edgeRep)
        return out, x

#training
hiddenDim = 32
model = FlowGCN(numNodes=numNodes + 10, hiddenDim=hiddenDim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

weights = data.edge_attr[:, 1].unsqueeze(1)  #normalized latency
weights = torch.clamp(weights, 0.01, 1.0)

#reduces learning rate if the loss slows down
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50)

print("\nTraining model . . . (1000 epochs)")
model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    predLoad, _ = model(data.edge_index, data.edge_attr)
    
    loss = (weights * (predLoad - data.y)**2).mean()
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    
    optimizer.step()
    scheduler.step(loss)
    
    if epoch % 100 == 0:
        print(f"epoch {epoch}, loss {loss.item():.4f}")


def makeSimulatedFeatures(tech):
    if tech == 'fiber':
        lat, bw, rel = 2.0, 30000.0, 0.999
    elif tech == 'ethernet':
        lat, bw, rel = 10.0, 1000.0, 0.98
    else:
        lat, bw, rel = 40.0, 150.0, 0.92
    normLat = lat / maxLatency
    normBw = bw / maxBandwidth
    return [0.5, normLat, normBw, rel, float(typeMap[tech])]

def evaluateNewNode(model, data, numNodes, installCosts, maxEdgesPerNode=2):
    print("\n--- running realistic optimization evaluation ---")
    model.eval()
    with torch.no_grad():
        baselinePred, _ = model(data.edge_index, data.edge_attr)
        baselineStress = baselinePred.max().item()
        print(f"current network max stress: {baselineStress:.4f}")

        newNodeId = numNodes
        #initialize new node embedding as mean of existing
        meanEmb = data.x[:numNodes].mean(dim=0, keepdim=True)
        model.nodeEmb.weight.data[newNodeId] = meanEmb[0]

        #select top candidate nodes
        degrees = torch.bincount(data.edge_index[0])
        candidateNodes = torch.topk(degrees, min(10, numNodes)).indices.tolist()

        results = []

        for target in candidateNodes:
            for tech, cost in installCosts.items():
                simFeat = makeSimulatedFeatures(tech)
                simFeatTensor = torch.tensor([simFeat, simFeat], dtype=torch.float)

                newEdges = torch.tensor(
                    [[newNodeId, target], [target, newNodeId]],
                    dtype=torch.long
                ).t()

                updatedEdgeIndex = torch.cat([data.edge_index, newEdges], dim=1)
                updatedEdgeAttr = torch.cat([data.edge_attr, simFeatTensor], dim=0)

                newPred, _ = model(updatedEdgeIndex, updatedEdgeAttr)
                newStress = newPred.max().item()
                improvement = baselineStress - newStress
                roi = improvement / (cost / 1000.0)
                results.append({
                    'connectTo': target + 1,
                    'technology': tech,
                    'cost': cost,
                    'newStress': newStress,
                    'improvement': improvement,
                    'roi': roi
                })

        results.sort(key=lambda x: x['roi'], reverse=True)

        print("-"*70)
        print(f"{'Node':<10} {'Tech':<10} {'Cost(in M)':<10} {'Stress':<10} {'Improve':<10} {'ROI':<10}")
        print("-"*70)
        for r in results:
            print(f"{r['connectTo']:<10} {r['technology']:<10} {r['cost']:<10.1f} "
                  f"{r['newStress']:<10.4f} {r['improvement']:<10.4f} {r['roi']:<10.4f}")

        best = results[0]
        print(f"\nThe best option is connecting new node to node: {best['connectTo']} using {best['technology']}")

        fieldnames = ['connectTo','technology','cost','newStress','improvement','roi']
        with open(outputCsvPath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to: {os.path.abspath(outputCsvPath)}")

evaluateNewNode(model, data, numNodes, installCosts)
