# Network Traffic Simulation and Graph Neural Network (GCN) Modeling

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-orange)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project simulates **network traffic** on an *Autonomous System (AS)*-level topology and applies a **Graph Convolutional Network (GCN)** model to learn and predict node-level properties such as network load, delay, or congestion.

The system uses:
- **PyTorch Geometric (PyG)** for graph processing  
- **NetworkX** for graph analysis and routing simulation  
- A real-world dataset `tech-as-topology.edges` describing AS-level connectivity

---

## Features

Load `.edges` topology files and build PyTorch Geometric graph objects  
Construct node features and edge weights for graph learning  
Implement and train a **2-layer GCN model** for node-level prediction  
Perform dummy training (regression on random targets)  
Display graph structure, training progress, and model output  

*Upcoming / Planned Features*  

- Dynamic packet routing simulation (discrete-time network model)  
- Integration of per-edge latency and capacity constraints  
- Real traffic generation and congestion visualization  
- Evaluation on real network performance data  

## Example Output

Data(x=[N, 3], edge_index=[2, E], edge_attr=[E, 1])
tensor([[0.0321],
        [-0.1192],
        [0.0577],
        ...])
Training done. Final loss: 0.0234

## Release Notes
Version 1.0 – Initial Graph Construction and GCN Training

Working in this submission:

Graph data loading and preprocessing

Node and edge tensor creation for PyTorch Geometric

Two-layer GCN model implementation and forward propagation

Dummy supervised training loop with MSE loss

Output of model predictions and training diagnostics

Not yet implemented (Next Milestone):

End-to-end packet-level traffic simulation

Dynamic latency and bandwidth constraints

Real-world target variable training (beyond random)

Visualization of graph metrics and network flow
