import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

# Load graph
G = load_pickle('./phishing_subgraph.pkl')
print(f"Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

# Extract node features
node_features = {}

for node in G.nodes():
    # Basic degrees
    in_degree = G.in_degree(node)
    out_degree = G.out_degree(node)
    total_degree = in_degree + out_degree
    
    # Transaction amounts (in-strength and out-strength)
    in_strength = sum(G[pred][node][key]["amount"] for pred in G.predecessors(node) for key in G[pred][node])
    out_strength = sum(G[node][succ][key]["amount"] for succ in G.successors(node) for key in G[node][succ])
    total_strength = in_strength + out_strength
    
    # Unique neighbors
    unique_neighbors = len(set(G.predecessors(node)) | set(G.successors(node)))
    
    # Transaction timing
    timestamps = []
    for pred in G.predecessors(node):
        for key in G[pred][node]:
            timestamps.append(G[pred][node][key]["timestamp"])
    for succ in G.successors(node):
        for key in G[node][succ]:
            timestamps.append(G[node][succ][key]["timestamp"])
    
    txn_frequency = (max(timestamps) - min(timestamps)) if len(timestamps) > 1 else 0
    
    # Store features
    node_features[node] = [
        in_degree, out_degree, total_degree,
        in_strength, out_strength, total_strength,
        unique_neighbors, txn_frequency
    ]

# Convert to DataFrame
df_features = pd.DataFrame.from_dict(
    node_features, 
    orient='index', 
    columns=['in_degree', 'out_degree', 'total_degree', 
             'in_strength', 'out_strength', 'total_strength',
             'unique_neighbors', 'txn_frequency']
)

print(f"Features extracted for {len(df_features)} nodes")
print(df_features.head())

# Extract labels from 'isp' field
labels = []
node_list = list(G.nodes())

for node in node_list:
    isp_value = G.nodes[node].get('isp', 0)  # Default to 0 if missing
    labels.append(isp_value)

print(f"Labels: {pd.Series(labels).value_counts()}")

# Create node mapping
node_mapping = {node: i for i, node in enumerate(node_list)}

# Extract edges
edge_index = []
edge_attr = []

for u, v, key, data in G.edges(keys=True, data=True):
    edge_index.append([node_mapping[u], node_mapping[v]])
    edge_attr.append([data.get("amount", 0), data.get("timestamp", 0)])

# Convert to tensors
edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long)
edge_attr = torch.tensor(edge_attr, dtype=torch.float)

# Normalize features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(df_features.values)
node_features_tensor = torch.tensor(features_normalized, dtype=torch.float)

# Convert labels
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Create PyG Data object
data = Data(
    x=node_features_tensor, 
    edge_index=edge_index, 
    edge_attr=edge_attr,
    y=labels_tensor
)

print(f"\nFinal data object:")
print(f"  Nodes: {data.x.shape}")
print(f"  Edges: {data.edge_index.shape}")
print(f"  Labels: {data.y.shape}")
print(f"  Label distribution: {torch.bincount(data.y)}")

# Save the data
torch.save(data, './data_simple.pt')
print("Saved as 'data_simple.pt'")